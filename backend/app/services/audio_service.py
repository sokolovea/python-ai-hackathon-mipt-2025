import os
import time
import datetime
import logging
import numpy as np
from app.config import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import whisper

# Настройка логгирования как в исходном коде
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

import os
import json
import time
import logging
from typing import List, Dict, Optional
from tqdm import tqdm


class TranscriptionPipeline:
    def __init__(
            self, video_path: str,
            audio_path: str = "output_audio.wav",
            chunk_length_ms: int = 30000,
            engine: str = "google",
            whisper_model: str = "base"):
        """
        Инициализация пайплайна транскрибации.

        :param video_path: Путь к видеофайлу.
        :param audio_path: Путь к временному аудиофайлу.
        :param chunk_length_ms: Длина фрагмента аудио в миллисекундах.
        """
        self.video_path = video_path
        self.audio_path = audio_path
        self.chunk_length_ms = chunk_length_ms

        self.engine = engine
        self.whisper_model_name = whisper_model
        if self.engine == "whisper":
            import whisper
            self.whisper = whisper.load_model(self.whisper_model_name)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def extract_audio(self):
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(
            self.audio_path,
            codec="pcm_s16le",
            fps=16000,
            nbytes=2,
            ffmpeg_params=["-ac", "1"]
        )

    def split_audio(self):
        audio = AudioSegment.from_wav(self.audio_path)
        chunks = []
        for i in range(0, len(audio), self.chunk_length_ms):
            chunk = audio[i:i + self.chunk_length_ms]
            chunk_path = f"chunk_{i // self.chunk_length_ms}.wav"
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        return chunks

    def format_milliseconds(self, ms):
        return str(datetime.timedelta(milliseconds=ms))

    def recognize_chunk_google(self, chunk_path, chunk_index, language="ru-RU"):
        recognizer = sr.Recognizer()
        start_ms = chunk_index * self.chunk_length_ms
        end_ms = start_ms + self.chunk_length_ms

        try:
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=language)
                return {
                    "start_time": self.format_milliseconds(start_ms),
                    "end_time": self.format_milliseconds(end_ms),
                    "text": text
                }
        except sr.UnknownValueError:
            logging.warning(f"Не удалось распознать речь в: {chunk_path}")
        except sr.RequestError as e:
            logging.error(f"Ошибка API для {chunk_path}: {e}")
        except Exception as e:
            logging.error(f"Непредвиденная ошибка при обработке {chunk_path}: {e}")

        return {
            "start_time": self.format_milliseconds(start_ms),
            "end_time": self.format_milliseconds(end_ms),
            "text": ""
        }

    def recognize_chunk_whisper(self, chunk_path, chunk_index, language="ru"):
        start_ms = chunk_index * self.chunk_length_ms
        end_ms = start_ms + self.chunk_length_ms

        try:
            result = self.whisper.transcribe(chunk_path, language=language, fp16=False, task="transcribe")
            text = result["text"].strip()

            return {
                "start_time": self.format_milliseconds(start_ms),
                "end_time": self.format_milliseconds(end_ms),
                "text": text
            }

        except Exception as e:
            logging.error(f"Ошибка в {chunk_path} [{self.engine}]: {e}")
            return {
                "start_time": self.format_milliseconds(start_ms),
                "end_time": self.format_milliseconds(end_ms),
                "text": ""
            }

    def transcribe_google(self):
        chunks = self.split_audio()
        transcript_segments = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.recognize_chunk_google, chunk, idx): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Транскрибация"):
                idx = futures[future]
                try:
                    transcript_segments[idx] = future.result()
                except Exception as e:
                    logging.error(f"Ошибка в потоке обработки {idx}: {e}")
                    transcript_segments[idx] = {
                        "start_time": self.format_milliseconds(idx * self.chunk_length_ms),
                        "end_time": self.format_milliseconds((idx + 1) * self.chunk_length_ms),
                        "text": ""
                    }

        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

        full_text = " ".join([seg["text"] for seg in transcript_segments if seg]).strip()
        return transcript_segments, full_text


    def transcribe_whisper(self):
        self.logger.info("[INFO] Распознавание с помощью Whisper без разбивки на чанки")
        result = self.whisper.transcribe(audio=self.audio_path, language="ru", fp16=False, task="transcribe")

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start_time": self.format_milliseconds(seg["start"] * 1000),
                "end_time": self.format_milliseconds(seg["end"] * 1000),
                "text": seg["text"].strip()
            })

        full_text = result["text"].strip()
        return segments, full_text


    def transcribe(self):
        if self.engine == "whisper":
            return self.transcribe_whisper()
        elif self.engine == "google":
            return self.transcribe_google()
        else:
            raise ValueError(f"Неизвестный движок: {self.engine}")


    def save_to_json(self, segments, path="transcription.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logging.info(f"Результат сохранён в {path}")

    def save_to_txt(self, full_text, path="transcription.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logging.info(f"Текст сохранён в {path}")

    def save_to_srt(self, segments, path="transcription.srt"):
        def format_srt_time(time_value):
            if isinstance(time_value, str):
                if not time_value:
                    return "00:00:00"
                base = time_value.split(".", 1)[0]
                parts = base.split(":")
                if len(parts) != 3:
                    return "00:00:00"
                h, m, s = parts
                try:
                    h, m, s = int(h), int(m), int(s)
                except ValueError:
                    return "00:00:00"
                return f"{h:02}:{m:02}:{s:02}"
            if isinstance(time_value, (int, float)):
                total = int(time_value)
                h, rem = divmod(total, 3600)
                m, s = divmod(rem, 60)
                return f"{h:02}:{m:02}:{s:02}"
            return "00:00:00"

        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = format_srt_time(seg["start_time"])
                end = format_srt_time(seg["end_time"])
                text = seg["text"]
                if text:  # пропускаем пустые
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        logging.info(f"Субтитры сохранены в {path}")

    def merge_incomplete_segments(self, segments: List[Dict]) -> List[Dict]:
        merged = []
        i = 0
        while i < len(segments):
            current = segments[i].copy()
            while (
                    i + 1 < len(segments)
                    and not current["text"].endswith(('.', '!', '?'))
                    and segments[i + 1]["text"]
                    and segments[i + 1]["text"][0].islower()
            ):
                next_seg = segments[i + 1]
                current["text"] = current["text"].rstrip() + " " + next_seg["text"].lstrip()
                current["end_time"] = next_seg["end_time"]
                i += 1
            merged.append(current)
            i += 1
        return merged

    def run(self):
        """
        Запускает полный процесс транскрибации: извлечение аудио, разбивка, распознавание речи.
        Обрабатывает ошибки на каждом этапе.
        """
        try:
            start = time.time()
            self.logger.info("[INFO] Извлечение аудио из видео...")
            self.extract_audio()
            self.logger.info(f"[INFO] Аудио извлечено за {time.time() - start:.2f} сек")

            start = time.time()
            self.logger.info("[INFO] Запуск транскрибации...")
            chunked_texts, full_text = self.transcribe()
            self.logger.info("[INFO] Транскрибация завершена.")
            if self.engine == "whisper":
                chunked_texts = self.merge_incomplete_segments(chunked_texts)
            self.logger.info(full_text)
            self.logger.info(f"[INFO] Время транскрибации: {time.time() - start:.2f} сек")

            return chunked_texts, full_text

        except Exception as e:
            self.logger.error(f"[ERROR] Ошибка в процессе транскрибации: {e.with_traceback()}")
            return [], ""

        finally:
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                    self.logger.info("[INFO] Временный аудиофайл удалён.")
                except Exception as e:
                    self.logger.warning(f"[WARNING] Не удалось удалить временный файл: {e}")


