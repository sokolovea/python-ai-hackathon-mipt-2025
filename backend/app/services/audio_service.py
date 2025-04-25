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
    level=logging.DEBUG,
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
            whisper_model: str = "tiny"):
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
        """Конвертирует миллисекунды в формат HH:MM:SS.sss"""
        logger.debug(f"Formatting milliseconds: {ms}, type: {type(ms)}")
        try:
            if isinstance(ms, str):
                if ms in ['00.000', '0.000', ''] or '.' in ms:
                    logger.debug(f"Converting string to float: {ms}")
                    ms = float(ms.replace(',', '.')) * 1000  # Секунды в миллисекунды
                else:
                    ms = int(ms)
            ms = float(ms)
            seconds, milliseconds = divmod(ms, 1000)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка форматирования времени: {ms} -> {e}")
            return "00:00:00.000"

    def recognize_chunk_google(self, chunk_path, chunk_index, language="ru-RU"):
        recognizer = sr.Recognizer()
        start_ms = chunk_index * self.chunk_length_ms
        end_ms = start_ms + self.chunk_length_ms

        try:
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=language)
                result = {
                    "start_time": self.format_milliseconds(start_ms),
                    "end_time": self.format_milliseconds(end_ms),
                    "text": text
                }
                self.logger.debug(f"Google chunk: {result}")
                return result
        except sr.UnknownValueError:
            logging.warning(f"Не удалось распознать речь в: {chunk_path}")
        except sr.RequestError as e:
            logging.error(f"Ошибка API для {chunk_path}: {e}")
        except Exception as e:
            logging.error(f"Непредвиденная ошибка при обработке {chunk_path}: {e}")

        result = {
            "start_time": self.format_milliseconds(start_ms),
            "end_time": self.format_milliseconds(end_ms),
            "text": ""
        }
        self.logger.debug(f"Google chunk (empty): {result}")
        return result

    def recognize_chunk_whisper(self, chunk_path, chunk_index, language="ru"):
        start_ms = chunk_index * self.chunk_length_ms
        end_ms = start_ms + self.chunk_length_ms

        try:
            result = self.whisper.transcribe(chunk_path, language=language, fp16=False)
            segments = []
            for seg in result["segments"]:
                # Явное преобразование времени
                start = float(seg["start"]) * 1000  # секунды -> миллисекунды
                end = float(seg["end"]) * 1000

                segments.append({
                    "start_time": self.format_milliseconds(start),
                    "end_time": self.format_milliseconds(end),
                    "text": seg["text"].strip()
                })

            return segments[-1] if segments else {
                "start_time": self.format_milliseconds(start_ms),
                "end_time": self.format_milliseconds(end_ms),
                "text": ""
            }

        except Exception as e:
            logger.error(f"Ошибка в {chunk_path}: {e}")
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
        self.logger.info("Распознавание с помощью Whisper")
        try:
            result = self.whisper.transcribe(
                audio=self.audio_path,
                language="ru",
                fp16=False,
                verbose=False
            )

            segments = []
            for seg in result["segments"]:
                start = float(seg["start"]) * 1000
                end = float(seg["end"]) * 1000
                self.logger.debug(f"Whisper segment: start={start}, end={end}, text={seg['text']}")
                start_time = self.format_milliseconds(start)
                end_time = self.format_milliseconds(end)
                self.logger.debug(f"Formatted times: start_time={start_time}, end_time={end_time}")
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": seg["text"].strip()
                })

            full_text = result["text"].strip()
            self.logger.debug(f"Final segments: {segments}")
            return segments, full_text

        except Exception as e:
            self.logger.error(f"Ошибка Whisper: {e}")
            return [], ""


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
            logger.debug(f"Formatting SRT time: {time_value}, type: {type(time_value)}")
            try:
                if isinstance(time_value, str):
                    time_value = time_value.replace(",", ".")
                    # Проверяем строки вида XX.XXX или пустые
                    if time_value in ['00.000', '0.000', ''] or '.' in time_value:
                        try:
                            logger.debug(f"Converting string to float: {time_value}")
                            time_value = float(time_value) * 1000  # Секунды в миллисекунды
                        except ValueError:
                            logger.warning(f"Invalid float string: {time_value}, defaulting to 0")
                            time_value = 0
                    elif ":" in time_value:
                        # Формат HH:MM:SS.sss
                        base, ms_part = time_value.split(".", 1) if "." in time_value else (time_value, "0")
                        logger.debug(f"Parsing time string: base={base}, ms_part={ms_part}")
                        h, m, s = map(int, base.split(":"))
                        ms = int(ms_part[:3].ljust(3, '0'))  # Нормализация миллисекунд
                        time_value = (h * 3600000) + (m * 60000) + (s * 1000) + ms
                    else:
                        logger.warning(f"Unexpected string format: {time_value}, defaulting to 0")
                        time_value = 0
                # Обрабатываем числовое значение (в миллисекундах)
                total_ms = int(time_value)
                h, rem = divmod(total_ms, 3600000)
                m, rem = divmod(rem, 60000)
                s, ms = divmod(rem, 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            except Exception as e:
                logger.error(f"Ошибка форматирования SRT времени: {time_value} -> {e}")
                return "00:00:00,000"

        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                logger.debug(f"Processing segment {i}: {seg}")
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
            self.logger.debug(f"Segments: {chunked_texts}")
            if self.engine == "whisper":
                chunked_texts = self.merge_incomplete_segments(chunked_texts)
                self.logger.debug(f"Merged segments: {chunked_texts}")
            self.logger.info(full_text)
            self.logger.info(f"[INFO] Время транскрибации: {time.time() - start:.2f} сек")

            return chunked_texts, full_text

        except Exception as e:
            self.logger.error(f"[ERROR] Ошибка в процессе транскрибации: {e}", exc_info=True)
            return [], ""

        finally:
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                    self.logger.info("[INFO] Временный аудиофайл удалён.")
                except Exception as e:
                    self.logger.warning(f"[WARNING] Не удалось удалить временный файл: {e}")


