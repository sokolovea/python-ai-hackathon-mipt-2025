import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import textwrap
from app.config import *

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from typing import List, Dict, Optional
import hashlib
import textwrap
import re
import random

logger = logging.getLogger(__name__)

class LectureGenerator:
    def __init__(self, api_key: str, output_file: str):
        self.delay = 10
        self.retries = 3
        if not api_key:
            raise ValueError("API key обязателен")
        self.client = GigaChat(credentials=api_key, verify_ssl_certs=False)
        self.lecture_content: List[str] = []
        self.cache_dir = "lecture_cache"
        self.output_file = output_file
        self.retries = 3
        self.delay = 10
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info("LectureGenerator инициализирован")

        self.BAD_ANSWER_PATTERNS = [
            "я не могу", 
            "как искусственный интеллект", 
            "я не имею доступа", 
            "выходит за рамки", 
            "мне жаль", 
            "не предусмотрено",
            "Не люблю менять тему разговора, но вот сейчас тот самый случай.",
            "Что-то в вашем вопросе меня смущает. Может, поговорим на другую тему?"
        ]

    def _send_prompt(self, prompt: str) -> str:
        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"[_send_prompt] Отправка запроса (попытка {attempt} из {self.retries})")
                response = self.client.chat(
                    Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)],
                        temperature=0.6, max_tokens=5000
                    ))

                if not response or not hasattr(response, "choices"):
                    raise ValueError("Ответ пустой или без choices")

                choice = response.choices[0]
                message = getattr(choice, "message", None)
                content = getattr(message, "content", "").strip() if message else ""

                if not content:
                    raise ValueError("Ответ не содержит текста")

                if any(bad in content.lower() for bad in self.BAD_ANSWER_PATTERNS):
                    raise ValueError("Обнаружен шаблон ответа-заглушки")

                return content

            except Exception as e:
                logger.warning(f"[_send_prompt] Ошибка генерации: {e}")
                if attempt < self.retries:
                    sleep_time = self.delay + random.uniform(0, 1)
                    logger.info(f"[_send_prompt] Повтор через {sleep_time:.1f} сек...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[_send_prompt] Не удалось получить валидный ответ после {self.retries} попыток.")
                    self.lecture_content.append("## Ошибка генерации\n")
                    return "⚠️ Ошибка генерации контента."


    def _get_cache_path(self, name: str, content: Optional[str] = None) -> str:
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
        safe_name = name.replace(" ", "_")[:50]
        if content:
            hash_digest = hashlib.md5(content.encode('utf-8')).hexdigest()[:10]
            safe_name += f"_{hash_digest}"
        return os.path.join(self.cache_dir, f"{safe_name}.json")

    def _load_cache(self, name: str, context: Optional[str] = None) -> Optional[str]:
        path = self._get_cache_path(name, context)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.logger.info(f"Загружаю из кеша: {path}")
                return json.load(f)['content']
        return None

    def _save_cache(self, name: str, content: str, context: Optional[str] = None):
        path = self._get_cache_path(name, context)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'content': content}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_source_text(file_path: str, retries: int = 3, delay: int = 5) -> str:
        logger = logging.getLogger(__name__)
        for attempt in range(retries):
            try:
                logger.info(f"Пробую загрузить файл: {file_path} (попытка {attempt + 1})")
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    raise ValueError("Файл пуст.")
                return text
            except Exception as e:
                logger.warning(f"Не удалось загрузить файл: {e}")
                if attempt == retries - 1:
                    logger.error("Файл не загружен после нескольких попыток.")
                    raise
                time.sleep(delay * (attempt + 1))

    "TODO: НОВЫЙ МЕТОД, СТОИТ ПРОВЕРИТЬ НА АДЕКВАТНОСТЬ, В ТЕСТАХ ПОКАЗАЛ СЕБЯ ХОРОШО"
    def estimate_section_timestamps(
        self,
        sections: List[Dict],
        segments: List[Dict],
        model_name: str = "intfloat/multilingual-e5-base",
        top_k: int = 1,
    ) -> List[Dict]:
        """
        Оценивает таймкоды начала тем на основе семантического сходства между названиями тем и сегментами транскрибации.

        :param sections: список тем лекции [{"title": "Введение"}, ...]
        :param segments: список сегментов с таймкодами и текстом [{"start_time": "00:00:10", "text": "..."}]
        :param model_name: SentenceTransformer model
        :param top_k: сколько ближайших сегментов учитывать (обычно 1)
        :return: обновлённый список sections с ключом 'start_time'
        """

        self.logger.info("Загрузка модели SentenceTransformer...")
        model = SentenceTransformer(model_name)

        section_titles = [s["title"] for s in sections]
        segment_texts = [s["text"] for s in segments]

        self.logger.info("Генерация эмбедингов для тем и сегментов...")
        section_embeddings = model.encode(section_titles, convert_to_tensor=True)
        segment_embeddings = model.encode(segment_texts, convert_to_tensor=True)

        self.logger.info("Расчёт сходства эмбедингов...")
        similarity_matrix = util.cos_sim(section_embeddings, segment_embeddings)

        updated_sections = []
        for i, section in enumerate(sections):
            top_matches = similarity_matrix[i].topk(k=top_k)
            best_idx = top_matches.indices[0].item()
            best_segment = segments[best_idx]

            section_with_time = section.copy()
            section_with_time["start_time"] = best_segment.get("start_time", "")
            self.logger.info(f"Тема '{section['title']}' -> {section_with_time['start_time']}")
            updated_sections.append(section_with_time)

        return updated_sections

    def generate_lecture(self, source_text: str, segments: Optional[List[Dict]] = None) -> None:
        self.logger.info("Запуск генерации структуры лекции...")

        if not source_text.strip():
            self.logger.error("Источник текста пуст. Прекращаю генерацию.")
            raise ValueError("Источник текста пуст. Пожалуйста, проверьте содержимое.")

        if len(source_text.strip()) < 100:
            self.logger.warning("Источник текста слишком короткий. Возможна низкая релевантность результата.")

        segment_info = ""
        if segments:
            preview = segments[:]
            lines = [f"[{seg['start_time']} - {seg['end_time']}] {seg['text'][:]}..." for seg in preview]
            joined = "".join(lines)
            segment_info = textwrap.dedent(f"""
            Ниже приведены фрагменты лекции и промежутки времени, в которые они читаются на видео:
            {joined}
            В дальнейшем используй их для примерного определения времени начала различных разделов.
            """)

        structure_cache = self._load_cache("structure", source_text)
        if structure_cache:
            structure = structure_cache
        else:
            structure_prompt = textwrap.dedent(f"""
            {segment_info}
            Ты — профессиональный преподаватель в ВУЗе. На основе следующего транскрибированного текста создай план лекции в формате Markdown.
            1. **Не добавляй** фраз, не относящихся к теме, таких как: 
                - "Что-то в вашем вопросе меня смущает"
                - "Не люблю менять тему"
                - обращения к собеседнику ("давайте обсудим", "как вы думаете?" и т.п.)
            2. Пиши в официально-деловом или вдохновляющем стиле, без попыток вести диалог.
            3. Не вставляй шаблонные фразы, которые не связаны с содержанием (например, защитные реплики ИИ).
            4. Текст должен быть цельным, без повтора одних и тех же фраз в конце (проверяй дубли).
            
            Действуй следующим образом:
            - Составь план лекции в виде нумерованного списка разделов. Просто перечисли названия разделов, не описывай их.
            - Для каждого раздела из плана добавь примерное время его начала на видео в формате [HH:MM:SS]. Используй приведенные выше фрагменты лекции.
            - Каждый раздел должен быть уникальным и содержать только одну тему.
            - Последний раздел плана должен быть "Выводы" или "Резюме".
            
            Транскрибированный текст лекции:
            {source_text}
            """
                                               )
            structure = self._send_prompt(structure_prompt)
            self._save_cache("structure", structure, source_text)

        # self.lecture_content.append(structure)
        sections = self._extract_sections_from_list(structure)
        self.logger.info(f"Найдено {len(sections)} разделов. Приступаю к генерации содержания...")

        "TODO: Вызов нового метода"
        if segments:
            sections = self.estimate_section_timestamps(sections, segments)

        structure = self._insert_timestamps_into_structure(structure, sections)
        self.lecture_content.append(structure)

        for section in tqdm(sections, desc="Генерация разделов"):
            title = section['title']
            content_cache = self._load_cache(title, source_text)
            if content_cache:
                self.lecture_content.append(content_cache)
                continue

            self.logger.info(f"Генерация раздела: {title}")
            content_prompt = textwrap.dedent(f"""
            Раздел: "{title}"
            Инструкции:
            - Напиши этот раздел на основе контекста ниже, используя академический стиль. Пиши от третьего лица.
            - Применяй Markdown. Как заголовок второго уровня (##) выделяй только название раздела.
            - Избегай "воды" и повторов, фокусируйся на сути.
            - Не выдумывай от себя — используй только информацию из контекста ниже.
            - Исправь ошибки или опечатки, если встретишь.
            - Если приведено несколько примеров или заданий, приводи только один пример. Просто добавь его в текст раздела, а не как отдельный раздел.
            Контекст:
            {source_text}
            """
                                             )
            content = self._send_prompt(content_prompt)
            self._save_cache(title, content, source_text)
            self.lecture_content.append(content)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    def _extract_sections_from_list(self, markdown_text: str) -> List[Dict]:
        sections = []
        for line in markdown_text.splitlines():
            if re.match(r"^\d+\.", line):
                title = line.split(".", 1)[1].strip()
                if "[" in title and "]" in title:
                    title = title[:title.find("[")].strip()
                sections.append({"title": title.strip("# ").strip("*"), "level": 2})
        return sections

    def _extract_sections(self, markdown_text: str) -> List[Dict]:
        sections = []
        for line in markdown_text.splitlines():
            if line.startswith("## "):
                level = 2
            elif line.startswith("### "):
                level = 3
            else:
                continue
            sections.append({"title": line.strip("# "), "level": level})
        return sections

    def save_to_file(self, filename: str) -> None:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(self.lecture_content))
            self.logger.info(f"Лекция успешно сохранена в файл: {filename}")
        except Exception as e:
            self.logger.error(f"Не удалось сохранить лекцию: {e}")

    def _insert_timestamps_into_structure(self, structure, sections):
        """
                Заменяет временные метки в квадратных скобках на соответствующие значения из sections.

                :param structure: Исходная структура плана лекции (строка).
                :param sections: Список словарей с информацией о секциях.
                :return: Обновленная структура плана лекции с замененными временными метками.
                """
        # Регулярное выражение для поиска временных меток в квадратных скобках
        timestamp_pattern = r"\[\d{2}:\d{2}:\d{2}\]"

        # Создаем копию структуры для модификации
        updated_structure = structure

        # Проходим по всем секциям и заменяем временные метки
        for section in sections:
            start_time = section.get("start_time", "")

            # Ищем первую метку в квадратных скобках
            match = re.search(timestamp_pattern, updated_structure)
            if match:
                # Заменяем найденную метку на значение start_time
                updated_structure = updated_structure.replace(match.group(0), f"[{start_time}]", 1)

        return updated_structure

# pipeline = TranscriptionPipeline(
#     video_path="физика.mp4",
#     engine="google",
#     whisper_model="medium"
# )
#
# segments, full_text = pipeline.run()
#
# pipeline.save_to_json(segments, path="output.json")
# pipeline.save_to_txt(full_text, path="output.txt")
# pipeline.save_to_srt(segments, path="output.srt")
#
# print("segments", segments)

import json
def read_json_to_list(file_path):
    try:
        # Открываем файл и загружаем данные
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Проверяем, что данные являются списком словарей
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        else:
            raise ValueError("Файл JSON не содержит ожидаемый формат списка словарей.")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка: Файл '{file_path}' содержит некорректный JSON.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Пример использования
file_path = "json физика whisper.txt"  # Укажите путь к вашему JSON-файлу
segments = read_json_to_list(file_path)

# Вывод первых нескольких записей для проверки
if segments:
    for i, entry in enumerate(segments[:5]):  # Выводим первые 5 записей
        print(f"Запись {i + 1}:")
        print(entry)

full_text = ""
with open("output whisper.txt", "r", encoding="utf-8") as file:
    full_text = file.read()
print(full_text)

gen = LectureGenerator(API_KEY, "lecture.md")
gen.generate_lecture(full_text, segments)
gen.save_to_file("lecture.md")
