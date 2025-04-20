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

logger = logging.getLogger(__name__)

class LectureGenerator:
    def __init__(self, api_key: str, output_file: str):
        if not api_key:
            raise ValueError("API key обязателен")
        self.client = GigaChat(credentials=api_key, verify_ssl_certs=False)
        self.lecture_content: List[str] = []
        self.cache_dir = "lecture_cache"
        self.output_file = output_file
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info("LectureGenerator инициализирован")

    def _send_prompt(self, prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat(
                    Chat(
                        messages=[Messages(role=MessagesRole.USER, content=prompt)],
                        temperature=0.6,
                        max_tokens=8000
                    )
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"Ошибка при запросе (попытка {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    self.logger.error("Превышено количество попыток. Прерывание.")
                    raise
                time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))

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

        self.lecture_content.append(structure)
        sections = self._extract_sections_from_list(structure)
        self.logger.info(f"Найдено {len(sections)} разделов. Приступаю к генерации содержания...")

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
            print(content)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    def _extract_sections_from_list(self, markdown_text: str) -> List[Dict]:
        sections = []
        for line in markdown_text.splitlines():
            # проверяем, что строка начинается с номера пункта (например, "1.", "2.", и т.д.)
            if re.match(r"^\d+\.", line):
                title = line.split(".", 1)[1].strip()
                if "[" in title and "]" in title:
                    title = title[:title.find("[")].strip()
                sections.append({"title": title.strip("# ").strip("*"), "level": 2})
        return sections

    # def _extract_sections_from_headers(self, markdown_text: str) -> List[Dict]:
    #     sections = []
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