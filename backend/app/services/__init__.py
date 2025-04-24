"""
services package - бизнес-логика приложения

Содержит:
- audio_service.py: обработка аудио
- lecture_service.py: генерация лекций
- video_service.py: обработка видео
- docx_service.py: работа с DOCX
"""

from .audio_service import TranscriptionPipeline
from .lecture_service import generate_lecture
from .video_service import process_video
from .docx_service import MarkdownConverter
from .frames_service import get_frames_information

all = [
    'extract_audio_from_video',
    'split_audio',
    'recognize_chunk',
    'transcribe_audio',
    'generate_lecture',
    'process_video',
    'markdown_to_docx'
]