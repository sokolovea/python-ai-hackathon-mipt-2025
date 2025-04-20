from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from pathlib import Path

from app.config import UPLOAD_DIR
from app.models.lecture_generator import *
from app.utils.logging_utils import setup_logger
from app.utils.file_utils import ensure_upload_dir
from app.services.video_service import process_video
from app.services.lecture_service import generate_lecture
from app.services.audio_service import TranscriptionPipeline
from app.services.docx_service import MarkdownConverter
from app.services.timecode_service import convert_segments_to_timecodes

logger = setup_logger()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_upload_dir(UPLOAD_DIR)

import json
from typing import List, Dict

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    logger.info(f"Получен файл: {file.filename}")
    file_id = str(uuid.uuid4())
    output_dir = os.path.join(UPLOAD_DIR, file_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    video_path = os.path.join(output_dir, "video.mp4")
    transcrib_path = os.path.join(output_dir, "transcrib.txt")
    TEXT_MD_PATH = os.path.join(output_dir, "summary.md")
    WORD_MD_PATH = os.path.join(output_dir, "summary.docx")
    JSON_PATH = os.path.join(output_dir, "timecodes.json")
    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        print(f"Видео сохранено: {video_path}")

        process_video(video_path, output_dir)
        pipeline = TranscriptionPipeline(video_path=video_path)
        segments, text = pipeline.run()

        with open(transcrib_path, 'w', encoding='UTF-8') as f:
            if isinstance(text, list):  
                f.writelines(text)
            else:  
                f.write(text)

        with open(TEXT_MD_PATH, 'w', encoding='UTF-8') as f:
            if isinstance(text, list):  
                f.writelines(text)
            else:  
                f.write(text)

        logger.info("0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        timecodes = convert_segments_to_timecodes(segments)
        logger.info("1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(timecodes, f, ensure_ascii=False, indent=4)
        logger.info("2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        # generate_lecture(text, TEXT_MD_PATH)


        converter = MarkdownConverter("app/config.json")
        converter.convert(TEXT_MD_PATH, WORD_MD_PATH, export_pdf=True)

        text_md = None
        with open(TEXT_MD_PATH, 'r', encoding='UTF-8') as f:
            text_md = f.readlines()
        
        return {"file_id": file_id, "text_md": text_md, "text_docx": WORD_MD_PATH}
    except Exception as e:
        print(f"Ошибка обработки видео: {e}")
        raise HTTPException(500, str(e))

@app.get("/frames/{file_id}")
def get_frames_info(file_id: str):
    frames_dir = os.path.join(UPLOAD_DIR, file_id, "frames")
    if not os.path.exists(frames_dir):
        raise HTTPException(404)
    
    frames = sorted(os.listdir(frames_dir))
    return {"frames": [f"frames/{f}" for f in frames]}  

@app.get("/files/{file_id}/{path:path}")
def get_file(file_id: str, path: str):
    file_path = os.path.join(UPLOAD_DIR, file_id, path)
    logger.info(f"Запрашиваемый путь: {file_path}")  
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        raise HTTPException(404)
    return FileResponse(file_path)