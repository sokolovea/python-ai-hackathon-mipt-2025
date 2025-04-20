from pathlib import Path
import os

def ensure_upload_dir(upload_dir: str):
    Path(upload_dir).mkdir(parents=True, exist_ok=True)