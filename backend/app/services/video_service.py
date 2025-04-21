import subprocess
from pathlib import Path
import os

def process_video(video_path: str, output_dir: str, interval: int = 10):
    audio_path = os.path.join(output_dir, "audio.wav")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a", audio_path
    ], check=True)

    frames_dir = os.path.join(output_dir, "frames")
    Path(frames_dir).mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        os.path.join(frames_dir, "frame_%04d.jpg")
    ], check=True)