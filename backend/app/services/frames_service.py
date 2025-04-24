import os
from typing import Dict, Optional, List

from app.models.VideoProcessor import VideoProcessor


def get_frames_information(video_path: str, output_dir: str, segments: Optional[List[Dict]] = None) -> Dict:
    processor = VideoProcessor(output_dir=os.path.join(output_dir, "output_frames"))
    result = processor.process_video(video_path, frame_skip=2 * 24, num_threads=4)
    return result