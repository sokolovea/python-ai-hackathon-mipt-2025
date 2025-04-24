from app.models.VideoProcessor import VideoProcessor


def get_frames_information(video_path: str, output_file: str, segments: Optional[List[Dict]] = None) -> None:
    processor = VideoProcessor(output_dir="output_frames")
    result = processor.process_video(video_path, frame_skip=2 * 24, num_threads=4)