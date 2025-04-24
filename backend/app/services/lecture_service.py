from app.models.lecture_generator import LectureGenerator
from app.config import API_KEY


def generate_lecture(source_text: str, output_file: str, segments: Optional[List[Dict]] = None) -> None:
    generator = LectureGenerator(api_key=API_KEY, output_file=output_file)
    generator.generate_lecture(source_text, analysis_json)
    generator.save_to_file(output_file)