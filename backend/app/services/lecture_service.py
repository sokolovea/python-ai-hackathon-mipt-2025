from app.models.lecture_generator import LectureGenerator
from app.config import API_KEY

def generate_lecture(source_text: str, output_file: str):
    generator = LectureGenerator(api_key=API_KEY, output_file=output_file)
    generator.generate_lecture(source_text)
    generator.save_to_file(output_file)