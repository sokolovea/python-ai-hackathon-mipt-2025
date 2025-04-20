import json
from typing import List, Dict

def convert_segments_to_timecodes(segments):
    """
    Конвертирует сегменты с субтитрами в структуру таймкодов для фронтенда
    """
    timecodes = []
    
    if segments:
        first_start = format_time(segments[0]['start_time'])
        timecodes.append({
            "time": first_start,
            "label": "Начало",
            "seconds": time_to_seconds(segments[0]['start_time'])
        })
    

    for i, seg in enumerate(segments, 1):
        formatted_time = format_time(seg['start_time'])
        timecodes.append({
            "time": formatted_time,
            "label": f"Действие {i}",
            "seconds": time_to_seconds(seg['start_time'])
        })
    
    if segments:
        last_end = format_time(segments[-1]['end_time'])
        timecodes.append({
            "time": last_end,
            "label": "Конец",
            "seconds": time_to_seconds(segments[-1]['end_time'])
        })
    
    return timecodes

def time_to_seconds(time_str: str) -> int:
    parts = list(map(int, time_str.split(':')))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def format_time(time_str: str) -> str:
    parts = time_str.split(':')
    return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"