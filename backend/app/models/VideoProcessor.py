import cv2
import numpy as np
import time
import json
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from pix2text import Pix2Text
from img2table.document import Image
from img2table.ocr import SuryaOCR
import re
import uuid


class VideoProcessor:
    def __init__(self, output_dir="output_frames", model_path="frozen_east_text_detection.pb"):
        """
        Инициализация VideoProcessor.

        Args:
            output_dir (str): Папка для сохранения кадров и графиков.
            model_path (str): Путь к модели EAST для детекции текста.
        """
        self.output_dir = output_dir
        self.model_path = model_path
        os.makedirs(self.output_dir, exist_ok=True)

        # Конфигурация для Pix2Text с локальными путями к моделям
        self.total_config = {
            "layout": {},
            "text_formula": {
                "languages": ("ru", "en"),
                "det_model_path": "./models/text_detection_2025_02_28.onnx",
                "rec_model_path": "./models/text_recognition_2025_02_18.onnx",
                "mfd_model_path": "./models/mfd-v20240618.onnx"
            }
        }

        # Инициализация моделей один раз
        self.p2t = Pix2Text.from_config(total_configs=self.total_config, return_text=True, auto_line_break=False)
        self.ocr = SuryaOCR(langs=["en", "ru"])

    def calculate_text_percentage_east_batch(self, frames, min_confidence=0.5):
        """
        Пакетная обработка кадров для вычисления процента текстовой области с помощью EAST.

        Args:
            frames (list): Список кадров (numpy.ndarray).
            min_confidence (float): Порог уверенности для детекции текста.

        Returns:
            list: Список процентов текстовой области для каждого кадра.
        """
        net = cv2.dnn.readNet(self.model_path)
        text_percentages = []

        for frame in frames:
            orig = frame.copy()
            (H, W) = frame.shape[:2]
            newW, newH = (320, 320)
            rW = W / float(newW)
            rH = H / float(newH)
            frame = cv2.resize(frame, (newW, newH))
            blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                         (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            try:
                scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
            except cv2.error as e:
                print(f"Ошибка в net.forward: {e}")
                text_percentages.append(0)
                continue
            (numRows, numCols) = scores.shape[2:4]
            text_area = 0
            for y in range(numRows):
                for x in range(numCols):
                    if scores[0, 0, y, x] < min_confidence:
                        continue
                    h = geometry[0, 0, y, x]
                    w = geometry[0, 1, y, x]
                    endX = int((x * 4.0 + geometry[0, 2, y, x]) * rW)
                    endY = int((y * 4.0 + geometry[0, 3, y, x]) * rH)
                    startX = int(endX - w * rW)
                    startY = int(endY - h * rH)
                    box_area = max(0, endX - startX) * max(0, endY - startY)
                    text_area += box_area
            total_area = W * H
            text_percentages.append((text_area / total_area) * 100)
        return text_percentages

    def process_frame_pair(self, prev_gray, frame, frame_index, timestamp, threshold):
        """
        Обработка пары кадров для определения уникальности и процента текста.

        Args:
            prev_gray (numpy.ndarray): Предыдущий кадр в градациях серого.
            frame (numpy.ndarray): Текущий кадр.
            frame_index (int): Индекс кадра.
            timestamp (str): Временная метка.
            threshold (float): Порог SSIM для уникальности.

        Returns:
            tuple or None: (frame_index, frame, text_percent, timestamp) если кадр уникален и содержит текст, иначе None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_gray, gray, full=True)
        if score < threshold:
            try:
                text_percent = self.calculate_text_percentage_east_batch([frame])[0]
                if text_percent > 10:
                    return (frame_index, frame, text_percent, timestamp)
            except Exception as e:
                print(f"Ошибка обработки кадра {frame_index}: {e}")
        return None

    def process_video_segment(self, video_path, start_frame, end_frame, threshold, frame_skip, fps):
        """
        Обработка сегмента видео от start_frame до end_frame, считывая каждый frame_skip кадр.

        Args:
            video_path (str): Путь к видеофайлу.
            start_frame (int): Начальный индекс кадра.
            end_frame (int): Конечный индекс кадра.
            threshold (float): Порог SSIM для уникальности.
            frame_skip (int): Количество пропускаемых кадров.
            fps (float): Частота кадров в секунду.

        Returns:
            list: Список кортежей (frame_index, frame_path, timestamp).
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

            saved_frames = []
            prev_gray = None
            frames_to_process = []

            # Определяем кадры, которые будем считывать (каждый frame_skip)
            frame_indices = range(start_frame, end_frame, frame_skip)
            total_frames_to_process = len(frame_indices)

            print(f"Обработка сегмента с кадра {start_frame} по {end_frame}, шаг {frame_skip}...")
            with tqdm(total=total_frames_to_process, desc=f"Сегмент {start_frame}-{end_frame}", unit="кадр") as pbar:
                for frame_count in frame_indices:
                    # Устанавливаем позицию на нужный кадр
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Вычисляем временную метку
                    timestamp = frame_count / fps
                    hours = int(timestamp // 3600)
                    minutes = int((timestamp % 3600) // 60)
                    seconds = int(timestamp % 60)
                    milliseconds = int((timestamp % 1) * 1000)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        frames_to_process.append((prev_gray.copy(), frame.copy(), frame_count, time_str))
                    prev_gray = gray
                    pbar.update(1)

            cap.release()

            # Пакетная обработка кадров
            frames = [args[1] for args in frames_to_process]
            if frames:
                text_percentages = self.calculate_text_percentage_east_batch(frames)
            else:
                text_percentages = []

            for (prev_gray, frame, frame_count, time_str), text_percent in zip(frames_to_process, text_percentages):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score, _ = ssim(prev_gray, gray, full=True)
                if score < threshold and text_percent > 10:
                    frame_filename = f"frame_{frame_count}_{uuid.uuid4().hex}.jpg"
                    frame_path = os.path.join(self.output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    print(f"Кадр {frame_count}: Покрытие текстом = {text_percent:.2f}%, Сохранён как {frame_path}")
                    saved_frames.append((frame_count, frame_path, time_str))

            return saved_frames
        except Exception as e:
            print(f"Ошибка в сегменте {start_frame}-{end_frame}: {e}")
            return []

    def extract_unique_frames(self, video_path, threshold=0.95, frame_skip=48, num_threads=4):
        """
        Извлечение уникальных кадров из видео с значительным текстовым содержимым.

        Args:
            video_path (str): Путь к видеофайлу.
            threshold (float): Порог SSIM для уникальности.
            frame_skip (int): Количество пропускаемых кадров (по умолчанию 2 секунды при 24 fps).
            num_threads (int): Количество потоков для обработки.

        Returns:
            list: Список кортежей (frame_index, frame_path, timestamp).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Разделяем видео на сегменты для параллельной обработки
        frames_per_thread = (total_frames // frame_skip) // num_threads * frame_skip  # Учитываем frame_skip
        segments = [
            (i * frames_per_thread, min((i + 1) * frames_per_thread, total_frames))
            for i in range(num_threads)
        ]

        saved_frames = []
        print(f"Разделение видео на {num_threads} сегментов: {segments}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    self.process_video_segment,
                    video_path, start, end, threshold, frame_skip, fps
                )
                for start, end in segments
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc="Обработка сегментов"):
                saved_frames.extend(future.result())

        saved_frames.sort(key=lambda x: x[0])
        return saved_frames

    def extract_all_formulas(self, text):
        """
        Извлечение всех возможных формул из текста.

        Args:
            text (str): Текст для извлечения формул.

        Returns:
            list: Список извлечённых формул в виде строк.
        """
        latex_formulas = re.findall(r'\$\$[\s\S]*?\$\$|\$[^\$]*?\$|\\\[.*?\\\]', text, re.MULTILINE)
        additional_candidates = [
            line.strip() for line in text.split('\n')
            if any(sym in line for sym in ['=', '+', '-', '*', '^', '_', '\\frac', '\\sum', '\\int', '△'])
               and len(line.strip()) > 3
               and not line.strip().startswith('Так как')
               and not re.match(r'^[a-zA-ZА-Яа-я\s]+$', line.strip())
        ]
        all_formulas = list(set(latex_formulas + additional_candidates))
        return all_formulas

    def extract_text_and_formulas(self, frame_path):
        """
        Извлечение текста и формул из кадра с помощью Pix2Text.

        Args:
            frame_path (str): Путь к изображению кадра.

        Returns:
            dict: Словарь с ключами 'text' и 'formulas'.
        """
        text = self.p2t.recognize(frame_path, file_type="text_formula", return_text=True, auto_line_break=True,
                                  use_fast=True)
        formulas = self.extract_all_formulas(text)
        return {'text': text, 'formulas': formulas}

    def is_bar_chart(self, edges, img_shape):
        """
        Проверка, содержит ли изображение столбчатую диаграмму.

        Args:
            edges (numpy.ndarray): Изображение с выделенными краями.
            img_shape (tuple): Размеры изображения (высота, ширина).

        Returns:
            bool: True, если обнаружена столбчатая диаграмма.
        """
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        return lines is not None and len(lines) > 5

    def is_tree_diagram(self, edges, gray_img, img_shape):
        """
        Проверка, содержит ли изображение древовидную диаграмму.

        Args:
            edges (numpy.ndarray): Изображение с выделенными краями.
            gray_img (numpy.ndarray): Изображение в градациях серого.
            img_shape (tuple): Размеры изображения (высота, ширина).

        Returns:
            bool: True, если обнаружена древовидная диаграмма.
        """
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nodes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) >= 4 or cv2.minEnclosingCircle(contour)[1] > 5:
                    nodes.append(contour)
        return len(nodes) > 2 and lines is not None and len(lines) > 1

    def get_chart_box(self, edges, img_shape):
        """
        Получение ограничивающего прямоугольника области диаграммы.

        Args:
            edges (numpy.ndarray): Изображение с выделенными краями.
            img_shape (tuple): Размеры изображения (высота, ширина).

        Returns:
            tuple or None: (x, y, w, h) области диаграммы или None, если не найдено.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        chart_box = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if area > max_area:
                    max_area = area
                    chart_box = (x, y, w, h)
        return chart_box

    def extract_charts(self, frame_path):
        """
        Обнаружение и извлечение диаграмм из кадра.

        Args:
            frame_path (str): Путь к изображению кадра.

        Returns:
            list: Список путей к сохранённым изображениям диаграмм.
        """
        img = cv2.imread(frame_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        height, width = img.shape[:2]
        chart_paths = []
        chart_box = None
        if self.is_bar_chart(edges, (height, width)) or self.is_tree_diagram(edges, gray, (height, width)):
            chart_box = self.get_chart_box(edges, (height, width))
        if chart_box:
            x, y, w, h = chart_box
            padding = 50
            x_new = max(x - padding, 0)
            y_new = max(y - padding, 0)
            w_new = min(w + 2 * padding, width - x_new)
            h_new = min(h + 2 * padding, height - y_new)
            chart_img = img[y_new:y_new + h_new, x_new:x_new + w_new]
            chart_filename = f"chart_{uuid.uuid4().hex}.jpg"
            chart_path = os.path.join(self.output_dir, chart_filename)
            cv2.imwrite(chart_path, chart_img)
            chart_paths.append(chart_path)
        return chart_paths

    def merge_cells(self, table):
        """
        Объединение ячеек таблицы с одинаковым текстом.

        Args:
            table: Объект таблицы, содержащий данные о ячейках.

        Returns:
            list: Список строк таблицы, где одинаковый текст объединён.
        """
        rows = list(table.content.values())
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows)
        merged_table = [
            [str(cell.value).replace("-\n", "-").replace("\n", " ") if hasattr(cell, 'value') and cell.value else ""
             for cell in row]
            for row in rows
        ]
        for row in merged_table:
            for i in range(len(row) - 1, 0, -1):
                if row[i] == row[i - 1]:
                    row[i] = ""
        for col in range(num_cols):
            for row in range(num_rows - 1, 0, -1):
                if merged_table[row][col] == merged_table[row - 1][col]:
                    merged_table[row][col] = ""
        return merged_table

    def table_to_markdown(self, table):
        """
        Преобразование таблицы в формат Markdown.

        Args:
            table: Объект таблицы.

        Returns:
            str: Таблица в формате Markdown.
        """
        markdown_output = ""
        merged_table = self.merge_cells(table)
        num_columns = max(len(row) for row in merged_table)
        for i, row in enumerate(merged_table):
            markdown_output += "| " + " | ".join(row) + " |\n"
            if i == 0:
                markdown_output += "| " + " | ".join(["---"] * num_columns) + " |\n"
        return markdown_output

    def extract_tables(self, frame_path, time_code):
        """
        Извлечение таблиц из изображения и преобразование в Markdown.

        Args:
            frame_path (str): Путь к изображению.
            time_code (str): Временной код для идентификации таблицы.

        Returns:
            list: Список таблиц в формате Markdown.
        """
        img_document = Image(frame_path)
        extracted_tables = img_document.extract_tables(ocr=self.ocr)
        markdown_tables = [self.table_to_markdown(table) for table in extracted_tables]
        return markdown_tables

    def process_video(self, video_path, threshold=0.95, frame_skip=48, num_threads=4):
        """
        Обработка видео для извлечения уникальных кадров, текста, формул, диаграмм и таблиц.

        Args:
            video_path (str): Путь к видеофайлу.
            threshold (float): Порог SSIM для уникальности.
            frame_skip (int): Количество пропускаемых кадров.
            num_threads (int): Количество потоков.

        Returns:
            dict: JSON-совместимый словарь с деталями кадров.
        """
        result = {}
        unique_frames = self.extract_unique_frames(video_path, threshold, frame_skip, num_threads)
        for frame_index, frame_path, timestamp in unique_frames:
            text_formulas = self.extract_text_and_formulas(frame_path)
            charts = self.extract_charts(frame_path)
            tables = self.extract_tables(frame_path, timestamp)
            result[f"frame_{frame_index}"] = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "frame_path": frame_path,
                "text": text_formulas['text'],
                "formulas": text_formulas['formulas'],
                "charts": charts,
                "tables": tables
            }
        with open("video_analysis.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

#
# if __name__ == "__main__":
#     processor = VideoProcessor(output_dir="output_frames")
#     video_path = "static/5.mp4"
#     result = processor.process_video(video_path, frame_skip=2 * 24, num_threads=4)
#     print("Результаты сохранены в video_analysis.json")
