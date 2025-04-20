from flask import render_template, request, session, redirect, url_for
import os
import requests
from flask import current_app as app

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
GET_URL = os.getenv("GET_URL", "http://localhost:8000")

timecodes = [
    {"time": "00:00:00", "label": "Начало", "seconds": 0},
    {"time": "00:00:30", "label": "Действие 1", "seconds": 30},
    {"time": "00:01:00", "label": "Действие 2", "seconds": 60},
    {"time": "00:01:54", "label": "Иван Иванович наконец-таки доехал до Клязьмы", "seconds": 116},
    {"time": "00:02:11", "label": "Закупился на базаре", "seconds": 131},
    {"time": "00:02:40", "label": "Возвращается с базара", "seconds": 160},
    {"time": "00:03:20", "label": "Конец", "seconds": 200}
]

def register_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        error = None
        file_info = None
        
        if request.method == 'POST':
            session.clear()
            if 'file' not in request.files:
                error = "Файл не выбран"
                return render_template('index.html', error=error)
            
            file = request.files['file']
            if file.filename == '':
                error = "Файл не выбран"
                return render_template('index.html', error=error)
            
            try:
                files = {'file': (file.filename, file.stream)}
                response = requests.post(f"{BACKEND_URL}/upload/", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    session['file_id'] = data['file_id']
                    # session['text_md'] = data['text_md']
                    return redirect(url_for('index'))
                else:
                    error = "Ошибка загрузки файла: некорректное содержимое"
            except Exception as e:
                app.logger.error(f"Error uploading file: {str(e)}")
                error = "Ошибка соединения с сервером"
        # Очищаем сессию при обновлении страницы
        # if request.method == 'GET' and 'file_id' in session:
        #     session.pop('file_id', None)

        file_id = session.get('file_id')
        return render_template(
            'index.html',
            error=error,
            file_id=file_id,
            timecodes=timecodes,
            get_url=GET_URL
        )
    
    @app.route('/reset', methods=['POST'])
    def reset():
        session.clear()
        return redirect(url_for('index'))