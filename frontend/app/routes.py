from flask import render_template, request, session, redirect, url_for
import os
import requests
import json

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
GET_URL = os.getenv("GET_URL", "http://localhost:8000")

timecodes = [{1: 2}]
timecodes_json = 0

def register_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        global timecodes_json
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
                    res = requests.get(f"{BACKEND_URL}/files/{session.get('file_id')}/timecodes.json")
                    
                    if res.status_code == 200:
                        timecodes_json = res.json()
                        print(f"Полученные таймкоды: {timecodes_json}")
                    else:
                        app.logger.error(f"Ошибка получения таймкодов: {res.status_code}")
                        timecodes_json = [{}]
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
        print(f'TIMECODES_JSON = {timecodes_json}')
        return render_template(
            'index.html',
            error=error,
            file_id=file_id,
            timecodes=timecodes_json if timecodes_json else timecodes,
            get_url=GET_URL
        )
    
    @app.route('/reset', methods=['POST'])
    def reset():
        session.clear()
        return redirect(url_for('index'))