FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PYTHONPATH="${PYTHONPATH}:/app"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]