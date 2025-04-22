FROM python:3.11-slim as base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice \
    texlive-lang-cyrillic \
    fonts-dejavu \
    pandoc texlive-xetex texlive-fonts-recommended texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

COPY requirements2.txt .

# Устанавливаем зависимости из второго файла
RUN pip install --user --no-cache-dir -r requirements2.txt

FROM base

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY ./app ./app

ENV PATH=/root/.local/bin:$PATH \
    UPLOAD_DIR=/uploads \
    MAX_FILE_SIZE=209715200 \
    PYTHONPATH=/app

VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]