FROM python:3.11-slim as base

RUN echo "deb http://ftp.debian.org/debian bookworm contrib" >> /etc/apt/sources.list && \
    apt-get update && \
    echo ttf-mscorefonts-installer msttcorefonts/accepted-msttcorefonts-eula select true | debconf-set-selections && \
    apt-get install -y --no-install-recommends \
        fontconfig \
        ttf-mscorefonts-installer \
        ffmpeg \
        libreoffice \
        pandoc \
        texlive-full \
        texlive-xetex \
        texlive-lang-cyrillic \
        fonts-dejavu \
        fonts-liberation \
        lmodern \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

COPY requirements2.txt .

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