# Образ ассистента корпоративной базы знаний.
# Собирается в два этапа, чтобы в готовый образ не попали компиляторы.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.12-slim

# ffmpeg — видео и голос, poppler — PDF, libarchive — архивы RAR,
# tesseract с русским пакетом — распознавание сканов сертификатов
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg poppler-utils libarchive-tools curl ca-certificates \
        tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    LOG_DIR=/data/logs \
    KB_ROOT=/kb \
    ADMIN_HOST=0.0.0.0

WORKDIR /app
COPY . /app
RUN mkdir -p /data /kb && \
    useradd -m -u 1000 kb && chown -R kb /app /data
USER kb

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8800/api/state >/dev/null || exit 1

CMD ["python", "webui.py"]
