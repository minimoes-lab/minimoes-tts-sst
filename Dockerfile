FROM python:3.9-slim

RUN useradd -ms /bin/bash appuser
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/generated_audio && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--timeout", "300000", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "api:app"]
