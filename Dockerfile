FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_DISABLE_CACHE=1

WORKDIR /app

# 🔥 REQUIRED for numba / scipy / librosa
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--timeout", "300000", "--bind", "0.0.0.0:7860", "--worker-class", "uvicorn.workers.UvicornWorker", "flask_Character:app"]
