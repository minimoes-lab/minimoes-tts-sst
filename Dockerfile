FROM python:3.11

ENV PYTHONUNBUFFERED=1
ENV NUMBA_DISABLE_CACHE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed for librosa / numba / audio
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy code
COPY . /app

EXPOSE 7860

# IMPORTANT: match RunPod command exactly
CMD ["gunicorn", "--workers", "1", "--timeout", "300000", "--bind", "0.0.0.0:7860", "--worker-class", "uvicorn.workers.UvicornWorker", "api:app"]
