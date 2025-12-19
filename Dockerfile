# Use a stable Python 3.9 slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1

# Install system dependencies required for packages like numba, librosa, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    libffi-dev \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the port for Hugging Face / RunPod
EXPOSE 7860

# Run gunicorn with uvicorn worker
CMD ["gunicorn", "--workers", "4", "--timeout", "30000", "--bind", "0.0.0.0:7860", "--worker-class", "uvicorn.workers.UvicornWorker", "flask_Character:app"]
