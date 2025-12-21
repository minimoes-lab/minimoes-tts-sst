# Use official lightweight Python 3.9 image
FROM python:3.9-slim

# Environment variables for Python & pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1

# Set working directory
WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching optimization)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port the app will run on
EXPOSE 7860

# Start the app using Gunicorn + Uvicorn worker
CMD ["gunicorn", "--workers", "2","--preload", "--timeout", "30000", "--bind", "0.0.0.0:7860", "--worker-class", "uvicorn.workers.UvicornWorker", "api:app"]
