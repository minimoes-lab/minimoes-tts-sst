# Use lightweight Python 3.9 image
FROM python:3.9-slim

# Environment variables for Python and pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Expose port
EXPOSE 7860

# Run the app using Gunicorn + Uvicorn worker for FastAPI/Flask
CMD ["gunicorn", "--workers", "4", "--timeout", "30000", "--bind", "0.0.0.0:7860", "--worker-class", "uvicorn.workers.UvicornWorker", "api:app"]
