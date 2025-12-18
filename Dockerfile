# Use Python 3.9 slim
FROM python:3.9-slim

# Create appuser and app directories
RUN useradd -ms /bin/bash appuser
RUN mkdir -p /app/generated_audio

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire repo
COPY . /app

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 7860

# Start FastAPI using Uvicorn
CMD ["uvicorn", "flask_Character:app", "--host", "0.0.0.0", "--port", "7860"]
