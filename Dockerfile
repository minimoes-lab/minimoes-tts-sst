# Dockerfile - serverless / local dev friendly
FROM python:3.9-slim

# create non-root user and app dir
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# envs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1 \
    PYTHONPATH=/app

# caches and huggingface caches (optional)
RUN mkdir -p /app/generated_audio /app/.cache/huggingface /app/.cache/torch \
 && chown -R appuser:appuser /app

# copy requirements and install as root (faster)
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y git build-essential ffmpeg \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && apt-get remove -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# copy app
COPY . /app
RUN chown -R appuser:appuser /app

# switch to non-root user
USER appuser

# expose port (match uvicorn cmd)
EXPOSE 7860

# start server (make sure module name matches your file)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--loop", "asyncio"]
