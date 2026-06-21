FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    wget \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir qwen-tts \
    && pip install --no-cache-dir "transformers==4.57.3" "tokenizers==0.22.2" "huggingface-hub==0.36.2" \
    && pip install --no-cache-dir flash-attn --no-build-isolation

COPY . /app

# Blendshape model is downloaded at runtime on first startup (see load_models in api.py)
RUN mkdir -p utils/model

RUN mkdir -p /app/generated_audio /app/demo_outputs

RUN chmod +x /app/*.py || true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "300"]
