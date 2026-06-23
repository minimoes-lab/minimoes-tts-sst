FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    NUMBA_DISABLE_CACHE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELSCOPE_CACHE=/tmp/modelscope_cache \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir qwen-tts \
    && pip install --no-cache-dir "transformers==4.57.3" "tokenizers==0.22.2" "huggingface-hub==0.36.2" \
    && pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

COPY . /app

# Blendshape model is downloaded at runtime on first startup (see load_models in api.py)
RUN mkdir -p utils/model

RUN mkdir -p /app/generated_audio /app/demo_outputs

RUN chmod +x /app/*.py || true

# Non-root user for security
RUN groupadd --system --gid 1001 appgroup \
 && useradd --system --uid 1001 --gid appgroup --no-create-home appuser \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "300", "--ws-ping-interval", "10", "--ws-ping-timeout", "30"]
