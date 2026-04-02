# Use official Python 3.10 image (langchain-classic requires 3.10+)

FROM python:3.10-slim



# Environment variables for Python & pip

ENV PYTHONUNBUFFERED=1 \

    PIP_NO_CACHE_DIR=off \

    PIP_DISABLE_PIP_VERSION_CHECK=on \

    PIP_DEFAULT_TIMEOUT=100 \

    NUMBA_DISABLE_CACHE=1 \

    TOKENIZERS_PARALLELISM=false \

    PYTHONDONTWRITEBYTECODE=1



# Set working directory

WORKDIR /app



# Install system dependencies

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



# Copy requirements first (Docker caching optimization)

COPY requirements.txt .



# Upgrade pip and install Python dependencies

RUN pip install --upgrade pip \

    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio \

    && pip install --no-cache-dir -r requirements.txt \

    && pip install --no-cache-dir qwen-tts

# PEP 318 FIX: Replace @check_model_inputs() with @check_model_inputs in rekuenkdr fork
RUN sed -i 's/@check_model_inputs()/@check_model_inputs/g' /usr/local/lib/python3.10/site-packages/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py || echo "[PATCH] File not found, may need manual fix"



# Copy the application code

COPY . /app



# Download the blendshape model from HuggingFace (force re-download)

RUN rm -f utils/model/model.pth || true && \

    echo "Downloading blendshape model..." && \

    wget -O utils/model/model.pth https://huggingface.co/KKKONNK/model/resolve/main/model.pth && \

    echo "Model downloaded successfully:" && \

    ls -lh utils/model/model.pth



# Create directories

RUN mkdir -p /app/generated_audio /app/demo_outputs



# Make demo scripts executable

RUN chmod +x /app/*.py || true



# Expose the port

EXPOSE 7860



# Health check

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \

    CMD curl -f http://localhost:7860/health || exit 1



# Run the application

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "300"]





