FROM python:3.9-slim

# ---- env ----
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_DISABLE_CACHE=1

# ---- workdir ----
WORKDIR /app

# ---- deps first (important for buildx) ----
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ---- copy app ----
COPY . /app

# ---- port (RunPod + HF compatible) ----
EXPOSE 7860

# ---- start server ----
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300000", "--bind", "0.0.0.0:7860", "api:app"]
