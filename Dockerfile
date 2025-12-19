# ---------- BASE IMAGE ----------
FROM python:3.9-slim

# ---------- ENVIRONMENT ----------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- WORKDIR ----------
WORKDIR /app

# ---------- SYSTEM DEPS (OPTIONAL BUT SAFE) ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- PYTHON DEPS ----------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install fastapi uvicorn gunicorn \
 && pip install -r requirements.txt

# ---------- APP CODE ----------
COPY . .

# ---------- PORT ----------
EXPOSE 7860

# ---------- START COMMAND ----------
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:7860", "api:app"]
