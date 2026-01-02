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
# 1. Ensure wget is installed
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*



# 3. Delete the fake pointer file using the path you requested
RUN rm utils/model/model.pth

# 4. Download the REAL 600MB model into that exact folder
RUN wget -O utils/model/model.pth https://huggingface.co/KKKONNK/model/resolve/main/model.pth

# 5. VERIFY: The logs should now show ~600M instead of 134
RUN ls -lh utils/model/model.pth
# Check if the file is actually there during the build
RUN ls -lh /utils/model/model.pth || echo "FILE NOT FOUND DURING BUILD"
# Expose the port the app will run on
EXPOSE 7860

CMD ["gunicorn", "api:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:7860", "--timeout", "300", "--graceful-timeout", "300"]


