FROM python:3.9-slim

# Create appuser and app directory
RUN useradd -ms /bin/bash appuser
RUN mkdir -p /app/generated_audio

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to appuser
USER appuser

EXPOSE 7860
ENV NUMBA_DISABLE_CACHE=1

CMD ["gunicorn", "--workers", "4", "--timeout", "300000", "--bind", "0.0.0.0:7860", "flask_Character:app"]
