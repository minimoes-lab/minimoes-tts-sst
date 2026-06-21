#!/bin/bash
set -e

# Install flash-attn at pod startup where CUDA headers are available.
# This runs once per pod lifecycle (~3-5 min first boot, cached after).
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "[start.sh] Installing flash-attn (requires CUDA headers — done at runtime)..."
    pip install flash-attn --no-build-isolation
    echo "[start.sh] flash-attn installed."
else
    echo "[start.sh] flash-attn already installed, skipping."
fi

exec uvicorn api:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 300
