# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import multiprocessing
import time
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config, get_blendshape_names, blendshapes_to_named_frames
import numpy as np

import core.state as state
from routers import rag, tts, stt, ws

if __name__ == '__main__' or __name__.startswith("api"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

app = FastAPI(
    title="Intelligent Document & Web API",
    description="RAG pipeline with Groq + Qwen3-TTS speech + Moonshine STT.",
    version="2.1.0",
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(rag.router)
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(ws.router)

# ── Static files ─────────────────────────────────────────────────────────────
STATIC_AUDIO_DIR = "generated_audio"
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=STATIC_AUDIO_DIR), name="audio")

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)
model_path = "utils/model/model.pth"


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"[{datetime.now()}] Loading HuggingFace embeddings model...")
    t0 = time.time()
    state.embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    )
    print(f"[{datetime.now()}] Embeddings loaded in {time.time() - t0:.2f}s")

    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    if model_size < 1_000_000:
        print(f"[{datetime.now()}] Blendshape model missing or invalid (size={model_size}), downloading...")
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/KKKONNK/model/resolve/main/model.pth",
            model_path,
        )
        print(f"[{datetime.now()}] Blendshape model downloaded ({os.path.getsize(model_path)} bytes).")

    print(f"[{datetime.now()}] Loading blendshape model from {model_path}...")
    state.blendshape_model = load_model(model_path, config, device)
    print(f"[{datetime.now()}] Blendshape model loaded.")

    print(f"[{datetime.now()}] Warming up TTS model...")
    t0 = time.time()
    from routers.tts import tts_warmup
    await tts_warmup()
    print(f"[{datetime.now()}] TTS warmed in {time.time() - t0:.2f}s")

    print(f"[{datetime.now()}] Warming up STT model...")
    t0 = time.time()
    try:
        from routers.stt import _get_stt_worker
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _get_stt_worker)
        print(f"[{datetime.now()}] STT warmed in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[{datetime.now()}] STT warmup failed (non-fatal): {e}")

    print(f"[{datetime.now()}] Startup complete. All models ready.")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


# ── Audio-to-blendshapes ──────────────────────────────────────────────────────
@app.post("/audio_to_blendshapes")
async def audio_to_blendshapes_route(request: Request):
    audio_bytes = await request.body()
    generated = generate_facial_data_from_bytes(audio_bytes, state.blendshape_model, device, config)
    generated_list = generated.tolist() if isinstance(generated, np.ndarray) else generated
    frames = blendshapes_to_named_frames(generated_list)
    return JSONResponse(content={
        'frame_rate': config['frame_rate'],
        'total_frames': len(frames),
        'frames': frames,
        'mapping': get_blendshape_names(),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
