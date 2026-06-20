import asyncio
import os
import uuid

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import core.state as state
from streaming.qwen_tts_worker import QwenTTSWorker

router = APIRouter()

# WAV magic bytes: "RIFF" at offset 0, "WAVE" at offset 8
_WAV_MAGIC_RIFF = b"RIFF"
_WAV_MAGIC_WAVE = b"WAVE"
_WAV_HEADER_MIN = 12  # minimum bytes to read for both markers


def _is_valid_wav(data: bytes) -> bool:
    """Verify the file is a real WAV by checking RIFF/WAVE magic bytes."""
    return (
        len(data) >= _WAV_HEADER_MIN
        and data[:4] == _WAV_MAGIC_RIFF
        and data[8:12] == _WAV_MAGIC_WAVE
    )


@router.post("/tts/warmup")
async def tts_warmup():
    with state._tts_worker_lock:
        if state._tts_model_worker is not None:
            return {"status": "ok", "warmed": True}

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    loop = asyncio.get_running_loop()

    def _build():
        return QwenTTSWorker(
            device=device_str,
            use_qwen3=True,
            reference_audio_path=None,
            reference_text=None,
            raise_on_error=True,
        )

    worker = await loop.run_in_executor(None, _build)

    if not worker.model_loaded:
        return {"status": "error", "warmed": False, "error": "Model failed to load, check logs"}

    with state._tts_worker_lock:
        state._tts_model_worker = worker

    return {"status": "ok", "warmed": True}


@router.post("/tts/reference_audio")
async def set_tts_reference_audio(
    audio: UploadFile = File(...),
    text: str = Form(...),
    voice_id: str = Form("default"),
):
    if audio is None:
        raise HTTPException(status_code=400, detail="Missing audio file")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Missing reference text")

    filename = (audio.filename or "").lower()
    if filename and not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Reference audio must be a .wav file")

    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(base_dir, "..", "tts_reference")
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, f"ref_{uuid.uuid4().hex}.wav")

    try:
        content = await audio.read(MAX_SIZE + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")
        if len(content) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="Audio file too large. Maximum 50 MB.")
        # Verify real WAV magic bytes — rejects executables/binaries disguised as .wav
        if not _is_valid_wav(content):
            raise HTTPException(status_code=400, detail="File is not a valid WAV (RIFF/WAVE header missing).")
        with open(ref_path, "wb") as f:
            f.write(content)
    finally:
        try:
            await audio.close()
        except Exception:
            pass

    with state._tts_worker_lock:
        model_worker = state._tts_model_worker

    if model_worker is None:
        await tts_warmup()
        with state._tts_worker_lock:
            model_worker = state._tts_model_worker

    if model_worker is None:
        raise HTTPException(status_code=500, detail="TTS warmup failed")

    try:
        prompt = model_worker.create_voice_clone_prompt(ref_path, text.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build voice clone prompt: {e}")

    async with state._voice_store_lock:
        state._voice_store[str(voice_id)] = {
            "audio_path": ref_path,
            "text": text.strip(),
            "prompt": prompt,
        }

        if str(voice_id) == "default":
            state._tts_reference_audio_path = ref_path
            state._tts_reference_text = text.strip()

    return {
        "status": "ok",
        "reference_configured": True,
        "reference_audio_path": ref_path,
        "voice_id": str(voice_id),
    }


@router.get("/tts/speakers")
def get_tts_speakers():
    return {
        "speakers": [],
        "default_speaker": None,
        "tts_mode": "base_voice_clone",
        "reference_configured": bool(state._tts_reference_audio_path) and bool(state._tts_reference_text),
        "voice_ids": sorted(list(state._voice_store.keys())),
    }