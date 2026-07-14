import asyncio
import base64
import os
import pickle
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
    # Fast-path: already warmed, no lock needed for read
    with state._tts_worker_lock:
        if state._tts_model_worker is not None:
            return {"status": "ok", "warmed": True}

    # Slow-path: serialise concurrent warmup calls so only one build runs at a time
    warmup_lock = state._tts_warmup_lock
    if warmup_lock is None:
        return {"status": "error", "warmed": False, "error": "Service not initialised yet"}

    async with warmup_lock:
        # Re-check inside lock — another caller may have finished while we waited
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
    return_prompt_b64: str = Form("0"),
):
    if audio is None:
        raise HTTPException(status_code=400, detail="Missing audio file")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Missing reference text")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Reference text too long (max 2000 chars)")
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9_-]{1,64}$', voice_id):
        raise HTTPException(status_code=400, detail="voice_id must be alphanumeric, underscore, or hyphen (max 64 chars)")

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
        import traceback
        print(f"[TTS] create_voice_clone_prompt failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Failed to build voice clone prompt: {e}")

    async with state._voice_store_lock:
        # Remove old WAV file for this voice_id before replacing it
        old_entry = state._voice_store.get(str(voice_id))
        if isinstance(old_entry, dict):
            old_path = old_entry.get("audio_path")
            if old_path and old_path != ref_path and os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        state._voice_store[str(voice_id)] = {
            "audio_path": ref_path,
            "text": text.strip(),
            "prompt": prompt,
        }

        if str(voice_id) == "default":
            state._tts_reference_audio_path = ref_path
            state._tts_reference_text = text.strip()

    response: dict = {
        "status": "ok",
        "reference_configured": True,
        "voice_id": str(voice_id),
    }

    # Relay requests the serialized prompt so it can forward it per-sentence
    if return_prompt_b64.strip() in ("1", "true", "True"):
        try:
            prompt_bytes = pickle.dumps(prompt)
            response["prompt_b64"] = base64.b64encode(prompt_bytes).decode()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to serialize voice prompt: {e}")

    return response


@router.get("/tts/speakers")
async def get_tts_speakers():
    async with state._voice_store_lock:
        voice_ids = sorted(list(state._voice_store.keys()))
        reference_configured = bool(state._tts_reference_audio_path) and bool(state._tts_reference_text)
    return {
        "speakers": [],
        "default_speaker": None,
        "tts_mode": "base_voice_clone",
        "reference_configured": reference_configured,
        "voice_ids": voice_ids,
    }