"""
WS /ws/infer/sentence — stateless TTS + blendshapes for one sentence.

Called by the VPS relay once per sentence via WebSocket.
Official RunPod load-balancing streaming pattern (worker-lb-websocket).

Protocol:
  Client → Worker: {"sentence": "...", "sentence_index": 0, "cumulative_time": 0.0,
                    "voice_clone_prompt_b64": "...", "language": "English", "return_audio": true}
  Worker → Client: {"type": "audio_chunk", ...}   (multiple, via make_audio_chunk_msg)
                   {"type": "blendshapes", ...}    (multiple, via make_blendshapes_msg)
                   {"type": "done"}
                   {"type": "error", "message": "..."}
"""
import asyncio
import base64
import io
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import core.state as state
from streaming.optimized_blendshape_worker import OptimizedBlendshapeWorker
from streaming.protocol import make_audio_chunk_msg, make_blendshapes_msg
from streaming.qwen_tts_worker import AudioChunk
from utils.config import config as _app_config

router = APIRouter()

# Singleton worker — reused across connections for CUDA graph thread locality.
# Per-connection state (_is_first_chunk, _previous_tail_frames) is reset on each request.
_BS_WORKER: Optional[OptimizedBlendshapeWorker] = None


def _get_bs_worker() -> OptimizedBlendshapeWorker:
    global _BS_WORKER
    if _BS_WORKER is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _BS_WORKER = OptimizedBlendshapeWorker(state.blendshape_model, device, _app_config)
    return _BS_WORKER


async def _send(ws: WebSocket, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


@router.websocket("/ws/infer/sentence")
async def ws_infer_sentence(websocket: WebSocket):
    await websocket.accept()

    # Auth: check Bearer token from HTTP upgrade headers (relay sends it via additional_headers)
    _api_key = state._http_api_key if hasattr(state, "_http_api_key") else ""
    if not _api_key:
        import os
        _api_key = os.getenv("RUNPOD_API_KEY", "")
    if _api_key:
        auth = websocket.headers.get("Authorization", "")
        if auth != f"Bearer {_api_key}":
            await websocket.close(code=1008, reason="Unauthorized")
            return

    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
    except asyncio.TimeoutError:
        await _send(websocket, {"type": "error", "message": "Init timeout"})
        await websocket.close()
        return
    except WebSocketDisconnect:
        return

    sentence: str = str(data.get("sentence") or "")[:500]
    sentence_index: int = int(data.get("sentence_index") or 0)
    cumulative_time: float = float(data.get("cumulative_time") or 0.0)
    voice_clone_prompt_b64: Optional[str] = data.get("voice_clone_prompt_b64")
    language: str = str(data.get("language") or "English")
    return_audio: bool = bool(data.get("return_audio", True))

    if not sentence:
        await _send(websocket, {"type": "error", "message": "sentence is required"})
        await websocket.close()
        return

    # Validate TTS model
    tts_worker = state._tts_model_worker
    if tts_worker is None:
        await _send(websocket, {"type": "error", "message": "TTS model not warmed — call /tts/warmup first"})
        await websocket.close()
        return

    # Decode voice prompt
    voice_clone_prompt = None
    if voice_clone_prompt_b64:
        try:
            raw = base64.b64decode(voice_clone_prompt_b64)
            voice_clone_prompt = pickle.loads(raw)
        except Exception as e:
            await _send(websocket, {"type": "error", "message": f"Invalid voice_clone_prompt_b64: {e}"})
            await websocket.close()
            return
    else:
        async with state._voice_store_lock:
            entry = state._voice_store.get("default")
        if entry:
            voice_clone_prompt = entry.get("prompt")

    if voice_clone_prompt is None:
        await _send(websocket, {"type": "error", "message": "No voice prompt available"})
        await websocket.close()
        return

    # Acquire GPU semaphore — respects MAX_CONCURRENT_PIPELINES
    sem = state._gpu_semaphore
    if sem is not None:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=10.0)
        except asyncio.TimeoutError:
            await _send(websocket, {"type": "error", "message": "GPU busy — too many concurrent requests"})
            await websocket.close()
            return

    try:
        bs_worker = _get_bs_worker()
        # Reset per-connection state on the singleton worker
        bs_worker._is_first_chunk = True
        bs_worker._previous_tail_frames = None

        bs_min_samples = max(1, int(24000 * 0.150))  # 150ms buffer

        chunk_index = 0
        bs_chunk_index = 0
        cumulative = cumulative_time

        bs_buf: list = []
        bs_buf_samples = 0
        bs_buf_start = cumulative
        sr = 24000

        async def flush_bs_buffer():
            nonlocal bs_chunk_index, bs_buf, bs_buf_samples, bs_buf_start
            if not bs_buf:
                return
            merged = np.concatenate(bs_buf)
            buf_wav = io.BytesIO()
            pcm16 = (np.clip(merged, -1.0, 1.0) * 32767.0).astype(np.int16)
            wavfile.write(buf_wav, sr, pcm16)
            buf_wav.seek(0)
            bs_audio = AudioChunk(
                sentence_index=sentence_index,
                audio_bytes=buf_wav.read(),
                audio_np=merged,
                sample_rate=sr,
                start_time=bs_buf_start,
                duration=float(bs_buf_samples) / float(sr),
            )
            bs_chunk = await bs_worker.process_audio_chunk(bs_audio)
            if bs_chunk is not None and len(bs_chunk.frames) > 0:
                await _send(websocket, make_blendshapes_msg(
                    chunk_index=bs_chunk_index,
                    sentence_index=sentence_index,
                    frames=bs_chunk.frames.tolist(),
                    start_time=bs_chunk.start_time,
                    end_time=bs_chunk.end_time,
                    frame_rate=bs_chunk.frame_rate,
                    is_final=False,
                ))
                bs_chunk_index += 1
            bs_buf_start = cumulative
            bs_buf = []
            bs_buf_samples = 0

        async for audio_chunk in tts_worker.stream_sentence(
            sentence=sentence,
            sentence_index=sentence_index,
            cumulative_time=cumulative_time,
            voice_clone_prompt=voice_clone_prompt,
            language=language,
        ):
            audio_np = audio_chunk.audio_np
            sr = audio_chunk.sample_rate
            dur = audio_chunk.duration

            if return_audio and audio_np is not None:
                pcm16 = (np.clip(audio_np.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
                audio_b64 = base64.b64encode(pcm16.tobytes()).decode()
                await _send(websocket, make_audio_chunk_msg(
                    chunk_index=chunk_index,
                    sentence_index=sentence_index,
                    audio_base64=audio_b64,
                    audio_bytes_base64=audio_b64,
                    start_time=audio_chunk.start_time,
                    end_time=audio_chunk.start_time + dur,
                    sample_rate=sr,
                    is_final=False,
                ))
                chunk_index += 1

            cumulative += dur

            if audio_np is not None:
                bs_buf.append(audio_np.astype(np.float32, copy=False))
                bs_buf_samples += int(audio_np.shape[0])
                if bs_buf_samples >= bs_min_samples:
                    await flush_bs_buffer()

        # Flush remaining
        await flush_bs_buffer()

        await _send(websocket, {"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[{datetime.now()}] [/ws/infer/sentence] ERROR: {e}")
        import traceback; traceback.print_exc()
        await _send(websocket, {"type": "error", "message": str(e)})
    finally:
        if sem is not None:
            sem.release()
        try:
            await websocket.close()
        except Exception:
            pass
