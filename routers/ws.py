import asyncio
import os
from datetime import datetime
from typing import Annotated, Optional

import torch
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, WebSocketException
from starlette.status import WS_1008_POLICY_VIOLATION

import core.state as state
from streaming.kyutai_coordinator import KyutaiStreamCoordinator
from streaming.optimized_blendshape_worker import OptimizedBlendshapeWorker
from streaming.performance_monitor import get_monitor
from utils.config import config

router = APIRouter()

_API_KEY = os.getenv("RUNPOD_API_KEY", "")


async def _require_ws_token(
    token: Annotated[Optional[str], Query()] = None,
) -> str:
    """Reject the WebSocket handshake before accept() if the token is wrong."""
    if not _API_KEY:
        return ""  # auth disabled when no key is configured (dev mode)
    if token != _API_KEY:
        raise WebSocketException(code=WS_1008_POLICY_VIOLATION, reason="Invalid or missing token")
    return token


@router.websocket("/ws/infer/kyutai")
async def websocket_infer_kyutai(
    websocket: WebSocket,
    _token: Annotated[str, Depends(_require_ws_token)],
):
    """
    Real-time streaming inference: RAG → TTS → blendshapes via PCM16.

    Client flow:
      1. Connect to ws://.../ws/infer/kyutai?token=<RUNPOD_API_KEY>
      2. Send: {"type": "start", "session_id": "...", "question": "...", ...}
      3. Receive: text_chunk, audio_chunk (PCM16), blendshapes, status
      4. Send: {"type": "interrupt"} to stop
    """
    await websocket.accept()
    monitor = get_monitor()
    monitor.reset()

    try:
        try:
            init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        except asyncio.TimeoutError:
            await websocket.close(code=1008, reason="Init timeout")
            return

        if init_msg.get("type") != "start":
            await websocket.send_json({"type": "status", "status": "error", "message": "First message must be type 'start'"})
            await websocket.close()
            return

        session_id  = init_msg.get("session_id")
        question    = init_msg.get("question")
        voice_id    = init_msg.get("voice_id") or "default"
        return_audio = init_msg.get("return_audio", True)
        chunk_ms    = init_msg.get("chunk_ms")

        if not session_id or not question:
            await websocket.send_json({"type": "status", "status": "error", "message": "session_id and question are required"})
            await websocket.close()
            return

        chain = state.get_conversation(session_id)
        if not chain:
            await websocket.send_json({"type": "status", "status": "error", "message": "Session not found. Call /process first."})
            await websocket.close()
            return

        with state._tts_worker_lock:
            model_worker = state._tts_model_worker

        if model_worker is None:
            await websocket.send_json({"type": "status", "status": "error", "message": "TTS model not warmed. Call POST /tts/warmup first."})
            await websocket.close()
            return

        async with state._voice_store_lock:
            voice_entry = state._voice_store.get(str(voice_id))
            voice_prompt = voice_entry.get("prompt") if isinstance(voice_entry, dict) else None

            # Fallback: if requested voice not found, try "default"
            if voice_prompt is None and str(voice_id) != "default":
                default_entry = state._voice_store.get("default")
                voice_prompt = default_entry.get("prompt") if isinstance(default_entry, dict) else None

        if voice_prompt is None and return_audio:
            await websocket.send_json({"type": "status", "status": "error", "message": "No voice configured. Upload a reference audio via POST /tts/reference_audio first."})
            await websocket.close()
            return

        # Enforce GPU concurrency limit — reject immediately if all slots are taken.
        # sem.locked() is True when _value == 0 (no slots available).
        sem = state._gpu_semaphore
        if sem is not None and sem.locked():
            slots = int(os.getenv("MAX_CONCURRENT_PIPELINES", "3"))
            await websocket.send_json({
                "type": "status", "status": "error",
                "message": f"Server busy: max {slots} concurrent sessions. Try again shortly."
            })
            await websocket.close()
            return

        async with (sem if sem is not None else asyncio.Semaphore(999)):
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            bs_worker = OptimizedBlendshapeWorker(state.blendshape_model, device_str, config)

            coordinator = KyutaiStreamCoordinator(
                websocket=websocket,
                tts_worker=model_worker,
                blendshape_worker=bs_worker,
                config=config,
            )

            await coordinator.run_streaming_pipeline(
                rag_chain=chain,
                question=question,
                voice_clone_prompt=voice_prompt,
                return_audio=return_audio,
                chunk_ms=chunk_ms,
            )

        monitor.print_summary()

    except WebSocketDisconnect:
        print(f"[{datetime.now()}] WebSocket client disconnected")
    except Exception as e:
        print(f"[{datetime.now()}] WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "status", "status": "error", "message": "An internal error occurred."})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/performance/summary")
async def get_performance_summary():
    monitor = get_monitor()
    return monitor.get_summary()