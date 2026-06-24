import asyncio
import json
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

import core.state as state
from streaming.stt_worker import STTWorker, StreamingSTTSession

router = APIRouter()

_API_KEY = os.getenv("RUNPOD_API_KEY", "")

# How often to send a WebSocket ping to keep the RunPod proxy alive (seconds).
# RunPod's reverse-proxy drops idle connections after ~45s; 20s gives a safe margin.
_PING_INTERVAL = 20.0

# Hard cap on how long a single STT session may run (seconds).
# Prevents zombie sessions from holding a thread forever.
_SESSION_MAX_DURATION = 600.0

# After sending {type:'stop'}, how long to wait for the transcription thread (seconds).
_FLUSH_TIMEOUT = 10.0



def _get_stt_worker() -> STTWorker:
    with state._stt_worker_lock:
        if state._stt_worker is None:
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model = os.getenv("STT_MODEL_SIZE", "base")
            state._stt_worker = STTWorker(
                model_size=model,
                device=device_str,
            )
            state._stt_worker.load()
    return state._stt_worker


@router.post("/stt/warmup")
async def stt_warmup():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _get_stt_worker)
    return {"status": "ok", "model": os.getenv("STT_MODEL_SIZE", "base")}


@router.websocket("/ws/stt")
async def websocket_stt(
    websocket: WebSocket,
    token: Annotated[Optional[str], Query()] = None,
):
    """
    Real-time streaming STT via WebSocket.

    Flow:
      1. Connect to ws://.../ws/stt?token=<RUNPOD_API_KEY>
      2. Send: {"type": "start", "language": "fr"}
      3. Send binary PCM16 mono 16kHz chunks
      4. Receive: {"type": "partial", "text": "..."} — incremental results
         Receive: {"type": "final",   "text": "..."} — end-of-utterance result
      5. Send: {"type": "stop"} to flush and close

    Auth: token validated after accept() so RunPod proxy can forward the handshake.
    """
    await websocket.accept()
    # Validate token after accept — RunPod proxy rejects pre-upgrade 403 on some paths
    if _API_KEY and token != _API_KEY:
        await websocket.close(code=1008, reason="Invalid or missing token")
        return

    # ── done_event: set by the STT thread when it has emitted its final result ──
    done_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # ── Receive the mandatory init message ────────────────────────────────────
    try:
        init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.close(code=1008, reason="Init timeout")
        return
    except WebSocketDisconnect:
        return

    if init_msg.get("type") != "start":
        await websocket.send_json({"type": "error", "message": "First message must be type 'start'"})
        await websocket.close()
        return

    language = init_msg.get("language") or None

    worker = _get_stt_worker()

    async def send_result(result: dict):
        try:
            await websocket.send_json(result)
        except Exception:
            pass
        # Signal the main loop that the transcription thread has finished
        # (only on terminal messages).
        if result.get("type") in ("final", "error"):
            done_event.set()

    session = StreamingSTTSession(
        worker=worker,
        on_result=send_result,
        loop=loop,
        language=language,
    )

    # ── Keepalive ping task ───────────────────────────────────────────────────
    # RunPod's proxy drops idle TCP connections; send a JSON ping every
    # _PING_INTERVAL seconds.  The client ignores unknown message types, so
    # this is safe even for old clients.
    async def _ping_loop():
        while True:
            await asyncio.sleep(_PING_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                return  # WS already closed

    ping_task = asyncio.create_task(_ping_loop())

    # ── Session timeout task ──────────────────────────────────────────────────
    async def _timeout_loop():
        await asyncio.sleep(_SESSION_MAX_DURATION)
        try:
            await websocket.send_json({"type": "error", "message": "Session timeout"})
            await websocket.close(code=1001, reason="Session timeout")
        except Exception:
            pass

    timeout_task = asyncio.create_task(_timeout_loop())

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"]:
                session.push_audio(msg["bytes"])
                continue

            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue

                if data.get("type") == "stop":
                    # Tell the STT thread to flush — it will call send_result
                    # with type='final', which sets done_event.
                    session.flush()
                    # Wait for the transcription thread to finish, with a hard
                    # timeout so we never block forever.
                    try:
                        await asyncio.wait_for(done_event.wait(), timeout=_FLUSH_TIMEOUT)
                    except asyncio.TimeoutError:
                        pass
                    break

                # Ignore {type:'ping'} echo-backs or other unknown text frames.

    except WebSocketDisconnect:
        # Client disconnected without sending stop — stop the STT thread cleanly
        # so it doesn't loop forever consuming CPU.
        session.stop()
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
        session.stop()
    finally:
        ping_task.cancel()
        timeout_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
