import os
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import core.state as state
from streaming.stt_worker import STTWorker, StreamingSTTSession

router = APIRouter()


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
async def websocket_stt(websocket: WebSocket):
    """
    Real-time streaming STT via WebSocket.

    Flow:
      1. Connect
      2. Send: {"type": "start", "language": "fr"}
      3. Send binary PCM16 mono 16kHz chunks
      4. Receive: {"type": "partial"/"final", "text": "..."}
      5. Send: {"type": "stop"} to flush and close
    """
    await websocket.accept()

    try:
        init_msg = await websocket.receive_json()
        if init_msg.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "First message must be type 'start'"})
            await websocket.close()
            return

        language = init_msg.get("language") or None
        loop = asyncio.get_event_loop()

        worker = _get_stt_worker()

        async def send_result(result: dict):
            try:
                await websocket.send_json(result)
            except Exception:
                pass

        session = StreamingSTTSession(worker=worker, on_result=send_result, loop=loop, language=language)

        await websocket.send_json({"type": "status", "status": "listening"})

        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"]:
                session.push_audio(msg["bytes"])
                continue

            if "text" in msg:
                try:
                    data = __import__("json").loads(msg["text"])
                except Exception:
                    continue
                if data.get("type") == "stop":
                    session.flush()
                    await asyncio.sleep(0.5)
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
