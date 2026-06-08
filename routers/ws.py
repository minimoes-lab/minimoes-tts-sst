from datetime import datetime

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import core.state as state
from streaming.performance_monitor import get_monitor
from streaming.blendshape_worker import BlendshapeWorker
from streaming.optimized_blendshape_worker import OptimizedBlendshapeWorker
from streaming.kyutai_coordinator import KyutaiStreamCoordinator
from utils.config import config

router = APIRouter()


@router.websocket("/ws/infer/kyutai")
async def websocket_infer_kyutai(websocket: WebSocket):
    """
    Real-time streaming inference: RAG → TTS → blendshapes via PCM16.

    Client flow:
      1. Connect to ws://.../ws/infer/kyutai
      2. Send: {"type": "start", "session_id": "...", "question": "...", ...}
      3. Receive: text_chunk, audio_chunk (PCM16), blendshapes, status
      4. Send: {"type": "interrupt"} to stop
    """
    await websocket.accept()
    monitor = get_monitor()
    monitor.reset()

    try:
        init_msg = await websocket.receive_json()

        if init_msg.get("type") != "start":
            await websocket.send_json({"type": "status", "status": "error", "message": "First message must be type 'start'"})
            await websocket.close()
            return

        session_id  = init_msg.get("session_id")
        question    = init_msg.get("question")
        voice_preset   = init_msg.get("voice_preset")
        tts_instruct   = init_msg.get("tts_instruct")
        voice_id       = init_msg.get("voice_id") or "default"
        return_audio   = init_msg.get("return_audio", True)
        use_optimized_bs = init_msg.get("use_optimized_bs", True)
        chunk_ms       = init_msg.get("chunk_ms")

        if not session_id or not question:
            await websocket.send_json({"type": "status", "status": "error", "message": "session_id and question are required"})
            await websocket.close()
            return

        chain = state.conversations.get(session_id)
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

        voice_entry = state._voice_store.get(str(voice_id))
        voice_prompt = voice_entry.get("prompt") if isinstance(voice_entry, dict) else None
        if voice_prompt is None:
            await websocket.send_json({"type": "status", "status": "error", "message": f"Unknown voice_id '{voice_id}'. Upload via POST /tts/reference_audio first."})
            await websocket.close()
            return

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        bs_worker = (
            OptimizedBlendshapeWorker(state.blendshape_model, device_str, config)
            if use_optimized_bs
            else BlendshapeWorker(state.blendshape_model, device_str, config)
        )

        coordinator = KyutaiStreamCoordinator(
            websocket=websocket,
            tts_worker=model_worker,
            blendshape_worker=bs_worker,
            config=config,
        )

        await coordinator.run_streaming_pipeline(
            rag_chain=chain,
            question=question,
            voice_preset=voice_preset,
            tts_instruct=tts_instruct,
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
