"""
Kyutai-inspired delayed-streams coordinator.
Reference: https://github.com/kyutai-labs/delayed-streams-modeling

Business logic lives in the two mixins:
  - PipelineStagesMixin  (streaming/pipeline_stages.py)
  - TransportMixin       (streaming/transport_mixin.py)

This file owns: init, run_streaming_pipeline, interrupt listener,
buffer-health monitor, error handler.
"""
import asyncio
from datetime import datetime
from typing import Any, Optional, List

from fastapi import WebSocket

from streaming.sentence_buffer import SentenceBuffer
from streaming.transport_mixin import TransportMixin
from streaming.pipeline_stages import PipelineStagesMixin


class DelayedStream:
    """Delayed buffer: items become available after `delay_frames` newer items arrive."""

    def __init__(self, delay_frames: int = 0):
        self.delay_frames = delay_frames
        self.buffer: List = []

    def push(self, item):
        self.buffer.append(item)

    def pop(self) -> Optional[Any]:
        if len(self.buffer) > self.delay_frames:
            return self.buffer.pop(0)
        return None

    def flush(self) -> List:
        items = self.buffer.copy()
        self.buffer.clear()
        return items


class KyutaiStreamCoordinator(TransportMixin, PipelineStagesMixin):
    """
    Coordinates three concurrent async stages:
      LLM  →  TTS  →  Blendshapes + Audio WS send
    """

    def __init__(self, websocket: WebSocket, tts_worker, blendshape_worker, config: dict):
        self.ws = websocket
        self.tts = tts_worker
        self.bs = blendshape_worker
        self.config = config

        self._voice_clone_prompt = None
        self.sentence_buffer = SentenceBuffer(min_chars=12, max_chars=160)

        self.audio_stream = DelayedStream(delay_frames=0)
        self.visual_stream = DelayedStream(delay_frames=2)

        self._cancelled = False
        self._cumulative_audio_time = 0.0
        self._sentence_index = 0
        self._last_blendshape_frame = None
        self._last_successful_frame = None

        self._buffer_health = 1.0
        self._target_buffer_size = 3
        self._min_buffer_size = 1
        self._max_buffer_size = 5

        self._sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_buffer_size)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=self._target_buffer_size)

        self._error_count = 0
        self._max_errors = 3
        self._chunk_ms: int = 50

    async def run_streaming_pipeline(
        self,
        rag_chain,
        question: str,
        voice_clone_prompt=None,
        return_audio: bool = True,
        chunk_ms: Optional[int] = None,
    ):
        self._cancelled = False
        self.tts.reset()
        self.bs.reset()

        if isinstance(chunk_ms, int) and chunk_ms > 0:
            self._chunk_ms = chunk_ms
        self._voice_clone_prompt = voice_clone_prompt

        await self._send_status("processing", "Starting Kyutai-optimized pipeline")
        print(f"[{datetime.now()}] [Kyutai] Pipeline start return_audio={return_audio} chunk_ms={self._chunk_ms}")

        llm_task        = asyncio.create_task(self._llm_stage(rag_chain, question))
        tts_task        = asyncio.create_task(self._tts_stage())
        blendshape_task = asyncio.create_task(self._blendshape_stage(return_audio))
        interrupt_task  = asyncio.create_task(self._listen_for_interrupts())
        monitor_task    = asyncio.create_task(self._monitor_buffer_health())

        tasks = [llm_task, tts_task, blendshape_task, interrupt_task, monitor_task]

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exc = task.exception()
                if exc and task not in (interrupt_task, monitor_task):
                    raise exc
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai] Pipeline error: {repr(e)}")
            try:
                await self._send_status("error", str(e))
            except Exception:
                pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                if self._cancelled:
                    await self._send_idle_transition()
                    await self._send_status("interrupted", "Generation interrupted")
                else:
                    await self._send_status("complete", "Generation complete")
            except Exception:
                pass
            print(f"[{datetime.now()}] [Kyutai] Pipeline end cancelled={self._cancelled}")

    # ── Control messages ──────────────────────────────────────────────────────

    async def _listen_for_interrupts(self):
        try:
            while not self._cancelled:
                msg = await self.ws.receive_json()

                if msg.get("type") == "interrupt":
                    print(f"[{datetime.now()}] [Kyutai] Interrupt received")
                    self._cancelled = True
                    self.tts.cancel()
                    self.bs.cancel()
                    for q in (self._sentence_queue, self._audio_queue):
                        while not q.empty():
                            try: q.get_nowait()
                            except asyncio.QueueEmpty: break
                    break

                elif msg.get("type") == "ping":
                    await self.ws.send_json({"type": "pong"})

                elif msg.get("type") == "buffer_adjust":
                    target = msg.get("target_size", self._target_buffer_size)
                    self._target_buffer_size = max(
                        self._min_buffer_size, min(target, self._max_buffer_size)
                    )
        except Exception:
            pass

    async def _monitor_buffer_health(self):
        while not self._cancelled:
            try:
                await asyncio.sleep(0.5)
                audio_fill    = self._audio_queue.qsize() / self._target_buffer_size
                sentence_fill = self._sentence_queue.qsize() / self._target_buffer_size
                self._buffer_health = (audio_fill + sentence_fill) / 2

                if self._buffer_health < 0.3:
                    self._target_buffer_size = min(self._target_buffer_size + 1, self._max_buffer_size)
                elif self._buffer_health > 0.8:
                    self._target_buffer_size = max(self._target_buffer_size - 1, self._min_buffer_size)
            except Exception:
                pass

    async def _handle_error(self, stage: str, error: Exception):
        self._error_count += 1
        if self._error_count >= self._max_errors:
            await self._send_status("error", f"Too many errors in {stage} stage. Stopping pipeline.")
            self._cancelled = True
        else:
            await self._send_status("warning", f"Error in {stage} (attempt {self._error_count}): {error}")
