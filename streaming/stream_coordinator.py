import asyncio
import base64
from datetime import datetime
from typing import Optional

from fastapi import WebSocket

from streaming.blendshape_worker import BlendshapeWorker
from streaming.idle_frames import generate_idle_frames
from streaming.protocol import (
    make_audio_chunk_msg,
    make_blendshapes_msg,
    make_idle_frames_msg,
    make_status_msg,
    make_text_chunk_msg,
)
from streaming.sentence_buffer import SentenceBuffer
from streaming.streaming_rag import streaming_rag_query
from streaming.tts_worker import AudioChunk, TTSWorker

import numpy as np


class StreamCoordinator:
    """
    Orchestrates the streaming pipeline:
    LLM tokens -> sentence buffer -> TTS -> features -> blendshapes -> WebSocket
    """

    def __init__(
        self,
        websocket: WebSocket,
        tts_worker: TTSWorker,
        blendshape_worker: BlendshapeWorker,
        config: dict,
    ):
        self.ws = websocket
        self.tts = tts_worker
        self.bs = blendshape_worker
        self.config = config
        self.sentence_buffer = SentenceBuffer(min_chars=20, max_chars=200)

        self._cancelled = False
        self._cumulative_audio_time = 0.0
        self._sentence_index = 0
        self._last_blendshape_frame: Optional[np.ndarray] = None

        self._sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=3)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_streaming_pipeline(
        self,
        rag_chain,
        question: str,
        voice_preset: Optional[str] = None,
        return_audio: bool = True,
    ):
        self._cancelled = False
        self.tts.reset()
        self.bs.reset()

        await self._send_status("processing", "Starting inference pipeline")

        llm_task = asyncio.create_task(self._llm_stage(rag_chain, question))
        tts_task = asyncio.create_task(self._tts_stage(voice_preset))
        blendshape_task = asyncio.create_task(
            self._blendshape_stage(return_audio)
        )
        interrupt_task = asyncio.create_task(self._listen_for_interrupts())

        tasks = [llm_task, tts_task, blendshape_task, interrupt_task]

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.exception() and task is not interrupt_task:
                    raise task.exception()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[{datetime.now()}] [Coordinator] Pipeline error: {repr(e)}")
            await self._send_status("error", str(e))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

            if self._cancelled:
                await self._send_idle_transition()
                await self._send_status(
                    "interrupted", "Generation interrupted by user"
                )
            else:
                await self._send_status("complete", "Generation complete")

    # ------------------------------------------------------------------
    # Stage 1: LLM streaming -> sentence buffer -> sentence queue
    # ------------------------------------------------------------------

    async def _llm_stage(self, rag_chain, question: str):
        try:
            async for token in streaming_rag_query(rag_chain, question):
                if self._cancelled:
                    break

                sentences = self.sentence_buffer.add_token(token)
                for sentence in sentences:
                    await self.ws.send_json(
                        make_text_chunk_msg(
                            self._sentence_index, sentence, is_final=False
                        )
                    )
                    await self._sentence_queue.put(sentence)
                    self._sentence_index += 1

            # Flush remaining buffer
            if not self._cancelled:
                remaining = self.sentence_buffer.flush()
                if remaining:
                    await self.ws.send_json(
                        make_text_chunk_msg(
                            self._sentence_index, remaining, is_final=True
                        )
                    )
                    await self._sentence_queue.put(remaining)
                    self._sentence_index += 1

        except Exception as e:
            print(f"[{datetime.now()}] [LLM Stage] ERROR: {repr(e)}")
            raise
        finally:
            await self._sentence_queue.put(None)  # sentinel

    # ------------------------------------------------------------------
    # Stage 2: sentence queue -> Bark TTS -> audio queue
    # ------------------------------------------------------------------

    async def _tts_stage(self, voice_preset: Optional[str]):
        sentence_idx = 0
        try:
            while True:
                if self._cancelled:
                    break

                sentence = await self._sentence_queue.get()
                if sentence is None:
                    break

                audio_chunk = await self.tts.process_sentence(
                    sentence,
                    sentence_idx,
                    self._cumulative_audio_time,
                    voice_preset,
                )

                if audio_chunk and not self._cancelled:
                    self._cumulative_audio_time += audio_chunk.duration
                    await self._audio_queue.put(audio_chunk)

                sentence_idx += 1

        except Exception as e:
            print(f"[{datetime.now()}] [TTS Stage] ERROR: {repr(e)}")
            raise
        finally:
            await self._audio_queue.put(None)  # sentinel

    # ------------------------------------------------------------------
    # Stage 3: audio queue -> feature extraction -> blendshapes -> WS
    # ------------------------------------------------------------------

    async def _blendshape_stage(self, return_audio: bool):
        chunk_idx = 0
        try:
            while True:
                if self._cancelled:
                    break

                audio_chunk: Optional[AudioChunk] = await self._audio_queue.get()
                if audio_chunk is None:
                    break

                # Send audio chunk to client
                if return_audio:
                    audio_b64 = base64.b64encode(
                        audio_chunk.audio_bytes
                    ).decode("utf-8")
                    await self.ws.send_json(
                        make_audio_chunk_msg(
                            chunk_index=chunk_idx,
                            sentence_index=audio_chunk.sentence_index,
                            audio_base64=audio_b64,
                            start_time=audio_chunk.start_time,
                            end_time=audio_chunk.start_time
                            + audio_chunk.duration,
                            sample_rate=audio_chunk.sample_rate,
                            is_final=False,
                        )
                    )

                # Generate blendshapes from audio
                bs_chunk = await self.bs.process_audio_chunk(audio_chunk)

                if bs_chunk and not self._cancelled:
                    if len(bs_chunk.frames) > 0:
                        self._last_blendshape_frame = bs_chunk.frames[
                            -1
                        ].copy()

                    await self.ws.send_json(
                        make_blendshapes_msg(
                            chunk_index=chunk_idx,
                            sentence_index=bs_chunk.sentence_index,
                            frames=bs_chunk.frames.tolist(),
                            start_time=bs_chunk.start_time,
                            end_time=bs_chunk.end_time,
                            frame_rate=bs_chunk.frame_rate,
                            is_final=False,
                        )
                    )

                chunk_idx += 1

            # Send final markers
            if not self._cancelled:
                await self.ws.send_json(
                    make_audio_chunk_msg(
                        chunk_index=chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        audio_base64="",
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        sample_rate=self.tts.sr or 24000,
                        is_final=True,
                    )
                )
                await self.ws.send_json(
                    make_blendshapes_msg(
                        chunk_index=chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        frames=[],
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        frame_rate=60,
                        is_final=True,
                    )
                )

        except Exception as e:
            print(
                f"[{datetime.now()}] [Blendshape Stage] ERROR: {repr(e)}"
            )
            raise

    # ------------------------------------------------------------------
    # Interrupt listener
    # ------------------------------------------------------------------

    async def _listen_for_interrupts(self):
        try:
            while not self._cancelled:
                msg = await self.ws.receive_json()
                if msg.get("type") == "interrupt":
                    print(
                        f"[{datetime.now()}] [Coordinator] "
                        "Interrupt received from client"
                    )
                    self._cancelled = True
                    self.tts.cancel()
                    self.bs.cancel()
                    # Drain queues
                    while not self._sentence_queue.empty():
                        try:
                            self._sentence_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    break
                elif msg.get("type") == "ping":
                    await self.ws.send_json({"type": "pong"})
        except Exception:
            pass  # WebSocket closed

    # ------------------------------------------------------------------
    # Idle frame transition
    # ------------------------------------------------------------------

    async def _send_idle_transition(self):
        idle_frames = generate_idle_frames(
            num_frames=30,  # 0.5s at 60 fps
            output_dim=self.config.get("output_dim", 68),
            last_active_frame=self._last_blendshape_frame,
            ease_to_neutral=True,
        )
        await self.ws.send_json(
            make_idle_frames_msg(
                frames=idle_frames.tolist(),
                start_time=self._cumulative_audio_time,
                end_time=self._cumulative_audio_time + 30 / 60.0,
                frame_rate=60,
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_status(self, status: str, message: str):
        try:
            await self.ws.send_json(make_status_msg(status, message))
        except Exception:
            pass
