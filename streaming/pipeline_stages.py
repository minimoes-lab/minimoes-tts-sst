"""
The three pipeline stages for KyutaiStreamCoordinator:
  _llm_stage   — RAG streaming → sentence queue
  _tts_stage   — sentence queue → audio queue
  _blendshape_stage — audio queue → WS blendshapes + audio chunks
"""
import asyncio
import io
from datetime import datetime
from typing import Optional, List

import numpy as np
import scipy.io.wavfile as wavfile

from streaming.protocol import make_text_chunk_msg, make_audio_chunk_msg, make_blendshapes_msg
from streaming.streaming_rag import streaming_rag_query


class PipelineStagesMixin:
    """
    Mixed into KyutaiStreamCoordinator.
    Requires all coordinator state attributes.
    """

    # ── Stage 1: LLM ────────────────────────────────────────────────────────

    async def _llm_stage(self, rag_chain, question: str):
        print(f"[{datetime.now()}] [Kyutai LLM] Stage start, question={question!r}")
        try:
            async for token in streaming_rag_query(rag_chain, question):
                if self._cancelled:
                    break
                # Forward every token immediately so the UI updates word-by-word.
                # Sentence boundaries are tracked separately for TTS chunking.
                if token:
                    await self.ws.send_json(make_text_chunk_msg(self._sentence_index, token, is_final=False))
                sentences = self.sentence_buffer.add_token(token or "")
                for sentence in sentences:
                    await self._sentence_queue.put(sentence)
                    self._sentence_index += 1

            if not self._cancelled:
                remaining = self.sentence_buffer.flush()
                if remaining:
                    # Tokens already sent above — only queue for TTS.
                    await self._sentence_queue.put(remaining)
                    self._sentence_index += 1

        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai LLM] ERROR: {repr(e)}")
            import traceback; traceback.print_exc()
            try:
                await self._handle_error("llm", e)
            except Exception:
                pass
        finally:
            print(f"[{datetime.now()}] [Kyutai LLM] Stage end")
            try:
                self._sentence_queue.put_nowait(None)
            except asyncio.QueueFull:
                # Queue is full; drain one item to make room, then push sentinel
                try:
                    self._sentence_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._sentence_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    # ── Stage 2: TTS ─────────────────────────────────────────────────────────

    async def _tts_stage(self):
        sentence_idx = 0
        max_retries = 2
        print(f"[{datetime.now()}] [Kyutai TTS] Stage start")

        try:
            while True:
                if self._cancelled:
                    break

                sentence = await self._sentence_queue.get()
                if sentence is None:
                    break

                # Streaming path (preferred)
                if hasattr(self.tts, "stream_sentence"):
                    try:
                        chunk_n = 0
                        async for audio_chunk in self.tts.stream_sentence(
                            sentence=sentence,
                            sentence_index=sentence_idx,
                            cumulative_time=self._cumulative_audio_time,
                            voice_clone_prompt=self._voice_clone_prompt,
                            language=getattr(self, "_language", "English"),
                        ):
                            if self._cancelled:
                                break
                            chunk_n += 1
                            self._cumulative_audio_time += audio_chunk.duration
                            await self._audio_queue.put(audio_chunk)
                        sentence_idx += 1
                        continue
                    except Exception as e:
                        print(f"[{datetime.now()}] [Kyutai TTS] Streaming path failed: {e}")
                        import traceback; traceback.print_exc()

                # Batch path with retries
                audio_chunk = None
                for attempt in range(max_retries + 1):
                    try:
                        audio_chunk = await self.tts.process_sentence(
                            sentence, sentence_idx, self._cumulative_audio_time,
                            voice_clone_prompt=self._voice_clone_prompt,
                        )
                        if audio_chunk:
                            break
                    except Exception as e:
                        print(f"[{datetime.now()}] [Kyutai TTS] Attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries:
                            await asyncio.sleep(0.1 * (attempt + 1))

                if audio_chunk and not self._cancelled:
                    self._cumulative_audio_time += audio_chunk.duration
                    await self._audio_queue.put(audio_chunk)
                    self._error_count = 0
                else:
                    await self._generate_silence_chunk(sentence_idx)
                    self._error_count += 1

                sentence_idx += 1

        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai TTS] ERROR: {repr(e)}")
            import traceback; traceback.print_exc()
            try:
                await self._handle_error("tts", e)
            except Exception:
                pass
        finally:
            print(f"[{datetime.now()}] [Kyutai TTS] Stage end")
            try:
                self._audio_queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._audio_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    # ── Stage 3: Blendshapes ──────────────────────────────────────────────────

    async def _blendshape_stage(self, return_audio: bool):
        from streaming.qwen_tts_worker import AudioChunk
        from streaming.protocol import make_blendshapes_msg, make_audio_chunk_msg

        print(f"[{datetime.now()}] [Kyutai BS] Stage start")
        audio_chunk_idx = 0
        bs_chunk_idx = 0

        bs_min_chunk_ms = int(self.config.get("bs_min_chunk_ms", 200) or 200)
        bs_min_samples = max(1, int((self.tts.sr or 24000) * (bs_min_chunk_ms / 1000.0)))

        bs_buf_audio: List[np.ndarray] = []
        bs_buf_samples = 0
        bs_buf_start_time: Optional[float] = None
        bs_buf_sentence_index: Optional[int] = None
        bs_buf_sample_rate: Optional[int] = None

        async def _flush_bs_buffer():
            nonlocal bs_chunk_idx, bs_buf_audio, bs_buf_samples
            nonlocal bs_buf_start_time, bs_buf_sentence_index, bs_buf_sample_rate

            if bs_buf_samples <= 0 or not bs_buf_audio:
                return

            audio_np = np.concatenate(bs_buf_audio, axis=0)
            sr = int(bs_buf_sample_rate or (self.tts.sr or 24000))
            duration = float(bs_buf_samples) / float(sr)
            start_time = float(bs_buf_start_time or 0.0)
            sentence_index = int(bs_buf_sentence_index or 0)

            if hasattr(audio_np, "ndim") and audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]

            audio_np_f32 = audio_np.astype(np.float32, copy=False)
            audio_int16 = (np.clip(audio_np_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, sr, audio_int16)
            buf.seek(0)

            bs_audio_chunk = AudioChunk(
                sentence_index=sentence_index,
                audio_bytes=buf.read(),
                audio_np=audio_np,
                sample_rate=sr,
                start_time=start_time,
                duration=duration,
            )

            try:
                bs_chunk = await self.bs.process_audio_chunk(bs_audio_chunk)

                if bs_chunk is None or not hasattr(bs_chunk, 'frames') or len(bs_chunk.frames) == 0:
                    try:
                        await self._send_fallback_frames(bs_chunk_idx, bs_audio_chunk)
                    except Exception:
                        pass
                else:
                    self._last_blendshape_frame = bs_chunk.frames[-1].copy()
                    self._last_successful_frame = bs_chunk.frames[-1].copy()
                    self.visual_stream.push(bs_chunk)
                    ready_chunk = self.visual_stream.pop()
                    if ready_chunk:
                        try:
                            await self.ws.send_json(make_blendshapes_msg(
                                chunk_index=bs_chunk_idx,
                                sentence_index=ready_chunk.sentence_index,
                                frames=ready_chunk.frames.tolist(),
                                start_time=ready_chunk.start_time,
                                end_time=ready_chunk.end_time,
                                frame_rate=ready_chunk.frame_rate,
                                is_final=False,
                            ))
                        except Exception:
                            pass
            except Exception as e:
                print(f"[{datetime.now()}] [Kyutai BS] Inference error: {e}")
                try:
                    await self._send_fallback_frames(bs_chunk_idx, bs_audio_chunk)
                except Exception:
                    pass

            bs_chunk_idx += 1
            bs_buf_audio = []
            bs_buf_samples = 0
            bs_buf_start_time = None
            bs_buf_sentence_index = None
            bs_buf_sample_rate = None

        try:
            while True:
                if self._cancelled:
                    break

                audio_chunk = await self._audio_queue.get()
                if audio_chunk is None:
                    break

                if return_audio:
                    audio_chunk_idx = await self._send_audio_pcm16(audio_chunk_idx, audio_chunk)

                try:
                    sr = int(audio_chunk.sample_rate or (self.tts.sr or 24000))
                    audio_np = audio_chunk.audio_np

                    if audio_np is None:
                        await self._send_fallback_frames(bs_chunk_idx, audio_chunk)
                        bs_chunk_idx += 1
                        continue

                    if hasattr(audio_np, "ndim") and audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                        if audio_np.ndim > 1:
                            audio_np = audio_np[:, 0]

                    if (bs_buf_sentence_index is not None
                            and int(audio_chunk.sentence_index) != int(bs_buf_sentence_index)):
                        await _flush_bs_buffer()

                    if bs_buf_start_time is None:
                        bs_buf_start_time = float(audio_chunk.start_time)
                        bs_buf_sentence_index = int(audio_chunk.sentence_index)
                        bs_buf_sample_rate = sr

                    bs_buf_audio.append(audio_np.astype(np.float32, copy=False))
                    bs_buf_samples += int(audio_np.shape[0])

                    if bs_buf_samples >= bs_min_samples:
                        await _flush_bs_buffer()

                except Exception as e:
                    print(f"[{datetime.now()}] [Kyutai BS] Buffer error: {e}")
                    try:
                        await self._send_fallback_frames(bs_chunk_idx, audio_chunk)
                    except Exception:
                        break
                    bs_chunk_idx += 1

            # Final flush
            if not self._cancelled and bs_buf_samples > 0:
                await _flush_bs_buffer()

            # Flush delayed visual stream
            for bs_chunk in self.visual_stream.flush():
                try:
                    await self.ws.send_json(make_blendshapes_msg(
                        chunk_index=bs_chunk_idx,
                        sentence_index=bs_chunk.sentence_index,
                        frames=bs_chunk.frames.tolist(),
                        start_time=bs_chunk.start_time,
                        end_time=bs_chunk.end_time,
                        frame_rate=bs_chunk.frame_rate,
                        is_final=False,
                    ))
                except Exception:
                    break
                bs_chunk_idx += 1

            if not self._cancelled:
                try:
                    await self.ws.send_json(make_audio_chunk_msg(
                        chunk_index=audio_chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        audio_base64="",
                        audio_bytes_base64="",
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        sample_rate=self.tts.sr or 24000,
                        audio_format="pcm_s16le",
                        channels=1,
                        is_final=True,
                    ))
                except Exception:
                    pass
                try:
                    await self.ws.send_json(make_blendshapes_msg(
                        chunk_index=bs_chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        frames=(
                            [self._last_successful_frame.tolist()]
                            if self._last_successful_frame is not None else []
                        ),
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        frame_rate=60,
                        is_final=True,
                    ))
                except Exception:
                    pass

        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai BS] ERROR: {repr(e)}")
            import traceback; traceback.print_exc()
            try:
                await self._handle_error("blendshape", e)
            except Exception:
                pass
        finally:
            print(f"[{datetime.now()}] [Kyutai BS] Stage end")
