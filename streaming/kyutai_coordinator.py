"""
Kyutai-inspired delayed streams coordinator for joint audio-visual modeling.
Reference: https://github.com/kyutai-labs/delayed-streams-modeling
"""
import asyncio
import base64
import io
from datetime import datetime
from typing import Optional, List
import numpy as np
import scipy.io.wavfile as wavfile

from fastapi import WebSocket

from streaming.protocol import (
    make_audio_chunk_msg,
    make_blendshapes_msg,
    make_idle_frames_msg,
    make_status_msg,
    make_text_chunk_msg,
)
from streaming.sentence_buffer import SentenceBuffer
from streaming.streaming_rag import streaming_rag_query
from streaming.idle_frames import generate_idle_frames


class DelayedStream:
    """
    Manages a delayed stream with configurable latency.
    Kyutai approach: audio and visual streams are synchronized with controlled delay.
    """
    
    def __init__(self, delay_frames: int = 0):
        self.delay_frames = delay_frames
        self.buffer: List = []
    
    def push(self, item):
        """Add item to delayed buffer."""
        self.buffer.append(item)
    
    def pop(self) -> Optional[any]:
        """Pop item if delay satisfied."""
        if len(self.buffer) > self.delay_frames:
            return self.buffer.pop(0)
        return None
    
    def flush(self) -> List:
        """Flush all remaining items."""
        items = self.buffer.copy()
        self.buffer.clear()
        return items


class KyutaiStreamCoordinator:
    """
    Enhanced coordinator using Kyutai delayed streams approach.
    
    Key improvements:
    - Joint audio-visual modeling with controlled delay
    - Adaptive buffering based on network conditions
    - Better synchronization between modalities
    - Graceful error recovery
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        tts_worker,
        blendshape_worker,
        config: dict,
    ):
        self.ws = websocket
        self.tts = tts_worker
        self.bs = blendshape_worker
        self.config = config

        self._voice_clone_prompt = None
        
        # Sentence buffering
        self.sentence_buffer = SentenceBuffer(min_chars=12, max_chars=160)
        
        # Delayed streams for synchronization
        self.audio_stream = DelayedStream(delay_frames=0)
        self.visual_stream = DelayedStream(delay_frames=2)  # 2-chunk visual delay
        
        # State tracking
        self._cancelled = False
        self._cumulative_audio_time = 0.0
        self._sentence_index = 0
        self._last_blendshape_frame: Optional[np.ndarray] = None
        
        # Adaptive buffering
        self._buffer_health = 1.0  # 0.0 = empty, 1.0 = full
        self._target_buffer_size = 3
        self._min_buffer_size = 1
        self._max_buffer_size = 5
        
        # Queues with adaptive sizing
        self._sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_buffer_size)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=self._target_buffer_size)
        
        # Error recovery
        self._error_count = 0
        self._max_errors = 3
        self._last_successful_frame = None

        # Real-time audio streaming (PCM16)
        self._chunk_ms: int = 50
    
    async def run_streaming_pipeline(
        self,
        rag_chain,
        question: str,
        voice_preset: Optional[str] = None,
        tts_instruct: Optional[str] = None,
        voice_clone_prompt=None,
        return_audio: bool = True,
        chunk_ms: Optional[int] = None,
    ):
        """Run the full streaming pipeline with Kyutai optimizations."""
        self._cancelled = False
        self.tts.reset()
        self.bs.reset()

        if isinstance(chunk_ms, int) and chunk_ms > 0:
            self._chunk_ms = chunk_ms
        
        self._voice_clone_prompt = voice_clone_prompt
        
        await self._send_status("processing", "Starting Kyutai-optimized pipeline")
        print(f"[{datetime.now()}] [Kyutai] Pipeline start")
        print(f"[{datetime.now()}] [Kyutai] return_audio={return_audio} chunk_ms={self._chunk_ms}")
        print(f"[{datetime.now()}] [Kyutai] voice_clone_prompt_present={self._voice_clone_prompt is not None}")
        
        # Create concurrent tasks
        llm_task = asyncio.create_task(self._llm_stage(rag_chain, question))
        tts_task = asyncio.create_task(self._tts_stage(voice_preset, tts_instruct))
        blendshape_task = asyncio.create_task(self._blendshape_stage(return_audio))
        interrupt_task = asyncio.create_task(self._listen_for_interrupts())
        monitor_task = asyncio.create_task(self._monitor_buffer_health())
        
        tasks = [llm_task, tts_task, blendshape_task, interrupt_task, monitor_task]
        
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.exception() and task not in [interrupt_task, monitor_task]:
                    raise task.exception()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai] Pipeline error: {repr(e)}")
            await self._send_status("error", str(e))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if self._cancelled:
                await self._send_idle_transition()
                await self._send_status("interrupted", "Generation interrupted")
            else:
                await self._send_status("complete", "Generation complete")
            print(f"[{datetime.now()}] [Kyutai] Pipeline end cancelled={self._cancelled}")
    
    async def _llm_stage(self, rag_chain, question: str):
        """Stage 1: LLM streaming with sentence buffering."""
        print(f"[{datetime.now()}] [Kyutai LLM] Stage start")
        print(f"[{datetime.now()}] [Kyutai LLM] question={question!r}")
        try:
            async for token in streaming_rag_query(rag_chain, question):
                if self._cancelled:
                    print(f"[{datetime.now()}] [Kyutai LLM] Cancelled")
                    break

                if token:
                    print(f"[{datetime.now()}] [Kyutai LLM] token_len={len(token)}")
                sentences = self.sentence_buffer.add_token(token)
                for sentence in sentences:
                    print(f"[{datetime.now()}] [Kyutai LLM] enqueue_sentence idx={self._sentence_index} len={len(sentence)}")
                    await self.ws.send_json(
                        make_text_chunk_msg(
                            self._sentence_index, sentence, is_final=False
                        )
                    )
                    await self._sentence_queue.put(sentence)
                    self._sentence_index += 1
            
            # Flush remaining
            if not self._cancelled:
                remaining = self.sentence_buffer.flush()
                if remaining:
                    print(f"[{datetime.now()}] [Kyutai LLM] flush_remaining idx={self._sentence_index} len={len(remaining)}")
                    await self.ws.send_json(
                        make_text_chunk_msg(
                            self._sentence_index, remaining, is_final=True
                        )
                    )
                    await self._sentence_queue.put(remaining)
                    self._sentence_index += 1
        
        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai LLM] ERROR: {repr(e)}")
            import traceback
            traceback.print_exc()
            await self._handle_error("llm", e)
        finally:
            print(f"[{datetime.now()}] [Kyutai LLM] Stage end")
            await self._sentence_queue.put(None)
    
    async def _tts_stage(self, voice_preset: Optional[str], tts_instruct: Optional[str]):
        """Stage 2: TTS with error recovery."""
        sentence_idx = 0
        retry_count = 0
        max_retries = 2

        print(f"[{datetime.now()}] [Kyutai TTS] Stage start")
        print(
            f"[{datetime.now()}] [Kyutai TTS] stream_sentence={hasattr(self.tts, 'stream_sentence')} "
            f"process_sentence={hasattr(self.tts, 'process_sentence')}"
        )
        print(f"[{datetime.now()}] [Kyutai TTS] voice_clone_prompt_present={self._voice_clone_prompt is not None}")
        
        try:
            while True:
                if self._cancelled:
                    print(f"[{datetime.now()}] [Kyutai TTS] Cancelled")
                    break
                
                sentence = await self._sentence_queue.get()
                if sentence is None:
                    print(f"[{datetime.now()}] [Kyutai TTS] Got None sentence, finishing")
                    break

                print(f"[{datetime.now()}] [Kyutai TTS] Got sentence_idx={sentence_idx} len={len(sentence)}")

                # Inference streaming path (Base voice-clone)
                if hasattr(self.tts, "stream_sentence"):
                    try:
                        print(f"[{datetime.now()}] [Kyutai TTS] stream_sentence start idx={sentence_idx}")
                        chunk_n = 0
                        async for audio_chunk in self.tts.stream_sentence(
                            sentence=sentence,
                            sentence_index=sentence_idx,
                            cumulative_time=self._cumulative_audio_time,
                            voice_clone_prompt=self._voice_clone_prompt,
                        ):
                            if self._cancelled:
                                break
                            chunk_n += 1
                            if chunk_n == 1:
                                print(f"[{datetime.now()}] [Kyutai TTS] first_audio_chunk idx={sentence_idx}")
                            self._cumulative_audio_time += audio_chunk.duration
                            await self._audio_queue.put(audio_chunk)
                        print(f"[{datetime.now()}] [Kyutai TTS] stream_sentence end idx={sentence_idx} chunks={chunk_n}")
                        sentence_idx += 1
                        continue
                    except Exception as e:
                        print(f"[{datetime.now()}] [Kyutai TTS] Streaming path failed, fallback: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Try to generate audio with retries
                audio_chunk = None
                for attempt in range(max_retries + 1):
                    try:
                        print(f"[{datetime.now()}] [Kyutai TTS] process_sentence attempt {attempt + 1}/{max_retries + 1} idx={sentence_idx}")
                        audio_chunk = await self.tts.process_sentence(
                            sentence,
                            sentence_idx,
                            self._cumulative_audio_time,
                            voice_preset,
                            tts_instruct,
                            self._voice_clone_prompt,
                        )
                        if audio_chunk:
                            retry_count = 0
                            print(f"[{datetime.now()}] [Kyutai TTS] process_sentence success idx={sentence_idx}")
                            break
                    except Exception as e:
                        print(f"[{datetime.now()}] [Kyutai TTS] Attempt {attempt + 1} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        if attempt < max_retries:
                            await asyncio.sleep(0.1 * (attempt + 1))
                
                if audio_chunk and not self._cancelled:
                    self._cumulative_audio_time += audio_chunk.duration
                    await self._audio_queue.put(audio_chunk)
                    self._error_count = 0
                else:
                    # Generate silence chunk as fallback
                    print(f"[{datetime.now()}] [Kyutai TTS] silence_fallback idx={sentence_idx}")
                    await self._generate_silence_chunk(sentence_idx)
                    self._error_count += 1
                
                sentence_idx += 1
        
        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai TTS] ERROR: {repr(e)}")
            import traceback
            traceback.print_exc()
            await self._handle_error("tts", e)
        finally:
            print(f"[{datetime.now()}] [Kyutai TTS] Stage end")
            await self._audio_queue.put(None)
    
    async def _blendshape_stage(self, return_audio: bool):
        """Stage 3: Blendshape generation with delayed stream sync."""
        from streaming.qwen_tts_worker import AudioChunk
        
        audio_chunk_idx = 0
        bs_chunk_idx = 0
        
        # Blendshape buffering: accumulate PCM until minimum duration for stable feature extraction
        # OPTIMIZATION: 800ms window based on KeyFace/SyncAnimation research for temporal coherence
        bs_min_chunk_ms = int(self.config.get("bs_min_chunk_ms", 800) or 800)
        bs_min_samples = max(1, int((self.tts.sr or 24000) * (bs_min_chunk_ms / 1000.0)))
        
        bs_buf_audio: List[np.ndarray] = []
        bs_buf_samples = 0
        bs_buf_start_time: Optional[float] = None
        bs_buf_sentence_index: Optional[int] = None
        bs_buf_sample_rate: Optional[int] = None
        
        async def _flush_bs_buffer():
            """Process accumulated PCM buffer into blendshapes."""
            nonlocal bs_chunk_idx, bs_buf_audio, bs_buf_samples, bs_buf_start_time, bs_buf_sentence_index, bs_buf_sample_rate
            
            if bs_buf_samples <= 0 or not bs_buf_audio:
                return
            
            # Concatenate buffered audio
            audio_np = np.concatenate(bs_buf_audio, axis=0)
            sr = int(bs_buf_sample_rate or (self.tts.sr or 24000))
            duration = float(bs_buf_samples) / float(sr)
            start_time = float(bs_buf_start_time or 0.0)
            sentence_index = int(bs_buf_sentence_index or 0)
            
            # Ensure mono
            if hasattr(audio_np, "ndim") and audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            
            # Reconstruct WAV bytes for BlendshapeWorker
            audio_np_f32 = audio_np.astype(np.float32, copy=False)
            audio_int16 = (np.clip(audio_np_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, sr, audio_int16)
            buf.seek(0)
            wav_bytes = buf.read()
            
            # Create AudioChunk for blendshape inference
            bs_audio_chunk = AudioChunk(
                sentence_index=sentence_index,
                audio_bytes=wav_bytes,
                audio_np=audio_np,
                sample_rate=sr,
                start_time=start_time,
                duration=duration,
            )
            
            # Run blendshape inference
            try:
                bs_chunk = await self.bs.process_audio_chunk(bs_audio_chunk)
                
                if bs_chunk is None or not hasattr(bs_chunk, 'frames') or len(bs_chunk.frames) == 0:
                    # Fallback: use last successful frame
                    await self._send_fallback_frames(bs_chunk_idx, bs_audio_chunk)
                else:
                    # Success: update last frames and push to delayed stream
                    self._last_blendshape_frame = bs_chunk.frames[-1].copy()
                    self._last_successful_frame = bs_chunk.frames[-1].copy()
                    
                    self.visual_stream.push(bs_chunk)
                    ready_chunk = self.visual_stream.pop()
                    if ready_chunk:
                        await self.ws.send_json(
                            make_blendshapes_msg(
                                chunk_index=bs_chunk_idx,
                                sentence_index=ready_chunk.sentence_index,
                                frames=ready_chunk.frames.tolist(),
                                start_time=ready_chunk.start_time,
                                end_time=ready_chunk.end_time,
                                frame_rate=ready_chunk.frame_rate,
                                is_final=False,
                            )
                        )
            except Exception as e:
                print(f"[{datetime.now()}] [Kyutai BS] Inference error: {e}")
                await self._send_fallback_frames(bs_chunk_idx, bs_audio_chunk)
            
            bs_chunk_idx += 1
            
            # Reset buffer
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
                
                # Send audio immediately (no delay)
                if return_audio:
                    audio_chunk_idx = await self._send_audio_pcm16(audio_chunk_idx, audio_chunk)
                
                # Buffer PCM for blendshape inference
                try:
                    sr = int(audio_chunk.sample_rate or (self.tts.sr or 24000))
                    audio_np = audio_chunk.audio_np
                    
                    if audio_np is None:
                        # No audio data, send fallback
                        await self._send_fallback_frames(bs_chunk_idx, audio_chunk)
                        bs_chunk_idx += 1
                        continue
                    
                    # Ensure mono
                    if hasattr(audio_np, "ndim") and audio_np.ndim > 1:
                        audio_np_mono = audio_np[:, 0]
                    else:
                        audio_np_mono = audio_np
                    
                    # Check if sentence changed: flush buffer before starting new sentence
                    if (bs_buf_sentence_index is not None 
                        and int(audio_chunk.sentence_index) != int(bs_buf_sentence_index)):
                        await _flush_bs_buffer()
                    
                    # Initialize buffer if empty
                    if bs_buf_start_time is None:
                        bs_buf_start_time = float(audio_chunk.start_time)
                        bs_buf_sentence_index = int(audio_chunk.sentence_index)
                        bs_buf_sample_rate = sr
                    
                    # Accumulate PCM samples
                    bs_buf_audio.append(audio_np_mono.astype(np.float32, copy=False))
                    bs_buf_samples += int(audio_np_mono.shape[0])
                    
                    # Flush buffer if minimum duration reached
                    if bs_buf_samples >= bs_min_samples:
                        await _flush_bs_buffer()
                
                except Exception as e:
                    print(f"[{datetime.now()}] [Kyutai BS] Buffer error: {e}")
                    await self._send_fallback_frames(bs_chunk_idx, audio_chunk)
                    bs_chunk_idx += 1
            
            # Flush any remaining buffered audio
            try:
                if not self._cancelled and bs_buf_samples > 0:
                    await _flush_bs_buffer()
            except Exception as e:
                print(f"[{datetime.now()}] [Kyutai BS] Final flush error: {e}")
            
            # Flush delayed visual stream
            remaining = self.visual_stream.flush()
            for bs_chunk in remaining:
                await self.ws.send_json(
                    make_blendshapes_msg(
                        chunk_index=bs_chunk_idx,
                        sentence_index=bs_chunk.sentence_index,
                        frames=bs_chunk.frames.tolist(),
                        start_time=bs_chunk.start_time,
                        end_time=bs_chunk.end_time,
                        frame_rate=bs_chunk.frame_rate,
                        is_final=False,
                    )
                )
                bs_chunk_idx += 1
            
            # Send final markers
            if not self._cancelled:
                await self.ws.send_json(
                    make_audio_chunk_msg(
                        chunk_index=audio_chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        audio_base64="",
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        sample_rate=self.tts.sr or 24000,
                        audio_format="pcm_s16le",
                        channels=1,
                        is_final=True,
                    )
                )
                await self.ws.send_json(
                    make_blendshapes_msg(
                        chunk_index=bs_chunk_idx,
                        sentence_index=max(0, self._sentence_index - 1),
                        frames=(
                            [self._last_successful_frame.tolist()]
                            if self._last_successful_frame is not None
                            else []
                        ),
                        start_time=self._cumulative_audio_time,
                        end_time=self._cumulative_audio_time,
                        frame_rate=60,
                        is_final=True,
                    )
                )
        
        except Exception as e:
            print(f"[{datetime.now()}] [Kyutai BS] ERROR: {repr(e)}")
            await self._handle_error("blendshape", e)

    async def _send_audio_pcm16(self, chunk_idx: int, audio_chunk) -> int:
        """Emit PCM16 little-endian chunks over WS for real-time playback."""
        sr = int(audio_chunk.sample_rate or (self.tts.sr or 24000))
        audio_np = audio_chunk.audio_np
        if audio_np is None:
            return chunk_idx

        if hasattr(audio_np, "ndim") and audio_np.ndim > 1:
            audio_np = audio_np[:, 0]

        audio_np = audio_np.astype(np.float32, copy=False)
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)

        samples_per = max(1, int(sr * (self._chunk_ms / 1000.0)))
        total_samples = int(audio_int16.shape[0])

        sample_cursor = 0
        while sample_cursor < total_samples and not self._cancelled:
            end = min(total_samples, sample_cursor + samples_per)
            seg_bytes = audio_int16[sample_cursor:end].tobytes(order="C")
            seg_b64 = base64.b64encode(seg_bytes).decode("utf-8")

            seg_start_time = audio_chunk.start_time + (sample_cursor / sr)
            seg_end_time = audio_chunk.start_time + (end / sr)

            await self.ws.send_json(
                make_audio_chunk_msg(
                    chunk_index=chunk_idx,
                    sentence_index=audio_chunk.sentence_index,
                    audio_base64="",
                    audio_bytes_base64=seg_b64,
                    start_time=seg_start_time,
                    end_time=seg_end_time,
                    sample_rate=sr,
                    audio_format="pcm_s16le",
                    channels=1,
                    is_final=False,
                )
            )

            if chunk_idx == 0 or chunk_idx % 20 == 0:
                print(
                    f"[{datetime.now()}] [Kyutai Audio] sent_chunk={chunk_idx} "
                    f"bytes={len(seg_bytes)} sr={sr} sentence_index={audio_chunk.sentence_index}"
                )

            chunk_idx += 1
            sample_cursor = end

        return chunk_idx
    
    async def _monitor_buffer_health(self):
        """Monitor and adapt buffer sizes based on queue health."""
        while not self._cancelled:
            try:
                await asyncio.sleep(0.5)
                
                # Calculate buffer health
                audio_fill = self._audio_queue.qsize() / self._target_buffer_size
                sentence_fill = self._sentence_queue.qsize() / self._target_buffer_size
                
                self._buffer_health = (audio_fill + sentence_fill) / 2
                
                # Adapt buffer sizes
                if self._buffer_health < 0.3:
                    # Buffer running low, increase target
                    self._target_buffer_size = min(
                        self._target_buffer_size + 1,
                        self._max_buffer_size
                    )
                elif self._buffer_health > 0.8:
                    # Buffer too full, decrease target
                    self._target_buffer_size = max(
                        self._target_buffer_size - 1,
                        self._min_buffer_size
                    )
                
            except Exception:
                pass
    
    async def _listen_for_interrupts(self):
        """Listen for client interrupts and control messages."""
        try:
            while not self._cancelled:
                msg = await self.ws.receive_json()
                
                if msg.get("type") == "interrupt":
                    print(f"[{datetime.now()}] [Kyutai] Interrupt received")
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
                
                elif msg.get("type") == "buffer_adjust":
                    # Client can request buffer adjustment
                    target = msg.get("target_size", self._target_buffer_size)
                    self._target_buffer_size = max(
                        self._min_buffer_size,
                        min(target, self._max_buffer_size)
                    )
        
        except Exception:
            pass
    
    async def _send_idle_transition(self):
        """Send smooth idle transition frames."""
        idle_frames = generate_idle_frames(
            num_frames=30,
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
    
    async def _send_status(self, status: str, message: str):
        """Send status message to client."""
        try:
            await self.ws.send_json(make_status_msg(status, message))
        except Exception:
            pass
    
    async def _handle_error(self, stage: str, error: Exception):
        """Handle errors with graceful degradation."""
        self._error_count += 1
        
        if self._error_count >= self._max_errors:
            await self._send_status(
                "error",
                f"Too many errors in {stage} stage. Stopping pipeline."
            )
            self._cancelled = True
        else:
            await self._send_status(
                "warning",
                f"Error in {stage} stage (attempt {self._error_count}): {str(error)}"
            )
    
    async def _generate_silence_chunk(self, sentence_idx: int):
        """Generate a silent audio chunk as fallback."""
        from streaming.qwen_tts_worker import AudioChunk
        
        duration = 0.5  # 500ms silence
        samples = int(duration * (self.tts.sr or 24000))
        audio_np = np.zeros(samples, dtype=np.float32)
        
        buf = io.BytesIO()
        audio_int16 = (audio_np * 32767.0).astype(np.int16)
        wavfile.write(buf, self.tts.sr or 24000, audio_int16)
        buf.seek(0)
        audio_bytes = buf.read()
        
        chunk = AudioChunk(
            sentence_index=sentence_idx,
            audio_bytes=audio_bytes,
            audio_np=audio_np,
            sample_rate=self.tts.sr or 24000,
            start_time=self._cumulative_audio_time,
            duration=duration,
        )
        
        self._cumulative_audio_time += duration
        await self._audio_queue.put(chunk)
    
    async def _send_fallback_frames(self, chunk_idx: int, audio_chunk):
        """Send fallback blendshape frames using last successful frame."""
        if self._last_successful_frame is not None:
            # Repeat last successful frame
            num_frames = int(audio_chunk.duration * 60)  # 60 fps
            frames = np.tile(self._last_successful_frame, (num_frames, 1))
            
            await self.ws.send_json(
                make_blendshapes_msg(
                    chunk_index=chunk_idx,
                    sentence_index=audio_chunk.sentence_index,
                    frames=frames.tolist(),
                    start_time=audio_chunk.start_time,
                    end_time=audio_chunk.start_time + audio_chunk.duration,
                    frame_rate=60,
                    is_final=False,
                )
            )
