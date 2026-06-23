"""
QwenTTSInferenceMixin — streaming and batch audio generation.
Imported by QwenTTSWorker in qwen_tts_worker.py.
"""
import asyncio
import concurrent.futures
import io
import time
import threading
from datetime import datetime
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from streaming.qwen_tts_model import AudioChunk, QwenTTSModelMixin


class QwenTTSInferenceMixin:
    """
    Audio generation on top of the loaded model from QwenTTSModelMixin.
    Requires self to also inherit QwenTTSModelMixin (guaranteed by QwenTTSWorker).
    Accesses model state via QwenTTSModelMixin class attributes directly to avoid
    dependency on self.__class__ MRO ordering.
    """

    async def stream_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        language: str = "English",
        voice_clone_prompt=None,
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 512,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream PCM chunks for one sentence using Base model voice-clone streaming.

        Runs GPU inference in a thread-pool executor so the asyncio event loop
        stays free for WebSocket sends, interrupt handling, and blendshape work.
        """
        if self._cancelled:
            return
        if self.model is None or not self.model_loaded:
            return

        prompt = voice_clone_prompt
        if prompt is None:
            raise ValueError(
                "voice_clone_prompt must be provided per-request; "
                "instance-level fallback is unsafe in a shared worker"
            )

        print(
            f"[{datetime.now()}] [Qwen TTS stream] sentence={sentence_index} lang={language!r} "
            f"first_chunk_emit={first_chunk_emit_every} emit={emit_every_frames} "
            f"dw={decode_window_frames} overlap={overlap_samples}"
        )

        # Safety cap: 1.5× word-count estimate (2.5 wps), min 3s, hard cap 30s.
        word_count = max(1, len(sentence.split()))
        max_duration = min(30.0, max(3.0, word_count / 2.5 * 1.5))
        print(f"[{datetime.now()}] [Qwen TTS stream] max_duration={max_duration:.1f}s for {word_count} words")

        loop = asyncio.get_running_loop()
        # maxsize=0 = unbounded; thread never blocks on put(), no deadlock risk.
        q: asyncio.Queue = asyncio.Queue(maxsize=0)
        _SENTINEL = object()
        _stop_event = threading.Event()

        def _run_sync():
            try:
                for chunk, sr in self.model.stream_generate_voice_clone(
                    text=sentence,
                    language=language,
                    voice_clone_prompt=prompt,
                    emit_every_frames=emit_every_frames,
                    decode_window_frames=decode_window_frames,
                    overlap_samples=overlap_samples,
                    first_chunk_emit_every=0,
                    first_chunk_decode_window=first_chunk_decode_window,
                    first_chunk_frames=first_chunk_frames,
                    max_frames=10000,
                ):
                    if _stop_event.is_set():
                        break
                    try:
                        asyncio.run_coroutine_threadsafe(q.put((chunk, sr)), loop).result(timeout=5.0)
                    except Exception:
                        _stop_event.set()
                        break
            except Exception:
                _stop_event.set()
                raise
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(q.put(_SENTINEL), loop).result(timeout=5.0)
                except Exception:
                    pass

        # Reuse the class-level persistent executor so every inference call runs
        # on the same background thread where CUDA graphs were compiled during warmup.
        executor = QwenTTSModelMixin._shared_executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="qwen_tts_gpu"
            )
            executor.submit(lambda: None).result()
            with QwenTTSModelMixin._shared_lock:
                if QwenTTSModelMixin._shared_executor is None:
                    QwenTTSModelMixin._shared_executor = executor
                else:
                    executor = QwenTTSModelMixin._shared_executor
        fut = loop.run_in_executor(executor, _run_sync)

        t_cursor = float(cumulative_time)
        generated_duration = 0.0
        chunk_count = 0
        start_time = time.time()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    print(
                        f"[{datetime.now()}] [Qwen TTS stream] Chunk wait timeout (30s) "
                        f"— aborting sentence {sentence_index}"
                    )
                    _stop_event.set()
                    break
                if item is _SENTINEL:
                    break
                if self._cancelled:
                    _stop_event.set()
                    break
                chunk, sr = item
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()

                chunk_count += 1
                if chunk_count <= 3:
                    print(
                        f"[{datetime.now()}] [Qwen TTS stream] raw chunk#{chunk_count} "
                        f"shape={chunk.shape} dtype={chunk.dtype} "
                        f"min={float(chunk.min()):.4f} max={float(chunk.max()):.4f} sr={sr}"
                    )

                # Flatten to 1-D
                if chunk.ndim > 1:
                    chunk = chunk.squeeze()
                    if chunk.ndim > 1:
                        chunk = chunk[0]

                chunk = chunk.astype(np.float32)
                dur = float(chunk.shape[0] / sr) if sr else 0.0

                if chunk_count == 1:
                    elapsed_first = time.time() - start_time
                    print(
                        f"[{datetime.now()}] [Qwen TTS stream] First chunk after {elapsed_first:.2f}s, "
                        f"flat shape={chunk.shape} dur={dur:.3f}s "
                        + (f"RTF={elapsed_first/dur:.2f}x" if dur > 0 else "")
                    )

                if chunk_count > 1:
                    elapsed_total = time.time() - start_time
                    rtf_running = elapsed_total / generated_duration if generated_duration > 0 else 0.0
                    if chunk_count <= 8 or chunk_count % 10 == 0:
                        print(
                            f"[{datetime.now()}] [Qwen TTS stream] chunk#{chunk_count} "
                            f"dur={dur:.3f}s generated={generated_duration:.2f}s "
                            f"elapsed={elapsed_total:.2f}s RTF={rtf_running:.2f}x"
                        )

                yield AudioChunk(
                    sentence_index=sentence_index,
                    audio_bytes=b"",
                    audio_np=chunk,
                    sample_rate=int(sr),
                    start_time=t_cursor,
                    duration=dur,
                )
                t_cursor += dur
                generated_duration += dur
                if generated_duration >= max_duration:
                    print(
                        f"[{datetime.now()}] [Qwen TTS stream] Safety cap: "
                        f"{generated_duration:.2f}s >= {max_duration:.1f}s after {chunk_count} chunks"
                    )
                    _stop_event.set()
                    break
        finally:
            _stop_event.set()
            await fut
            # Do NOT shut down the shared executor — it must stay alive for future requests.

        total_elapsed = time.time() - start_time
        print(
            f"[{datetime.now()}] [Qwen TTS stream] Completed {chunk_count} chunks "
            f"in {total_elapsed:.2f}s ({generated_duration:.2f}s audio)"
        )

    async def process_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        voice_clone_prompt=None,
    ) -> Optional[AudioChunk]:
        """Generate full audio for a single sentence (fallback path)."""
        if self._cancelled:
            return None

        loop = asyncio.get_running_loop()
        executor = QwenTTSModelMixin._shared_executor
        result = await loop.run_in_executor(
            executor,
            self._generate_audio_sync,
            sentence,
            voice_clone_prompt,
        )

        if result is None or self._cancelled:
            return None

        audio_np, audio_bytes = result
        duration = len(audio_np) / self.sr

        return AudioChunk(
            sentence_index=sentence_index,
            audio_bytes=audio_bytes,
            audio_np=audio_np,
            sample_rate=self.sr,
            start_time=cumulative_time,
            duration=duration,
        )

    def _generate_audio_sync(
        self,
        text: str,
        voice_clone_prompt=None,
    ) -> Optional[Tuple[np.ndarray, bytes]]:
        """Synchronous audio generation with Qwen3-TTS (Base voice-clone fallback)."""
        try:
            start = time.time()

            if self.model is None or not self.model_loaded:
                return self._fallback_synthesis(text)

            prompt = voice_clone_prompt if voice_clone_prompt is not None else self.voice_clone_prompt
            if prompt is None:
                raise ValueError("voice_clone_prompt not initialized (missing reference audio/text)")

            with torch.no_grad():
                audio_tuple = self.model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=prompt,
                )

                if audio_tuple is None or len(audio_tuple) < 2:
                    return self._fallback_synthesis(text)

                audio_list, sr = audio_tuple

                if not audio_list or len(audio_list) == 0:
                    return self._fallback_synthesis(text)

                audio_np = audio_list[0]

                if isinstance(audio_np, torch.Tensor):
                    audio_np = audio_np.cpu().numpy()

                audio_np = audio_np.astype(np.float32)
                audio_np = self._normalize_audio(audio_np)
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)

                buf = io.BytesIO()
                wavfile.write(buf, sr, audio_int16)
                buf.seek(0)
                wav_bytes = buf.read()

                end = time.time()
                audio_duration = len(audio_np) / sr
                rtf = (end - start) / audio_duration if audio_duration > 0 else 0
                print(
                    f"[{datetime.now()}] [Qwen TTS] Generated in "
                    f"{end - start:.2f}s, {len(audio_np)} samples at {sr}Hz (RTF: {rtf:.2f}x)"
                )

                return audio_np, wav_bytes

        except Exception as e:
            print(f"[{datetime.now()}] [Qwen TTS] ERROR: {repr(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_synthesis(text)

    def _fallback_synthesis(self, text: str) -> Optional[Tuple[np.ndarray, bytes]]:
        """Sine-wave fallback for testing when the real model is unavailable."""
        duration = min(len(text) * 0.05, 10.0)
        samples = int(self.sr * duration)
        t = np.linspace(0, duration, samples)

        audio = np.zeros(samples)
        audio += 0.3 * np.sin(2 * np.pi * 200 * t)
        audio += 0.2 * np.sin(2 * np.pi * 400 * t)
        audio += 0.1 * np.sin(2 * np.pi * 800 * t)
        envelope = np.exp(-3 * t / duration)
        audio *= envelope
        audio = audio.astype(np.float32)

        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, self.sr, audio_int16)
        buf.seek(0)
        return audio, buf.read()

    @staticmethod
    def _normalize_audio(audio_np: np.ndarray) -> np.ndarray:
        """Normalize audio to float32 range [-1, 1]."""
        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2 and audio_np.shape[0] < audio_np.shape[-1]:
                audio_np = audio_np.mean(axis=0)
            else:
                audio_np = audio_np.mean(axis=-1)

        if np.issubdtype(audio_np.dtype, np.floating):
            return audio_np.astype(np.float32)
        return audio_np.astype(np.float32) / 32768.0
