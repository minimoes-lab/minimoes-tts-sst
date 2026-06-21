"""
Qwen3-TTS Worker for streaming audio generation.
Reference: https://github.com/QwenLM/Qwen3-TTS
"""
import asyncio
import io
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch


@dataclass
class AudioChunk:
    sentence_index: int
    audio_bytes: bytes
    audio_np: np.ndarray
    sample_rate: int
    start_time: float
    duration: float


class QwenTTSWorker:
    """Streaming TTS worker using Qwen3-TTS for lower latency."""

    _shared_lock = threading.Lock()
    _shared_model = None
    _shared_speakers = None
    _shared_default_speaker = None
    _shared_loaded_device = None
    _shared_model_loaded = False
    # Single persistent thread for all GPU inference.
    # torch.compile(mode='reduce-overhead') stores CUDA graph state in thread-local
    # storage (TLS). Creating a new ThreadPoolExecutor per request spawns a new thread
    # that has no TLS state → AssertionError on the 2nd request.  Reusing the same
    # executor guarantees all inference (warmup + runtime) runs on the same thread.
    _shared_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    
    def __init__(self, device="cuda", use_qwen3=True, reference_audio_path=None, reference_text: Optional[str] = None, raise_on_error: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.sr = 24000
        self._cancelled = False
        self.use_qwen3 = use_qwen3
        self.model_loaded = False
        self.speakers = []
        self.default_speaker = None
        self.reference_audio_path = reference_audio_path
        self.reference_text: Optional[str] = reference_text
        self.raise_on_error = raise_on_error
        self.voice_clone_prompt = None  # set by _load_model() if reference audio provided
        self._load_model()
    
    def _load_model(self):
        """Load Qwen3-TTS model using qwen_tts library with 6x optimizations."""
        if not self.use_qwen3:
            print(f"[{datetime.now()}] [Qwen TTS] Qwen3 disabled, using fallback")
            return
            
        try:
            reuse_shared = False
            with self.__class__._shared_lock:
                if (
                    self.__class__._shared_model_loaded
                    and self.__class__._shared_model is not None
                    and self.__class__._shared_loaded_device == self.device
                ):
                    self.model = self.__class__._shared_model
                    self.speakers = self.__class__._shared_speakers or []
                    self.default_speaker = (
                        self.__class__._shared_default_speaker
                        if self.__class__._shared_default_speaker
                        else (self.speakers[0] if self.speakers else "aiden")
                    )
                    self.model_loaded = True
                    self.sr = 24000
                    reuse_shared = True

            if not reuse_shared:
                print(f"[{datetime.now()}] [Qwen TTS] Loading Qwen3-TTS model...")
            
            from qwen_tts import Qwen3TTSModel
            
            # Base model supports streaming inference (voice clone)
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

            if not reuse_shared:
                self.speakers = []
                self.default_speaker = None
                torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                try:
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=self.device,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(f"[{datetime.now()}] [Qwen TTS] Warning: flash_attention_2 not available: {e}")
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=self.device,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )
            
            # OPTIMIZATION 6x: Enable streaming optimizations (torch.compile + CUDA graphs)
            if (not reuse_shared) and self.device == "cuda":
                print(f"[{datetime.now()}] [Qwen TTS] Enabling 6x streaming optimizations...")
                try:
                    # After warmup we will have pre-compiled graphs for all common input
                    # sizes.  Tell the inductor to skip recording new graphs for any
                    # unseen size at runtime — fall back to eager instead of adding
                    # latency on the first call with an unexpected shape.
                    try:
                        import torch._inductor.config as _ic
                        _ic.triton.cudagraph_skip_dynamic_graphs = True
                    except Exception:
                        pass

                    if hasattr(self.model, "enable_streaming_optimizations"):
                        self.model.enable_streaming_optimizations(
                            decode_window_frames=80,
                            use_compile=True,
                            use_cuda_graphs=False,  # reduce-overhead already captures CUDA graphs; enabling both causes nested-graph RuntimeError
                            compile_mode="reduce-overhead",
                            use_fast_codebook=False,
                            compile_codebook_predictor=True,
                        )
                        print(f"[{datetime.now()}] [Qwen TTS] Two-phase streaming optimizations enabled")
                    else:
                        print(f"[{datetime.now()}] [Qwen TTS] Warning: enable_streaming_optimizations not available")
                except Exception as opt_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warning: Could not enable optimizations: {opt_err}")
            
            # Build voice clone prompt if provided (optional for warmup / model preload)
            if self.reference_audio_path and self.reference_text:
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=self.reference_audio_path,
                    ref_text=self.reference_text,
                )

            # Create the shared persistent executor now (before warmup) so that
            # warmup JIT-compiles CUDA graphs on the SAME thread that all future
            # inference calls will use.  This must happen exactly once.
            with self.__class__._shared_lock:
                if self.__class__._shared_executor is None:
                    self.__class__._shared_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="qwen_tts_gpu",
                    )
                    # Eagerly initialise the thread so it exists before warmup.
                    self.__class__._shared_executor.submit(lambda: None).result()

            if (not reuse_shared) and self.device == "cuda":
                try:
                    warmup_prompt = self.voice_clone_prompt

                    # If no real reference audio yet, build a synthetic one (1s of silence)
                    # so the JIT compilation and CUDA graph recording happen at startup,
                    # not on the first real user request (which would add 30s+ latency).
                    if warmup_prompt is None and hasattr(self.model, "create_voice_clone_prompt"):
                        try:
                            import tempfile, struct
                            _sr_w = 24000
                            _n_w  = _sr_w  # 1 second of silence
                            _wav_header = struct.pack(
                                "<4sI4s4sIHHIIHH4sI",
                                b"RIFF", 36 + _n_w * 2, b"WAVE",
                                b"fmt ", 16, 1, 1, _sr_w, _sr_w * 2, 2, 16,
                                b"data", _n_w * 2,
                            )
                            _wav_data = b"\x00" * (_n_w * 2)
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
                                _f.write(_wav_header + _wav_data)
                                _syn_path = _f.name
                            warmup_prompt = self.model.create_voice_clone_prompt(
                                ref_audio=_syn_path,
                                ref_text="Hello, this is a warmup.",
                            )
                            print(f"[{datetime.now()}] [Qwen TTS] Synthetic warmup prompt created")
                        except Exception as _we:
                            print(f"[{datetime.now()}] [Qwen TTS] Synthetic warmup prompt failed: {_we}")

                    if warmup_prompt is not None:
                        def _warmup():
                            for wtext in ["Hello.", "This is a warmup."]:
                                for _chunk, _sr in self.model.stream_generate_voice_clone(
                                    text=wtext,
                                    language="English",
                                    voice_clone_prompt=warmup_prompt,
                                    emit_every_frames=12,
                                    decode_window_frames=80,
                                    overlap_samples=512,
                                    first_chunk_emit_every=5,
                                    first_chunk_decode_window=48,
                                    first_chunk_frames=48,
                                ):
                                    pass  # consume fully to ensure complete graph capture
                        # Run warmup on the persistent thread so CUDA graphs are
                        # recorded on the exact thread that will serve all requests.
                        self.__class__._shared_executor.submit(_warmup).result()
                        print(f"[{datetime.now()}] [Qwen TTS] Warmup complete (CUDA graphs recorded)")
                    else:
                        print(f"[{datetime.now()}] [Qwen TTS] Warmup skipped (no prompt available)")
                except Exception as warmup_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warmup error: {warmup_err}")
            
            self.sr = 24000  # Qwen3-TTS uses 24kHz
            self.model_loaded = True

            with self.__class__._shared_lock:
                self.__class__._shared_model = self.model
                self.__class__._shared_speakers = self.speakers
                self.__class__._shared_default_speaker = self.default_speaker
                self.__class__._shared_loaded_device = self.device
                self.__class__._shared_model_loaded = True

            print(f"[{datetime.now()}] [Qwen TTS] Qwen3-TTS loaded successfully on {self.device}")
            print(f"[{datetime.now()}] [Qwen TTS] Available speakers: {self.speakers}")
            
        except Exception as e:
            print(f"[{datetime.now()}] [Qwen TTS] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            self.sr = 24000  # Fallback uses 24kHz
            if self.raise_on_error:
                raise RuntimeError(f"Qwen3-TTS model failed to load: {e}") from e
            print(f"[{datetime.now()}] [Qwen TTS] Using fallback synthesis")

    def create_voice_clone_prompt(self, ref_audio_path: str, ref_text: str):
        if self.model is None or not self.model_loaded:
            raise RuntimeError("Model not loaded")
        return self.model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
    
    async def stream_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        language: str = "English",
        voice_clone_prompt=None,
        emit_every_frames: int = 12,
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

        print(f"[{datetime.now()}] [Qwen TTS stream] sentence={sentence_index} lang={language!r} "
              f"first_chunk_emit={first_chunk_emit_every} emit={emit_every_frames} dw={decode_window_frames} overlap={overlap_samples}")

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
                    first_chunk_emit_every=first_chunk_emit_every,
                    first_chunk_decode_window=first_chunk_decode_window,
                    first_chunk_frames=first_chunk_frames,
                    max_frames=10000,
                ):
                    if _stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(q.put((chunk, sr)), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(_SENTINEL), loop).result()

        # Reuse the class-level persistent executor so every inference call runs
        # on the same background thread where CUDA graphs were compiled during warmup.
        executor = self.__class__._shared_executor
        if executor is None:
            # Fallback: executor not yet created (model loaded without warmup path).
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="qwen_tts_gpu"
            )
            executor.submit(lambda: None).result()
            with self.__class__._shared_lock:
                if self.__class__._shared_executor is None:
                    self.__class__._shared_executor = executor
                else:
                    executor = self.__class__._shared_executor
        fut = loop.run_in_executor(executor, _run_sync)

        t_cursor = float(cumulative_time)
        generated_duration = 0.0
        chunk_count = 0
        start_time = time.time()

        try:
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    break
                if self._cancelled:
                    _stop_event.set()
                    break
                chunk, sr = item
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                chunk = chunk.astype(np.float32)
                dur = float(len(chunk) / sr) if sr else 0.0

                chunk_count += 1
                if chunk_count == 1:
                    print(f"[{datetime.now()}] [Qwen TTS stream] First chunk after {time.time()-start_time:.2f}s")

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
                    print(f"[{datetime.now()}] [Qwen TTS stream] Safety cap: {generated_duration:.2f}s >= {max_duration:.1f}s after {chunk_count} chunks")
                    _stop_event.set()
                    break
        finally:
            _stop_event.set()
            await fut
            # Do NOT shut down the shared executor — it must stay alive for future requests.

        total_elapsed = time.time() - start_time
        print(f"[{datetime.now()}] [Qwen TTS stream] Completed {chunk_count} chunks in {total_elapsed:.2f}s ({generated_duration:.2f}s audio)")

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
        result = await loop.run_in_executor(
            None,
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
                
                # Get first audio from list
                audio_np = audio_list[0]
                
                # Convert to numpy if it's a tensor
                if isinstance(audio_np, torch.Tensor):
                    audio_np = audio_np.cpu().numpy()
                
                # Ensure it's float32
                audio_np = audio_np.astype(np.float32)
                
                # Normalize and convert to WAV
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
        """Simple fallback synthesis for testing."""
        # Generate simple sine wave based on text length
        duration = min(len(text) * 0.05, 10.0)  # ~50ms per char, max 10s
        samples = int(self.sr * duration)
        t = np.linspace(0, duration, samples)
        
        # Simple speech-like synthesis with formants
        audio = np.zeros(samples)
        audio += 0.3 * np.sin(2 * np.pi * 200 * t)  # F1
        audio += 0.2 * np.sin(2 * np.pi * 400 * t)  # F2
        audio += 0.1 * np.sin(2 * np.pi * 800 * t)  # F3
        
        # Add envelope
        envelope = np.exp(-3 * t / duration)
        audio *= envelope
        
        audio = audio.astype(np.float32)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        
        buf = io.BytesIO()
        wavfile.write(buf, self.sr, audio_int16)
        buf.seek(0)
        wav_bytes = buf.read()
        
        return audio, wav_bytes
    
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
    
    def cancel(self):
        """Cancel ongoing generation."""
        self._cancelled = True
    
    def reset(self):
        """Reset worker state."""
        self._cancelled = False
