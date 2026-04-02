"""
Qwen3-TTS Worker for streaming audio generation.
Reference: https://github.com/QwenLM/Qwen3-TTS
"""
import asyncio
import io
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch

# PEP 318 FIX: Monkey-patch broken decorator in rekuenkdr fork before importing qwen_tts
try:
    import sys
    import types
    import functools
    
    # Create a mock module with the fixed decorator
    mock_module = types.ModuleType('mock_modeling')
    
    def fixed_check_model_inputs(**kwargs):
        """PEP 318 compliant factory decorator."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                return func(*args, **kwargs_inner)
            return wrapper
        return decorator
    
    mock_module.check_model_inputs = fixed_check_model_inputs
    
    # Pre-populate the import cache to intercept the broken decorator
    sys.modules['qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2'] = mock_module
    
except Exception:
    pass  # If patching fails, continue anyway


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
    
    def __init__(self, device="cuda", use_qwen3=True, reference_audio_path=None, reference_text: Optional[str] = None, raise_on_error: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.sr = 24000
        self._cancelled = False
        self.use_qwen3 = use_qwen3
        self.model_loaded = False
        self.speakers = []
        self.default_speaker = None
        self.reference_audio_path = reference_audio_path
        self.reference_text: Optional[str] = reference_text
        self.raise_on_error = raise_on_error
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
                    if hasattr(self.model, "enable_streaming_optimizations"):
                        self.model.enable_streaming_optimizations(
                            decode_window_frames=80,
                            use_compile=True,
                            use_cuda_graphs=False,  # Disabled for stability
                            compile_mode="reduce-overhead",
                            use_fast_codebook=True,
                            compile_codebook_predictor=True,
                            compile_talker=True,
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

            if (not reuse_shared) and self.device == "cuda":
                try:
                    warmup_texts = [
                        "Hello.",
                        "This is a warmup.",
                    ]
                    if self.voice_clone_prompt is not None:
                        for wtext in warmup_texts:
                            _ = self.model.generate_voice_clone(
                                text=wtext,
                                language="English",
                                voice_clone_prompt=self.voice_clone_prompt,
                            )
                    print(f"[{datetime.now()}] [Qwen TTS] Warmup complete")
                except Exception as warmup_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warmup skipped: {warmup_err}")
            
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
        overlap_samples: int = 128,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream PCM chunks for one sentence using Base model voice-clone streaming."""
        if self._cancelled:
            return
        if self.model is None or not self.model_loaded:
            return

        prompt = voice_clone_prompt if voice_clone_prompt is not None else self.voice_clone_prompt
        if prompt is None:
            raise ValueError("voice_clone_prompt not initialized (missing reference audio/text)")

        print(f"[{datetime.now()}] [Qwen TTS stream] Starting stream_generate_voice_clone for sentence {sentence_index}")
        print(f"[{datetime.now()}] [Qwen TTS stream] Phase 1: first_chunk_emit_every={first_chunk_emit_every}, first_chunk_decode_window={first_chunk_decode_window}")
        print(f"[{datetime.now()}] [Qwen TTS stream] Phase 2: emit_every_frames={emit_every_frames}, decode_window_frames={decode_window_frames}")
        
        t_cursor = float(cumulative_time)
        chunk_count = 0
        start_time = time.time()
        
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
        ):
            if self._cancelled:
                break
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.cpu().numpy()
            chunk = chunk.astype(np.float32)
            dur = float(len(chunk) / sr) if sr else 0.0
            
            chunk_count += 1
            if chunk_count == 1:
                elapsed = time.time() - start_time
                print(f"[{datetime.now()}] [Qwen TTS stream] First chunk after {elapsed:.2f}s")
            
            yield AudioChunk(
                sentence_index=sentence_index,
                audio_bytes=b"",
                audio_np=chunk,
                sample_rate=int(sr),
                start_time=t_cursor,
                duration=dur,
            )
            t_cursor += dur
        
        total_elapsed = time.time() - start_time
        print(f"[{datetime.now()}] [Qwen TTS stream] Completed {chunk_count} chunks in {total_elapsed:.2f}s")

    async def process_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        voice_preset: Optional[str] = None,
        tts_instruct: Optional[str] = None,
        voice_clone_prompt=None,
    ) -> Optional[AudioChunk]:
        """Generate full audio for a single sentence (fallback path)."""
        if self._cancelled:
            return None
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_audio_sync,
            sentence,
            voice_preset,
            tts_instruct,
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
        voice_preset: Optional[str],
        tts_instruct: Optional[str],
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
