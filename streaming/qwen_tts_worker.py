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
from typing import Optional, Tuple

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
    
    def __init__(self, device="cuda", use_qwen3=True, reference_audio_path=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.sr = 24000
        self._cancelled = False
        self.use_qwen3 = use_qwen3
        self.model_loaded = False
        self.reference_audio_path = reference_audio_path
        self._load_model()
    
    def _load_model(self):
        """Load Qwen3-TTS model using qwen_tts library with 6x optimizations."""
        if not self.use_qwen3:
            print(f"[{datetime.now()}] [Qwen TTS] Qwen3 disabled, using fallback")
            return
            
        try:
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
                    return

            print(f"[{datetime.now()}] [Qwen TTS] Loading Qwen3-TTS model...")
            
            from qwen_tts import Qwen3TTSModel
            
            # Use ModelScope model path
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            
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
            if self.device == "cuda":
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
                        print(f"[{datetime.now()}] [Qwen TTS] Streaming optimizations enabled (6x faster)")
                    else:
                        print(f"[{datetime.now()}] [Qwen TTS] Warning: enable_streaming_optimizations not available")
                except Exception as opt_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warning: Could not enable optimizations: {opt_err}")
            
            # Get available speakers
            self.speakers = self.model.get_supported_speakers() or []
            self.default_speaker = self.speakers[0] if self.speakers else "aiden"

            if self.device == "cuda":
                try:
                    warmup_texts = [
                        "Hello.",
                        "This is a warmup.",
                    ]
                    for wtext in warmup_texts:
                        _ = self.model.generate_custom_voice(
                            text=wtext,
                            language="English",
                            speaker=self.default_speaker,
                            instruct=None,
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
            print(f"[{datetime.now()}] [Qwen TTS] Using fallback synthesis")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            self.sr = 24000  # Fallback uses 24kHz
    
    async def process_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        voice_preset: Optional[str] = None,
        tts_instruct: Optional[str] = None,
    ) -> Optional[AudioChunk]:
        """Generate TTS audio for a single sentence with streaming."""
        if self._cancelled:
            return None
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_audio_sync,
            sentence,
            voice_preset,
            tts_instruct,
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
        self, text: str, voice_preset: Optional[str], tts_instruct: Optional[str]
    ) -> Optional[Tuple[np.ndarray, bytes]]:
        """Synchronous audio generation with Qwen3-TTS (optimized 6x)."""
        try:
            start = time.time()
            
            if self.model is None or not self.model_loaded:
                return self._fallback_synthesis(text)
            
            with torch.no_grad():
                # Use optimized generate_custom_voice method
                # With torch.compile enabled, this is ~6x faster
                speaker = voice_preset if voice_preset else self.default_speaker
                audio_tuple = self.model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker=speaker,
                    instruct=tts_instruct,
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
