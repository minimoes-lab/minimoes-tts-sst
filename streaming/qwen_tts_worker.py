"""
Qwen3-TTS Worker for streaming audio generation.
Reference: https://github.com/QwenLM/Qwen3-TTS
"""
import asyncio
import io
import time
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
    
    def __init__(self, device="cuda", use_qwen3=True, reference_audio_path=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.sr = 24000
        self._cancelled = False
        self.use_qwen3 = use_qwen3
        self.model_loaded = False
        self.reference_audio_path = reference_audio_path
        self._load_model()
    
    def _load_model(self):
        """Load Qwen3-TTS model."""
        if not self.use_qwen3:
            print(f"[{datetime.now()}] [Qwen TTS] Qwen3 disabled, using fallback")
            return
            
        try:
            print(f"[{datetime.now()}] [Qwen TTS] Loading Qwen3-TTS model...")
            
            from qwen_tts import Qwen3TTSModel
            
            model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
            )
            
            self.sr = 12000  # Qwen3-TTS uses 12kHz
            self.model_loaded = True
            print(f"[{datetime.now()}] [Qwen TTS] Qwen3-TTS loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[{datetime.now()}] [Qwen TTS] Failed to load model: {e}")
            print(f"[{datetime.now()}] [Qwen TTS] Using fallback synthesis")
            self.model = None
            self.model_loaded = False
            self.sr = 24000  # Fallback uses 24kHz
    
    async def process_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        voice_preset: Optional[str] = None,
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
        self, text: str, voice_preset: Optional[str]
    ) -> Optional[Tuple[np.ndarray, bytes]]:
        """Synchronous audio generation with Qwen3-TTS."""
        try:
            start = time.time()
            
            if self.model is None or not self.model_loaded:
                return self._fallback_synthesis(text)
            
            with torch.no_grad():
                # Create a more expressive reference audio
                # Using varied frequencies and amplitude modulation for prosody
                ref_sr = 24000
                ref_duration = 3.0
                ref_samples = int(ref_sr * ref_duration)
                t = np.linspace(0, ref_duration, ref_samples)
                
                # Create expressive voice with pitch variation and prosody
                ref_audio_np = np.zeros(ref_samples, dtype=np.float32)
                
                # Base frequency with pitch contour (rising and falling)
                pitch_contour = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Pitch variation
                ref_audio_np += 0.4 * np.sin(2 * np.pi * pitch_contour * t)
                
                # Add harmonics for richer voice
                ref_audio_np += 0.25 * np.sin(2 * np.pi * pitch_contour * 2 * t)
                ref_audio_np += 0.15 * np.sin(2 * np.pi * pitch_contour * 3 * t)
                ref_audio_np += 0.08 * np.sin(2 * np.pi * pitch_contour * 4 * t)
                
                # Add amplitude modulation for prosody (stress patterns)
                stress_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * 2 * t)
                ref_audio_np *= stress_pattern
                
                # Add natural envelope with attack and decay
                attack = np.linspace(0, 1, int(ref_sr * 0.1))
                decay = np.linspace(1, 0.3, int(ref_sr * 0.5))
                sustain = np.ones(ref_samples - len(attack) - len(decay)) * 0.3
                envelope = np.concatenate([attack, sustain, decay])
                ref_audio_np *= envelope
                
                # Add slight noise for naturalness
                noise = np.random.normal(0, 0.02, ref_samples).astype(np.float32)
                ref_audio_np += noise
                
                # Normalize
                ref_audio_np = ref_audio_np / np.max(np.abs(ref_audio_np)) * 0.8
                
                # More expressive reference text with emotion markers
                ref_text = "Hello! I'm excited to help you today. How are you doing?"
                
                # Use Qwen3-TTS generate_voice_clone with tuple (audio, sr)
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="English",
                    ref_audio=(ref_audio_np, ref_sr),
                    ref_text=ref_text
                )
                
                if wavs is None or len(wavs) == 0:
                    return self._fallback_synthesis(text)
                
                # Get first audio output
                audio_np = wavs[0]
                
                # Normalize and convert to WAV
                audio_np = self._normalize_audio(audio_np)
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                
                buf = io.BytesIO()
                wavfile.write(buf, sr, audio_int16)
                buf.seek(0)
                wav_bytes = buf.read()
                
                end = time.time()
                print(
                    f"[{datetime.now()}] [Qwen TTS] Generated in "
                    f"{end - start:.2f}s, {len(audio_np)} samples"
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
