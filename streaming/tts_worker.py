import asyncio
import io
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch


@dataclass
class AudioChunk:
    sentence_index: int
    audio_bytes: bytes        # raw WAV bytes
    audio_np: np.ndarray      # float32 numpy array (for feature extraction)
    sample_rate: int
    start_time: float         # seconds from conversation start
    duration: float           # seconds


class TTSWorker:
    def __init__(self, bark_processor, bark_model, bark_device, bark_sr):
        self.processor = bark_processor
        self.model = bark_model
        self.device = bark_device
        self.sr = bark_sr
        self._cancelled = False

    async def process_sentence(
        self,
        sentence: str,
        sentence_index: int,
        cumulative_time: float,
        voice_preset: Optional[str] = None,
    ) -> Optional[AudioChunk]:
        """Generate TTS audio for a single sentence. Runs in thread pool."""
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

    # ------------------------------------------------------------------
    # Synchronous Bark generation (runs inside thread-pool executor)
    # Core logic extracted from api.py generate_speech_content_base64
    # ------------------------------------------------------------------

    def _generate_audio_sync(
        self, text: str, voice_preset: Optional[str]
    ) -> Optional[Tuple[np.ndarray, bytes]]:
        try:
            start = time.time()

            inputs = self._prepare_inputs(text, voice_preset)
            if inputs is None:
                return None

            converted_inputs = self._convert_inputs_to_device(inputs)
            self._ensure_attention_mask(converted_inputs)

            pad_token_id = self._get_pad_token_id()

            # Generate audio
            self.model.eval()
            with torch.no_grad():
                try:
                    gen_kwargs = (
                        {"pad_token_id": pad_token_id}
                        if pad_token_id is not None
                        else {}
                    )
                    bark_output = self.model.generate(
                        **converted_inputs, **gen_kwargs
                    )
                except TypeError:
                    bark_output = self.model.generate(**converted_inputs)

            audio_np = self._extract_audio_array(bark_output)
            if audio_np is None:
                return None

            # Normalize
            audio_np = self._normalize_audio(audio_np)

            # Build WAV bytes
            sr = int(self.sr) if self.sr else 24000
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(
                np.int16
            )
            buf = io.BytesIO()
            wavfile.write(buf, sr, audio_int16)
            buf.seek(0)
            wav_bytes = buf.read()

            end = time.time()
            print(
                f"[{datetime.now()}] [TTS Worker] Sentence generated in "
                f"{end - start:.2f}s, {len(audio_np)} samples"
            )

            return audio_np, wav_bytes

        except Exception as e:
            print(
                f"[{datetime.now()}] [TTS Worker] ERROR: {repr(e)}"
            )
            return None

    # ------------------------------------------------------------------
    # Helper: prepare processor inputs (mirrors api.py:484-511)
    # ------------------------------------------------------------------

    def _prepare_inputs(self, text: str, voice_preset: Optional[str]):
        attempts = [
            {"voice_preset": voice_preset, "return_tensors": "pt", "padding": True},
            {"voice_preset": voice_preset, "return_tensors": "pt"},
            {"return_tensors": "pt", "padding": True},
            {"return_tensors": "pt"},
            {},
        ]
        last_exc = None
        for kw in attempts:
            try:
                safe_kw = {k: v for k, v in kw.items() if v is not None}
                inputs = self.processor(text, **safe_kw)
                return inputs
            except Exception as e:
                last_exc = e

        # Tokenizer fallback
        if hasattr(self.processor, "tokenizer"):
            try:
                return self.processor.tokenizer(
                    text, return_tensors="pt", padding=True
                )
            except Exception:
                pass

        print(
            f"[{datetime.now()}] [TTS Worker] Failed to prepare inputs: "
            f"{repr(last_exc)}"
        )
        return None

    # ------------------------------------------------------------------
    # Helper: recursive device conversion (mirrors api.py:436-478)
    # ------------------------------------------------------------------

    def _to_tensor_strict(self, obj, device):
        if torch.is_tensor(obj):
            if obj.dtype in (torch.int8, torch.int16, torch.int32):
                obj = obj.long()
            return obj.to(device)
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.integer):
                return torch.tensor(obj, dtype=torch.long, device=device)
            return torch.tensor(obj, dtype=torch.float32, device=device)
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return torch.tensor([], device=device)
            if all(isinstance(i, int) for i in obj):
                return torch.tensor(list(obj), dtype=torch.long, device=device)
            if all(isinstance(i, (float, int)) for i in obj):
                return torch.tensor(list(obj), dtype=torch.float32, device=device)
            return type(obj)([self._to_tensor_strict(x, device) for x in obj])
        if hasattr(obj, "__array__"):
            try:
                return self._to_tensor_strict(np.asarray(obj), device)
            except Exception:
                pass
        return obj

    def _recursively_convert(self, obj, device):
        if isinstance(obj, Mapping):
            return {
                k: self._recursively_convert(v, device)
                for k, v in obj.items()
            }
        if hasattr(obj, "to") and callable(obj.to) and not torch.is_tensor(obj):
            try:
                moved = obj.to(device)
                if isinstance(moved, Mapping):
                    return {
                        k: self._recursively_convert(v, device)
                        for k, v in moved.items()
                    }
                obj = moved
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            return type(obj)(
                [self._recursively_convert(v, device) for v in obj]
            )
        return self._to_tensor_strict(obj, device)

    def _convert_inputs_to_device(self, inputs):
        # Try .to(device) on the whole object first
        if hasattr(inputs, "to") and callable(inputs.to):
            try:
                inputs = inputs.to(self.device)
            except Exception:
                pass
        return self._recursively_convert(inputs, self.device)

    # ------------------------------------------------------------------
    # Helper: ensure attention mask exists (mirrors api.py:532-544)
    # ------------------------------------------------------------------

    def _ensure_attention_mask(self, converted_inputs):
        if (
            isinstance(converted_inputs, dict)
            and "input_ids" in converted_inputs
            and "attention_mask" not in converted_inputs
        ):
            try:
                input_ids = converted_inputs["input_ids"]
                if torch.is_tensor(input_ids):
                    pad_id = getattr(self.model.config, "pad_token_id", None)
                    if pad_id is not None:
                        mask = (input_ids != int(pad_id)).long().to(self.device)
                    else:
                        mask = torch.ones_like(
                            input_ids, dtype=torch.long, device=self.device
                        )
                    converted_inputs["attention_mask"] = mask
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helper: get pad_token_id (mirrors api.py:576-582)
    # ------------------------------------------------------------------

    def _get_pad_token_id(self) -> Optional[int]:
        try:
            cfg_pad = getattr(self.model.config, "pad_token_id", None)
            if cfg_pad is not None and int(cfg_pad) >= 0:
                return int(cfg_pad)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Helper: extract audio from Bark output (mirrors api.py:596-616)
    # ------------------------------------------------------------------

    def _extract_audio_array(self, bark_output) -> Optional[np.ndarray]:
        audio_array = None
        if isinstance(bark_output, dict):
            for key in ("audio", "audios", "waveform", "wav", "output_audio"):
                if key in bark_output:
                    audio_array = bark_output[key]
                    break
            if audio_array is None and "outputs" in bark_output:
                cand = bark_output["outputs"]
                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                    audio_array = cand[0]
        elif isinstance(bark_output, (list, tuple)):
            audio_array = bark_output[0]
        else:
            audio_array = bark_output

        if audio_array is None:
            return None

        if hasattr(audio_array, "cpu"):
            return audio_array.cpu().numpy().squeeze()
        return np.asarray(audio_array).squeeze()

    # ------------------------------------------------------------------
    # Helper: normalize audio (mirrors api.py:618-631)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_audio(audio_np: np.ndarray) -> np.ndarray:
        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2 and audio_np.shape[0] < audio_np.shape[-1]:
                audio_np = audio_np.mean(axis=0)
            else:
                audio_np = audio_np.mean(axis=-1)
        if np.issubdtype(audio_np.dtype, np.floating):
            return audio_np.astype(np.float32)
        return audio_np.astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self):
        self._cancelled = True

    def reset(self):
        self._cancelled = False
