import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from streaming.qwen_tts_worker import AudioChunk

BLEND_FRAMES = 16  # Increased from 8 to 16 frames for smoother temporal transitions (KeyFace method)


@dataclass
class BlendshapeChunk:
    sentence_index: int
    frames: np.ndarray   # shape (N, 68)
    start_time: float    # seconds
    end_time: float      # seconds
    frame_rate: int      # 60


class BlendshapeWorker:
    def __init__(self, blendshape_model, device, config):
        self.model = blendshape_model
        self.device = device
        self.config = config
        self._previous_tail_frames: Optional[np.ndarray] = None
        self._cancelled = False
        self._is_first_chunk = True

    async def process_audio_chunk(
        self, audio_chunk: AudioChunk
    ) -> Optional[BlendshapeChunk]:
        """Process an AudioChunk into blendshapes. Runs in thread pool."""
        if self._cancelled:
            return None

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._process_sync, audio_chunk
        )
        return result

    def _process_sync(self, audio_chunk: AudioChunk) -> Optional[BlendshapeChunk]:
        from utils.audio.extraction.extract_features import extract_audio_features
        from utils.audio.processing.audio_processing import process_audio_features

        try:
            # Step 1: Extract features from this sentence's WAV audio
            audio_features, y = extract_audio_features(
                audio_chunk.audio_bytes, from_bytes=True
            )

            if audio_features is None or y is None:
                print(
                    f"[{datetime.now()}] [BS Worker] Feature extraction "
                    f"returned None for sentence {audio_chunk.sentence_index}"
                )
                return None

            # DEBUG: Verify feature dimensions match model input
            expected_dim = self.config.get('input_dim', 256)
            actual_dim = audio_features.shape[1] if len(audio_features.shape) > 1 else audio_features.shape[0]
            print(f"[{datetime.now()}] [BS Worker] Features: {actual_dim}D (expected: {expected_dim}D)")
            
            if actual_dim != expected_dim:
                print(f"⚠️  DIMENSION MISMATCH! Model expects {expected_dim}, got {actual_dim}")

            # Step 2: Run blendshape inference
            # Only apply easing fade-in on the very first chunk
            bs_min_chunk_ms = int(self.config.get("bs_min_chunk_ms", 800) or 800)
            bs_min_samples = max(1, int((self.tts.sr or 24000) * (bs_min_chunk_ms / 1000.0)))
            blendshapes = process_audio_features(
                audio_features,
                self.model,
                self.device,
                self.config,
                apply_easing=self._is_first_chunk,
            )

            if self._is_first_chunk:
                self._is_first_chunk = False

            # Step 3: Cross-sentence blending with previous chunk's tail
            if self._previous_tail_frames is not None and len(blendshapes) > 0:
                blend_n = min(
                    BLEND_FRAMES,
                    len(self._previous_tail_frames),
                    len(blendshapes),
                )
                for i in range(blend_n):
                    alpha = (i + 1) / (blend_n + 1)
                    blendshapes[i] = (
                        (1 - alpha)
                        * self._previous_tail_frames[-(blend_n - i)]
                        + alpha * blendshapes[i]
                    )

            # Save tail frames for next chunk
            if len(blendshapes) >= BLEND_FRAMES:
                self._previous_tail_frames = blendshapes[-BLEND_FRAMES:].copy()

            frame_rate = self.config.get("frame_rate", 60)
            duration = len(blendshapes) / frame_rate

            print(
                f"[{datetime.now()}] [BS Worker] Sentence "
                f"{audio_chunk.sentence_index}: {len(blendshapes)} frames"
            )

            return BlendshapeChunk(
                sentence_index=audio_chunk.sentence_index,
                frames=blendshapes,
                start_time=audio_chunk.start_time,
                end_time=audio_chunk.start_time + duration,
                frame_rate=frame_rate,
            )

        except Exception as e:
            print(
                f"[{datetime.now()}] [BS Worker] ERROR: {repr(e)}"
            )
            return None

    def cancel(self):
        self._cancelled = True

    def reset(self):
        self._cancelled = False
        self._previous_tail_frames = None
        self._is_first_chunk = True
