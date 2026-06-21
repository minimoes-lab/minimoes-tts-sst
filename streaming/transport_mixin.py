"""
Transport helpers for KyutaiStreamCoordinator.
Handles all WebSocket send operations so the coordinator stays free of wire-format details.
"""
import base64
import io
from datetime import datetime
from typing import Optional, List

import numpy as np
import scipy.io.wavfile as wavfile

from streaming.protocol import (
    make_audio_chunk_msg,
    make_blendshapes_msg,
    make_idle_frames_msg,
    make_status_msg,
)
from streaming.idle_frames import generate_idle_frames


class TransportMixin:
    """
    Mixed into KyutaiStreamCoordinator.
    Requires self.ws, self.tts, self._chunk_ms, self._cancelled,
    self._cumulative_audio_time, self._sentence_index,
    self._last_blendshape_frame, self._last_successful_frame,
    self.config.
    """

    async def _send_status(self, status: str, message: str):
        try:
            await self.ws.send_json(make_status_msg(status, message))
        except Exception:
            pass

    async def _send_audio_pcm16(self, chunk_idx: int, audio_chunk) -> int:
        """Emit PCM16 LE chunks over WS for real-time playback."""
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
            seg_end_time   = audio_chunk.start_time + (end / sr)

            try:
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
            except Exception:
                return chunk_idx

            if chunk_idx == 0 or chunk_idx % 20 == 0:
                print(
                    f"[{datetime.now()}] [Kyutai Audio] sent_chunk={chunk_idx} "
                    f"bytes={len(seg_bytes)} sr={sr} sentence_index={audio_chunk.sentence_index}"
                )

            chunk_idx += 1
            sample_cursor = end

        return chunk_idx

    async def _send_fallback_frames(self, chunk_idx: int, audio_chunk):
        if self._last_successful_frame is None:
            return
        num_frames = int(audio_chunk.duration * 60)
        frames = np.tile(self._last_successful_frame, (num_frames, 1))
        try:
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
        except Exception:
            pass

    async def _send_idle_transition(self):
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

    async def _generate_silence_chunk(self, sentence_idx: int):
        from streaming.qwen_tts_worker import AudioChunk
        duration = 0.5
        sr = self.tts.sr or 24000
        samples = int(duration * sr)
        audio_np = np.zeros(samples, dtype=np.float32)

        buf = io.BytesIO()
        wavfile.write(buf, sr, (audio_np * 32767.0).astype(np.int16))
        buf.seek(0)

        chunk = AudioChunk(
            sentence_index=sentence_idx,
            audio_bytes=buf.read(),
            audio_np=audio_np,
            sample_rate=sr,
            start_time=self._cumulative_audio_time,
            duration=duration,
        )
        self._cumulative_audio_time += duration
        await self._audio_queue.put(chunk)
