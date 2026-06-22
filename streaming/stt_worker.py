import numpy as np
import asyncio
import threading
import queue
import struct
from typing import Optional, Callable


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512          # ~32ms per chunk at 16kHz
PARTIAL_EVERY_N_CHUNKS = 8   # emit partial every ~256ms
SILENCE_THRESHOLD = 0.01     # RMS below this = silence
SILENCE_CHUNKS = 20          # ~640ms silence = end of utterance
NO_SPEECH_THRESHOLD = 0.85   # Whisper no_speech_prob above this → discard
# Regex for Whisper hallucinations: only dots, ellipsis, whitespace, or dashes
import re as _re
_HALLUCINATION_RE = _re.compile(r'^[\s.…\-–—*()[\]]+$')
# Max 5 minutes of PCM16 mono 16kHz = 5*60*16000*2 bytes = 9.6 MB
_MAX_PCM_BUFFER_BYTES = 5 * 60 * SAMPLE_RATE * 2


class STTWorker:

    def __init__(self, model_size: str = "base", device: str = "cpu", language: Optional[str] = None):
        self.model_name = model_size
        self.device = device
        self.language = language
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)

    def transcribe_audio(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        segments, info = self.model.transcribe(
            audio,
            language=language or self.language,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"threshold": 0.3},
        )
        if info.no_speech_prob > NO_SPEECH_THRESHOLD:
            return ""
        text = "".join(s.text for s in segments).strip()
        if _HALLUCINATION_RE.fullmatch(text):
            return ""
        return text


class StreamingSTTSession:
    """
    Per-WebSocket session.
    - Accumulates PCM16 chunks
    - Emits partial transcripts every ~256ms
    - Detects end-of-utterance via silence and emits final transcript
    """

    def __init__(self, worker: STTWorker, on_result: Callable, loop: asyncio.AbstractEventLoop, language: Optional[str] = None):
        self.worker = worker
        self.on_result = on_result
        self.loop = loop
        self.language = language

        self._pcm_buffer: bytes = b""          # raw PCM16 bytes
        self._chunk_count = 0
        self._silence_count = 0
        self._last_partial = ""
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def push_audio(self, pcm_bytes: bytes):
        self._q.put(("audio", pcm_bytes))

    def flush(self):
        self._q.put(("flush", b""))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pcm_to_float32(self, pcm: bytes) -> np.ndarray:
        n = len(pcm) // 2
        samples = struct.unpack(f"<{n}h", pcm[:n * 2])
        return np.array(samples, dtype=np.float32) / 32768.0

    def _rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    def _emit(self, result: dict):
        asyncio.run_coroutine_threadsafe(self.on_result(result), self.loop)

    def _transcribe_buffer(self, is_final: bool):
        if not self._pcm_buffer:
            return
        audio = self._pcm_to_float32(self._pcm_buffer)
        try:
            text = self.worker.transcribe_audio(audio, language=self.language).strip()
        except Exception as e:
            self._emit({"type": "error", "text": "", "error": str(e)})
            return

        if not text:
            return

        if is_final:
            self._emit({"type": "final", "text": text})
            self._pcm_buffer = b""
            self._last_partial = ""
            self._silence_count = 0
            self._chunk_count = 0
        else:
            if text != self._last_partial:
                self._last_partial = text
                self._emit({"type": "partial", "text": text})

    def _loop(self):
        while True:
            try:
                event, data = self._q.get(timeout=1.0)
            except Exception:
                # Timeout or queue error — check if we should keep waiting
                continue

            if event == "flush":
                self._transcribe_buffer(is_final=True)
                break

            # event == "audio"
            # Cap buffer to prevent OOM on long/abandoned sessions
            if len(self._pcm_buffer) + len(data) > _MAX_PCM_BUFFER_BYTES:
                # Drop oldest data to make room (keep most recent audio)
                keep = _MAX_PCM_BUFFER_BYTES - len(data)
                self._pcm_buffer = self._pcm_buffer[-keep:] if keep > 0 else b""
            self._pcm_buffer += data
            self._chunk_count += 1

            # Detect silence on this chunk
            audio_chunk = self._pcm_to_float32(data)
            if self._rms(audio_chunk) < SILENCE_THRESHOLD:
                self._silence_count += 1
            else:
                self._silence_count = 0

            # End-of-utterance: enough silence after some speech
            if self._silence_count >= SILENCE_CHUNKS and len(self._pcm_buffer) > SAMPLE_RATE * 2 * 0.3:
                self._transcribe_buffer(is_final=True)
                continue

            # Emit partial every N chunks
            if self._chunk_count % PARTIAL_EVERY_N_CHUNKS == 0:
                self._transcribe_buffer(is_final=False)
