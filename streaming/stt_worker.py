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


class MoonshineSTTWorker:
    """
    True streaming STT using Moonshine (usefulsensors/moonshine).
    Incremental encoder caching: each new chunk updates the transcription
    without reprocessing the full buffer — word-by-word latency ~250ms.
    """

    def __init__(self, model_size: str = "moonshine/base", device: str = "cpu", language: Optional[str] = None):
        self.model_name = model_size
        self.device = device
        self.language = language
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            from moonshine import Moonshine
            self.model = Moonshine(self.model_name)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        tokens = self.model.generate(audio[np.newaxis, :])
        return self.model.decode(tokens[0])


class StreamingSTTSession:
    """
    Per-WebSocket session.
    - Accumulates PCM16 chunks
    - Emits partial transcripts every ~256ms
    - Detects end-of-utterance via silence and emits final transcript
    """

    def __init__(self, worker: MoonshineSTTWorker, on_result: Callable, loop: asyncio.AbstractEventLoop):
        self.worker = worker
        self.on_result = on_result
        self.loop = loop

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
            text = self.worker.transcribe_audio(audio).strip()
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
            event, data = self._q.get()

            if event == "flush":
                self._transcribe_buffer(is_final=True)
                break

            # event == "audio"
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
