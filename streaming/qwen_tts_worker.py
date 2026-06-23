"""
QwenTTSWorker — public entry point.

Split into two focused modules:
  qwen_tts_model.py     — model loading, warmup, voice clone prompt, lifecycle
  qwen_tts_inference.py — streaming and batch audio generation

All existing imports remain valid:
  from streaming.qwen_tts_worker import QwenTTSWorker, AudioChunk
"""
from streaming.qwen_tts_model import AudioChunk, QwenTTSModelMixin  # noqa: F401 (re-export AudioChunk)
from streaming.qwen_tts_inference import QwenTTSInferenceMixin


class QwenTTSWorker(QwenTTSModelMixin, QwenTTSInferenceMixin):
    """
    Streaming TTS worker using Qwen3-TTS.
    Reference: https://github.com/QwenLM/Qwen3-TTS

    QwenTTSModelMixin   provides: __init__, _load_model, create_voice_clone_prompt, cancel, reset
    QwenTTSInferenceMixin provides: stream_sentence, process_sentence, _generate_audio_sync,
                                    _fallback_synthesis, _normalize_audio
    """
