"""
Global mutable state shared across all routers.
Import these objects — never re-create them.
"""
import os
import threading
from typing import Dict, Any, Optional

# ── RAG sessions ────────────────────────────────────────────────────────────
# Values are either a ConversationalRetrievalChain (RAG mode)
# or a dict {"type": "direct", "history": [...], "system_prompt": str} (direct LLM mode)
conversations: Dict[str, Any] = {}
embeddings_model = None
blendshape_model = None

# ── TTS ─────────────────────────────────────────────────────────────────────
_tts_worker_lock = threading.Lock()
_tts_model_worker = None
_voice_store: Dict[str, Dict[str, Any]] = {}
_tts_reference_audio_path: Optional[str] = os.getenv("TTS_REF_AUDIO_PATH")
_tts_reference_text: Optional[str] = os.getenv("TTS_REF_TEXT")

# ── STT ─────────────────────────────────────────────────────────────────────
_stt_worker = None
_stt_worker_lock = threading.Lock()
