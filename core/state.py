"""
Global mutable state shared across all routers.
Import these objects — never re-create them.

Locks:
  _tts_worker_lock   threading.Lock  — protects _tts_model_worker (sync executor context)
  _voice_store_lock  asyncio.Lock    — protects _voice_store (async routes)
  _stt_worker_lock   threading.Lock  — protects _stt_worker (sync executor context)

_voice_store_lock is assigned during app lifespan startup so the asyncio event loop
is guaranteed to exist when the Lock is created.
"""
import os
import threading
from typing import Any, Dict, Optional

# ── RAG sessions ────────────────────────────────────────────────────────────
# Values: ConversationalRetrievalChain (RAG) or {"type":"direct",...} (direct LLM)
conversations: Dict[str, Any] = {}
embeddings_model = None
blendshape_model = None

# ── TTS ─────────────────────────────────────────────────────────────────────
_tts_worker_lock = threading.Lock()
_tts_model_worker = None
_voice_store: Dict[str, Dict[str, Any]] = {}
_voice_store_lock = None  # asyncio.Lock — set by lifespan startup in api.py
_tts_reference_audio_path: Optional[str] = os.getenv("TTS_REF_AUDIO_PATH")
_tts_reference_text: Optional[str] = os.getenv("TTS_REF_TEXT")

# ── STT ─────────────────────────────────────────────────────────────────────
_stt_worker = None
_stt_worker_lock = threading.Lock()
