"""
Global mutable state shared across all routers.
Import these objects — never re-create them.

Locks:
  _tts_worker_lock   threading.Lock  — protects _tts_model_worker (sync executor context)
  _voice_store_lock  asyncio.Lock    — protects _voice_store (async routes)
  _stt_worker_lock   threading.Lock  — protects _stt_worker (sync executor context)

Async primitives (_voice_store_lock, _gpu_semaphore) are assigned during app
lifespan startup so the asyncio event loop is guaranteed to exist when created.
"""
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

# ── RAG sessions ────────────────────────────────────────────────────────────
# Values: (chain, created_at_epoch) where chain is either a
#   ConversationalRetrievalChain (RAG) or {"type":"direct",...} (direct LLM)
# TTL is enforced by set_conversation() / get_conversation() helpers below.
_SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 3600)))  # default 2h

conversations: Dict[str, Tuple[Any, float]] = {}
_conversations_lock = threading.Lock()  # protects all reads/writes to conversations dict
embeddings_model = None
blendshape_model = None


def set_conversation(session_id: str, chain: Any) -> None:
    """Store a RAG/direct session with the current timestamp."""
    with _conversations_lock:
        conversations[session_id] = (chain, time.monotonic())


def get_conversation(session_id: str) -> Optional[Any]:
    """Return the chain if it exists and has not expired, else None."""
    with _conversations_lock:
        entry = conversations.get(session_id)
        if entry is None:
            return None
        chain, created_at = entry
        if time.monotonic() - created_at > _SESSION_TTL_SECONDS:
            del conversations[session_id]
            return None
        return chain


def purge_expired_conversations() -> int:
    """Remove all expired sessions. Returns number of sessions removed."""
    now = time.monotonic()
    with _conversations_lock:
        expired = [
            sid for sid, (_, created_at) in list(conversations.items())
            if now - created_at > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            conversations.pop(sid, None)
    return len(expired)


# ── TTS ─────────────────────────────────────────────────────────────────────
_tts_worker_lock = threading.Lock()
_tts_model_worker = None
_voice_store: Dict[str, Dict[str, Any]] = {}
_voice_store_lock = None       # asyncio.Lock — set by lifespan; guard: if None, service not ready
_tts_warmup_lock = None        # asyncio.Lock — serialises concurrent warmup calls (set by lifespan)
_gpu_semaphore = None          # asyncio.Semaphore — limits concurrent GPU pipelines
_tts_reference_audio_path: Optional[str] = os.getenv("TTS_REF_AUDIO_PATH")
_tts_reference_text: Optional[str] = os.getenv("TTS_REF_TEXT")

# ── STT ─────────────────────────────────────────────────────────────────────
_stt_worker = None
_stt_worker_lock = threading.Lock()