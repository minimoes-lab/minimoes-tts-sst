"""
Simple coordinator for testing without WebSocket dependency.
"""
import asyncio
from typing import List, AsyncIterator

from streaming.qwen_tts_worker import AudioChunk


class KyutaiCoordinator:
    """Simplified coordinator for testing."""
    
    def __init__(self, tts_worker, blendshape_worker=None, performance_monitor=None):
        self.tts = tts_worker
        self.bs = blendshape_worker
        self.monitor = performance_monitor
        self._cancelled = False
        self._cumulative_time = 0.0
    
    async def stream_audio(self, text_chunks: List[str]) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from text."""
        self._cancelled = False
        self._cumulative_time = 0.0
        
        for idx, text in enumerate(text_chunks):
            if self._cancelled:
                break
            
            chunk = await self.tts.process_sentence(
                sentence=text,
                sentence_index=idx,
                cumulative_time=self._cumulative_time,
                voice_preset=None
            )
            
            if chunk:
                self._cumulative_time += chunk.duration
                yield chunk
    
    def cancel(self):
        """Cancel streaming."""
        self._cancelled = True
        if self.tts:
            self.tts.cancel()
