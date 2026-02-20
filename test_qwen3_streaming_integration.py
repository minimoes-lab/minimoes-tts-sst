"""
Integration tests for Qwen3-TTS in the full streaming pipeline.
"""
import asyncio
import json
import time
from typing import List, Dict

import pytest
import numpy as np

from streaming.qwen_tts_worker import QwenTTSWorker
from streaming.test_coordinator import KyutaiCoordinator


class TestQwen3StreamingIntegration:
    """Test Qwen3-TTS integration with streaming coordinator."""
    
    @pytest.fixture
    def tts_worker(self):
        """Create TTS worker."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    @pytest.fixture
    def coordinator(self, tts_worker):
        """Create coordinator with Qwen TTS worker."""
        coord = KyutaiCoordinator(
            tts_worker=tts_worker,
            blendshape_worker=None,  # Not needed for TTS tests
            performance_monitor=None
        )
        yield coord
        coord.cancel()
    
    @pytest.mark.asyncio
    async def test_text_to_audio_stream(self, coordinator):
        """Test streaming text to audio conversion."""
        text_chunks = [
            "Hello, this is the first sentence.",
            "Here comes the second sentence.",
            "And finally the third one."
        ]
        
        audio_chunks = []
        
        async for chunk in coordinator.stream_audio(text_chunks):
            audio_chunks.append(chunk)
        
        assert len(audio_chunks) == len(text_chunks)
        
        for idx, chunk in enumerate(audio_chunks):
            assert chunk.sentence_index == idx
            assert chunk.duration > 0
            assert len(chunk.audio_bytes) > 0
            assert chunk.sample_rate == 24000
    
    @pytest.mark.asyncio
    async def test_audio_visual_sync(self, coordinator):
        """Test audio-visual synchronization timing."""
        text_chunks = ["First.", "Second.", "Third."]
        
        audio_chunks = []
        cumulative_time = 0.0
        
        async for chunk in coordinator.stream_audio(text_chunks):
            # Verify timing continuity
            assert abs(chunk.start_time - cumulative_time) < 0.001
            cumulative_time += chunk.duration
            audio_chunks.append(chunk)
        
        # Verify no gaps in timing
        for i in range(len(audio_chunks) - 1):
            expected_next_start = audio_chunks[i].start_time + audio_chunks[i].duration
            actual_next_start = audio_chunks[i + 1].start_time
            assert abs(expected_next_start - actual_next_start) < 0.001
    
    @pytest.mark.asyncio
    async def test_streaming_latency(self, coordinator):
        """Test first chunk latency in streaming."""
        text_chunks = ["Quick test sentence."]
        
        start_time = time.time()
        first_chunk_time = None
        
        async for chunk in coordinator.stream_audio(text_chunks):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            break
        
        latency = first_chunk_time - start_time
        
        # First chunk should arrive quickly (< 2s for fallback)
        assert latency < 2.0, f"First chunk latency: {latency:.2f}s"
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, tts_worker):
        """Test error recovery in TTS generation."""
        # Test with problematic input
        result = tts_worker._generate_audio_sync(
            text="Test" * 1000,  # Very long text
            voice_preset=None
        )
        
        # Should still generate audio via fallback
        assert result is not None
        audio_np, audio_bytes = result
        assert len(audio_np) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self, coordinator):
        """Test concurrent audio generation."""
        text_chunks_1 = ["Stream 1 sentence 1.", "Stream 1 sentence 2."]
        text_chunks_2 = ["Stream 2 sentence 1.", "Stream 2 sentence 2."]
        
        async def collect_stream(chunks):
            result = []
            async for chunk in coordinator.stream_audio(chunks):
                result.append(chunk)
            return result
        
        # Run two streams concurrently
        results = await asyncio.gather(
            collect_stream(text_chunks_1),
            collect_stream(text_chunks_2)
        )
        
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2


class TestQwen3OutputFormat:
    """Test Qwen3-TTS output format compliance."""
    
    @pytest.fixture
    def worker(self):
        """Create worker for format testing."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    @pytest.mark.asyncio
    async def test_audio_chunk_format(self, worker):
        """Test AudioChunk format matches specification."""
        chunk = await worker.process_sentence(
            sentence="Format test",
            sentence_index=5,
            cumulative_time=2.5
        )
        
        assert chunk is not None
        
        # Verify all required fields
        assert hasattr(chunk, 'sentence_index')
        assert hasattr(chunk, 'audio_bytes')
        assert hasattr(chunk, 'audio_np')
        assert hasattr(chunk, 'sample_rate')
        assert hasattr(chunk, 'start_time')
        assert hasattr(chunk, 'duration')
        
        # Verify field types
        assert isinstance(chunk.sentence_index, int)
        assert isinstance(chunk.audio_bytes, bytes)
        assert isinstance(chunk.audio_np, np.ndarray)
        assert isinstance(chunk.sample_rate, int)
        assert isinstance(chunk.start_time, float)
        assert isinstance(chunk.duration, float)
        
        # Verify field values
        assert chunk.sentence_index == 5
        assert chunk.sample_rate == 24000
        assert chunk.start_time == 2.5
        assert chunk.duration > 0
    
    @pytest.mark.asyncio
    async def test_wav_format_compliance(self, worker):
        """Test WAV output format compliance."""
        import wave
        import io
        
        chunk = await worker.process_sentence(
            sentence="WAV format test",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        
        # Parse WAV file
        with io.BytesIO(chunk.audio_bytes) as buf:
            with wave.open(buf, 'rb') as wav:
                # Verify WAV parameters
                assert wav.getnchannels() == 1, "Should be mono"
                assert wav.getsampwidth() == 2, "Should be 16-bit"
                assert wav.getframerate() == 24000, "Should be 24kHz"
                assert wav.getnframes() > 0, "Should have frames"
                
                # Verify audio data
                frames = wav.readframes(wav.getnframes())
                assert len(frames) > 0
    
    @pytest.mark.asyncio
    async def test_numpy_array_format(self, worker):
        """Test numpy array format compliance."""
        chunk = await worker.process_sentence(
            sentence="NumPy format test",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        
        # Verify numpy array properties
        assert chunk.audio_np.ndim == 1, "Should be 1D array"
        assert chunk.audio_np.dtype == np.float32, "Should be float32"
        assert chunk.audio_np.min() >= -1.0, "Values should be >= -1.0"
        assert chunk.audio_np.max() <= 1.0, "Values should be <= 1.0"
        assert not np.isnan(chunk.audio_np).any(), "Should not contain NaN"
        assert not np.isinf(chunk.audio_np).any(), "Should not contain Inf"
    
    @pytest.mark.asyncio
    async def test_timestamp_format(self, worker):
        """Test timestamp format (decimal seconds)."""
        chunks = []
        cumulative = 0.0
        
        for i in range(3):
            chunk = await worker.process_sentence(
                sentence=f"Sentence {i}",
                sentence_index=i,
                cumulative_time=cumulative
            )
            chunks.append(chunk)
            cumulative += chunk.duration
        
        # Verify timestamp format
        for chunk in chunks:
            # Should be decimal seconds (float)
            assert isinstance(chunk.start_time, float)
            assert isinstance(chunk.duration, float)
            
            # Should have reasonable precision
            assert chunk.start_time >= 0
            assert chunk.duration > 0
            
            # Duration should match audio length
            expected_duration = len(chunk.audio_np) / chunk.sample_rate
            assert abs(chunk.duration - expected_duration) < 0.001


class TestQwen3VoicePresets:
    """Test voice preset functionality."""
    
    @pytest.fixture
    def worker(self):
        """Create worker for voice preset testing."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    @pytest.mark.asyncio
    async def test_default_voice(self, worker):
        """Test generation with default voice."""
        chunk = await worker.process_sentence(
            sentence="Default voice test",
            sentence_index=0,
            cumulative_time=0.0,
            voice_preset=None
        )
        
        assert chunk is not None
        assert chunk.duration > 0
    
    @pytest.mark.asyncio
    async def test_custom_voice_preset(self, worker):
        """Test generation with custom voice preset."""
        chunk = await worker.process_sentence(
            sentence="Custom voice test",
            sentence_index=0,
            cumulative_time=0.0,
            voice_preset="female"
        )
        
        assert chunk is not None
        assert chunk.duration > 0
    
    @pytest.mark.asyncio
    async def test_multiple_voice_presets(self, worker):
        """Test generation with different voice presets."""
        presets = ["default", "male", "female", None]
        
        for preset in presets:
            chunk = await worker.process_sentence(
                sentence=f"Voice preset: {preset}",
                sentence_index=0,
                cumulative_time=0.0,
                voice_preset=preset
            )
            
            assert chunk is not None
            assert chunk.duration > 0


class TestQwen3EdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def worker(self):
        """Create worker for edge case testing."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    @pytest.mark.asyncio
    async def test_empty_text(self, worker):
        """Test handling of empty text."""
        chunk = await worker.process_sentence(
            sentence="",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        assert chunk.duration >= 0
    
    @pytest.mark.asyncio
    async def test_whitespace_only(self, worker):
        """Test handling of whitespace-only text."""
        chunk = await worker.process_sentence(
            sentence="   \t\n  ",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
    
    @pytest.mark.asyncio
    async def test_special_characters(self, worker):
        """Test handling of special characters."""
        special_texts = [
            "Hello! How are you?",
            "Test... with... ellipsis...",
            "Question? Answer!",
            "Numbers: 123, 456.78",
            "Symbols: @#$%^&*()",
        ]
        
        for text in special_texts:
            chunk = await worker.process_sentence(
                sentence=text,
                sentence_index=0,
                cumulative_time=0.0
            )
            
            assert chunk is not None
            assert chunk.duration > 0
    
    @pytest.mark.asyncio
    async def test_unicode_text(self, worker):
        """Test handling of Unicode text."""
        unicode_texts = [
            "Hello 世界",
            "Café résumé",
            "Emoji: 😀🎉",
        ]
        
        for text in unicode_texts:
            chunk = await worker.process_sentence(
                sentence=text,
                sentence_index=0,
                cumulative_time=0.0
            )
            
            assert chunk is not None
    
    @pytest.mark.asyncio
    async def test_very_short_text(self, worker):
        """Test handling of very short text."""
        chunk = await worker.process_sentence(
            sentence="Hi",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        assert chunk.duration > 0
    
    @pytest.mark.asyncio
    async def test_maximum_length_text(self, worker):
        """Test handling of maximum length text."""
        long_text = "This is a very long sentence. " * 100
        
        chunk = await worker.process_sentence(
            sentence=long_text,
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        # Fallback caps at 10 seconds
        assert chunk.duration <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
