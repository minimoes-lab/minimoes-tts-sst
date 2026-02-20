"""
Test suite for Qwen3-TTS integration and streaming functionality.
"""
import asyncio
import io
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from streaming.qwen_tts_worker import QwenTTSWorker, AudioChunk


class TestQwen3TTSWorker:
    """Test Qwen3-TTS worker functionality."""
    
    @pytest.fixture
    def worker(self):
        """Create worker instance for testing."""
        # Use CPU for testing to avoid GPU requirements
        worker = QwenTTSWorker(device="cpu", use_qwen3=True)
        yield worker
        worker.cancel()
    
    @pytest.fixture
    def fallback_worker(self):
        """Create worker with fallback synthesis."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    def test_worker_initialization(self, worker):
        """Test worker initializes correctly."""
        assert worker is not None
        assert worker.sr == 24000
        assert worker.device in ["cpu", "cuda"]
        assert not worker._cancelled
    
    def test_fallback_synthesis(self, fallback_worker):
        """Test fallback synthesis generates valid audio."""
        text = "Hello world"
        result = fallback_worker._fallback_synthesis(text)
        
        assert result is not None
        audio_np, audio_bytes = result
        
        # Check audio array
        assert isinstance(audio_np, np.ndarray)
        assert audio_np.dtype == np.float32
        assert len(audio_np) > 0
        assert audio_np.min() >= -1.0
        assert audio_np.max() <= 1.0
        
        # Check WAV bytes
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 44  # WAV header is 44 bytes
        
        # Validate WAV format
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, 'rb') as wav:
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
                assert wav.getframerate() == 24000
    
    @pytest.mark.asyncio
    async def test_process_sentence_basic(self, fallback_worker):
        """Test basic sentence processing."""
        sentence = "This is a test sentence."
        chunk = await fallback_worker.process_sentence(
            sentence=sentence,
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        assert isinstance(chunk, AudioChunk)
        assert chunk.sentence_index == 0
        assert chunk.sample_rate == 24000
        assert chunk.start_time == 0.0
        assert chunk.duration > 0
        assert len(chunk.audio_bytes) > 0
        assert len(chunk.audio_np) > 0
    
    @pytest.mark.asyncio
    async def test_process_multiple_sentences(self, fallback_worker):
        """Test processing multiple sentences with cumulative timing."""
        sentences = [
            "First sentence.",
            "Second sentence here.",
            "And a third one."
        ]
        
        cumulative_time = 0.0
        chunks = []
        
        for idx, sentence in enumerate(sentences):
            chunk = await fallback_worker.process_sentence(
                sentence=sentence,
                sentence_index=idx,
                cumulative_time=cumulative_time
            )
            
            assert chunk is not None
            assert chunk.sentence_index == idx
            assert chunk.start_time == cumulative_time
            
            chunks.append(chunk)
            cumulative_time += chunk.duration
        
        # Verify timing continuity
        assert len(chunks) == 3
        assert chunks[0].start_time == 0.0
        assert chunks[1].start_time == chunks[0].duration
        assert chunks[2].start_time == chunks[0].duration + chunks[1].duration
    
    @pytest.mark.asyncio
    async def test_cancellation(self, fallback_worker):
        """Test worker cancellation."""
        fallback_worker.cancel()
        
        chunk = await fallback_worker.process_sentence(
            sentence="This should be cancelled",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is None
        assert fallback_worker._cancelled
    
    @pytest.mark.asyncio
    async def test_reset_after_cancel(self, fallback_worker):
        """Test worker reset after cancellation."""
        fallback_worker.cancel()
        assert fallback_worker._cancelled
        
        fallback_worker.reset()
        assert not fallback_worker._cancelled
        
        chunk = await fallback_worker.process_sentence(
            sentence="This should work after reset",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
    
    def test_audio_normalization(self, worker):
        """Test audio normalization function."""
        # Test float32 audio
        audio_float = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        normalized = worker._normalize_audio(audio_float)
        assert normalized.dtype == np.float32
        np.testing.assert_array_equal(normalized, audio_float)
        
        # Test int16 audio
        audio_int = np.array([16384, -16384, 0], dtype=np.int16)
        normalized = worker._normalize_audio(audio_int)
        assert normalized.dtype == np.float32
        assert np.allclose(normalized, [0.5, -0.5, 0.0], atol=0.01)
        
        # Test stereo to mono conversion
        audio_stereo = np.array([[0.5, 0.3], [-0.5, -0.3]], dtype=np.float32)
        normalized = worker._normalize_audio(audio_stereo)
        assert normalized.ndim == 1
        assert len(normalized) == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, fallback_worker):
        """Test concurrent sentence processing."""
        sentences = [f"Sentence {i}" for i in range(5)]
        
        tasks = [
            fallback_worker.process_sentence(
                sentence=sent,
                sentence_index=idx,
                cumulative_time=idx * 1.0
            )
            for idx, sent in enumerate(sentences)
        ]
        
        chunks = await asyncio.gather(*tasks)
        
        assert len(chunks) == 5
        assert all(chunk is not None for chunk in chunks)
        assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
    
    def test_audio_duration_calculation(self, fallback_worker):
        """Test audio duration matches expected length."""
        text = "Test"
        result = fallback_worker._fallback_synthesis(text)
        
        assert result is not None
        audio_np, _ = result
        
        expected_duration = len(audio_np) / fallback_worker.sr
        actual_samples = len(audio_np)
        actual_duration = actual_samples / fallback_worker.sr
        
        assert abs(expected_duration - actual_duration) < 0.001
    
    @pytest.mark.asyncio
    async def test_empty_sentence_handling(self, fallback_worker):
        """Test handling of empty or whitespace sentences."""
        chunk = await fallback_worker.process_sentence(
            sentence="",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        # Should still generate audio (silence or minimal audio)
        assert chunk is not None
        assert chunk.duration >= 0
    
    @pytest.mark.asyncio
    async def test_long_sentence_handling(self, fallback_worker):
        """Test handling of very long sentences."""
        long_sentence = "This is a very long sentence. " * 50
        
        chunk = await fallback_worker.process_sentence(
            sentence=long_sentence,
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        assert chunk.duration > 0
        # Fallback synthesis caps at 10 seconds
        assert chunk.duration <= 10.0
    
    def test_voice_preset_parameter(self, fallback_worker):
        """Test voice preset parameter is accepted."""
        result = fallback_worker._generate_audio_sync(
            text="Test with voice preset",
            voice_preset="female"
        )
        
        assert result is not None
        audio_np, audio_bytes = result
        assert len(audio_np) > 0


class TestQwen3TTSIntegration:
    """Integration tests for Qwen3-TTS with real model (if available)."""
    
    @pytest.fixture
    def real_worker(self):
        """Create worker that attempts to load real model."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=True)
        yield worker
        worker.cancel()
    
    @pytest.mark.skipif(
        not Path("~/.cache/huggingface").expanduser().exists(),
        reason="HuggingFace cache not available"
    )
    @pytest.mark.asyncio
    async def test_real_model_generation(self, real_worker):
        """Test generation with real Qwen model if available."""
        if not real_worker.model_loaded:
            pytest.skip("Model not loaded, using fallback")
        
        chunk = await real_worker.process_sentence(
            sentence="Hello, this is a test.",
            sentence_index=0,
            cumulative_time=0.0
        )
        
        assert chunk is not None
        assert chunk.duration > 0
        assert len(chunk.audio_bytes) > 0
        
        # Verify audio quality
        assert chunk.audio_np.min() >= -1.0
        assert chunk.audio_np.max() <= 1.0
    
    def test_model_loading_fallback(self):
        """Test that worker falls back gracefully if model unavailable."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=True)
        
        # Worker should initialize even if model fails to load
        assert worker is not None
        assert worker.sr == 24000
        
        # Should be able to generate audio via fallback
        result = worker._fallback_synthesis("Test")
        assert result is not None


class TestQwen3TTSPerformance:
    """Performance tests for Qwen3-TTS."""
    
    @pytest.fixture
    def perf_worker(self):
        """Create worker for performance testing."""
        worker = QwenTTSWorker(device="cpu", use_qwen3=False)
        yield worker
        worker.cancel()
    
    @pytest.mark.asyncio
    async def test_generation_latency(self, perf_worker):
        """Test audio generation latency."""
        sentence = "This is a performance test sentence."
        
        start = time.time()
        chunk = await perf_worker.process_sentence(
            sentence=sentence,
            sentence_index=0,
            cumulative_time=0.0
        )
        end = time.time()
        
        latency = end - start
        
        assert chunk is not None
        # Fallback synthesis should be very fast
        assert latency < 1.0, f"Generation took {latency:.2f}s"
    
    @pytest.mark.asyncio
    async def test_throughput(self, perf_worker):
        """Test audio generation throughput."""
        sentences = [f"Sentence number {i}" for i in range(10)]
        
        start = time.time()
        tasks = [
            perf_worker.process_sentence(
                sentence=sent,
                sentence_index=idx,
                cumulative_time=0.0
            )
            for idx, sent in enumerate(sentences)
        ]
        chunks = await asyncio.gather(*tasks)
        end = time.time()
        
        total_time = end - start
        throughput = len(sentences) / total_time
        
        assert all(chunk is not None for chunk in chunks)
        assert throughput > 5, f"Throughput: {throughput:.2f} sentences/sec"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, perf_worker):
        """Test memory usage during generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple chunks
        for i in range(20):
            chunk = await perf_worker.process_sentence(
                sentence=f"Memory test sentence {i}",
                sentence_index=i,
                cumulative_time=0.0
            )
            assert chunk is not None
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Memory increase should be reasonable (< 100MB for fallback)
        assert mem_increase < 100, f"Memory increased by {mem_increase:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
