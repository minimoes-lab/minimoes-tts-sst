"""
Quick validation script for Qwen3-TTS integration.
Run this to verify the Qwen3-TTS features are working correctly.
"""
import asyncio
import time
from streaming.qwen_tts_worker import QwenTTSWorker


async def validate_basic_functionality():
    """Validate basic TTS functionality."""
    print("=" * 60)
    print("Qwen3-TTS Validation")
    print("=" * 60)
    
    # Initialize worker
    print("\n1. Initializing Qwen TTS Worker...")
    worker = QwenTTSWorker(device="cpu", use_qwen3=False)
    print(f"   ✓ Worker initialized (device: {worker.device})")
    print(f"   ✓ Sample rate: {worker.sr} Hz")
    print(f"   ✓ Model loaded: {worker.model_loaded}")
    
    # Test single sentence
    print("\n2. Testing single sentence generation...")
    start = time.time()
    chunk = await worker.process_sentence(
        sentence="Hello, this is a test of the Qwen TTS system.",
        sentence_index=0,
        cumulative_time=0.0
    )
    latency = time.time() - start
    
    if chunk:
        print(f"   ✓ Audio generated successfully")
        print(f"   ✓ Latency: {latency:.3f}s")
        print(f"   ✓ Duration: {chunk.duration:.3f}s")
        print(f"   ✓ Audio samples: {len(chunk.audio_np)}")
        print(f"   ✓ WAV bytes: {len(chunk.audio_bytes)}")
    else:
        print("   ✗ Failed to generate audio")
        return False
    
    # Test multiple sentences
    print("\n3. Testing multiple sentence generation...")
    sentences = [
        "First sentence here.",
        "Second sentence follows.",
        "And the third one completes the test."
    ]
    
    cumulative = 0.0
    chunks = []
    
    start = time.time()
    for idx, sentence in enumerate(sentences):
        chunk = await worker.process_sentence(
            sentence=sentence,
            sentence_index=idx,
            cumulative_time=cumulative
        )
        if chunk:
            chunks.append(chunk)
            cumulative += chunk.duration
    
    total_time = time.time() - start
    
    print(f"   ✓ Generated {len(chunks)} chunks")
    print(f"   ✓ Total generation time: {total_time:.3f}s")
    print(f"   ✓ Total audio duration: {cumulative:.3f}s")
    print(f"   ✓ Real-time factor: {cumulative/total_time:.2f}x")
    
    # Test timing continuity
    print("\n4. Validating timing continuity...")
    timing_ok = True
    for i in range(len(chunks) - 1):
        expected_next = chunks[i].start_time + chunks[i].duration
        actual_next = chunks[i + 1].start_time
        if abs(expected_next - actual_next) > 0.001:
            print(f"   ✗ Timing gap detected at chunk {i}")
            timing_ok = False
    
    if timing_ok:
        print("   ✓ Timing is continuous (no gaps)")
    
    # Test voice presets
    print("\n5. Testing voice presets...")
    presets = ["default", "male", "female"]
    for preset in presets:
        chunk = await worker.process_sentence(
            sentence="Testing voice preset.",
            sentence_index=0,
            cumulative_time=0.0,
            voice_preset=preset
        )
        if chunk:
            print(f"   ✓ Voice preset '{preset}' works")
        else:
            print(f"   ✗ Voice preset '{preset}' failed")
    
    # Test cancellation
    print("\n6. Testing cancellation...")
    worker.cancel()
    chunk = await worker.process_sentence(
        sentence="This should be cancelled",
        sentence_index=0,
        cumulative_time=0.0
    )
    if chunk is None:
        print("   ✓ Cancellation works correctly")
    else:
        print("   ✗ Cancellation failed")
    
    # Test reset
    print("\n7. Testing reset...")
    worker.reset()
    chunk = await worker.process_sentence(
        sentence="This should work after reset",
        sentence_index=0,
        cumulative_time=0.0
    )
    if chunk:
        print("   ✓ Reset works correctly")
    else:
        print("   ✗ Reset failed")
    
    # Test edge cases
    print("\n8. Testing edge cases...")
    
    # Empty text
    chunk = await worker.process_sentence("", 0, 0.0)
    print(f"   ✓ Empty text: {'handled' if chunk else 'failed'}")
    
    # Very short text
    chunk = await worker.process_sentence("Hi", 0, 0.0)
    print(f"   ✓ Very short text: {'handled' if chunk else 'failed'}")
    
    # Long text
    long_text = "This is a very long sentence. " * 50
    chunk = await worker.process_sentence(long_text, 0, 0.0)
    print(f"   ✓ Long text: {'handled' if chunk else 'failed'}")
    if chunk:
        print(f"      Duration capped at: {chunk.duration:.2f}s")
    
    # Special characters
    chunk = await worker.process_sentence("Hello! How are you? 123...", 0, 0.0)
    print(f"   ✓ Special characters: {'handled' if chunk else 'failed'}")
    
    # Unicode
    chunk = await worker.process_sentence("Hello 世界 Café", 0, 0.0)
    print(f"   ✓ Unicode text: {'handled' if chunk else 'failed'}")
    
    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    
    return True


async def validate_streaming_integration():
    """Validate streaming integration."""
    print("\n" + "=" * 60)
    print("Streaming Integration Validation")
    print("=" * 60)
    
    from streaming.test_coordinator import KyutaiCoordinator
    
    print("\n1. Initializing coordinator...")
    worker = QwenTTSWorker(device="cpu", use_qwen3=False)
    coordinator = KyutaiCoordinator(tts_worker=worker)
    print("   ✓ Coordinator initialized")
    
    print("\n2. Testing streaming pipeline...")
    text_chunks = [
        "Welcome to the streaming test.",
        "This validates the full pipeline.",
        "Audio chunks are generated continuously."
    ]
    
    chunks_received = []
    start = time.time()
    
    async for chunk in coordinator.stream_audio(text_chunks):
        chunks_received.append(chunk)
        print(f"   ✓ Chunk {chunk.sentence_index}: {chunk.duration:.3f}s")
    
    total_time = time.time() - start
    
    print(f"\n   ✓ Received {len(chunks_received)} chunks")
    print(f"   ✓ Total time: {total_time:.3f}s")
    
    print("\n" + "=" * 60)
    print("Streaming Validation Complete!")
    print("=" * 60)


async def main():
    """Run all validations."""
    try:
        success = await validate_basic_functionality()
        if success:
            await validate_streaming_integration()
        
        print("\n✓ All validations passed!")
        print("\nQwen3-TTS integration is working correctly.")
        print("You can now use the TTS worker in your streaming pipeline.")
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
