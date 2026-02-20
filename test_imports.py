"""
Test script to verify all imports and basic functionality.
"""
import sys

def test_basic_imports():
    """Test basic Python imports."""
    print("="*60)
    print("Testing Basic Imports")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import fastapi
        print(f"✅ FastAPI")
    except Exception as e:
        print(f"❌ FastAPI: {e}")
        return False
    
    try:
        import websockets
        print(f"✅ WebSockets")
    except Exception as e:
        print(f"❌ WebSockets: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers")
    except Exception as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain")
    except Exception as e:
        print(f"❌ LangChain: {e}")
        return False
    
    try:
        from langchain_groq import ChatGroq
        print(f"✅ LangChain Groq")
    except Exception as e:
        print(f"❌ LangChain Groq: {e}")
        return False
    
    print()
    return True


def test_streaming_modules():
    """Test streaming module imports."""
    print("="*60)
    print("Testing Streaming Modules")
    print("="*60)
    
    try:
        from streaming.sentence_buffer import SentenceBuffer
        print(f"✅ SentenceBuffer")
    except Exception as e:
        print(f"❌ SentenceBuffer: {e}")
        return False
    
    try:
        from streaming.idle_frames import generate_idle_frames
        print(f"✅ Idle Frames")
    except Exception as e:
        print(f"❌ Idle Frames: {e}")
        return False
    
    try:
        from streaming.protocol import make_status_msg
        print(f"✅ Protocol")
    except Exception as e:
        print(f"❌ Protocol: {e}")
        return False
    
    try:
        from streaming.streaming_rag import streaming_rag_query
        print(f"✅ Streaming RAG")
    except Exception as e:
        print(f"❌ Streaming RAG: {e}")
        return False
    
    try:
        from streaming.performance_monitor import get_monitor
        print(f"✅ Performance Monitor")
    except Exception as e:
        print(f"❌ Performance Monitor: {e}")
        return False
    
    try:
        from streaming.qwen_tts_worker import QwenTTSWorker
        print(f"✅ Qwen TTS Worker")
    except Exception as e:
        print(f"❌ Qwen TTS Worker: {e}")
        return False
    
    try:
        from streaming.kyutai_coordinator import KyutaiStreamCoordinator
        print(f"✅ Kyutai Coordinator")
    except Exception as e:
        print(f"❌ Kyutai Coordinator: {e}")
        return False
    
    try:
        from streaming.optimized_blendshape_worker import OptimizedBlendshapeWorker
        print(f"✅ Optimized Blendshape Worker")
    except Exception as e:
        print(f"❌ Optimized Blendshape Worker: {e}")
        return False
    
    print()
    return True


def test_utils_modules():
    """Test utility module imports."""
    print("="*60)
    print("Testing Utility Modules")
    print("="*60)
    
    try:
        from utils.config import config, get_blendshape_names
        print(f"✅ Config")
        print(f"   Frame rate: {config['frame_rate']}")
        print(f"   Output dim: {config['output_dim']}")
        print(f"   Blendshape names: {len(get_blendshape_names())}")
    except Exception as e:
        print(f"❌ Config: {e}")
        return False
    
    print()
    return True


def test_sentence_buffer():
    """Test sentence buffer functionality."""
    print("="*60)
    print("Testing Sentence Buffer Functionality")
    print("="*60)
    
    try:
        from streaming.sentence_buffer import SentenceBuffer
        
        buffer = SentenceBuffer(min_chars=20, max_chars=200)
        
        # Test adding tokens
        sentences = buffer.add_token("This is a test sentence. ")
        print(f"✅ Added token, got {len(sentences)} sentences")
        
        sentences = buffer.add_token("Another sentence here.")
        print(f"✅ Added token, got {len(sentences)} sentences")
        
        # Test flush
        remaining = buffer.flush()
        print(f"✅ Flushed buffer: '{remaining}'")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Sentence Buffer Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor():
    """Test performance monitor functionality."""
    print("="*60)
    print("Testing Performance Monitor")
    print("="*60)
    
    try:
        from streaming.performance_monitor import get_monitor
        import time
        
        monitor = get_monitor()
        monitor.reset()
        
        # Test stage timing
        start = monitor.start_stage("test_stage")
        time.sleep(0.1)
        monitor.end_stage("test_stage", start)
        
        # Test metrics
        monitor.record_sentence(1.5)
        monitor.record_frames(60)
        monitor.record_e2e_latency(2.0)
        monitor.record_buffer_health(0.5)
        
        # Get summary
        summary = monitor.get_summary()
        print(f"✅ Performance Monitor Working")
        print(f"   Total sentences: {summary['total_sentences']}")
        print(f"   Total frames: {summary['total_frames']}")
        print(f"   Stages tracked: {len(summary['stages'])}")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Performance Monitor Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_protocol_messages():
    """Test protocol message generation."""
    print("="*60)
    print("Testing Protocol Messages")
    print("="*60)
    
    try:
        from streaming.protocol import (
            make_status_msg,
            make_text_chunk_msg,
            make_audio_chunk_msg,
            make_blendshapes_msg,
            make_idle_frames_msg,
        )
        
        # Test status message
        msg = make_status_msg("processing", "Test message")
        print(f"✅ Status message: {msg['type']}")
        
        # Test text chunk
        msg = make_text_chunk_msg(0, "Test text", False)
        print(f"✅ Text chunk message: {msg['type']}")
        
        # Test audio chunk
        msg = make_audio_chunk_msg(0, 0, "base64data", 0.0, 1.0, 24000, False)
        print(f"✅ Audio chunk message: {msg['type']}")
        
        # Test blendshapes
        frames = [[0.1] * 68 for _ in range(10)]
        msg = make_blendshapes_msg(0, 0, frames, 0.0, 1.0, 60, False)
        print(f"✅ Blendshapes message: {msg['type']}, {len(msg['frames'])} frames")
        
        # Test idle frames
        msg = make_idle_frames_msg(frames, 0.0, 1.0, 60)
        print(f"✅ Idle frames message: {msg['type']}")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Protocol Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE IMPORT AND FUNCTIONALITY TEST")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Basic Imports", test_basic_imports()))
    results.append(("Streaming Modules", test_streaming_modules()))
    results.append(("Utility Modules", test_utils_modules()))
    results.append(("Sentence Buffer", test_sentence_buffer()))
    results.append(("Performance Monitor", test_performance_monitor()))
    results.append(("Protocol Messages", test_protocol_messages()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
