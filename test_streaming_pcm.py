import asyncio
import base64
import json
import os
import struct
import sys
import threading
from typing import Optional

import numpy as np
import requests
import websockets

try:
    import winsound  # type: ignore
except Exception:
    winsound = None


def print_header(title):
    """Print a formatted header for test sections."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health(pod_url):
    """Test health endpoint."""
    print_header("HEALTH CHECK")
    try:
        response = requests.get(f"{pod_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_warmup(pod_url):
    """Test TTS warmup."""
    print_header("TTS WARMUP")
    try:
        response = requests.post(f"{pod_url}/tts/warmup", 
                                json={}, 
                                timeout=120)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200 and response.json().get("warmed", False)
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return False


def test_voice_cloning(pod_url, audio_path, voice_id="actor"):
    """Test voice cloning."""
    print_header("VOICE CLONING")
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            data = {
                'voice_id': voice_id,
                'text': 'The quick brown fox jumps over the lazy dog.'
            }
            response = requests.post(f"{pod_url}/tts/reference_audio",
                                    files=files,
                                    data=data,
                                    timeout=120)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")
        return False


async def test_websocket_streaming(pod_url, question="Hello, this is a streaming test.", voice_id="actor"):
    """Test WebSocket streaming with detailed logging."""
    print_header("WEBSOCKET STREAMING TEST")
    
    # First get session_id
    try:
        req = {"question": question, "return_audio": True, "return_csv": False}
        response = requests.post(f"{pod_url}/process",
                                files={"request_raw": ("request_raw.json", json.dumps(req), "application/json")},
                                data={"url": "text://Hello world"},
                                timeout=60)
        session_id = response.json().get("session_id")
        print(f"Session ID: {session_id}")
        
        if not session_id:
            print("❌ No session_id received")
            return False
    except Exception as e:
        print(f"❌ Failed to get session_id: {e}")
        return False
    
    # Connect WebSocket
    ws_base = pod_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_base}/ws/infer/kyutai"
    
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "session_id": session_id,
                "question": question,
                "return_audio": True,
                "chunk_ms": 50,
                "voice_id": voice_id
            }))
            
            print("🔌 WebSocket connected, waiting for messages...")
            message_count = 0
            audio_chunks = 0
            text_chunks = 0
            blendshape_chunks = 0
            
            while message_count < 50:  # Limit to prevent infinite loop
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    msg = json.loads(raw)
                    message_count += 1
                    
                    mtype = msg.get("type")
                    
                    if mtype == "text_chunk":
                        text_chunks += 1
                        print(f"📝 Text chunk {text_chunks}:")
                        for key, value in msg.items():
                            if key != "type":
                                print(f"  {key}: {str(value)[:100]}...")
                    
                    elif mtype == "audio_chunk":
                        audio_chunks += 1
                        print(f"🔊 Audio chunk {audio_chunks}:")
                        print(f"  Sample rate: {msg.get('sample_rate')}")
                        print(f"  Channels: {msg.get('channels')}")
                        print(f"  Chunk index: {msg.get('chunk_index')}")
                        print(f"  Audio bytes: {len(msg.get('audio_bytes_base64', ''))}")
                        print(f"  Is final: {msg.get('is_final', False)}")
                    
                    elif mtype == "blendshapes":
                        blendshape_chunks += 1
                        print(f"😊 Blendshapes chunk {blendshape_chunks}:")
                        print(f"  Sentence: {msg.get('sentence_index')}")
                        print(f"  Chunk: {msg.get('chunk_index')}")
                        print(f"  Is final: {msg.get('is_final', False)}")
                    
                    elif mtype == "status":
                        print(f"ℹ️ Status: {msg.get('status')} - {msg.get('message')}")
                        if msg.get("status") == "completed":
                            break
                        elif msg.get("status") == "error":
                            print(f"❌ Server error: {msg.get('message')}")
                            break
                    
                    else:
                        print(f"❓ Unknown message type: {mtype}")
                        
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for message")
                    break
                    
            print(f"\n📊 SUMMARY:")
            print(f"  Total messages: {message_count}")
            print(f"  Text chunks: {text_chunks}")
            print(f"  Audio chunks: {audio_chunks}")
            print(f"  Blendshapes chunks: {blendshape_chunks}")
            
            return audio_chunks > 0
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False


async def main():
    """Comprehensive test suite."""
    pod_url = os.environ.get("POD_URL", "http://127.0.0.1:7860").rstrip("/")
    audio_path = os.environ.get("VOICE_WAV", r"C:\Users\ernes\Downloads\actor_4.wav")
    voice_id = os.environ.get("VOICE_ID", "actor")
    test_question = os.environ.get("QUESTION", "Hello, this is a comprehensive test of the streaming system.")
    
    print(f"🧪 TESTING POD: {pod_url}")
    print(f"🎵 AUDIO FILE: {audio_path}")
    print(f"🗣️  VOICE ID: {voice_id}")
    
    # Test suite
    tests = [
        ("Health Check", lambda: test_health(pod_url)),
        ("TTS Warmup", lambda: test_warmup(pod_url)),
        ("Voice Cloning", lambda: test_voice_cloning(pod_url, audio_path, voice_id)),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print(f"\n{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
    
    # WebSocket streaming test
    streaming_result = await test_websocket_streaming(pod_url, test_question, voice_id)
    results.append(("WebSocket Streaming", streaming_result))
    
    # Final summary
    print_header("FINAL RESULTS")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
