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


def _assemble_chunks_with_crossfade(pcm_chunks: dict, sample_rate: int = 24000, blend_samples: int = 512) -> bytes:
    """
    Assemble PCM chunks with Hann window crossfade to eliminate boundary discontinuities.
    Based on Qwen3-TTS-streaming documentation: https://github.com/rekuenkdr/Qwen3-TTS-streaming
    
    Algorithm:
    1. Convert each chunk bytes to int16 numpy array
    2. For each chunk after the first:
       - Apply Hann crossfade between previous chunk tail and current chunk head
       - Trim the overlap region from the previous chunk
    3. Concatenate all processed chunks
    4. Apply fade-in to first chunk and fade-out to last chunk
    
    Args:
        pcm_chunks: Dict mapping chunk_index -> pcm_bytes
        sample_rate: Audio sample rate (default 24000)
        blend_samples: Number of samples for crossfade zone (default 512 = ~21ms at 24kHz)
    
    Returns:
        bytes: Assembled PCM data ready for WAV writing
    """
    import numpy as np
    
    if not pcm_chunks:
        return b""
    
    # Sort chunks by index
    sorted_indices = sorted(pcm_chunks.keys())
    
    # Convert all chunks to numpy int16 arrays
    chunk_arrays = []
    for idx in sorted_indices:
        pcm_data = pcm_chunks[idx]
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        chunk_arrays.append(audio_int16)
    
    if len(chunk_arrays) == 1:
        # Only one chunk - just apply fade in/out
        audio = chunk_arrays[0].astype(np.float32)
        fade_len = min(blend_samples, len(audio) // 4)
        if fade_len > 0:
            # Fade in
            fade_in = np.sin(np.linspace(0, np.pi/2, fade_len)) ** 2
            audio[:fade_len] *= fade_in
            # Fade out
            fade_out = np.sin(np.linspace(np.pi/2, 0, fade_len)) ** 2
            audio[-fade_len:] *= fade_out
        return audio.astype(np.int16).tobytes()
    
    # Multiple chunks - apply crossfade
    processed_chunks = []
    
    for i, current_chunk in enumerate(chunk_arrays):
        current = current_chunk.astype(np.float32)
        
        if i == 0:
            # First chunk: apply fade-in, keep full chunk (except tail will be trimmed by next)
            fade_len = min(blend_samples, len(current) // 4)
            if fade_len > 0:
                # Hann fade-in: 0.5 * (1 - cos(π*t))
                t = np.linspace(0, 1, fade_len)
                fade_in = 0.5 * (1 - np.cos(np.pi * t))
                current[:fade_len] *= fade_in
            processed_chunks.append(current)
        else:
            # Get previous chunk
            prev_chunk = processed_chunks[-1]
            
            # Determine blend length (can't exceed either chunk length)
            actual_blend = min(blend_samples, len(prev_chunk), len(current))
            
            if actual_blend > 0:
                # Extract tail of previous chunk and head of current
                prev_tail = prev_chunk[-actual_blend:].copy()
                curr_head = current[:actual_blend].copy()
                
                # Hann crossfade
                # fade_out = 0.5 * (1 + cos(π*t)) for previous
                # fade_in = 0.5 * (1 - cos(π*t)) for current
                t = np.linspace(0, 1, actual_blend)
                fade_out = 0.5 * (1 + np.cos(np.pi * t))
                fade_in = 0.5 * (1 - np.cos(np.pi * t))
                
                # Blend: prev_tail * fade_out + curr_head * fade_in
                blended = prev_tail * fade_out + curr_head * fade_in
                
                # Update previous chunk: remove overlap tail, add blended region
                processed_chunks[-1] = np.concatenate([prev_chunk[:-actual_blend], blended])
                
                # Current chunk: remove the head (already blended)
                current = current[actual_blend:]
            
            # Apply fade-out if this is the last chunk
            if i == len(chunk_arrays) - 1:
                fade_len = min(blend_samples, len(current) // 4)
                if fade_len > 0:
                    # Hann fade-out
                    t = np.linspace(0, 1, fade_len)
                    fade_out = 0.5 * (1 + np.cos(np.pi * t))
                    current[-fade_len:] *= fade_out
            
            processed_chunks.append(current)
    
    # Concatenate all processed chunks
    final_audio = np.concatenate(processed_chunks)
    
    # Convert back to int16 bytes
    return final_audio.astype(np.int16).tobytes()


def _write_wav_pcm16(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Write PCM data to WAV file with proper header."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm)
    riff_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # fmt chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


def print_header(title):
    """Print a formatted section header."""
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
        print(f"[OK] Health Check: PASSED" if result else f"[FAIL] Health check failed: {e}")
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
        print(f"[OK] TTS Warmup: PASSED" if result else f"[FAIL] Warmup failed: {e}")
        return False


def test_voice_cloning(pod_url, audio_path, voice_id="actor"):
    """Test voice cloning."""
    print_header("VOICE CLONING")
    try:
        # Get transcript file path
        transcript_path = audio_path.replace('.wav', '.txt')
        
        with open(audio_path, 'rb') as audio_file:
            with open(transcript_path, 'r') as transcript_file:
                files = {'audio': audio_file}
                data = {
                    'voice_id': voice_id,
                    'text': transcript_file.read()  # Read transcript from file
                }
                response = requests.post(f"{pod_url}/tts/reference_audio",
                                        files=files,
                                        data=data,
                                        timeout=120)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Audio file: {audio_path}")
        print(f"Transcript file: {transcript_path}")
        return response.status_code == 200
    except FileNotFoundError as e:
        print(f"[FAIL] File not found: {e}")
        print("[FAIL] Make sure both .wav and .txt files exist:")
        print(f"   - {audio_path}")
        print(f"   - {audio_path.replace('.wav', '.txt')}")
        return False
    except Exception as e:
        print(f"[FAIL] Voice cloning failed: {e}")
        return False


async def test_websocket_streaming(pod_url, question="Hello, this is a streaming test.", voice_id="actor"):
    """Test WebSocket streaming with detailed logging and proper audio assembly."""
    print_header("WEBSOCKET STREAMING TEST")
    
    # First get session_id
    try:
        req = {"question": question, "return_audio": True, "return_csv": False}
        response = requests.post(f"{pod_url}/process",
                                files={"request_raw": ("request_raw.json", json.dumps(req), "application/json")},
                                data={"url": "https://httpbin.org/html"},  # Use valid HTTP URL
                                timeout=60)
        session_id = response.json().get("session_id")
        print(f"Session ID: {session_id}")
        
        if not session_id:
            print("[FAIL] No session_id received")
            print(f"Response: {response.json()}")
            return False
    except Exception as e:
        print(f"[FAIL] Failed to get session_id: {e}")
        return False
    
    # Connect WebSocket
    ws_base = pod_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_base}/ws/infer/kyutai"
    
    # Storage for proper assembly
    pcm_chunks = {}
    sample_rate = 24000
    channels = 1
    audio_received = False
    
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
            
            print("[PLUG] WebSocket connected, waiting for messages...")
            message_count = 0
            audio_chunks = 0
            text_chunks = 0
            blendshape_chunks = 0
            last_audio_chunk = -1
            final_received = False
            arrival_order = []  # Track chunk arrival order
            all_blendshapes = []  # Store all blendshape frames
            
            while message_count < 100 and not final_received:  # Limit to prevent infinite loop
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    msg = json.loads(raw)
                    message_count += 1
                    
                    mtype = msg.get("type")
                    
                    if mtype == "text_chunk":
                        text_chunks += 1
                        print(f"[TEXT] Text chunk {text_chunks}:")
                        print(f"  [SEARCH] FULL MESSAGE: {json.dumps(msg, indent=2)}")
                        for key, value in msg.items():
                            if key != "type":
                                print(f"  {key}: {str(value)[:200]}...")
                        print(f"  [TEXT] EXTRACTED TEXT: {msg.get('text', 'NO TEXT FIELD')}")
                        print(f"  [TARGET] SENTENCE INDEX: {msg.get('sentence_index', 'NO INDEX')}")
                        print(f"  [OK] IS FINAL: {msg.get('is_final', False)}")
                    
                    elif mtype == "audio_chunk":
                        audio_chunks += 1
                        chunk_idx = msg.get("chunk_index")
                        is_final = msg.get("is_final", False)
                        
                        # DEBUG: Track arrival order
                        arrival_order.append(chunk_idx)
                        
                        print(f"[SOUND] Audio chunk #{audio_chunks} (index={chunk_idx}, final={is_final}):")
                        
                        # Check if arrived in order
                        if arrival_order and chunk_idx != arrival_order[-2] + 1 if len(arrival_order) > 1 else False:
                            print(f"  [WARN] OUT OF ORDER! Expected {arrival_order[-2] + 1 if len(arrival_order) > 1 else 0}, got {chunk_idx}")
                        
                        b64 = msg.get("audio_bytes_base64", "")
                        if b64 and isinstance(chunk_idx, int):
                            pcm_data = base64.b64decode(b64)
                            pcm_chunks[chunk_idx] = pcm_data
                            last_audio_chunk = chunk_idx
                            audio_received = True
                            
                            # [BUG] BUG DETECTION: Check for chunks after final
                            if final_received:
                                print(f"  [ALERT] WARNING: Received chunk {chunk_idx} AFTER final signal!")
                                print(f"  [ALERT] This indicates a TTS streaming bug!")
                        
                        if is_final:
                            final_received = True
                    
                    elif mtype == "blendshapes":
                        blendshape_chunks += 1
                        print(f"[BLEND] Blendshapes chunk {blendshape_chunks}:")
                        print(f"  Sentence: {msg.get('sentence_index')}")
                        print(f"  Chunk: {msg.get('chunk_index')}")
                        print(f"  Is final: {msg.get('is_final', False)}")
                        
                        # Store blendshape data for saving
                        frames_data = msg.get("frames", [])
                        if frames_data:
                            all_blendshapes.extend(frames_data)
                            print(f"  Frames in chunk: {len(frames_data)}")
                    
                    elif mtype == "status":
                        print(f"[INFO] Status: {msg.get('status')} - {msg.get('message')}")
                        if msg.get("status") == "completed":
                            break
                        elif msg.get("status") == "error":
                            print(f"[FAIL] Server error: {msg.get('message')}")
                            break
                    
                    else:
                        print(f"[Q] Unknown message type: {mtype}")
                        
                except asyncio.TimeoutError:
                    print("[TIME] Timeout waiting for message")
                    break
                    
            print(f"\n[STATS] SUMMARY:")
            print(f"  Total messages: {message_count}")
            print(f"  Text chunks: {text_chunks}")
            print(f"  Audio chunks: {audio_chunks}")
            print(f"  Blendshapes chunks: {blendshape_chunks}")
            print(f"  Last audio chunk: {last_audio_chunk}")
            print(f"  Audio received: {audio_received}")
            
            # Assemble audio if chunks received
            if audio_received and pcm_chunks:
                print(f"\n[AUDIO] ASSEMBLING AUDIO...")
                
                # COMPREHENSIVE DIAGNOSTIC
                chunk_indices = sorted(pcm_chunks.keys())
                print(f"\n[STATS] CHUNK ANALYSIS:")
                print(f"  Total chunks stored: {len(pcm_chunks)}")
                print(f"  Index range: {min(chunk_indices)} to {max(chunk_indices)}")
                print(f"  First 10 indices: {chunk_indices[:10]}")
                print(f"  Last 10 indices: {chunk_indices[-10:]}")
                
                # Check for missing chunks
                expected_count = max(chunk_indices) - min(chunk_indices) + 1
                missing_count = expected_count - len(chunk_indices)
                print(f"  Expected chunks: {expected_count}")
                print(f"  Missing chunks: {missing_count}")
                
                if missing_count > 0:
                    expected_indices = set(range(min(chunk_indices), max(chunk_indices) + 1))
                    actual_indices = set(chunk_indices)
                    missing = sorted(expected_indices - actual_indices)
                    print(f"  [ALERT] MISSING INDICES: {missing[:20]}{'...' if len(missing) > 20 else ''}")
                
                # Check chunk sizes
                sizes = [len(pcm_chunks[i]) for i in chunk_indices]
                avg_size = sum(sizes) / len(sizes)
                min_size = min(sizes)
                max_size = max(sizes)
                print(f"\n[BOX] CHUNK SIZE ANALYSIS:")
                print(f"  Average: {avg_size:.0f} bytes")
                print(f"  Min: {min_size} bytes, Max: {max_size} bytes")
                
                # Find anomalous chunks
                small_chunks = [(i, len(pcm_chunks[i])) for i in chunk_indices if len(pcm_chunks[i]) < 1000]
                if small_chunks:
                    print(f"  [WARN] Small chunks (<1000 bytes): {len(small_chunks)}")
                    for idx, size in small_chunks[:5]:
                        print(f"    Chunk {idx}: {size} bytes")
                
                # Check for gaps in sequence
                gaps = []
                for i in range(1, len(chunk_indices)):
                    if chunk_indices[i] != chunk_indices[i-1] + 1:
                        gaps.append((chunk_indices[i-1], chunk_indices[i]))
                if gaps:
                    print(f"\n[RED] SEQUENCE GAPS FOUND: {len(gaps)}")
                    for prev, curr in gaps[:5]:
                        print(f"  Gap: chunk {prev} -> {curr} (missing {curr - prev - 1} chunks)")
                else:
                    print(f"\n[OK] No sequence gaps - chunks in perfect order")
                
                # Deep diagnostic - analyze PCM values
                print(f"\n[SCIENCE] DEEP PCM ANALYSIS:")
                import numpy as np
                
                # Define output directory early
                out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "ws_streaming_out")
                os.makedirs(out_dir, exist_ok=True)
                
                # Analyze first few chunks
                for idx in chunk_indices[:3]:
                    pcm_data = pcm_chunks[idx]
                    audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
                    print(f"  Chunk {idx}:")
                    print(f"    Samples: {len(audio_int16)}")
                    print(f"    Min: {audio_int16.min()}, Max: {audio_int16.max()}")
                    print(f"    Mean: {audio_int16.mean():.1f}, Std: {audio_int16.std():.1f}")
                    print(f"    First 10 samples: {audio_int16[:10].tolist()}")
                    print(f"    Last 10 samples: {audio_int16[-10:].tolist()}")
                
                # Save individual chunks for inspection
                chunks_dir = os.path.join(out_dir, "raw_chunks")
                os.makedirs(chunks_dir, exist_ok=True)
                for idx in [0, 1, 2, 10, 20, len(chunk_indices)//2, len(chunk_indices)-1]:
                    if idx < len(chunk_indices):
                        real_idx = chunk_indices[idx]
                        pcm_data = pcm_chunks[real_idx]
                        chunk_path = os.path.join(chunks_dir, f"chunk_{real_idx:03d}.bin")
                        with open(chunk_path, 'wb') as f:
                            f.write(pcm_data)
                        print(f"  Saved chunk {real_idx} to {chunk_path}")
                
                # [DIAGNOSTIC] Test single chunk - if this is corrupted, TTS is the problem
                print(f"\n[DIAGNOSTIC] Testing single chunk (chunk 0)...")
                if 0 in pcm_chunks:
                    single_chunk_path = os.path.join(out_dir, "single_chunk_0.wav")
                    _write_wav_pcm16(single_chunk_path, pcm_chunks[0], sample_rate, channels)
                    print(f"  Single chunk 0: {single_chunk_path}")
                    print(f"  Size: {len(pcm_chunks[0])} bytes = {len(pcm_chunks[0])/2} samples = {len(pcm_chunks[0])/2/24000:.3f}s")
                    
                    # Analyze chunk 0
                    audio_int16 = np.frombuffer(pcm_chunks[0], dtype=np.int16)
                    print(f"  Chunk 0 analysis:")
                    print(f"    Min: {audio_int16.min()}, Max: {audio_int16.max()}")
                    print(f"    Mean: {audio_int16.mean():.1f}, Std: {audio_int16.std():.1f}")
                    print(f"    Values: {audio_int16[:20].tolist()}")
                    print(f"  >> Listen to this file - if it's garbled, the TTS is generating bad audio")
                    print(f"  >> If it's clear but the assembled version is garbled, the problem is in assembly")
                
                # Assemble with Hann crossfade (from Qwen3-TTS-streaming docs)
                use_crossfade = True  # Set to False to test raw assembly
                if use_crossfade:
                    print(f"\n[ASSEMBLY] Using Hann crossfade with 512 samples blend zone")
                    assembled_pcm = _assemble_chunks_with_crossfade(pcm_chunks, sample_rate, blend_samples=512)
                else:
                    print(f"\n[ASSEMBLY] Simple concatenation (NO crossfade)")
                    sorted_chunks = sorted(pcm_chunks.items())
                    assembled_pcm = b"".join(chunk for _, chunk in sorted_chunks)
                
                # Also create simple concatenated version for comparison
                print(f"\n[DIAGNOSTIC] Creating comparison files...")
                sorted_chunks = sorted(pcm_chunks.items())
                raw_assembled = b"".join(chunk for _, chunk in sorted_chunks)
                
                # Write raw version (no crossfade)
                raw_wav_path = os.path.join(out_dir, "assembled_audio_RAW.wav")
                _write_wav_pcm16(raw_wav_path, raw_assembled, sample_rate, channels)
                print(f"  Raw (no crossfade): {raw_wav_path}")
                
                print(f"\n[AUDIO] ASSEMBLED AUDIO:")
                print(f"  Total PCM size: {len(assembled_pcm)} bytes")
                print(f"  Duration: {len(assembled_pcm) / (sample_rate * channels * 2):.2f}s")
                print(f"  First 16 bytes: {assembled_pcm[:16].hex()}")
                
                # Write WAV
                wav_path = os.path.join(out_dir, "assembled_audio.wav")
                
                _write_wav_pcm16(wav_path, assembled_pcm, sample_rate, channels)
                print(f"\n[OK] WAV written: {wav_path}")
                
                # Audio quality assessment
                if missing_count > 0:
                    missing_ratio = missing_count / expected_count
                    print(f"\n[WARN] AUDIO QUALITY WARNING:")
                    print(f"  Missing {missing_ratio*100:.1f}% of chunks ({missing_count}/{expected_count})")
                    print(f"  This causes audio dropouts/corruption!")
                else:
                    print(f"\n[OK] AUDIO COMPLETENESS: 100% - No missing chunks")
                
                # Save blendshapes to JSON
                if all_blendshapes:
                    blendshapes_dir = os.path.join(out_dir, "blendshapes")
                    os.makedirs(blendshapes_dir, exist_ok=True)
                    blendshapes_path = os.path.join(blendshapes_dir, "blendshapes.json")
                    with open(blendshapes_path, 'w') as f:
                        json.dump({"frames": all_blendshapes}, f, indent=2)
                    print(f"\n[OK] Blendshapes saved: {blendshapes_path}")
                    print(f"  Total frames: {len(all_blendshapes)}")
                
                # Try to play the file
                try:
                    if winsound:
                        print("  [SOUND] Playing audio...")
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                except Exception as e:
                    print(f"  [WARN] Could not play audio: {e}")
                
                return True
            else:
                print("[FAIL] No audio data received to assemble")
                return False
            
    except Exception as e:
        print(f"[FAIL] WebSocket error: {e}")
        return False


async def main():
    """Comprehensive test suite."""
    pod_url = os.environ.get("POD_URL", "http://127.0.0.1:7860").rstrip("/")
    audio_path = os.environ.get("VOICE_WAV", r"C:\Users\ernes\Downloads\actor_4.wav")
    voice_id = os.environ.get("VOICE_ID", "actor")
    test_question = os.environ.get("QUESTION", "Hello, this is a comprehensive test of the streaming system.")
    
    print(f"[INFO] TESTING POD: {pod_url}")
    print(f"[INFO] AUDIO FILE: {audio_path}")
    print(f"[INFO] VOICE ID: {voice_id}")
    
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
        print(f"\n[OK] {test_name}: PASSED" if result else f"\n[FAIL] {test_name}: FAILED")
    
    # WebSocket streaming test
    streaming_result = await test_websocket_streaming(pod_url, test_question, voice_id)
    results.append(("WebSocket Streaming", streaming_result))
    
    # Final summary
    print_header("FINAL RESULTS")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n[STATS] OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("[WIN] All tests passed! The system is working correctly.")
    else:
        print("[WARN]  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
