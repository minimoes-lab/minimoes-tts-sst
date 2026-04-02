import asyncio
import json
import websockets

async def test_websocket():
    session_id = "6be77c07-8c00-4327-98b4-ffde24480217"
    ws_url = "wss://febz26zz1u3uc8-7860.proxy.runpod.net/ws/infer/kyutai"
    
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "session_id": session_id,
                "question": "Hello, this is a test with my cloned voice.",
                "return_audio": True,
                "chunk_ms": 50,
                "voice_id": "actor"
            }))
            
            print("WebSocket connected, waiting for messages...")
            message_count = 0
            
            while message_count < 10:  # Limit to 10 messages for testing
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    msg = json.loads(raw)
                    message_count += 1
                    
                    mtype = msg.get("type")
                    print(f"Message {message_count}: {mtype}")
                    
                    if mtype == "audio_chunk":
                        print(f"  - Sample rate: {msg.get('sample_rate')}")
                        print(f"  - Channels: {msg.get('channels')}")
                        print(f"  - Chunk index: {msg.get('chunk_index')}")
                        print(f"  - Audio bytes: {len(msg.get('audio_bytes_base64', ''))}")
                    elif mtype == "blendshapes":
                        print(f"  - Sentence: {msg.get('sentence_index')}")
                        print(f"  - Chunk: {msg.get('chunk_index')}")
                        print(f"  - Is final: {msg.get('is_final')}")
                    elif mtype == "status":
                        print(f"  - Status: {msg.get('status')}")
                        print(f"  - Message: {msg.get('message')}")
                    
                    if msg.get("type") == "status" and msg.get("status") == "completed":
                        break
                        
                except asyncio.TimeoutError:
                    print("Timeout waiting for message")
                    break
                    
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
