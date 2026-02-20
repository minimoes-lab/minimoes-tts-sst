"""
Test Qwen TTS via the API endpoint.
This uses the fallback synthesis which is currently working.
"""
import requests
import base64
import json

print("Testing Qwen TTS via API...")
print("=" * 60)

# Step 1: Create session
print("\n1. Creating session...")
response = requests.post(
    'http://localhost:7860/process',
    data={'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence'}
)

if response.status_code != 200:
    print(f"Failed to create session: {response.text}")
    exit(1)

session_data = response.json()
session_id = session_data['session_id']
print(f"   ✓ Session created: {session_id}")

# Step 2: Query with Qwen TTS enabled
print("\n2. Requesting audio with Qwen TTS...")
print("   (Using fallback synthesis - real Qwen3-TTS model not available)")

response = requests.post(
    'http://localhost:7860/query',
    json={
        'session_id': session_id,
        'question': 'What is artificial intelligence?',
        'use_qwen': True  # Enable Qwen TTS
    }
)

print(f"   Response status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    # Check if audio was generated
    audio_b64 = data.get('audio_base64', '')
    
    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        
        # Save audio
        with open('qwen_tts_output.wav', 'wb') as f:
            f.write(audio_bytes)
        
        print(f"\n✓ Success!")
        print(f"   Audio size: {len(audio_bytes)} bytes")
        print(f"   Saved to: qwen_tts_output.wav")
        print(f"   Answer: {data.get('answer', 'N/A')[:100]}...")
    else:
        print(f"\n✗ No audio generated")
        print(f"   Response: {json.dumps(data, indent=2)}")
else:
    print(f"\n✗ Request failed: {response.text}")

print("\n" + "=" * 60)
print("Note: Qwen TTS is using fallback synthesis because")
print("the actual Qwen3-TTS model doesn't exist yet.")
print("The fallback generates simple audio for testing.")
print("=" * 60)
