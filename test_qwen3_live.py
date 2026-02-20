"""
Test Qwen3-TTS with the live API.
"""
import requests
import base64

print("Testing Qwen3-TTS...")
print("=" * 60)

# Create session
print("\n1. Creating session...")
response = requests.post(
    'http://localhost:7860/process',
    data={'url': 'https://en.wikipedia.org/wiki/Python_(programming_language)'}
)

if response.status_code != 200:
    print(f"Failed: {response.text}")
    exit(1)

session_data = response.json()
session_id = session_data['session_id']
print(f"   ✓ Session: {session_id}")

# Query with Qwen3-TTS
print("\n2. Generating audio with Qwen3-TTS...")
print("   (This will download the model on first run - may take a few minutes)")

response = requests.post(
    'http://localhost:7860/query',
    json={
        'session_id': session_id,
        'question': 'What is Python programming language?',
        'use_qwen': True
    }
)

print(f"   Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    audio_b64 = data.get('audio_base64', '')
    
    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        
        with open('qwen3_output.wav', 'wb') as f:
            f.write(audio_bytes)
        
        print(f"\n✓ Success!")
        print(f"   Audio size: {len(audio_bytes)} bytes")
        print(f"   Saved to: qwen3_output.wav")
        print(f"   Answer: {data.get('answer', '')[:150]}...")
    else:
        print(f"\n✗ No audio in response")
        print(f"   Response keys: {list(data.keys())}")
else:
    print(f"\n✗ Failed: {response.text}")

print("\n" + "=" * 60)
