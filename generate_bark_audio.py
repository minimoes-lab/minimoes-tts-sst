import requests
import base64

print("Step 1: Creating session...")
# Create session first
response = requests.post(
    'http://localhost:7860/process',
    data={'url': 'https://en.wikipedia.org/wiki/Machine_learning'}
)

if response.status_code != 200:
    print(f"Failed to create session: {response.text}")
    exit(1)

session_data = response.json()
session_id = session_data['session_id']
print(f"Session created: {session_id}")

print("\nStep 2: Requesting Bark TTS audio...")
print("This will take 3-4 minutes...")

# Query with Bark TTS
response = requests.post(
    'http://localhost:7860/query',
    json={
        'session_id': session_id,
        'question': 'What is machine learning in simple terms?'
    }
)

print(f"Response status: {response.status_code}")

data = response.json()
audio_b64 = data.get('audio_base64', '')

if audio_b64:
    audio_bytes = base64.b64decode(audio_b64)
    with open('bark_audio_sample.wav', 'wb') as f:
        f.write(audio_bytes)
    print(f'Bark audio saved! Size: {len(audio_bytes)} bytes')
else:
    print('No audio generated')
    print(f'Response: {data}')
