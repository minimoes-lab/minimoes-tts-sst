"""Test voice cloning with actor_4.wav"""
import requests

url = "https://opyctbyuqsdcju-7860.proxy.runpod.net/tts/reference_audio"
audio_path = r"C:\Users\ernes\Downloads\actor_4.wav"
text = "The quick brown fox jumps over the lazy dog."

with open(audio_path, 'rb') as f:
    files = {'audio': ('actor_4.wav', f, 'audio/wav')}
    data = {'text': text}
    print(f"Sending voice clone request...")
    response = requests.post(url, files=files, data=data, timeout=60)

print(f"Status: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    print("\n✅ Voice cloning successful!")
else:
    print("\n❌ Voice cloning failed")
