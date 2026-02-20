"""
Create an expressive reference audio for Qwen3-TTS voice cloning.
This will download a sample with natural prosody and emotion.
"""
import requests
import os

# Try multiple sources for expressive reference audio
reference_sources = [
    {
        "url": "https://github.com/suno-ai/bark/raw/main/assets/prompts/v2/en_speaker_6.npz",
        "name": "expressive_female.npz",
        "description": "Expressive female voice from Bark"
    },
    {
        "url": "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav",
        "name": "expressive_sample.wav",
        "description": "Sample expressive speech"
    }
]

print("Downloading expressive reference audio...")
print("=" * 60)

for source in reference_sources:
    try:
        print(f"\nTrying: {source['description']}")
        print(f"URL: {source['url']}")
        
        response = requests.get(source['url'], timeout=10)
        
        if response.status_code == 200:
            output_path = f"reference_audio/{source['name']}"
            os.makedirs("reference_audio", exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded: {output_path} ({len(response.content)} bytes)")
        else:
            print(f"✗ Failed: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Alternative: Record your own expressive audio!")
print("1. Record 3-5 seconds of expressive speech")
print("2. Save as 'reference_audio/custom_voice.wav'")
print("3. Update the Qwen TTS worker to use it")
print("=" * 60)
