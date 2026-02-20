"""
Generate Qwen3-TTS audio samples with blendshapes for sample_outputs folder.
"""
import requests
import base64
import json
import csv

print("Generating Qwen3-TTS Samples")
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
print(f"   ✓ Session: {session_id}")

# Step 2: Generate audio with Qwen3-TTS
print("\n2. Generating Qwen3-TTS audio...")
response = requests.post(
    'http://localhost:7860/query',
    json={
        'session_id': session_id,
        'question': 'What is artificial intelligence in simple terms?',
        'use_qwen': True
    }
)

print(f"   Status: {response.status_code}")

if response.status_code != 200:
    print(f"Failed: {response.text}")
    exit(1)

data = response.json()
audio_b64 = data.get('audio_base64', '')

if not audio_b64:
    print("No audio generated")
    exit(1)

audio_bytes = base64.b64decode(audio_b64)
print(f"   ✓ Audio generated: {len(audio_bytes)} bytes")

# Save audio to sample_outputs
with open('sample_outputs/qwen3_sample_audio.wav', 'wb') as f:
    f.write(audio_bytes)
print(f"   ✓ Saved: sample_outputs/qwen3_sample_audio.wav")

# Step 3: Generate blendshapes from the audio
print("\n3. Generating blendshapes...")
response = requests.post(
    'http://localhost:7860/audio_to_blendshapes',
    data=audio_bytes,
    headers={'Content-Type': 'application/octet-stream'}
)

print(f"   Status: {response.status_code}")

if response.status_code != 200:
    print(f"Failed: {response.text}")
    exit(1)

blendshape_data = response.json()
print(f"   ✓ Generated {blendshape_data.get('total_frames', 0)} frames")

# Save JSON
with open('sample_outputs/qwen3_sample_blendshapes.json', 'w') as f:
    json.dump(blendshape_data, f, indent=2)
print(f"   ✓ Saved: sample_outputs/qwen3_sample_blendshapes.json")

# Save CSV
frames = blendshape_data.get('frames', [])
if frames:
    with open('sample_outputs/qwen3_sample_blendshapes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get blendshape names
        mapping = blendshape_data.get('mapping', [])
        if isinstance(mapping, list):
            blendshape_names = mapping
        else:
            blendshape_names = list(mapping.keys())
        
        # Write header
        header = ['timestamp'] + blendshape_names
        writer.writerow(header)
        
        # Write data
        for frame in frames:
            timestamp = frame['timestamp']
            blendshapes = frame['blendshapes']
            
            if isinstance(blendshapes, dict):
                row = [timestamp] + [blendshapes.get(name, 0.0) for name in blendshape_names]
            else:
                row = [timestamp] + blendshapes
            
            writer.writerow(row)
    
    print(f"   ✓ Saved: sample_outputs/qwen3_sample_blendshapes.csv")

print("\n" + "=" * 60)
print("✓ Qwen3-TTS samples generated successfully!")
print("\nFiles created:")
print("  - sample_outputs/qwen3_sample_audio.wav")
print("  - sample_outputs/qwen3_sample_blendshapes.json")
print("  - sample_outputs/qwen3_sample_blendshapes.csv")
print("=" * 60)
