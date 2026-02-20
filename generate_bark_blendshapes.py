import requests
import json
import csv

print("Generating blendshapes from Bark audio...")

# Read the Bark audio file
with open('bark_audio_sample.wav', 'rb') as f:
    audio_bytes = f.read()

print(f"Audio size: {len(audio_bytes)} bytes")
print("Sending to /audio_to_blendshapes endpoint...")
print("This may take 1-2 minutes...")

# Send to blendshape endpoint
response = requests.post(
    'http://localhost:7860/audio_to_blendshapes',
    data=audio_bytes,
    headers={'Content-Type': 'application/octet-stream'}
)

print(f"Response status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    print(f"\n✅ Success!")
    print(f"   Total frames: {data.get('total_frames', len(data.get('frames', [])))}")
    print(f"   Frame rate: {data.get('frame_rate', 'N/A')} fps")
    print(f"   Blendshape names: {len(data.get('mapping', []))}")
    
    # Save JSON
    with open('bark_blendshapes.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Saved: bark_blendshapes.json")
    
    # Save CSV
    frames = data.get('frames', [])
    if frames:
        with open('bark_blendshapes.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Get blendshape names from mapping
            if isinstance(data['mapping'], list):
                blendshape_names = data['mapping']
            else:
                blendshape_names = list(data['mapping'].keys())
            
            header = ['timestamp'] + blendshape_names
            writer.writerow(header)
            
            # Write data
            for frame in frames:
                timestamp = frame['timestamp']
                blendshapes = frame['blendshapes']
                
                # Handle both dict and list formats
                if isinstance(blendshapes, dict):
                    row = [timestamp] + [blendshapes.get(name, 0.0) for name in blendshape_names]
                else:
                    row = [timestamp] + blendshapes
                
                writer.writerow(row)
        
        print(f"💾 Saved: bark_blendshapes.csv")
        print(f"\n🎯 Generated {len(frames)} frames at {data['frame_rate']} fps")
    
else:
    print(f"❌ Failed: {response.text}")
