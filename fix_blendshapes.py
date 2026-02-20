import json
import csv

print("Fixing blendshape files...")

# Load the original JSON
with open('bark_blendshapes.json', 'r') as f:
    data = json.load(f)

print(f"Original frames: {len(data['blendshapes'])}")

# Fix the data
fixed_frames = []
for frame in data['blendshapes']:
    timestamp = frame['timestamp']
    blendshapes = frame['blendshapes']
    
    # Clamp all blendshape values to [0.0, 1.0]
    if isinstance(blendshapes, dict):
        clamped = {k: max(0.0, min(1.0, v)) for k, v in blendshapes.items()}
    else:
        clamped = [max(0.0, min(1.0, v)) for v in blendshapes]
    
    fixed_frames.append({
        'timestamp': timestamp,
        'blendshapes': clamped
    })

# Create properly formatted JSON
fixed_json = {
    'frame_rate': 60,
    'total_frames': len(fixed_frames),
    'frames': fixed_frames  # Changed from 'blendshapes' to 'frames'
}

# Save fixed JSON
with open('bark_blendshapes_fixed.json', 'w') as f:
    json.dump(fixed_json, f, indent=2)

print(f"✅ Fixed JSON saved: bark_blendshapes_fixed.json")
print(f"   - Added frame_rate: 60")
print(f"   - Added total_frames: {len(fixed_frames)}")
print(f"   - Renamed 'blendshapes' to 'frames'")
print(f"   - Clamped all values to [0.0, 1.0]")

# Create fixed CSV with clamped values
with open('bark_blendshapes_fixed.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Get blendshape names
    first_frame = fixed_frames[0]['blendshapes']
    if isinstance(first_frame, dict):
        blendshape_names = list(first_frame.keys())
    else:
        blendshape_names = [f'blendshape_{i}' for i in range(len(first_frame))]
    
    # Write header
    header = ['timestamp'] + blendshape_names
    writer.writerow(header)
    
    # Write data with clamped values
    for frame in fixed_frames:
        timestamp = frame['timestamp']
        blendshapes = frame['blendshapes']
        
        if isinstance(blendshapes, dict):
            row = [timestamp] + [blendshapes[name] for name in blendshape_names]
        else:
            row = [timestamp] + blendshapes
        
        writer.writerow(row)

print(f"✅ Fixed CSV saved: bark_blendshapes_fixed.csv")
print(f"   - All values clamped to [0.0, 1.0]")

# Verify the fixes
print(f"\n📊 Verification:")
print(f"   Total frames: {len(fixed_frames)}")
print(f"   Frame rate: 60 fps")
print(f"   Duration: {fixed_frames[-1]['timestamp']:.2f}s")

# Check value ranges
all_values = []
for frame in fixed_frames:
    bs = frame['blendshapes']
    if isinstance(bs, dict):
        all_values.extend(bs.values())
    else:
        all_values.extend(bs)

print(f"   Value range: [{min(all_values):.4f}, {max(all_values):.4f}]")
print(f"   ✅ All values within [0.0, 1.0]")

print(f"\n🎯 Files are now production-ready!")
