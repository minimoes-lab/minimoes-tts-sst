import json
import csv

# Load JSON
with open('demo_outputs/demo_blendshapes.json', 'r') as f:
    data = json.load(f)

# Write CSV
with open('demo_outputs/demo_blendshapes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header
    if data['frames']:
        first_frame = data['frames'][0]
        blendshapes = first_frame['blendshapes']
        
        # Handle both dict and list formats
        if isinstance(blendshapes, dict):
            num_bs = len(blendshapes)
            header = ['timestamp'] + [f'blendshape_{i}' for i in range(num_bs)]
        else:
            num_bs = len(blendshapes)
            header = ['timestamp'] + [f'blendshape_{i}' for i in range(num_bs)]
        
        writer.writerow(header)
        
        # Data
        for frame in data['frames']:
            blendshapes = frame['blendshapes']
            
            # Convert dict to list if needed
            if isinstance(blendshapes, dict):
                blendshapes = list(blendshapes.values())
            
            row = [frame['timestamp']] + blendshapes
            writer.writerow(row)

print('CSV generated successfully!')
print(f"Total frames: {len(data['frames'])}")
