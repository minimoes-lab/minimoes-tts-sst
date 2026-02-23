"""
Test script for the /infer endpoint - Full RAG + TTS + Blendshapes pipeline
"""
import requests
import json
import base64
import os

# API endpoint
API_URL = "http://localhost:7860/infer"

# Test data - we'll use a URL as the knowledge source
test_request = {
    "question": "What is artificial intelligence and how does it work?",
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "voice_preset": None,  # Use default voice
    "return_audio": True,  # Get audio back
    "return_csv": True     # Get blendshapes as CSV too
}

print("=" * 80)
print("TESTING FULL RAG + TTS + BLENDSHAPES PIPELINE")
print("=" * 80)
print(f"\nQuestion: {test_request['question']}")
print(f"Knowledge Source: {test_request['url']}")
print("\nSending request to API...")
print("-" * 80)

# Prepare the multipart form data
# Extract URL from request and pass it separately
url_param = test_request.pop('url', None)

files = []
data = {
    'request_raw': json.dumps(test_request),
    'url': url_param  # Pass URL as separate form field
}

try:
    # Send the request (no timeout - let it run as long as needed)
    response = requests.post(API_URL, data=data, files=files)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n✅ SUCCESS! Response received:")
        print("=" * 80)
        
        # 1. Show the answer
        print("\n📝 RAG ANSWER:")
        print("-" * 80)
        print(result['answer'])
        
        # 2. Show blendshape info
        print("\n\n🎭 BLENDSHAPES INFO:")
        print("-" * 80)
        print(f"Frame Rate: {result['frame_rate']} fps")
        print(f"Total Frames: {len(result['blendshapes'])}")
        print(f"Number of Blendshapes: {len(result['mapping'])}")
        print(f"\nFirst 5 blendshape names: {result['mapping'][:5]}")
        
        # Show first frame as example
        if result['blendshapes']:
            print(f"\nExample - First frame blendshapes:")
            first_frame = result['blendshapes'][0]
            # Show the structure
            print(f"  Keys in frame: {list(first_frame.keys())}")
            if 'blendshapes' in first_frame:
                bs_values = first_frame['blendshapes']
                print(f"  Sample values: {list(bs_values.items())[:3]}")
        
        # 3. Save audio if present
        if result.get('audio_base64'):
            print("\n\n🔊 AUDIO:")
            print("-" * 80)
            audio_bytes = base64.b64decode(result['audio_base64'])
            audio_path = "output_audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"✅ Audio saved to: {audio_path}")
            print(f"   Size: {len(audio_bytes):,} bytes")
        
        # 4. Save CSV if present
        if result.get('csv'):
            csv_path = "output_blendshapes.csv"
            with open(csv_path, 'w') as f:
                f.write(result['csv'])
            print(f"\n📊 CSV:")
            print("-" * 80)
            print(f"✅ Blendshapes CSV saved to: {csv_path}")
            # Show first few lines
            lines = result['csv'].split('\n')[:5]
            print(f"   Preview (first 5 lines):")
            for line in lines:
                print(f"   {line[:100]}...")
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)
        
    else:
        print(f"\n❌ ERROR: Status code {response.status_code}")
        print(response.text)
        
except requests.exceptions.Timeout:
    print("\n❌ ERROR: Request timed out (this can happen on first run)")
    print("   The API might still be initializing models. Try again in a moment.")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
