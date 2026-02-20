#!/usr/bin/env python3
"""
Quick script to download the blendshape model from HuggingFace.
Usage: python download_model.py
"""
import os
import sys
import requests
from pathlib import Path

def download_model():
    url = "https://huggingface.co/KKKONNK/model/resolve/main/model.pth"
    output_path = Path("utils/model/model.pth")
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        size = output_path.stat().st_size
        if size > 1_000_000:  # > 1MB
            print(f"✅ Model already exists ({size / 1_000_000:.1f}MB)")
            response = input("Download again? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                return True
    
    print("="*60)
    print("📥 Downloading Blendshape Model from HuggingFace")
    print("="*60)
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print()
    
    try:
        print("Connecting...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Size: {total_size / 1_000_000:.1f}MB")
        print()
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            last_percent = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress every 1%
                    progress = (downloaded / total_size) * 100
                    if int(progress) > last_percent:
                        last_percent = int(progress)
                        mb_downloaded = downloaded / 1_000_000
                        mb_total = total_size / 1_000_000
                        print(f"Progress: {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f}MB)", end='\r')
        
        print("\n")
        print("="*60)
        print("✅ Model downloaded successfully!")
        print("="*60)
        
        # Verify
        final_size = output_path.stat().st_size
        print(f"Final size: {final_size / 1_000_000:.1f}MB")
        
        if final_size < 1_000_000:
            print()
            print("⚠️  WARNING: File seems too small!")
            print("   Expected: ~600MB")
            print(f"   Got: {final_size / 1_000_000:.1f}MB")
            print("   Download may have failed. Please try again.")
            return False
        
        print()
        print("Next steps:")
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Run: uvicorn api:app --host 0.0.0.0 --port 7860")
        print("3. Test: curl http://localhost:7860/health")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network error: {e}")
        print("\nTroubleshooting:")
        print("- Check your internet connection")
        print("- Try using a VPN if HuggingFace is blocked")
        print("- Download manually from: https://huggingface.co/KKKONNK/model")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print()
    success = download_model()
    print()
    sys.exit(0 if success else 1)
