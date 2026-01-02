import runpod
import torch
import os
from pathlib import Path
import requests
import time
from datetime import datetime

# Import your existing modules
from transformers import AutoProcessor, BarkModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import numpy as np
import scipy.io.wavfile as wavfile

# =====================================================================================
# GLOBAL MODEL LOADING (Outside handler - as per RunPod best practices)
# =====================================================================================

print(f"[{datetime.now()}] Initializing models...")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")

# Initialize global variables
embeddings_model = None
bark_processor = None
bark_model = None
bark_sr = None
bark_device = None
bark_available = False

def ensure_model_downloaded():
    """Ensure the face model is downloaded."""
    model_path = "utils/model/model.pth"
    model_url = "https://huggingface.co/KKKONNK/model/resolve/main/model.pth"
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        if file_size > 500_000_000:  # 500MB
            print(f"✅ Model already exists ({file_size / 1_000_000:.1f}MB)")
            return True
    
    print(f"📥 Downloading model from {model_url}...")
    try:
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(model_path)
        if file_size > 500_000_000:
            print(f"✅ Model downloaded successfully ({file_size / 1_000_000:.1f}MB)")
            return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def load_models():
    """Load all models globally (RunPod best practice)."""
    global embeddings_model, bark_processor, bark_model, bark_sr, bark_device, bark_available
    
    # Download face model
    ensure_model_downloaded()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[{datetime.now()}] Loading HuggingFace embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    print(f"[{datetime.now()}] Embeddings model loaded successfully.")

    try:
        print(f"[{datetime.now()}] Loading Bark TTS models...")
        bark_processor = AutoProcessor.from_pretrained("suno/bark")
        bark_model = BarkModel.from_pretrained("suno/bark").to(device)
        bark_sr = bark_model.generation_config.sample_rate
        bark_device = device
        bark_available = True
        print(f"[{datetime.now()}] Bark models loaded successfully!")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to load Bark model: {e}. TTS will be disabled.")
        bark_available = False

# Load models at startup (outside handler)
load_models()

# =====================================================================================
# RUNPOD HANDLER FUNCTION (Official pattern)
# =====================================================================================

def handler(job):
    """
    Official RunPod Serverless handler function.
    
    Args:
        job (dict): Contains 'input' with request data and 'id' for job tracking
        
    Returns:
        dict: Response data
    """
    try:
        # Extract input data (official RunPod pattern)
        job_input = job["input"]
        job_id = job.get("id", "unknown")
        
        print(f"[{datetime.now()}] Processing job {job_id}")
        
        # Determine the action to perform
        action = job_input.get("action", "query")
        
        if action == "query":
            return handle_query(job_input)
        elif action == "tts":
            return handle_tts(job_input)
        elif action == "health":
            return handle_health(job_input)
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        print(f"[{datetime.now()}] Handler error: {e}")
        return {"error": str(e)}

def handle_query(input_data):
    """Handle query requests."""
    try:
        query = input_data.get("query", "")
        if not query:
            return {"error": "No query provided"}
        
        # Simple response for now - you can expand this with your full logic
        response = f"Processed query: {query}"
        
        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {"error": f"Query processing failed: {str(e)}"}

def handle_tts(input_data):
    """Handle TTS requests."""
    try:
        if not bark_available:
            return {"error": "TTS not available"}
        
        text = input_data.get("text", "")
        if not text:
            return {"error": "No text provided"}
        
        # TTS processing would go here
        return {
            "message": f"TTS would process: {text}",
            "status": "success"
        }
    except Exception as e:
        return {"error": f"TTS processing failed: {str(e)}"}

def handle_health(input_data):
    """Handle health check requests."""
    return {
        "status": "healthy",
        "models_loaded": {
            "embeddings": embeddings_model is not None,
            "bark": bark_available
        },
        "device": str(device)
    }

# =====================================================================================
# RUNPOD SERVERLESS START (Official pattern)
# =====================================================================================

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
