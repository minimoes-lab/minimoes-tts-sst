

# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
from fastapi import Request  # Capital "R" for FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import io
import base64
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import uuid
import shutil
import uvicorn
import requests
import re
import tempfile
import threading
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Core Machine Learning & NLP Imports ---
import torch
import numpy as np
import pandas as pd
import docx
from pptx import Presentation
import zipfile
from PyPDF2 import PdfReader
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, BarkModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# =====================================================================================
# 1. API Initialization & Configuration
# =====================================================================================


from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config
import multiprocessing
app = FastAPI(
    title="Intelligent Document & Web API",
    description="A high-quality API for querying documents and websites using a RAG pipeline with Groq, and generating speech with Bark TTS.",
    version="2.0.1" # Version updated
)


if __name__ == '__main__' or __name__.startswith("api"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Join that with the filename
model_path = "utils/model/model.pth"

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}
print(f"--- ATTEMPTING TO LOAD: {model_path} ---")
blendshape_model = load_model(model_path, config, device)
print(f"DEBUG: Absolute path is: {model_path}")
@app.post("/audio_to_blendshapes")
async def audio_to_blendshapes_route(request: Request):

    audio_bytes = await request.body()    
    generated_facial_data = generate_facial_data_from_bytes(audio_bytes, blendshape_model, device, config)
    generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

    return JSONResponse(content={'blendshapes': generated_facial_data_list})


STATIC_AUDIO_DIR = "/app/generated_audio"



# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**









# --- Configuration & Global State ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please set it as an environment variable.")

conversations: Dict[str, ConversationalRetrievalChain] = {}
embeddings_model = None
bark_processor, bark_model, bark_sr, bark_device = None, None, None, None
bark_available = False

# =====================================================================================
# 2. High-Quality RAG Prompt Template
# =====================================================================================

PROFESSIONAL_RAG_PROMPT_TEMPLATE = """
You are a highly intelligent and diligent AI research assistant. Your primary goal is to provide accurate, concise, and helpful answers based *only* on the context provided.

**Instructions:**
1.  **Analyze the Context:** Carefully read and understand the `context` provided below. It is your only source of truth.
2.  **Answer the Question:** Use the context to answer the user's `question`.
3.  **Strict Grounding:** Do not use any external knowledge. If the answer is not in the context, you MUST state: "I am sorry, but the information required to answer your question is not available in the provided documents." Do not try to guess or infer information that isn't explicitly stated.
4.  **Synthesize Information:** If the question requires combining information from multiple parts of the context, synthesize a coherent answer.
5.  **Clarity and Conciseness:** Provide a clear and direct answer. If appropriate, use bullet points to structure complex information.
6.  **Cite Sources (if applicable):** While not strictly required, if you can identify the source of a piece of information within the context, it's good practice to mention it.

**Context:**
{context}

**Chat History:**
{chat_history}

**Question:**
{question}

**Answer:**
"""
RAG_PROMPT = PromptTemplate.from_template(PROFESSIONAL_RAG_PROMPT_TEMPLATE)

# =====================================================================================
# 3. Model Loading (at Startup)
# =====================================================================================
from fastapi.responses import PlainTextResponse

@app.get("/tts_diagnose")
async def tts_diagnose():
    """Run a short TTS diagnostic and return useful debug details."""
    debug = {}
    debug["bark_available"] = bool(bark_available)
    debug["bark_processor_type"] = repr(type(bark_processor))
    debug["bark_model_type"] = repr(type(bark_model))
    debug["bark_device"] = repr(bark_device)
    debug["bark_sr"] = repr(bark_sr)

    # Short test text
    test_text = "Diagnostic test. If you receive audio, the TTS pipeline works."

    try:
        print(f"[{datetime.now()}] TTS DIAGNOSE: starting generation test...")
        b64 = generate_speech_content_base64(test_text, voice_preset=None)
        if b64:
            debug["audio_base64_len"] = len(b64)
            # Save to file for convenience (optional)
            try:
                import base64 as _b64, os as _os
                out = "tts_diagnose_out.wav"
                with open(out, "wb") as f:
                    f.write(_b64.b64decode(b64))
                debug["wrote_file"] = out
                debug["wrote_file_size"] = _os.path.getsize(out)
            except Exception as exf:
                debug["save_file_error"] = repr(exf)
        else:
            debug["audio_base64_len"] = 0
            debug["note"] = "generate_speech_content_base64 returned None"
    except Exception as e:
        debug["exception"] = repr(e)

    # Return a plain JSON-like text for easy copying into the chat
    import json
    return PlainTextResponse(json.dumps(debug, indent=2))

@app.on_event("startup")

    
async def load_models():
    """Load heavy ML models once when the API starts up."""
    global embeddings_model, bark_processor, bark_model, bark_sr, bark_device, bark_available

    print(f"[{datetime.now()}] Starting model loading...")
    
    print(f"[{datetime.now()}] Loading HuggingFace embeddings model...")
    embeddings_start_time = time.time()
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    embeddings_end_time = time.time()
    print(f"[{datetime.now()}] Embeddings model loaded successfully in {embeddings_end_time - embeddings_start_time:.2f} seconds.")

    try:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{datetime.now()}] Using device for Bark TTS: {device_str}")
        device = torch.device(device_str)

        print(f"[{datetime.now()}] Loading Bark processor...")
        bark_processor_start_time = time.time()
        processor = AutoProcessor.from_pretrained("suno/bark")
        bark_processor_end_time = time.time()
        print(f"[{datetime.now()}] Bark processor loaded in {bark_processor_end_time - bark_processor_start_time:.2f} seconds.")

        print(f"[{datetime.now()}] Loading Bark model...")
        bark_model_start_time = time.time()
        model = BarkModel.from_pretrained("suno/bark").to(device)
        bark_model_end_time = time.time()
        print(f"[{datetime.now()}] Bark model loaded in {bark_model_end_time - bark_model_start_time:.2f} seconds.")

        bark_processor, bark_model, bark_sr, bark_device = processor, model, model.generation_config.sample_rate, device
        bark_available = True
        print(f"[{datetime.now()}] Bark model and processor loaded successfully!")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to load Bark model: {e}. TTS will be disabled.")
        bark_available = False
    print(f"[{datetime.now()}] Model loading complete.")

# =====================================================================================
# 4. Pydantic Models for API Data Validation
# =====================================================================================

class ProcessResponse(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the processing session.")
    message: str = Field(..., description="A confirmation message.")
    filenames: List[str] = Field(..., description="List of filenames processed.")
    processed_url: Optional[str] = Field(None, description="The URL that was processed, if any.")

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="The session ID obtained from the /process endpoint.")
    question: str = Field(..., description="The question to ask the RAG system.")
    voice_preset: Optional[str] = Field(None, description="Optional voice preset for Bark TTS (e.g., 'v2/en_speaker_6').")

class SourceDocument(BaseModel):
    page_content: str = Field(..., description="The text content of the source chunk.")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the source, like file name or page number.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the RAG model.")
    source_documents: List[SourceDocument] = Field(..., description="List of source document chunks used for the answer.")
    # Changed from audio_url to audio_base64
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio content.")

# =====================================================================================
# 5. Helper Functions for Document and Web Parsing
# =====================================================================================

class ContentExtractor:
    """A centralized class for extracting text from various sources."""
    def from_url(self, url: str) -> str:
        try:
            print(f"[{datetime.now()}] Extracting content from URL: {url}")
            response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
            text = soup.get_text(separator='\n', strip=True)
            print(f"[{datetime.now()}] URL content extracted successfully.")
            return text
        except requests.RequestException as e:
            print(f"[{datetime.now()}] ERROR: Failed to fetch or parse URL {url}. Error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL {url}. Error: {e}")

    def from_pdf(self, file_stream) -> str:
        try:
            print(f"[{datetime.now()}] Extracting content from PDF.")
            reader = PdfReader(file_stream)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            print(f"[{datetime.now()}] PDF content extracted successfully.")
            return text
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: PDF Parsing Error: {e}")
            return f"[PDF Parsing Error: {e}]"

    def from_docx(self, file_stream) -> str:
        try:
            print(f"[{datetime.now()}] Extracting content from DOCX.")
            doc = docx.Document(file_stream)
            text = "\n".join(para.text for para in doc.paragraphs if para.text)
            print(f"[{datetime.now()}] DOCX content extracted successfully.")
            return text
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: DOCX Parsing Error: {e}")
            return f"[DOCX Parsing Error: {e}]"

    def from_pptx(self, file_stream) -> str:
        try:
            print(f"[{datetime.now()}] Extracting content from PPTX.")
            prs = Presentation(file_stream)
            text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
            print(f"[{datetime.now()}] PPTX content extracted successfully.")
            return text
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: PPTX Parsing Error: {e}")
            return f"[PPTX Parsing Error: {e}]"

    def from_excel(self, file_stream) -> str:
        try:
            print(f"[{datetime.now()}] Extracting content from Excel.")
            xls = pd.ExcelFile(file_stream)
            text = ""
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                text += f"--- Sheet: {sheet_name} ---\n{df.to_string()}\n\n"
            print(f"[{datetime.now()}] Excel content extracted successfully.")
            return text
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: Excel Parsing Error: {e}")
            return f"[Excel Parsing Error: {e}]"

    def from_zip(self, file_stream, temp_dir) -> str:
        print(f"[{datetime.now()}] Extracting content from ZIP.")
        text = ""
        try:
            with zipfile.ZipFile(file_stream) as z:
                z.extractall(path=temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        text += self.from_file_path(file_path) + "\n\n"
            print(f"[{datetime.now()}] ZIP content extracted successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: ZIP Parsing Error: {e}")
            text += f"[ZIP Parsing Error: {e}]"
        return text

    def from_file_path(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[{datetime.now()}] Processing file: {os.path.basename(file_path)} with extension {ext}")
        with open(file_path, "rb") as f:
            if ext == ".pdf": return self.from_pdf(f)
            elif ext == ".docx": return self.from_docx(f)
            elif ext == ".pptx": return self.from_pptx(f)
            elif ext in [".xls", ".xlsx"]: return self.from_excel(f)
            else:
                print(f"[{datetime.now()}] INFO: Unsupported File in ZIP: {os.path.basename(file_path)}")
                return f"[Unsupported File in ZIP: {os.path.basename(file_path)}]"

# =====================================================================================
# 6. Core RAG and TTS Logic
# =====================================================================================
def get_rag_chain(text_chunks: List[str]) -> ConversationalRetrievalChain:
    print(f"[{datetime.now()}] Creating RAG chain...")
    if not text_chunks:
        raise ValueError("Cannot create RAG chain with no text chunks.")
    if not embeddings_model:
        raise RuntimeError("Embeddings model not loaded.")
    try:
        vector_store_start_time = time.time()
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
        vector_store_end_time = time.time()
        print(f"[{datetime.now()}] FAISS vector store created in {vector_store_end_time - vector_store_start_time:.2f} seconds.")

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        llm = ChatGroq(
                     model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.7,
                     max_tokens=800,
                     groq_api_key=GROQ_API_KEY
        )

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": RAG_PROMPT}
        )
        print(f"[{datetime.now()}] RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to create RAG chain: {e}")
        raise RuntimeError(f"Failed to create RAG chain: {e}")

# Add this import near the top of the file
from collections.abc import Mapping

def generate_speech_content_base64(text: str, voice_preset: Optional[str]) -> Optional[str]:
    print(f"[{datetime.now()}] [IN-MEMORY TTS] Starting speech generation (Base64 encoding).")
    if not bark_available or not text:
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Bark TTS not available or no text provided. Skipping.")
        return None

    # helpers (unchanged except for small improvements)
    def to_tensor_strict(obj, device):
        if torch.is_tensor(obj):
            # convert small int types to long for indexing
            if obj.dtype in (torch.int8, torch.int16, torch.int32):
                obj = obj.long()
            return obj.to(device)
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.integer):
                return torch.tensor(obj, dtype=torch.long, device=device)
            else:
                return torch.tensor(obj, dtype=torch.float32, device=device)
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return torch.tensor([], device=device)
            if all(isinstance(i, int) for i in obj):
                return torch.tensor(list(obj), dtype=torch.long, device=device)
            if all(isinstance(i, (float, int)) for i in obj):
                return torch.tensor(list(obj), dtype=torch.float32, device=device)
            converted = [to_tensor_strict(x, device) for x in obj]
            return type(obj)(converted)
        try:
            import pandas as _pd
            if isinstance(obj, _pd.Series):
                return to_tensor_strict(obj.to_numpy(), device)
            if isinstance(obj, _pd.DataFrame):
                return to_tensor_strict(obj.to_numpy(), device)
        except Exception:
            pass
        try:
            if hasattr(obj, "__array__"):
                arr = np.asarray(obj)
                return to_tensor_strict(arr, device)
        except Exception:
            pass
        return obj

    def recursively_convert(obj, device, path="root"):
        inventory = []
        # Mapping-like objects (dict, BatchEncoding, etc.)
        if isinstance(obj, Mapping):
            converted = {}
            for k, v in obj.items():
                conv, inv = recursively_convert(v, device, path + f".{k}")
                converted[k] = conv
                inventory.extend(inv)
            return converted, inventory
        # Objects that offer .to(device) (BatchEncoding has .to)
        if hasattr(obj, "to") and callable(obj.to) and not torch.is_tensor(obj):
            try:
                moved = obj.to(device)
                # If it's a mapping after moving, convert its items
                if isinstance(moved, Mapping):
                    converted = {}
                    for k, v in moved.items():
                        conv, inv = recursively_convert(v, device, path + f".{k}")
                        converted[k] = conv
                        inventory.extend(inv)
                    return converted, inventory
                # otherwise treat the moved object as a leaf (fall back)
                obj = moved
            except Exception as e:
                # ignore failures to .to(); we'll still try to convert internal pieces
                print(f"[{datetime.now()}] [IN-MEMORY TTS] .to(device) attempt failed for {type(obj)}: {e}")

        if isinstance(obj, (list, tuple)):
            converted_list = []
            for i, v in enumerate(obj):
                conv, inv = recursively_convert(v, device, path + f"[{i}]")
                converted_list.append(conv)
                inventory.extend(inv)
            return (type(obj)(converted_list), inventory)

        # leaf node: convert arrays/tensors where possible
        converted = to_tensor_strict(obj, device)
        if torch.is_tensor(converted):
            inv = [(path, tuple(converted.shape), str(converted.dtype), str(converted.device))]
        else:
            inv = [(path, None, repr(type(converted)), None)]
        return converted, inv

    try:
        start = time.time()
        # prepare inputs using multiple safe signatures (your existing pattern)
        inputs = None
        attempts = [
            {"voice_preset": voice_preset, "return_tensors": "pt", "padding": True},
            {"voice_preset": voice_preset, "return_tensors": "pt"},
            {"return_tensors": "pt", "padding": True},
            {"return_tensors": "pt"},
            {}
        ]
        last_exc = None
        for kw in attempts:
            try:
                safe_kw = {k: v for k, v in kw.items() if v is not None}
                inputs = bark_processor(text, **safe_kw)
                print(f"[{datetime.now()}] [IN-MEMORY TTS] Processor success with args: {list(safe_kw.keys())}")
                break
            except Exception as e:
                last_exc = e
                print(f"[{datetime.now()}] [IN-MEMORY TTS] Processor attempt failed with {list(kw.keys())}: {e}")

        if inputs is None:
            # tokenizer fallback
            if hasattr(bark_processor, "tokenizer"):
                try:
                    inputs = bark_processor.tokenizer(text, return_tensors="pt", padding=True)
                    print(f"[{datetime.now()}] [IN-MEMORY TTS] Tokenizer fallback used.")
                except Exception as e:
                    print(f"[{datetime.now()}] [IN-MEMORY TTS] Tokenizer fallback failed: {e}")
            if inputs is None:
                raise RuntimeError("Failed to prepare processor inputs: " + repr(last_exc))

        # IMPORTANT: Many transformers objects (BatchEncoding) implement `.to(device)`.
        # Try to move the whole object first — this usually solves mixed-device issues.
        try:
            if hasattr(inputs, "to") and callable(inputs.to):
                try:
                    inputs = inputs.to(bark_device)
                    print(f"[{datetime.now()}] [IN-MEMORY TTS] Called inputs.to({bark_device}) successfully.")
                except Exception as e:
                    print(f"[{datetime.now()}] [IN-MEMORY TTS] inputs.to(...) failed: {e}")
        except Exception:
            pass

        # Convert EVERYTHING recursively into tensors on device
        converted_inputs, inventory = recursively_convert(inputs, bark_device, path="inputs")
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Inventory of converted inputs (path, shape, dtype, device):")
        for item in inventory:
            print("  ", item)

        # Ensure attention_mask exists and is long on device
        if isinstance(converted_inputs, dict) and "input_ids" in converted_inputs and "attention_mask" not in converted_inputs:
            try:
                input_ids = converted_inputs["input_ids"]
                if torch.is_tensor(input_ids):
                    pad_id = getattr(bark_model.config, "pad_token_id", None)
                    if pad_id is not None:
                        mask = (input_ids != int(pad_id)).long().to(bark_device)
                    else:
                        mask = torch.ones_like(input_ids, dtype=torch.long, device=bark_device)
                    converted_inputs["attention_mask"] = mask
                    print(f"[{datetime.now()}] [IN-MEMORY TTS] Built attention_mask shape {mask.shape} on {mask.device}.")
            except Exception as e:
                print(f"[{datetime.now()}] [IN-MEMORY TTS] Failed building attention_mask: {e}")

        # Off-device detection (improved to inspect Mapping objects)
        def find_off_device(obj, desired_device, prefix=""):
            off = []
            if torch.is_tensor(obj):
                if str(obj.device) != str(desired_device):
                    off.append((prefix or "leaf", tuple(obj.shape), str(obj.dtype), str(obj.device)))
                return off
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    off.extend(find_off_device(v, desired_device, prefix + f".{k}"))
                return off
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    off.extend(find_off_device(v, desired_device, prefix + f"[{i}]"))
                return off
            # try attributes for objects that may contain tensors
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    off.extend(find_off_device(v, desired_device, prefix + f".{k}"))
            return off

        off = find_off_device(converted_inputs, bark_device, "inputs")
        if off:
            print(f"[{datetime.now()}] [IN-MEMORY TTS] WARNING: Found tensors off-device after conversion: {off}")
            # Optionally raise here to fail-fast and debug
            # raise RuntimeError(f"Found off-device tensors: {off}")
        else:
            print(f"[{datetime.now()}] [IN-MEMORY TTS] All tensors appear on {bark_device}.")

        # Prepare pad_token_id if available
        pad_token_id = None
        try:
            cfg_pad = getattr(bark_model.config, "pad_token_id", None)
            if cfg_pad is not None and int(cfg_pad) >= 0:
                pad_token_id = int(cfg_pad)
        except Exception:
            pad_token_id = None

        # Generate audio
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Calling Bark model.generate(...)")
        bark_model.eval()
        with torch.no_grad():
            try:
                gen_kwargs = {"pad_token_id": pad_token_id} if pad_token_id is not None else {}
                bark_output = bark_model.generate(**converted_inputs, **gen_kwargs)
            except TypeError as e:
                print(f"[{datetime.now()}] [IN-MEMORY TTS] generate(...) TypeError with gen_kwargs: {e}. Retrying without gen_kwargs.")
                bark_output = bark_model.generate(**converted_inputs)

        # (the rest of your output handling -> to numpy, int16, wav writing) unchanged...
        audio_array = None
        if isinstance(bark_output, dict):
            for key in ("audio", "audios", "waveform", "wav", "output_audio"):
                if key in bark_output:
                    audio_array = bark_output[key]; break
            if audio_array is None and "outputs" in bark_output:
                cand = bark_output["outputs"]
                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                    audio_array = cand[0]
        elif isinstance(bark_output, (list, tuple)):
            audio_array = bark_output[0]
        else:
            audio_array = bark_output

        if audio_array is None:
            raise RuntimeError("Could not locate waveform in Bark model output.")

        if hasattr(audio_array, "cpu"):
            audio_np = audio_array.cpu().numpy().squeeze()
        else:
            audio_np = np.asarray(audio_array).squeeze()

        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2 and audio_np.shape[0] < audio_np.shape[-1]:
                audio_np = audio_np.mean(axis=0)
            else:
                audio_np = audio_np.mean(axis=-1)
        try:
            sr = int(bark_sr)
        except Exception:
            sr = 24000
        if np.issubdtype(audio_np.dtype, np.floating):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767.0).astype(np.int16)
        else:
            audio_int16 = audio_np.astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, sr, audio_int16)
        buf.seek(0)
        bts = buf.read()
        b64 = base64.b64encode(bts).decode("utf-8")
        end = time.time()
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Finished in {end-start:.2f}s; base64 len {len(b64)}")
        return b64

    except Exception as e:
        print(f"[{datetime.now()}] [IN-MEMORY TTS] ERROR during speech generation/encoding: {repr(e)}")
        return None

# =====================================================================================
# 7. API Endpoints
# =====================================================================================

# The /audio mount is technically no longer needed for TTS audio, but kept for static files
# if other non-TTS static files are needed.
STATIC_AUDIO_DIR = "generated_audio"
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=STATIC_AUDIO_DIR), name="audio")

@app.post("/process", response_model=ProcessResponse)
async def process_content(
    files: Optional[List[UploadFile]] = File(None, description="A list of documents to process."),
    url: Optional[str] = Form(None, description="A URL to a website to scrape for text."),
    prompt_template: Optional[str] = Form(None, description="Optional custom RAG prompt template. Must include {context}, {chat_history}, and {question}.")
):
    print(f"[{datetime.now()}] /process endpoint called.")
    if not files and not url:
        raise HTTPException(status_code=400, detail="Please provide at least one document or a URL.")
    
    extractor = ContentExtractor()
    temp_dir = f"temp_{uuid.uuid4().hex}"
    os.makedirs(temp_dir)
    raw_text = ""
    processed_files = []
    
    try:
        process_start_time = time.time()
        if url:
            print(f"[{datetime.now()}] Processing URL: {url}")
            raw_text += extractor.from_url(url) + "\n\n"
        
        if files:
            print(f"[{datetime.now()}] Processing files: {[f.filename for f in files]}")
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                ext = os.path.splitext(file.filename)[1].lower()
                if ext == ".zip":
                    raw_text += extractor.from_zip(file_path, os.path.join(temp_dir, "unzipped"))
                else:
                    raw_text += extractor.from_file_path(file_path) + "\n\n"
                processed_files.append(file.filename)
        
        if not raw_text.strip():
            print(f"[{datetime.now()}] ERROR: No text extracted from sources.")
            raise HTTPException(status_code=400, detail="No text could be extracted from the provided sources.")
        
        print(f"[{datetime.now()}] Splitting text into chunks...")
        text_splitter_start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(raw_text)
        text_splitter_end_time = time.time()
        print(f"[{datetime.now()}] Text split into {len(text_chunks)} chunks in {text_splitter_end_time - text_splitter_start_time:.2f} seconds.")
        
        session_id = str(uuid.uuid4())
        print(f"[{datetime.now()}] Creating RAG chain for session: {session_id}")
        conversations[session_id] = get_rag_chain(text_chunks)
        
        if prompt_template:
            print(f"[{datetime.now()}] Applying custom prompt template.")
            conversations[session_id].combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
        
        process_end_time = time.time()
        print(f"[{datetime.now()}] /process endpoint finished in {process_end_time - process_start_time:.2f} seconds.")
        return ProcessResponse(
            session_id=session_id,
            message="Content processed successfully. You can now use the /query endpoint.",
            filenames=processed_files,
            processed_url=url
        )
    except Exception as e:
        print(f"[{datetime.now()}] ERROR in /process endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        print(f"[{datetime.now()}] Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

@app.post("/query", response_model=QueryResponse)
async def query_session(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Ask a question within an existing session.
    This endpoint uses the session's context to generate an answer and optionally creates a TTS audio file.
    """
    print(f"[{datetime.now()}] /query endpoint called for session: {request.session_id}")
    query_start_time = time.time()

    chain = conversations.get(request.session_id)
    if not chain:
        print(f"[{datetime.now()}] ERROR: Session {request.session_id} not found.")
        raise HTTPException(status_code=404, detail="Session not found. Please process content first.")

    try:
        print(f"[{datetime.now()}] Invoking RAG chain for question: '{request.question[:50]}...'")
        rag_invoke_start_time = time.time()
        result = chain.invoke({"question": request.question})
        rag_invoke_end_time = time.time()
        print(f"[{datetime.now()}] RAG chain invoked in {rag_invoke_end_time - rag_invoke_start_time:.2f} seconds.")
        
        answer = result.get("answer", "No answer could be generated.")
        print(f"[{datetime.now()}] RAG Answer generated. Length: {len(answer)} chars.")

        # --- FIX: Manually convert LangChain documents to Pydantic models ---
        source_docs_result = result.get("source_documents", [])
        validated_sources = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in source_docs_result
        ]
        # --- END FIX ---

        audio_base64_content = None # Changed from audio_url
        if bark_available:
            print(f"[{datetime.now()}] Bark TTS is available. Generating audio (Base64).")
            # Call the new function that returns base64 content
            audio_base64_content = generate_speech_content_base64(text=answer, voice_preset=request.voice_preset)
            if audio_base64_content:
                print(f"[{datetime.now()}] Base64 audio generated and ready for response.")
            else:
                print(f"[{datetime.now()}] Failed to generate Base64 audio content.")
        else:
            print(f"[{datetime.now()}] Bark TTS not available. Skipping audio generation.")

        query_end_time = time.time()
        print(f"[{datetime.now()}] /query endpoint finished in {query_end_time - query_start_time:.2f} seconds.")
        return QueryResponse(
            answer=answer,
            source_documents=validated_sources, # Use the validated list here
            audio_base64=audio_base64_content # Changed from audio_url
        )
    except Exception as e:
        print(f"[{datetime.now()}] ERROR in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {e}")