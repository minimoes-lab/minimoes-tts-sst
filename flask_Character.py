
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
from fastapi import Request  # Capital "R" for FastAPI
from fastapi.responses import JSONResponse

import os
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
from langchain_community.embeddings import HuggingFaceEmbeddings
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

app = FastAPI(
    title="Intelligent Document & Web API",
    description="A high-quality API for querying documents and websites using a RAG pipeline with Groq, and generating speech with Bark TTS.",
    version="2.0.1" # Version updated
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    audio_bytes = request.data
    generated_facial_data = generate_facial_data_from_bytes(audio_bytes, blendshape_model, device, config)
    generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

    return jsonify({'blendshapes': generated_facial_data_list})


# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**




app = FastAPI(
    title="Intelligent Document & Web API",
    description="A high-quality API for querying documents and websites using a RAG pipeline with Groq, and generating speech with Bark TTS.",
    version="2.0.1" # Version updated
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)


# =====================================================================================
# 1. API Initialization & Configuration
# =====================================================================================



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

# Modified: Now returns base64 encoded audio instead of saving a file
def generate_speech_content_base64(text: str, voice_preset: Optional[str]) -> Optional[str]:
    print(f"[{datetime.now()}] [IN-MEMORY TTS] Starting speech generation (Base64 encoding).")
    if not bark_available or not text:
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Bark TTS not available or no text provided. Skipping.")
        return None
    try:
        speech_gen_start_time = time.time()
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Preparing Bark inputs... (Ensuring all tensors are on {bark_device})")
        inputs_prepare_start_time = time.time()
        
        # Get inputs first
        inputs = bark_processor(text, voice_preset=voice_preset, return_tensors="pt")
        # Explicitly move each tensor to the bark_device, like in your working code
        inputs = {k: v.to(bark_device) for k, v in inputs.items()}
        
        inputs_prepare_end_time = time.time()
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Bark inputs prepared in {inputs_prepare_end_time - inputs_prepare_start_time:.2f} seconds.")

        print(f"[{datetime.now()}] [IN-MEMORY TTS] Generating audio with Bark model...")
        audio_gen_start_time = time.time()
        with torch.no_grad():
            # Check for multiple outputs and select the first one, or flatten if only one
            # The error suggests the output might be a single tensor within a tuple or list
            bark_output = bark_model.generate(**inputs)
            # Ensure we're selecting the correct tensor and flattening
            if isinstance(bark_output, tuple) or isinstance(bark_output, list):
                audio_array = bark_output[0].cpu().numpy().flatten()
            else:
                audio_array = bark_output.cpu().numpy().flatten()
                
        audio_gen_end_time = time.time()
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Bark audio generated in {audio_gen_end_time - audio_gen_start_time:.2f} seconds.")

        # Convert numpy array to WAV bytes in memory
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Converting audio to WAV bytes and Base64 encoding...")
        buffer = io.BytesIO()
        wavfile.write(buffer, bark_sr, audio_array)
        audio_bytes = buffer.getvalue()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Audio converted and Base64 encoded. Size: {len(base64_audio)} bytes (Base64).")
        
        speech_gen_end_time = time.time()
        print(f"[{datetime.now()}] [IN-MEMORY TTS] Speech generation and encoding complete (Total: {speech_gen_end_time - speech_gen_start_time:.2f} seconds).")
        return base64_audio
    except Exception as e:
        print(f"[{datetime.now()}] [IN-MEMORY TTS] ERROR during speech generation/encoding: {e}")
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


