# 🚀 AI Inference API — RAG → TTS → Blendshapes

This repository contains a GPU-accelerated AI inference service that processes documents or URLs, answers questions using an LLM, generates speech audio, and finally converts that audio into facial blendshape animation data — all in a single API request.

The system is designed for production deployment using a persistent GPU Pod with automated CI/CD.

---

## ✨ Key Features

- **Single `/infer` API endpoint**
  - One request → one response
  - No intermediate steps required

- **End-to-end pipeline**
  - RAG (document understanding)
  - LLM inference
  - Text-to-Speech (Bark)
  - Audio → Facial Blendshapes

- **GPU-accelerated**
  - Models are loaded once and kept warm in memory

- **Supports multiple input types**
  - URL
  - File uploads

- **JSON & CSV output**
  - Blendshapes returned directly

- **Automated CI/CD**
  - GitHub → Docker Hub → RunPod GPU Pod

---

## 🧠 High-Level Architecture
```
Client Request
   |
   v
/infer API
   |
   ├── Text extraction (URL / Files)
   ├── RAG (Embeddings + FAISS + LLM)
   ├── Bark TTS (Text → Audio)
   ├── Blendshape Model (Audio → Facial Data)
   |
   v
JSON / CSV Response
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI |
| **LLM** | Groq (via LangChain) |
| **RAG** | HuggingFace Embeddings + FAISS |
| **TTS** | Bark (Transformers) |
| **Facial Animation** | Custom blendshape model |
| **ML Framework** | PyTorch |
| **Deployment** | RunPod GPU Pod |
| **CI/CD** | GitHub Actions + Docker Hub |

---

## GPU Requirements

Based on real production testing:

| Resource | Usage |
|----------|-------|
| **Peak VRAM** | ~8 GB |
| **Recommended GPU** | 12–16 GB VRAM |
| **Tested GPU** | RTX 4000 Ada (20 GB) |
| **Avg /infer latency** | ~60 seconds |

The application runs safely with significant GPU headroom.

---

## API Overview

### `POST /infer`

Unified inference endpoint.

```markdown
NOTE: For hitting individual APIs, refer to the Swagger documentation at `http://localhost:7860/docs` or your deployed host URL.
```

#### What it does

- Accepts a question
- Accepts either URL or files (user choice)
- Runs the full AI pipeline
- Returns final blendshapes (and optional audio / CSV)

---

## 📥 Request Format

### Required

- `request_raw` → JSON string with inference parameters
- `url` → Website URL or `files` → One or more documents or `both`

---

# API Usage Examples

## Example CURL Requests

## ✅ Minimal working CURL (URL only)
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Summarize this","return_csv":true}' \
  -F 'url=https://example.com'
```

---

## ✅ Files only
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Summarize this"}' \
  -F 'files=@document.pdf'
```

---

## ✅ Files + URL
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Compare these sources"}' \
  -F 'url=https://example.com' \
  -F 'files=@doc1.pdf'
```

---

## 📝 Notes

- Replace `localhost:7860` with your deployed host/IP
- Replace `@document.pdf` with the actual path to your file
- The `request_raw` field must be a valid JSON string
- Multiple files can be uploaded by adding more `-F 'files=@...'` flags

---

## 🔧 Additional Options

### Return audio in response
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Explain this","return_audio":true}' \
  -F 'url=https://example.com'
```

### Custom voice preset
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Read this aloud","voice_preset":"v2/en_speaker_6"}' \
  -F 'files=@document.pdf'
```

### Multiple files
```bash
curl -X POST http://localhost:7860/infer \
  -F 'request_raw={"question":"Analyze all documents"}' \
  -F 'files=@doc1.pdf' \
  -F 'files=@doc2.docx' \
  -F 'files=@data.xlsx'
```

---

## 📤 Response Format
```json
{
  "answer": "Generated answer text",
  "blendshapes": [[0.01, 0.03, ...]],
  "audio_base64": null,
  "csv": "0.01,0.03,..."
}
```

---

## 🛠 Local Development

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run API
```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

### 4️⃣ Swagger UI
```
http://localhost:7860/docs
```

---

## 🐳 Docker

### Build locally
```bash
docker build -t infer-api .
```

### Run container
```bash
docker run --rm -it -p 7860:7860 infer-api uvicorn api:app --host 0.0.0.0 --port 7860

```

---

## 🚀 Deployment on RunPod (Production)

- Deployed as a **persistent GPU Pod**
- GPU stays warm (no serverless cold starts)
- Port **7860** exposed
- Docker image pulled from **Docker Hub**

---

## 🔄 CI/CD Flow
```
GitHub push
   ↓
GitHub Actions
   ↓
Docker image build
   ↓
Push to Docker Hub (:latest)
   ↓
RunPod Pod restart (Manually)
   ↓
Updated service live
```

### GitHub Actions

- Automatically builds and pushes Docker images on every push to `main`
- Disk cleanup included for large ML builds

---

## 🔐 Credentials Required (for CI/CD)

To finalize CI/CD under the client's account:

- Docker Hub username
- Docker Hub access token

> **Note:** No secrets are hard-coded in the repository.

---

## 📧 Contact

contact@sorayia.com
