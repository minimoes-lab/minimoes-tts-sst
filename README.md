# Streaming Avatar API with Qwen3-TTS

A high-performance FastAPI server for generating realistic avatar animations with synchronized speech and facial blendshapes using Qwen3-TTS and RAG (Retrieval-Augmented Generation).

## Features

- 🎙️ **Qwen3-TTS Integration**: Voice cloning with natural prosody
- 🎭 **52 ARKit Blendshapes**: Real-time facial animation generation
- 🔄 **WebSocket Streaming**: Low-latency audio and blendshape streaming
- 📚 **RAG Pipeline**: Context-aware responses using Groq LLM
- 🐳 **Docker Ready**: Easy deployment with Docker
- ⚡ **Optimized Performance**: CPU and CUDA support

## Architecture

```
User Query → RAG (Groq LLM) → Answer Text
                                    ↓
                            Qwen3-TTS (Voice Cloning)
                                    ↓
                              Audio (12kHz WAV)
                                    ↓
                          Blendshape Model (Transformer)
                                    ↓
                        52 ARKit Facial Blendshapes
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- GROQ API Key (for LLM)
- 4GB+ RAM
- (Optional) NVIDIA GPU for faster inference

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fastapi-main
```

2. Set up environment variables:
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

3. Build and run with Docker:
```bash
# Linux/Mac
./docker-run.sh

# Windows
docker-run.bat
```

4. Access the API:
```
http://localhost:7860
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Process Content
```bash
POST /process
Content-Type: multipart/form-data

{
  "url": "https://example.com/article"
}
```

### Query with TTS
```bash
POST /query
Content-Type: application/json

{
  "session_id": "uuid",
  "question": "What is this about?"
}
```

### Audio to Blendshapes
```bash
POST /audio_to_blendshapes
Content-Type: application/octet-stream

<audio_bytes>
```

### WebSocket Streaming
```javascript
ws://localhost:7860/ws/infer/kyutai

// Send
{
  "type": "start",
  "session_id": "uuid",
  "question": "Tell me more",
  "use_qwen": true
}

// Receive
{
  "type": "audio_chunk",
  "audio_base64": "...",
  "blendshapes": {...}
}
```

## Configuration

### Blendshape Model Config
Edit `utils/model/config.json`:
```json
{
  "input_dim": 80,
  "output_dim": 52,
  "hidden_dim": 512,
  "n_layers": 6,
  "num_heads": 8,
  "use_half_precision": false
}
```

### TTS Settings
- **Model**: Qwen3-TTS-12Hz-1.7B-CustomVoice
- **Size**: 1.7 billion parameters (~3.4GB)
- **Sample Rate**: 12kHz
- **Voice Options**: 9 predefined speakers
- **Languages**: 10+ supported
- **Quality**: Higher quality than 0.6B Base model

#### Available Voices
- **Female**: `serena` (default, expressive), `vivian` (friendly), `ono_anna` (warm), `sohee` (Asian accent)
- **Male**: `ryan` (standard), `eric` (professional), `dylan` (casual), `aiden` (young), `uncle_fu` (older)

#### Switching Voices
```python
# In API request
{
  "session_id": "...",
  "question": "...",
  "voice_preset": "ryan"  # Change to any speaker
}
```

## Project Structure

```
.
├── api.py                          # Main FastAPI application
├── streaming/
│   ├── qwen_tts_worker.py         # Qwen3-TTS integration
│   ├── blendshape_worker.py       # Blendshape generation
│   ├── kyutai_coordinator.py      # Streaming coordinator
│   └── optimized_blendshape_worker.py
├── utils/
│   └── model/
│       ├── model.py               # Blendshape model architecture
│       ├── model.pth              # Trained weights (download separately)
│       └── config.json            # Model configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── docker-compose.yml            # Docker Compose setup
```

## Model Downloads

The blendshape model is downloaded automatically on first run. If needed, download manually:

```bash
python download_model.py
```

## Performance

- **Startup Time**: ~10 seconds (model download on first run)
- **TTS Generation**: ~8 minutes (CPU) / ~1 minute (GPU) - 1.7B model
- **Blendshape Generation**: ~1 second per second of audio
- **Memory Usage**: ~4GB (CPU) / ~6GB (GPU)

## Improving Voice Expressiveness

The default voice uses synthetic reference audio. For more natural speech:

1. Record 3-5 seconds of expressive speech
2. Save as WAV (24kHz recommended)
3. Update the worker to use your recording

See `IMPROVING_VOICE_EXPRESSIVENESS.md` for details.

## Development

### Running Tests
```bash
pytest
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```

## Docker Commands

```bash
# Build image
docker build -t streaming-avatar-api .

# Run container
docker run -p 7860:7860 --env-file .env streaming-avatar-api

# View logs
docker logs streaming-avatar-api

# Restart
docker restart streaming-avatar-api
```

## Troubleshooting

### Model Not Loading
- Check `utils/model/model.pth` exists
- Verify file size (~600MB)
- Run `python download_model.py`

### Out of Memory
- Reduce batch size in config
- Use CPU instead of GPU
- Close other applications

### Slow Generation
- Use GPU if available
- Reduce audio length
- Enable half-precision (GPU only)

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:7860/docs`
- ReDoc: `http://localhost:7860/redoc`

## Technologies

- **FastAPI**: Web framework
- **Qwen3-TTS**: Text-to-speech with voice cloning
- **PyTorch**: Deep learning framework
- **Transformers**: Model architecture
- **LangChain**: RAG pipeline
- **Groq**: LLM inference
- **FAISS**: Vector similarity search
- **WebSockets**: Real-time streaming

## License

[Your License Here]

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues and questions:
- GitHub Issues: [Your repo issues]
- Documentation: See `/docs` folder

## Acknowledgments

- Qwen Team for Qwen3-TTS
- Groq for LLM API
- HuggingFace for model hosting
- Community contributors

---

**Note**: This project uses Qwen3-TTS for voice generation. Bark TTS has been completely removed.
