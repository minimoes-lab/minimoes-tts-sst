# AI Avatar RAG + TTS + Blendshapes API

Real-time AI avatar API that combines RAG (Retrieval-Augmented Generation), Qwen3-TTS, and facial blendshapes generation.

## Features

- **RAG Pipeline**: Process documents/websites and answer questions using Groq's Llama 4
- **Text-to-Speech**: Qwen3-TTS with 9 voice speakers
- **Facial Animation**: Generate 68 ARKit-compatible facial blendshapes from audio
- **Streaming Support**: Real-time WebSocket streaming

## Quick Start

### Using Docker (Recommended)

```bash
# Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start the API
docker-compose up --build

# API will be available at http://localhost:7860
```

### Test the API

```bash
python test_infer.py
```

This will:
1. Fetch content from Wikipedia about AI
2. Generate an answer using RAG
3. Convert the answer to speech using Qwen3-TTS
4. Generate facial blendshapes for avatar animation
5. Save `output_audio.wav` and `output_blendshapes.csv`

## API Endpoints

### `/infer` - Complete Pipeline
POST request with:
- `question`: Your question
- `url`: URL to scrape for context (or upload files)
- `return_audio`: Get audio as base64
- `return_csv`: Get blendshapes as CSV

Returns:
- `answer`: RAG-generated answer
- `audio_base64`: Speech audio (24kHz WAV)
- `blendshapes`: 68 facial animation frames at 60fps
- `csv`: Optional CSV export

### `/health` - Health Check
GET request to check if API is running.

## Available Voice Speakers

- aiden
- dylan
- eric
- ono_anna
- ryan
- serena
- sohee
- uncle_fu
- vivian

## Project Structure

```
├── api.py                  # Main FastAPI application
├── streaming/              # Streaming components
│   ├── qwen_tts_worker.py # Qwen3-TTS integration
│   ├── blendshape_worker.py
│   └── kyutai_coordinator.py
├── utils/                  # Utilities
│   ├── config.py          # Blendshape configuration
│   ├── generate_face_shapes.py
│   └── model/             # Blendshape neural network
├── test_infer.py          # Test script
├── docker-compose.yml     # Docker configuration
└── requirements.txt       # Python dependencies
```

## Performance

- **CPU**: ~13-15 minutes per TTS generation
- **GPU**: Significantly faster (recommended for production)

## Tech Stack

- FastAPI + Uvicorn
- PyTorch
- LangChain + Groq (Llama 4)
- Qwen3-TTS (1.7B parameters)
- FAISS vector embeddings
- Custom blendshape neural network

## License

Dual-license:
- MIT License for individuals and businesses earning under $1M/year
- Commercial license required for businesses with $1M+ annual revenue
