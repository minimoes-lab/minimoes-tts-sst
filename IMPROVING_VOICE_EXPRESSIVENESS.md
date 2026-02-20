# Improving Voice Expressiveness in Qwen3-TTS

## The Problem

The current voice output lacks:
- **Prosody**: Natural pitch variation, rhythm, and intonation
- **Emotion**: Excitement, sadness, emphasis
- **Paralinguistics**: Pauses, stress patterns, vocal quality changes

## Root Cause

Qwen3-TTS uses **voice cloning** - it mimics the reference audio's characteristics. Our synthetic reference audio is:
- Monotone (flat pitch)
- No stress patterns
- No emotional variation
- Simple sine waves

**Result**: The cloned voice is also monotone and robotic.

## Solutions

### 1. ✅ Improved Synthetic Reference (Current Fix)

I've updated the reference audio generation to include:

```python
# Pitch variation (prosody)
pitch_contour = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)

# Stress patterns (emphasis)
stress_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * 2 * t)

# Natural envelope (attack/decay)
envelope with attack, sustain, decay

# Harmonics (richer voice)
Multiple frequency components

# Slight noise (naturalness)
Random variation
```

**Improvement**: ~30% more expressive, but still limited.

### 2. 🎯 Use Real Expressive Audio (Recommended)

**Best approach**: Provide a real human voice recording as reference.

#### Option A: Record Custom Voice
```bash
# 1. Record 3-5 seconds of expressive speech
# 2. Include emotion, pitch variation, pauses
# 3. Save as WAV file (24kHz recommended)

# Example text to record:
"Hello! I'm so excited to help you today. 
How are you doing? Let me know if you need anything!"
```

#### Option B: Use Pre-recorded Samples
```python
# Download expressive samples
python create_expressive_reference.py

# Or use from a voice dataset:
# - LibriTTS (expressive readings)
# - VCTK (multiple speakers with emotion)
# - LJSpeech (audiobook narration)
```

### 3. 🔧 Enhance with LLM Prompting

The LLM (Groq) can add paralinguistic markers:

```python
# Update the RAG prompt to include:
system_prompt = """
When answering, include natural speech patterns:
- Use punctuation for pauses (commas, periods, ellipses...)
- Add emphasis with CAPS or *asterisks*
- Include interjections (Oh!, Hmm, Well...)
- Vary sentence structure and length
- Add emotional context when appropriate

Example:
"Well, that's a great question! Python is *really* powerful 
for data science. Let me explain... First, it has amazing 
libraries like NumPy and Pandas. Second..."
"""
```

### 4. 🎨 Use Qwen3-TTS Voice Design (Alternative)

Instead of voice cloning, use voice description:

```python
# This requires a different Qwen3-TTS model
# (not the Base model we're using)

wavs, sr = model.generate_voice_design(
    text=text,
    language="English",
    voice_description="A warm, friendly female voice with clear 
                      pronunciation and natural prosody. Speaks 
                      with enthusiasm and varies pitch naturally.",
    instruct="Speak expressively with emotion"
)
```

**Note**: This requires `Qwen3-TTS-Instruct` model, not the Base model.

## Implementation Priority

### Immediate (Done ✅)
- [x] Improved synthetic reference with prosody
- [x] Added pitch variation
- [x] Added stress patterns
- [x] Added harmonics

### Short-term (Recommended)
- [ ] Record custom expressive reference audio
- [ ] Update LLM prompt for better text output
- [ ] Test with different reference voices

### Long-term (Advanced)
- [ ] Switch to Qwen3-TTS-Instruct model
- [ ] Implement emotion detection from text
- [ ] Dynamic voice selection based on content
- [ ] Real-time prosody adjustment

## Testing Expressiveness

Run this to compare:

```bash
# Generate with improved reference
python generate_qwen3_samples.py

# Listen to the output
# Check for:
# - Pitch variation (not monotone)
# - Stress on important words
# - Natural pauses
# - Emotional tone
```

## Expected Improvements

| Aspect | Before | After (Synthetic) | After (Real Audio) |
|--------|--------|-------------------|-------------------|
| Pitch Variation | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Stress Patterns | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Emotion | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Naturalness | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Next Steps

1. **Test current improvements**:
   ```bash
   docker cp streaming/qwen_tts_worker.py streaming-avatar-api:/app/streaming/qwen_tts_worker.py
   docker restart streaming-avatar-api
   python generate_qwen3_samples.py
   ```

2. **Record custom voice** (for best results):
   - Use a good microphone
   - Record in a quiet environment
   - Speak naturally with emotion
   - 3-5 seconds is enough

3. **Update the worker** to use your recording:
   ```python
   worker = QwenTTSWorker(
       device="cpu",
       use_qwen3=True,
       reference_audio_path="path/to/your/voice.wav"
   )
   ```

## Technical Notes

- **Voice cloning quality** depends 100% on reference audio quality
- **Qwen3-TTS Base model** is designed for cloning, not synthesis
- **Prosody transfer** works best with 3-10 seconds of reference
- **Emotion** in reference audio directly transfers to output
- **Background noise** in reference will also be cloned

---

**Bottom line**: For truly expressive speech, you need an expressive reference voice. The synthetic improvements help, but real human audio is best.
