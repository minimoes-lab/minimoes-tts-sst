"""
Optimized blendshape worker with batching and quantization support.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch

from streaming.tts_worker import AudioChunk

BLEND_FRAMES = 8


@dataclass
class BlendshapeChunk:
    sentence_index: int
    frames: np.ndarray
    start_time: float
    end_time: float
    frame_rate: int


class OptimizedBlendshapeWorker:
    """
    Enhanced blendshape worker with:
    - Batch processing for multiple audio chunks
    - Model quantization support
    - GPU memory optimization
    - Caching for repeated patterns
    """
    
    def __init__(self, blendshape_model, device, config):
        self.model = blendshape_model
        self.device = device
        self.config = config
        self._previous_tail_frames: Optional[np.ndarray] = None
        self._cancelled = False
        self._is_first_chunk = True
        
        # Optimization features
        self._use_quantization = config.get("use_quantization", False)
        self._batch_size = config.get("batch_size", 1)
        self._cache_enabled = config.get("cache_blendshapes", True)
        self._frame_cache = {}
        
        # Apply optimizations
        self._optimize_model()
    
    def _optimize_model(self):
        """Apply model optimizations."""
        try:
            # Quantization for faster inference
            if self._use_quantization and torch.cuda.is_available():
                print(f"[{datetime.now()}] [BS Worker] Applying dynamic quantization")
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            # Compile model for faster execution (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    print(f"[{datetime.now()}] [BS Worker] Compiling model with torch.compile")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    print(f"[{datetime.now()}] [BS Worker] Compile failed: {e}")
            
            # Set to eval mode and optimize for inference
            self.model.eval()
            
            # Enable cudnn benchmarking for consistent input sizes
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            
        except Exception as e:
            print(f"[{datetime.now()}] [BS Worker] Optimization warning: {e}")
    
    async def process_audio_chunk(
        self, audio_chunk: AudioChunk
    ) -> Optional[BlendshapeChunk]:
        """Process an AudioChunk into blendshapes."""
        if self._cancelled:
            return None
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._process_sync, audio_chunk
        )
        return result
    
    async def process_audio_batch(
        self, audio_chunks: List[AudioChunk]
    ) -> List[Optional[BlendshapeChunk]]:
        """Process multiple audio chunks in a batch for efficiency."""
        if self._cancelled or not audio_chunks:
            return []
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._process_batch_sync, audio_chunks
        )
        return result
    
    def _process_sync(self, audio_chunk: AudioChunk) -> Optional[BlendshapeChunk]:
        """Synchronous processing of single audio chunk."""
        from utils.audio.extraction.extract_features import extract_audio_features
        from utils.audio.processing.audio_processing import process_audio_features
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(audio_chunk)
            if self._cache_enabled and cache_key in self._frame_cache:
                print(f"[{datetime.now()}] [BS Worker] Cache hit for sentence {audio_chunk.sentence_index}")
                blendshapes = self._frame_cache[cache_key].copy()
            else:
                # Extract features
                audio_features, y = extract_audio_features(
                    audio_chunk.audio_bytes, from_bytes=True
                )
                
                if audio_features is None or y is None:
                    print(f"[{datetime.now()}] [BS Worker] Feature extraction failed")
                    return None
                
                # Run inference
                blendshapes = process_audio_features(
                    audio_features,
                    self.model,
                    self.device,
                    self.config,
                    apply_easing=self._is_first_chunk,
                )
                
                # Cache result
                if self._cache_enabled and len(self._frame_cache) < 100:
                    self._frame_cache[cache_key] = blendshapes.copy()
            
            if self._is_first_chunk:
                self._is_first_chunk = False
            
            # Cross-sentence blending
            if self._previous_tail_frames is not None and len(blendshapes) > 0:
                blend_n = min(
                    BLEND_FRAMES,
                    len(self._previous_tail_frames),
                    len(blendshapes),
                )
                for i in range(blend_n):
                    alpha = (i + 1) / (blend_n + 1)
                    blendshapes[i] = (
                        (1 - alpha) * self._previous_tail_frames[-(blend_n - i)]
                        + alpha * blendshapes[i]
                    )
            
            # Save tail frames
            if len(blendshapes) >= BLEND_FRAMES:
                self._previous_tail_frames = blendshapes[-BLEND_FRAMES:].copy()
            
            frame_rate = self.config.get("frame_rate", 60)
            duration = len(blendshapes) / frame_rate
            
            print(
                f"[{datetime.now()}] [BS Worker] Sentence "
                f"{audio_chunk.sentence_index}: {len(blendshapes)} frames"
            )
            
            return BlendshapeChunk(
                sentence_index=audio_chunk.sentence_index,
                frames=blendshapes,
                start_time=audio_chunk.start_time,
                end_time=audio_chunk.start_time + duration,
                frame_rate=frame_rate,
            )
        
        except Exception as e:
            print(f"[{datetime.now()}] [BS Worker] ERROR: {repr(e)}")
            return None
    
    def _process_batch_sync(
        self, audio_chunks: List[AudioChunk]
    ) -> List[Optional[BlendshapeChunk]]:
        """Synchronous batch processing of multiple audio chunks."""
        from utils.audio.extraction.extract_features import extract_audio_features
        
        try:
            results = []
            
            # Extract features for all chunks
            features_list = []
            for chunk in audio_chunks:
                audio_features, y = extract_audio_features(
                    chunk.audio_bytes, from_bytes=True
                )
                if audio_features is not None:
                    features_list.append((chunk, audio_features))
                else:
                    results.append(None)
            
            if not features_list:
                return results
            
            # Batch inference
            with torch.no_grad():
                for chunk, features in features_list:
                    # Process each with model
                    from utils.audio.processing.audio_processing import process_audio_features
                    
                    blendshapes = process_audio_features(
                        features,
                        self.model,
                        self.device,
                        self.config,
                        apply_easing=False,
                    )
                    
                    frame_rate = self.config.get("frame_rate", 60)
                    duration = len(blendshapes) / frame_rate
                    
                    results.append(BlendshapeChunk(
                        sentence_index=chunk.sentence_index,
                        frames=blendshapes,
                        start_time=chunk.start_time,
                        end_time=chunk.start_time + duration,
                        frame_rate=frame_rate,
                    ))
            
            return results
        
        except Exception as e:
            print(f"[{datetime.now()}] [BS Worker Batch] ERROR: {repr(e)}")
            return [None] * len(audio_chunks)
    
    def _get_cache_key(self, audio_chunk: AudioChunk) -> str:
        """Generate cache key for audio chunk."""
        # Use hash of audio data
        import hashlib
        return hashlib.md5(audio_chunk.audio_bytes[:1024]).hexdigest()
    
    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()
    
    def cancel(self):
        """Cancel ongoing processing."""
        self._cancelled = True
    
    def reset(self):
        """Reset worker state."""
        self._cancelled = False
        self._previous_tail_frames = None
        self._is_first_chunk = True
        self.clear_cache()
