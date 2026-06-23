"""
QwenTTSModelMixin — model loading, warmup, voice clone prompt, lifecycle.
Imported by QwenTTSWorker in qwen_tts_worker.py.
"""
import concurrent.futures
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch

# Use TensorFloat32 cores for float32 matrix multiplications — free ~15% speedup on Ampere+
torch.set_float32_matmul_precision('high')

# Patch librosa.resample to use scipy instead of numba (numba's @guvectorize is broken
# with the installed numpy version). This prevents the 'get_call_template' crash when
# qwen_tts internally calls librosa.resample for audio normalization.
def _patched_librosa_resample(y=None, *, orig_sr=None, target_sr=None, **kwargs):
    if orig_sr == target_sr:
        return y
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(target_sr), int(orig_sr))
    return resample_poly(y, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)

try:
    import librosa
    librosa.resample = _patched_librosa_resample
    if hasattr(librosa, 'core'):
        librosa.core.resample = _patched_librosa_resample
except ImportError:
    pass


@dataclass
class AudioChunk:
    sentence_index: int
    audio_bytes: bytes
    audio_np: np.ndarray
    sample_rate: int
    start_time: float
    duration: float


class QwenTTSModelMixin:
    """
    Manages the Qwen3-TTS model lifecycle:
      - Shared class-level singleton (model, executor) so all instances share one GPU load
      - __init__: sets instance state and triggers _load_model()
      - _load_model: loads / reuses the shared model, runs warmup on first load
      - create_voice_clone_prompt: wraps model.create_voice_clone_prompt
      - cancel / reset: session-level lifecycle flags
    """

    _shared_lock = threading.Lock()
    _shared_model = None
    _shared_speakers = None
    _shared_default_speaker = None
    _shared_loaded_device = None
    _shared_model_loaded = False
    # Single persistent thread for all GPU inference.
    # torch.compile(mode='reduce-overhead') stores CUDA graph state in thread-local
    # storage (TLS). Creating a new ThreadPoolExecutor per request spawns a new thread
    # that has no TLS state → AssertionError on the 2nd request.  Reusing the same
    # executor guarantees all inference (warmup + runtime) runs on the same thread.
    _shared_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def __init__(
        self,
        device="cuda",
        use_qwen3=True,
        reference_audio_path=None,
        reference_text: Optional[str] = None,
        raise_on_error: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.sr = 24000
        self._cancelled = False
        self.use_qwen3 = use_qwen3
        self.model_loaded = False
        self.speakers = []
        self.default_speaker = None
        self.reference_audio_path = reference_audio_path
        self.reference_text: Optional[str] = reference_text
        self.raise_on_error = raise_on_error
        self.voice_clone_prompt = None  # set by _load_model() if reference audio provided
        self._load_model()

    def _load_model(self):
        """Load Qwen3-TTS model using qwen_tts library with 6x optimizations."""
        if not self.use_qwen3:
            print(f"[{datetime.now()}] [Qwen TTS] Qwen3 disabled, using fallback")
            return

        try:
            reuse_shared = False
            with self.__class__._shared_lock:
                if (
                    self.__class__._shared_model_loaded
                    and self.__class__._shared_model is not None
                    and self.__class__._shared_loaded_device == self.device
                ):
                    self.model = self.__class__._shared_model
                    self.speakers = self.__class__._shared_speakers or []
                    self.default_speaker = (
                        self.__class__._shared_default_speaker
                        if self.__class__._shared_default_speaker
                        else (self.speakers[0] if self.speakers else "aiden")
                    )
                    self.model_loaded = True
                    self.sr = 24000
                    reuse_shared = True

            if not reuse_shared:
                print(f"[{datetime.now()}] [Qwen TTS] Loading Qwen3-TTS model...")

            from qwen_tts import Qwen3TTSModel

            # Base model supports streaming inference (voice clone)
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

            if not reuse_shared:
                self.speakers = []
                self.default_speaker = None
                torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                try:
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=self.device,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(f"[{datetime.now()}] [Qwen TTS] Warning: flash_attention_2 not available: {e}")
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=self.device,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )

                # Patch the processor CLASS __call__ to bypass _merge_kwargs which
                # calls get_call_template — removed in newer transformers versions.
                # Instance.__call__ assignment doesn't work in Python (type().__call__ wins).
                _proc_cls = type(self.model.processor)
                if not getattr(_proc_cls, '_patched_for_compat', False):
                    def _patched_processor_call(self_proc, text=None, **kwargs):
                        from transformers.feature_extraction_utils import BatchFeature
                        if text is None:
                            raise ValueError("text is required")
                        if not isinstance(text, list):
                            text = [text]
                        result = self_proc.tokenizer(text, return_tensors=kwargs.get("return_tensors"), padding=kwargs.get("padding", False))
                        return BatchFeature(data=dict(result), tensor_type=kwargs.get("return_tensors"))
                    _proc_cls.__call__ = _patched_processor_call
                    _proc_cls._patched_for_compat = True
                print(f"[{datetime.now()}] [Qwen TTS] Processor patched for transformers compatibility")

            # OPTIMIZATION 6x: Enable streaming optimizations (torch.compile + CUDA graphs)
            if (not reuse_shared) and self.device == "cuda":
                print(f"[{datetime.now()}] [Qwen TTS] Enabling 6x streaming optimizations...")
                try:
                    if hasattr(self.model, "enable_streaming_optimizations"):
                        self.model.enable_streaming_optimizations(
                            decode_window_frames=80,
                            use_compile=True,
                            use_cuda_graphs=True,
                            compile_mode="reduce-overhead",
                            use_fast_codebook=False,
                            compile_codebook_predictor=True,
                        )
                        print(f"[{datetime.now()}] [Qwen TTS] Two-phase streaming optimizations enabled")
                    else:
                        print(f"[{datetime.now()}] [Qwen TTS] Warning: enable_streaming_optimizations not available")
                except Exception as opt_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warning: Could not enable optimizations: {opt_err}")

            # Build voice clone prompt if provided
            if self.reference_audio_path and self.reference_text:
                import soundfile as sf
                from scipy.signal import resample_poly
                from math import gcd
                _ref_audio, _ref_sr = sf.read(self.reference_audio_path, dtype="float32", always_2d=False)
                if _ref_audio.ndim > 1:
                    _ref_audio = _ref_audio.mean(axis=1)
                _target_sr = 24000
                if _ref_sr != _target_sr:
                    _g = gcd(_target_sr, _ref_sr)
                    _ref_audio = resample_poly(_ref_audio, _target_sr // _g, _ref_sr // _g).astype(np.float32)
                    _ref_sr = _target_sr
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=(_ref_audio, _ref_sr),
                    ref_text=self.reference_text,
                )

            # Create the shared persistent executor now (before warmup) so that
            # warmup JIT-compiles CUDA graphs on the SAME thread that all future
            # inference calls will use.  This must happen exactly once.
            with self.__class__._shared_lock:
                if self.__class__._shared_executor is None:
                    self.__class__._shared_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="qwen_tts_gpu",
                    )
                    self.__class__._shared_executor.submit(lambda: None).result()

            if (not reuse_shared) and self.device == "cuda":
                try:
                    warmup_prompt = self.voice_clone_prompt

                    # Synthetic 1s silence warmup so CUDA graphs are recorded at startup
                    if warmup_prompt is None and hasattr(self.model, "create_voice_clone_prompt"):
                        try:
                            _sr_w = 24000
                            _silence = np.zeros(_sr_w, dtype=np.float32)
                            warmup_prompt = self.model.create_voice_clone_prompt(
                                ref_audio=(_silence, _sr_w),
                                ref_text="Hello, this is a warmup.",
                            )
                            print(f"[{datetime.now()}] [Qwen TTS] Synthetic warmup prompt created")
                        except Exception as _we:
                            print(f"[{datetime.now()}] [Qwen TTS] Synthetic warmup prompt failed: {_we}")

                    if warmup_prompt is not None:
                        def _warmup():
                            for wtext in [
                                "Hello.",
                                "This is a warmup.",
                                "Hey! How can I help you today? Do you have a question?",
                                "I'd be happy to help you with that. Let me explain how this works in detail.",
                            ]:
                                for _chunk, _sr in self.model.stream_generate_voice_clone(
                                    text=wtext,
                                    language="English",
                                    voice_clone_prompt=warmup_prompt,
                                    emit_every_frames=8,
                                    decode_window_frames=80,
                                    overlap_samples=512,
                                    first_chunk_emit_every=0,
                                    first_chunk_decode_window=48,
                                    first_chunk_frames=48,
                                ):
                                    pass  # consume fully to ensure complete graph capture

                        self.__class__._shared_executor.submit(_warmup).result()
                        print(f"[{datetime.now()}] [Qwen TTS] Warmup complete (CUDA graphs recorded)")
                    else:
                        print(f"[{datetime.now()}] [Qwen TTS] Warmup skipped (no prompt available)")
                except Exception as warmup_err:
                    print(f"[{datetime.now()}] [Qwen TTS] Warmup error: {warmup_err}")

            self.sr = 24000
            self.model_loaded = True

            with self.__class__._shared_lock:
                self.__class__._shared_model = self.model
                self.__class__._shared_speakers = self.speakers
                self.__class__._shared_default_speaker = self.default_speaker
                self.__class__._shared_loaded_device = self.device
                self.__class__._shared_model_loaded = True

            print(f"[{datetime.now()}] [Qwen TTS] Qwen3-TTS loaded successfully on {self.device}")
            print(f"[{datetime.now()}] [Qwen TTS] Available speakers: {self.speakers}")

        except Exception as e:
            print(f"[{datetime.now()}] [Qwen TTS] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            self.sr = 24000
            if self.raise_on_error:
                raise RuntimeError(f"Qwen3-TTS model failed to load: {e}") from e
            print(f"[{datetime.now()}] [Qwen TTS] Using fallback synthesis")

    def create_voice_clone_prompt(self, ref_audio_path: str, ref_text: str):
        if self.model is None or not self.model_loaded:
            raise RuntimeError("Model not loaded")
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        audio_np, sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        target_sr = 24000
        if sr != target_sr:
            g = gcd(target_sr, sr)
            audio_np = resample_poly(audio_np, target_sr // g, sr // g).astype(np.float32)
            sr = target_sr
        return self.model.create_voice_clone_prompt(ref_audio=(audio_np, sr), ref_text=ref_text)

    def cancel(self):
        """Cancel ongoing generation."""
        self._cancelled = True

    def reset(self):
        """Reset worker state."""
        self._cancelled = False
