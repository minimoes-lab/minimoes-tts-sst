# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# extract_features.py
import io
import librosa
import numpy as np
import scipy.signal


def extract_audio_features(audio_input, sr=88200, from_bytes=False):
    try:
        if from_bytes:
            y, sr = load_audio_from_bytes(audio_input, sr)
        else:
            y, sr = load_and_preprocess_audio(audio_input, sr)
    except Exception as e:
            print(f"Loading as WAV failed: {e}\nFalling back to PCM loading.")
            y = load_pcm_audio_from_bytes(audio_input)  
    
    frame_length = int(0.01667 * sr)  # Frame length set to 0.01667 seconds (~60 fps)
    hop_length = frame_length // 2  # 2x overlap for smoother transitions
    min_frames = 9  # Minimum number of frames needed for delta calculation

    num_frames = (len(y) - frame_length) // hop_length + 1

    if num_frames < min_frames:
        print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
        return None, None

    combined_features = extract_and_combine_features(y, sr, frame_length, hop_length)
    
    return combined_features, y

def extract_zcr_features(y, sr, frame_length, hop_length):
    """
    Extract Zero Crossing Rate - measures signal noisiness vs periodicity.
    Adds 1 dimension to match model's expected 256 input features.
    """
    try:
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=frame_length, 
            hop_length=hop_length
        )
        return zcr.T  # Return shape: (frames, 1)
    except Exception as e:
        print(f"[extract_zcr_features] Error: {e}")
        return None


def extract_and_combine_features(y, sr, frame_length, hop_length, include_autocorr=True):
    """
    Extract comprehensive audio features for facial animation.
    Based on VOCASET/UniTalker research: MFCC + F0 + energy for dynamic expressions.
    """
    all_features = []
    
    # 1. MFCC features (standard for speech)
    mfcc_features = extract_mfcc_features(y, sr, frame_length, hop_length)
    all_features.append(mfcc_features)

    # 2. F0 (fundamental frequency) - adds prosody and emotional dynamics
    f0_features = extract_f0_features(y, sr, frame_length, hop_length)
    if f0_features is not None:
        all_features.append(f0_features)
    
    # 3. Energy features - for amplitude/intensity of movements
    energy_features = extract_energy_features(y, sr, frame_length, hop_length)
    if energy_features is not None:
        all_features.append(energy_features)

    # 4. Autocorrelation features - for pitch/periodicity
    if include_autocorr:
        autocorr_features = extract_autocorrelation_features(
            y, sr, frame_length, hop_length
        )
        all_features.append(autocorr_features)
    
    combined_features = np.hstack(all_features)
    
    # DEBUG: Detailed feature breakdown
    print(f"[Features] MFCC: {mfcc_features.shape[1]}, F0: {f0_features.shape[1] if f0_features is not None else 0}, "
          f"Energy: {energy_features.shape[1] if energy_features is not None else 0}, "
          f"Autocorr: {autocorr_features.shape[1] if include_autocorr else 0}, "
          f"TOTAL: {combined_features.shape[1]}")

    return combined_features


def extract_mfcc_features(y, sr, frame_length, hop_length, num_mfcc=23):
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)
    reduced_mfcc_features = reduce_features(mfcc_features)
    return reduced_mfcc_features.T

def extract_f0_features(y, sr, frame_length, hop_length):
    """
    Extract fundamental frequency (F0) for prosody and emotional dynamics.
    Based on VOCASET research: F0 correlates with facial expression intensity.
    """
    try:
        # Use librosa.pyin for robust F0 estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Replace NaN with 0 for unvoiced regions
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Get number of frames from MFCC to match dimensions
        mfcc_dummy = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1, 
                                           n_fft=frame_length, hop_length=hop_length)
        target_frames = mfcc_dummy.shape[1]
        
        # Ensure F0 has same number of frames
        if len(f0) < target_frames:
            # Pad if needed
            f0 = np.pad(f0, (0, target_frames - len(f0)), mode='edge')
        elif len(f0) > target_frames:
            # Trim if needed
            f0 = f0[:target_frames]
        
        # Add voiced flag as additional feature
        voiced_flag = voiced_flag.astype(float)
        if len(voiced_flag) < target_frames:
            voiced_flag = np.pad(voiced_flag, (0, target_frames - len(voiced_flag)), mode='edge')
        elif len(voiced_flag) > target_frames:
            voiced_flag = voiced_flag[:target_frames]
        
        # Stack F0 and voiced flag
        f0_features = np.vstack([f0.reshape(1, -1), voiced_flag.reshape(1, -1)])
        
        return f0_features.T  # Return shape: (frames, 2)
    except Exception as e:
        print(f"[extract_f0_features] Error: {e}")
        return None


def extract_energy_features(y, sr, frame_length, hop_length):
    """
    Extract RMS energy for amplitude/intensity of facial movements.
    High energy = more intense expressions.
    """
    try:
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        
        # Add delta for energy changes
        delta_rms = librosa.feature.delta(rms)
        
        # Stack original and delta
        energy_features = np.vstack([rms, delta_rms])
        
        return energy_features.T  # Return shape: (frames, 2)
    except Exception as e:
        print(f"[extract_energy_features] Error: {e}")
        return None
def cepstral_mean_variance_normalization(mfcc, mean_only=False, preserve_energy_coeff=True):
    """
    Cepstral Mean and Variance Normalization (CMVN).
    
    Args:
        mean_only: If True, only remove mean (preserve variance/energy)
        preserve_energy_coeff: If True, don't normalize first MFCC coefficient (energy)
    """
    mean = np.mean(mfcc, axis=1, keepdims=True)
    
    if preserve_energy_coeff:
        # Keep first coefficient (approximates energy) unnormalized
        mfcc_normalized = mfcc.copy()
        mfcc_normalized[1:, :] = (mfcc[1:, :] - mean[1:, :]) / (np.std(mfcc[1:, :], axis=1, keepdims=True) + 1e-10)
        return mfcc_normalized
    
    if mean_only:
        # Mean subtraction only - preserves variance structure
        return mfcc - mean
    
    # Full CMVN (original behavior)
    std = np.std(mfcc, axis=1, keepdims=True)
    return (mfcc - mean) / (std + 1e-10)


def extract_overlapping_mfcc(chunk, sr, num_mfcc, frame_length, hop_length, include_deltas=True, include_cepstral=True, threshold=1e-5):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc, n_fft=frame_length, hop_length=hop_length)
    if include_cepstral:
        mfcc = cepstral_mean_variance_normalization(mfcc)

    if include_deltas:
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])  # Stack original MFCCs with deltas
        return combined_mfcc
    else:
        return mfcc


def reduce_features(features):
    """
    VOCASET-style: Preserve full temporal resolution.
    Previous version averaged pairs of frames, causing loss of dynamic movements.
    Now returns features as-is to maintain high-resolution for facial animation.
    """
    # OPTIMIZATION: Removed frame averaging to preserve full temporal resolution
    # This maintains the 60 FPS output without reduction
    return features



def extract_overlapping_autocorr(y, sr, frame_length, hop_length, num_autocorr_coeff=183, pad_signal=True, padding_mode="reflect", trim_padded=False):
    if pad_signal:
        pad = frame_length // 2
        y_padded = np.pad(y, pad_width=pad, mode=padding_mode)
    else:
        y_padded = y

    frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
    if pad_signal and trim_padded:
        num_frames = frames.shape[1]
        start_indices = np.arange(num_frames) * hop_length
        valid_idx = np.where((start_indices >= pad) & (start_indices + frame_length <= len(y) + pad))[0]
        frames = frames[:, valid_idx]

    frames = frames - np.mean(frames, axis=0, keepdims=True)
    hann_window = np.hanning(frame_length)
    windowed_frames = frames * hann_window[:, np.newaxis]

    autocorr_list = []
    for frame in windowed_frames.T:
        full_corr = np.correlate(frame, frame, mode='full')
        mid = frame_length - 1  # Zero-lag index.
        # Extract `num_autocorr_coeff + 1` to include the first column initially
        wanted = full_corr[mid: mid + num_autocorr_coeff + 1]
        # Normalize by the zero-lag (energy) if nonzero.
        if wanted[0] != 0:
            wanted = wanted / wanted[0]
        autocorr_list.append(wanted)

    # Convert list to array and transpose so that shape is (num_autocorr_coeff + 1, num_valid_frames)
    autocorr_features = np.array(autocorr_list).T
    # Remove the first coefficient to avoid redundancy
    autocorr_features = autocorr_features[1:, :]

    autocorr_features = fix_edge_frames_autocorr(autocorr_features)
                                     
    return autocorr_features


def fix_edge_frames_autocorr(autocorr_features, zero_threshold=1e-7):
    """If the first or last frame is near all-zero, replicate from adjacent frames."""
    # Check first frame energy
    if np.all(np.abs(autocorr_features[:, 0]) < zero_threshold):
        autocorr_features[:, 0] = autocorr_features[:, 1]
    # Check last frame energy
    if np.all(np.abs(autocorr_features[:, -1]) < zero_threshold):
        autocorr_features[:, -1] = autocorr_features[:, -2]
    return autocorr_features

def extract_autocorrelation_features(
    y, sr, frame_length, hop_length, include_deltas=False
):
    """
    Extract autocorrelation features, optionally with deltas/delta-deltas,
    then align with the MFCC frame count, reduce, and handle first/last frames.
    """
    autocorr_features = extract_overlapping_autocorr(
        y, sr, frame_length, hop_length
    )
    
    if include_deltas:
        autocorr_features = compute_autocorr_with_deltas(autocorr_features)

    autocorr_features_reduced = reduce_features(autocorr_features)

    return autocorr_features_reduced.T


def compute_autocorr_with_deltas(autocorr_base):
    delta_ac = librosa.feature.delta(autocorr_base)
    delta2_ac = librosa.feature.delta(autocorr_base, order=2)
    combined_autocorr = np.vstack([autocorr_base, delta_ac, delta2_ac])
    return combined_autocorr

def load_and_preprocess_audio(audio_path, sr=88200):
    y, sr = load_audio(audio_path, sr)
    if sr != 88200:
        y = librosa.resample(y, orig_sr=sr, target_sr=88200)
        sr = 88200
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr

def load_audio(audio_path, sr=88200):
    y, sr = librosa.load(audio_path, sr=sr)
    print(f"Loaded audio file '{audio_path}' with sample rate {sr}")
    return y, sr

def load_audio_from_bytes(audio_bytes, sr=88200):
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=sr)
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr

def load_audio_file_from_memory(audio_bytes, sr=88200):
    """Load audio from memory bytes."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    print(f"Loaded audio data with sample rate {sr}")
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr




def load_pcm_audio_from_bytes(audio_bytes, sr=22050, channels=1, sample_width=2):
    """
    Load raw PCM bytes into a normalized numpy array and upsample to 88200 Hz.
    Uses librosa.resample for better quality than scipy.signal.resample.
    Assumes little-endian, 16-bit PCM data.
    """
    # Determine the appropriate numpy dtype.
    if sample_width == 2:
        dtype = np.int16
        max_val = 32768.0
    else:
        raise ValueError("Unsupported sample width")
    
    # Convert bytes to numpy array.
    data = np.frombuffer(audio_bytes, dtype=dtype)
    
    # If stereo or more channels, reshape accordingly.
    if channels > 1:
        data = data.reshape(-1, channels)
    
    # Normalize the data to range [-1, 1]
    y = data.astype(np.float32) / max_val
    
    # Upsample the audio from the current sample rate to 88200 Hz.
    target_sr = 88200
    if sr != target_sr:
        # Use librosa.resample for better quality (kaiser_best window)
        if channels > 1:
            y_resampled = np.zeros((int(len(y) * target_sr / sr), channels), dtype=np.float32)
            for ch in range(channels):
                y_resampled[:, ch] = librosa.resample(
                    y[:, ch], 
                    orig_sr=sr, 
                    target_sr=target_sr,
                    res_type='kaiser_best'  # High quality resampling
                )
            y = y_resampled
        else:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')
        sr = target_sr

    return y
