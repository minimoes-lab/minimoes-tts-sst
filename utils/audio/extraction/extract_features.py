# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# extract_features.py
import io
import numpy as np
import scipy.signal
import soundfile as sf
from math import gcd


def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    g = gcd(int(target_sr), int(orig_sr))
    return scipy.signal.resample_poly(y, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)


def _mfcc(y, sr, n_mfcc=23, n_fft=1024, hop_length=512):
    """Compute MFCCs using scipy/numpy (no librosa/numba dependency)."""
    n_mels = 128
    fmin = 0.0
    fmax = sr / 2.0

    # STFT
    stft = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(
            np.pad(y, (n_fft // 2, n_fft // 2), mode='reflect'),
            n_fft
        )[::hop_length] * np.hanning(n_fft),
        axis=-1
    )) ** 2

    # Mel filterbank
    n_freqs = n_fft // 2 + 1
    mel_low = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_left = bins[m - 1]
        f_center = bins[m]
        f_right = bins[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    # Apply mel filterbank and log
    mel_spec = np.dot(stft, fbank.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT-II to get MFCCs
    from scipy.fft import dct
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc.T  # shape: (n_mfcc, n_frames)


def _delta(data, width=9, order=1):
    """Compute delta features (replaces librosa.feature.delta)."""
    result = data
    for _ in range(order):
        half = width // 2
        padded = np.pad(result, ((0, 0), (half, half)), mode='edge')
        denom = 2 * sum(n ** 2 for n in range(1, half + 1))
        d = np.zeros_like(result)
        for n in range(1, half + 1):
            d += n * (padded[:, half + n:half + n + result.shape[1]] - padded[:, half - n:half - n + result.shape[1]])
        result = d / denom
    return result


def _frame(y, frame_length, hop_length):
    """Frame a signal (replaces librosa.util.frame)."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(frame_length, n_frames),
        strides=(y.strides[0], y.strides[0] * hop_length)
    )
    return frames


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

def extract_and_combine_features(y, sr, frame_length, hop_length, include_autocorr=True):

    all_features = []
    mfcc_features = extract_mfcc_features(y, sr, frame_length, hop_length)
    all_features.append(mfcc_features)

    if include_autocorr:
        autocorr_features = extract_autocorrelation_features(
            y, sr, frame_length, hop_length
        )
        all_features.append(autocorr_features)

    combined_features = np.hstack(all_features)

    return combined_features


def extract_mfcc_features(y, sr, frame_length, hop_length, num_mfcc=23):
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)
    reduced_mfcc_features = reduce_features(mfcc_features)
    return reduced_mfcc_features.T

def cepstral_mean_variance_normalization(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    return (mfcc - mean) / (std + 1e-10)


def extract_overlapping_mfcc(chunk, sr, num_mfcc, frame_length, hop_length, include_deltas=True, include_cepstral=True, threshold=1e-5):
    mfcc = _mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc, n_fft=frame_length, hop_length=hop_length)
    if include_cepstral:
        mfcc = cepstral_mean_variance_normalization(mfcc)

    if include_deltas:
        delta_mfcc = _delta(mfcc, order=1)
        delta2_mfcc = _delta(mfcc, order=2)
        combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        return combined_mfcc
    else:
        return mfcc


def reduce_features(features):
    num_frames = features.shape[1]
    paired_frames = features[:, :num_frames // 2 * 2].reshape(features.shape[0], -1, 2)
    reduced_frames = paired_frames.mean(axis=2)

    if num_frames % 2 == 1:
        last_frame = features[:, -1].reshape(-1, 1)
        reduced_final_features = np.hstack((reduced_frames, last_frame))
    else:
        reduced_final_features = reduced_frames

    return reduced_final_features



def extract_overlapping_autocorr(y, sr, frame_length, hop_length, num_autocorr_coeff=187, pad_signal=True, padding_mode="reflect", trim_padded=False):
    if pad_signal:
        pad = frame_length // 2
        y_padded = np.pad(y, pad_width=pad, mode=padding_mode)
    else:
        y_padded = y

    frames = _frame(y_padded, frame_length=frame_length, hop_length=hop_length)
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
    delta_ac = _delta(autocorr_base, order=1)
    delta2_ac = _delta(autocorr_base, order=2)
    combined_autocorr = np.vstack([autocorr_base, delta_ac, delta2_ac])
    return combined_autocorr

def load_and_preprocess_audio(audio_path, sr=88200):
    y, orig_sr = load_audio(audio_path, sr)
    if orig_sr != 88200:
        y = _resample(y, orig_sr, 88200)
        sr = 88200
    else:
        sr = orig_sr

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr

def load_audio(audio_path, sr=88200):
    y, file_sr = sf.read(audio_path, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = _resample(y, file_sr, sr)
    print(f"Loaded audio file '{audio_path}' with sample rate {sr}")
    return y, sr

def load_audio_from_bytes(audio_bytes, sr=88200):
    audio_file = io.BytesIO(audio_bytes)
    y, file_sr = sf.read(audio_file, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = _resample(y, file_sr, sr)

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr


def load_pcm_audio_from_bytes(audio_bytes, sr=22050, channels=1, sample_width=2):
    """
    Load raw PCM bytes into a normalized numpy array and upsample to 88200 Hz.
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
        # Calculate the number of samples in the resampled signal.
        num_samples = int(len(y) * target_sr / sr)
        if channels > 1:
            # Resample each channel separately.
            y_resampled = np.zeros((num_samples, channels), dtype=np.float32)
            for ch in range(channels):
                y_resampled[:, ch] = scipy.signal.resample(y[:, ch], num_samples)
        else:
            y_resampled = scipy.signal.resample(y, num_samples)
        y = y_resampled
        sr = target_sr

    return y
