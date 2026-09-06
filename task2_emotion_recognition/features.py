"""
features.py
------------
Audio feature extraction for speech emotion recognition.

Extracts MFCCs (plus deltas, chroma, and mel-spectrogram) from a .wav
file and returns a fixed-length feature vector/matrix suitable for a
CNN or LSTM.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 3.0          # seconds — clips are padded/truncated to this length
N_MFCC = 40
MAX_FRAMES = 130         # ~3s at default hop_length=512, sr=22050


def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Load audio, resample, and pad/truncate to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc_sequence(path: str) -> np.ndarray:
    """Return a (MAX_FRAMES, N_MFCC) MFCC sequence — good input for an LSTM/CRNN.

    Also augments each frame with delta and delta-delta coefficients,
    which are standard for speech emotion recognition.
    """
    y = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([mfcc, delta, delta2])  # (3*N_MFCC, T)
    feat = feat.T  # (T, 3*N_MFCC) — time-major for LSTM

    # Pad/truncate the time dimension to a fixed length
    if feat.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - feat.shape[0], feat.shape[1]))
        feat = np.vstack([feat, pad])
    else:
        feat = feat[:MAX_FRAMES, :]

    return feat.astype(np.float32)


def extract_mfcc_image(path: str) -> np.ndarray:
    """Return a (N_MFCC, MAX_FRAMES, 1) MFCC 'image' — good input for a 2D CNN."""
    y = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_FRAMES:
        pad = np.zeros((N_MFCC, MAX_FRAMES - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)  # normalize
    return mfcc[..., np.newaxis].astype(np.float32)
