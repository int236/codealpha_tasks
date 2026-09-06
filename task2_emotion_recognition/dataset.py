"""
dataset.py
----------
Builds a (features, labels) dataset from a folder of RAVDESS-style .wav
files.

RAVDESS filename convention (each file has 7 dash-separated numbers), e.g.:
    03-01-06-01-02-01-12.wav
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor

Emotion code -> label:
    01 neutral, 02 calm, 03 happy, 04 sad, 05 angry,
    06 fearful, 07 disgust, 08 surprised

If you're using TESS or EMO-DB instead, edit `parse_label()` below to
match that dataset's naming convention — everything else in the
pipeline (features, model, training) is dataset-agnostic.
"""

import glob
import os

import numpy as np
from tqdm import tqdm

from features import extract_mfcc_image

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_label(filename: str) -> str:
    """Extract the emotion label from a RAVDESS filename."""
    parts = os.path.basename(filename).split("-")
    code = parts[2]
    return RAVDESS_EMOTION_MAP[code]


def build_dataset(data_dir: str, cache_path: str = None):
    """Walk `data_dir` for .wav files, extract MFCC-image features, and
    return (X, y, label_names).

    Set cache_path to reuse a previously extracted .npz instead of
    re-processing audio (feature extraction is the slow part).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["labels"])

    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {data_dir}. Download RAVDESS from "
            "https://zenodo.org/record/1188976 and point --data-dir at it."
        )

    X, y = [], []
    for path in tqdm(wav_files, desc="Extracting MFCC features"):
        try:
            label = parse_label(path)
        except (IndexError, KeyError):
            continue  # skip files that don't match the naming convention
        feat = extract_mfcc_image(path)
        X.append(feat)
        y.append(label)

    X = np.array(X)
    labels = sorted(set(y))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[label] for label in y])

    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y_idx, labels=labels)
        print(f"Cached extracted features -> {cache_path}")

    return X, y_idx, labels
