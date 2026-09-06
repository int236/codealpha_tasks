"""
train.py
--------
CodeAlpha Task 2: Emotion Recognition from Speech

Trains a CNN (default) or LSTM on MFCC features extracted from a
RAVDESS-style dataset, then reports precision/recall/F1/accuracy and
a confusion matrix.

Usage:
    python train.py --data-dir data/RAVDESS --epochs 40
    python train.py --data-dir data/RAVDESS --arch lstm --epochs 40

Download RAVDESS (Audio_Speech_Actors_01-24.zip, ~200MB) from:
    https://zenodo.org/record/1188976
Unzip it so .wav files live under data-dir (subfolders are fine — the
loader searches recursively).
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataset import build_dataset
from features import extract_mfcc_sequence
from model import build_cnn_model, build_lstm_model


def load_sequence_dataset(data_dir, cache_path):
    """Same as build_dataset but using the sequence (LSTM) feature extractor."""
    import glob
    from dataset import parse_label
    from tqdm import tqdm

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["labels"])

    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    X, y = [], []
    for path in tqdm(wav_files, desc="Extracting MFCC sequences"):
        try:
            label = parse_label(path)
        except (IndexError, KeyError):
            continue
        X.append(extract_mfcc_sequence(path))
        y.append(label)

    X = np.array(X)
    labels = sorted(set(y))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])

    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y_idx, labels=labels)
    return X, y_idx, labels


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition")
    parser.add_argument("--data-dir", default="data/RAVDESS")
    parser.add_argument("--arch", choices=["cnn", "lstm"], default="cnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--cache", default="features_cache.npz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building dataset ({args.arch} features) from {args.data_dir} ...")
    if args.arch == "cnn":
        X, y, labels = build_dataset(args.data_dir, cache_path=args.cache)
    else:
        X, y, labels = load_sequence_dataset(args.data_dir, cache_path=args.cache)

    print(f"Dataset: {X.shape[0]} samples, {len(labels)} classes: {labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    if args.arch == "cnn":
        model = build_cnn_model(X.shape[1:], len(labels))
    else:
        model = build_lstm_model(X.shape[1:], len(labels))

    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_history(history, os.path.join(args.out_dir, "training_history.png"))

    model.save(os.path.join(args.out_dir, f"emotion_{args.arch}_model.keras"))
    print(f"\nSaved model, confusion matrix, and training curves to {args.out_dir}/")


if __name__ == "__main__":
    main()
