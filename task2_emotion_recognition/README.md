# Task 2 — Emotion Recognition from Speech

Recognizes emotion (happy, sad, angry, fearful, disgust, surprised,
calm, neutral) from speech audio using MFCC features and a CNN
(with an LSTM alternative), on the RAVDESS dataset.

## Pipeline

1. **`features.py`** — loads each `.wav`, resamples to 22.05kHz, pads/
   truncates to 3s, and extracts:
   - `extract_mfcc_image()`: a 40×130 MFCC "image" for the CNN
   - `extract_mfcc_sequence()`: a 130-frame × 120-dim sequence (MFCC +
     delta + delta-delta) for the LSTM
2. **`dataset.py`** — walks a data directory for `.wav` files and parses
   the emotion label straight from the RAVDESS filename convention
   (`Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav`).
   Extracted features are cached to `.npz` so re-runs are instant.
3. **`model.py`** — a 3-block Conv2D CNN (BatchNorm + Dropout, 32→64→128
   filters) by default, or a 2-layer LSTM (`--arch lstm`).
4. **`train.py`** — stratified 80/20 split, class-balanced loss weights
   (RAVDESS has 2x as many neutral/calm samples), early stopping +
   LR reduction on plateau, then reports Precision/Recall/F1/Accuracy
   per class plus a confusion matrix.

## Setup

```bash
pip install -r requirements.txt
```

**Download RAVDESS** (~200MB, 24 actors, 1440 clips):
https://zenodo.org/record/1188976 — grab `Audio_Speech_Actors_01-24.zip`,
unzip into `data/RAVDESS/` (subfolders are fine, the loader recurses).

Alternatives supported by the same feature/model code (just point
`--data-dir` at them and adjust `parse_label()` in `dataset.py` to match
their filename/folder convention):
- **TESS** — https://tspace.library.utoronto.ca/handle/1807/24487
- **EMO-DB** — http://emodb.bilderbar.info/

## Run it

```bash
# CNN on MFCC "images" (default)
python train.py --data-dir data/RAVDESS --epochs 40

# LSTM on MFCC sequences, for comparison
python train.py --data-dir data/RAVDESS --arch lstm --epochs 40
```

Outputs land in `outputs/`: the trained model (`.keras`), a confusion
matrix, and training accuracy/loss curves.

## Notes for your submission video

- Show the confusion matrix — RAVDESS's biggest real-world confusion is
  usually **calm vs. neutral** and **fearful vs. sad**, which is worth
  calling out since it makes the demo look considered rather than just
  "it worked."
- Mention the class-weighting: RAVDESS has double the neutral/calm
  samples of the other 6 emotions, and without weighting, models tend
  to over-predict those two.
