# Indian Sign Language Recognition — INCLUDE / MLOps

A modular research implementation for **isolated Indian Sign Language word recognition** using the INCLUDE dataset. The current benchmark represents each video as a temporal sequence of MediaPipe hand landmarks and hand-presence flags, then classifies the sequence with a BiLSTM + learned temporal attention model.

> **Scope:** this project classifies isolated sign clips. It is not a continuous sentence translator and does not claim to predict arbitrary ISL sentences.

## Current research baseline

The Master Evaluation notebook's selected model is configured as follows:

- Input: 128 values per sampled frame (126 landmark coordinates + 2 hand-presence flags)
- Temporal sampling: comparison of 16, 32 and 48 frames
- Model: 2-layer bidirectional LSTM, hidden size 128 per direction
- Attention: learned temporal attention (`Linear → Tanh → Linear → Softmax`)
- Classifier: 262 output classes
- Dropout: 0.3; optimizer: AdamW; loss: cross entropy
- Model selection: mean internal-validation macro-F1; early stopping uses validation loss

The reported final 48-frame test result from the existing experiment is **50.49% accuracy (412/816)**, macro-F1 **44.61%**, and weighted F1 **48.11%**. These are historical experiment results, not a guarantee that a fresh environment or reimplementation will reproduce them exactly. The working split is reconstructed/provisional; it must not be described as the official Zenodo split. Validation coverage is sparse (about 1–3 validation examples per class), and signer-independent evaluation has not been established.

## Repository design

```text
configs/                 YAML experiment settings
src/data/                manifest validation and split helpers
src/preprocessing/       deterministic frame sampling and sequence packing
src/features/            MediaPipe detection, landmark transform, extraction
src/models/              BiLSTM-Attention architecture
src/training/            training loop and safe checkpoint handling
src/evaluation/          held-out evaluation metrics
src/inference/           single-clip sequence prediction
notebooks/               reproducible Colab entry points (to be added)
tests/                   lightweight unit tests
 docs/                    methodology and experiment notes
```

## Cloud-first data policy

Keep the INCLUDE archives, extracted videos, feature arrays, model checkpoints and reports in Google Drive or another approved cloud storage location. Do **not** commit raw videos, feature arrays, checkpoints, MediaPipe task assets, credentials or private data to GitHub. The paths in `configs/dataset.yaml` are Colab examples; edit a local copy or pass paths at runtime rather than embedding personal credentials.

The project working manifest is expected at:

`/content/gdrive/MyDrive/ISL_Sign_Recognition/audit/full_include_working_manifest.csv`

Extracted videos are expected under:

`/content/gdrive/MyDrive/ISL_Sign_Recognition/dataset/extracted`

The working manifest contains 4,257 unique video paths and 262 labels. It is a reconstructed working benchmark, not an official split file.

## Install and test

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The unit tests exercise frame sampling, sequence shape and model output shape. MediaPipe extraction requires a compatible MediaPipe installation and the HandLandmarker `.task` asset. GPU is optional; CPU execution is supported but slower.

## Important safety before running expensive work

The modules expose extraction and training functions, but this repository intentionally has **no default command that starts extraction or training**. Build and test the repository first. Do not rerun extraction or final test evaluation merely to validate imports. Use explicit output paths and never point new checkpoint writes at existing historical checkpoints unless you intentionally want to replace them.

## Methodology notes

- Full-frame detection is attempted first; if no hands are detected, a central 65%-width crop is resized and retried.
- Crop landmarks are transformed back into full-frame normalized x coordinates; z is scaled by the resize factor.
- MediaPipe VIDEO timestamps must be strictly increasing. The notebook's caller spaces timestamps by multiplying the frame timestamp by two because crop fallback may make a second call at `timestamp + 1`.
- Hand slots are assigned from MediaPipe handedness. Unknown/ambiguous handedness is not silently guessed.
- The held-out test set is reserved for final reporting after validation-based model selection.

## Known limitations

1. The INCLUDE train/validation/test metadata was reconstructed because the official split link was unavailable; report this accurately.
2. Validation has only a small number of examples per class, so macro metrics are uncertain.
3. Two random seeds provide preliminary, not definitive, stability evidence.
4. Signer-independent generalization has not been verified.
5. The extraction function mirrors the notebook's timestamp and crop fallback logic; verify compatibility against the exact MediaPipe version before a fresh extraction run.
6. The repository currently provides a modular foundation. The complete experiment orchestration, final-model retraining script, webcam application and CI deployment workflow still require integration and validation.

## Citation and dataset

INCLUDE dataset: https://zenodo.org/records/4010759  
DOI: `10.1145/3394171.3413528`

Please follow the dataset's original license and citation requirements.
