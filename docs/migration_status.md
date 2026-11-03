# Migration status

## Implemented in this repository

- Manifest schema and split validation helpers
- Uniform temporal frame sampling
- Landmark conversion and crop-coordinate restoration
- MediaPipe VIDEO-mode full-frame detection with central crop fallback
- Resumable per-video compressed checkpoint writer and validator
- Feature/flag sequence packing to 128-dimensional frame vectors
- BiLSTM-Attention model architecture
- Validation-driven training function with early stopping and gradient clipping
- Atomic PyTorch checkpoint saving/loading
- Held-out evaluation metrics and single-sequence prediction helper
- Basic unit tests and cloud-first repository documentation

## Still required before claiming full reproducibility

- Compare the extracted modular code against the original notebook outputs on a controlled small fixture.
- Validate exact MediaPipe task asset/version and timestamp behavior.
- Add batch-feature loader with strict index and content validation.
- Add the notebook's stratified development split routine and experiment aggregation logic.
- Add final-fit orchestration that consumes the selected validation configuration and median best epoch.
- Add report export, confusion/error analysis, and full prediction CSV integration.
- Add a Colab entry notebook and verify all paths against the user's Drive layout.
- Add CI with dependency-compatible CPU tests.
- Validate checkpoint compatibility against the actual existing `.pth` file before loading it in the application.

No extraction or training has been run as part of this migration.
