# Repository audit and merged work

## Verification performed

- Ran `python -m pytest -q` against the supplied ZIP: **3 tests passed**.
- Reviewed source layout, configuration, dependency declarations, data governance, experiment report, migration status, and CI workflow.
- Added the recovered Master Evaluation notebook code-cell export under `notebooks/recovered/` as a provenance/reference artifact. It contains 88 recovered matching code cells; it is not represented as a clean executable notebook.
- Corrected the malformed `src/{__init__.py}` path to the conventional `src/__init__.py` package initializer.
- Removed generated Python and pytest cache files from the packaged repository.

## Existing implemented components

- Manifest validation for required columns, unique video paths, allowed splits, and expected row count.
- Deterministic uniform frame sampling.
- MediaPipe hand detection in VIDEO mode with central-crop fallback and coordinate restoration.
- Per-video landmark extraction and checkpoint helpers.
- Landmark/hand-flag sequence packing to 128 features per frame.
- BiLSTM with learned temporal attention.
- Validation-based training, early stopping, gradient clipping, and atomic checkpoint save/load helpers.
- Held-out evaluation metrics and single-sequence prediction helper.
- YAML configurations, basic tests, CI workflow, and cloud-first data governance documentation.

## Important limitations / follow-up

- Only three lightweight tests are present and pass. This is not end-to-end validation of extraction, training, checkpoint compatibility, or application inference.
- Dependencies in `requirements.txt` use minimum versions rather than a fully locked, tested environment. Pin exact versions after validating the target Colab/runtime.
- `src/{__init__.py}` in the input archive was a malformed filename; corrected here to `src/__init__.py`.
- The notebook export is a recovered code-cell reference. It may contain Drive-specific paths and notebook-only assumptions; inspect/adapt before running.
- A full application/webcam workflow, complete experiment orchestration, and reproducibility comparison against notebook outputs remain incomplete, as recorded in `docs/migration_status.md`.
- The reported benchmark is transcribed in `docs/experiment_results.md`; this audit did not rerun training or final test evaluation.
- The working split is reconstructed/provisional, not the official INCLUDE split. Signer-independent performance is not established.
- Dataset, feature arrays, checkpoints, and task assets remain external to Git. DVC remote configuration and access permissions must be set for collaborators before `dvc pull` can work.

## Suggested next checks before public push

1. Review the recovered notebook export for account-specific paths, tokens, or private information before publishing.
2. Create a compatible locked dependency file and test in the target environment.
3. Add integration tests with a small synthetic/approved fixture for feature extraction, checkpoint round-trip, and evaluation.
4. Validate existing checkpoints against the model code before application use.
5. Configure DVC with the actual shared Drive remote and grant team members access; never put credentials in Git.
