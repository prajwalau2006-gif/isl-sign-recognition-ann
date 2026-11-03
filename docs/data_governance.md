# Data and artifact governance

- Keep original INCLUDE archives and extracted videos in cloud storage.
- Keep feature batches, per-video extraction checkpoints, trained model checkpoints and generated reports outside Git.
- Commit only source code, small configuration files, tests and documentation.
- Never commit API keys, access tokens, `.env` files or credentials.
- Do not overwrite existing project checkpoints during repository migration.
- Treat the reconstructed working manifest as a project artifact with documented provenance; it is not the official dataset split.
