"""Safe checkpoint writing and loading utilities."""
from pathlib import Path
import os
import torch


def atomic_torch_save(payload: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Checkpoints are trusted project artifacts. For untrusted files, do not
    # load arbitrary pickle-based PyTorch checkpoints.
    return torch.load(path, map_location=map_location, weights_only=False)
