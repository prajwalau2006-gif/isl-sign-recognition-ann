"""Deterministic uniform temporal frame sampling."""
import numpy as np


def sample_frame_indices(total_frames: int, count: int = 16) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    return np.linspace(0, total_frames - 1, count).round().astype(int)
