"""Convert hand landmarks and presence flags into model sequences."""
import numpy as np


def make_sequence(features: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Convert [N,T,2,21,3] + [N,T,2] into float32 [N,T,128]."""
    features = np.asarray(features)
    flags = np.asarray(flags)
    if features.ndim != 5 or features.shape[2:] != (2, 21, 3):
        raise ValueError(f"Expected features [N,T,2,21,3], got {features.shape}")
    if flags.shape != features.shape[:2] + (2,):
        raise ValueError(f"Flags shape {flags.shape} does not match features {features.shape}")
    sequence = np.concatenate(
        [features.reshape(*features.shape[:2], 126), flags.reshape(*flags.shape[:2], 2)],
        axis=2,
    )
    result = sequence.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Sequence contains non-finite values")
    return result
