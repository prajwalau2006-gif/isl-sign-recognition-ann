"""Single-sequence prediction helpers."""
import numpy as np
import torch


def predict_sequence(model, sequence: np.ndarray, class_names, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 128:
        raise ValueError(f"Expected one [T,128] sequence, got {array.shape}")
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(array).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())
    return {"class_index": index, "label": class_names[index], "confidence": float(probabilities[index].item())}
