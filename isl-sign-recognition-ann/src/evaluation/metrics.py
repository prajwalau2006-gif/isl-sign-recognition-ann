"""Metrics and one-shot held-out test evaluation."""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def evaluate_classifier(model, X_test, y_test, class_names, *, batch_size=64, device=None):
    """Evaluate a previously selected model on the held-out test set."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    predictions, probabilities = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().numpy())
    elapsed = time.perf_counter() - start
    predictions = np.asarray(predictions)
    probabilities = np.concatenate(probabilities, axis=0)
    labels = np.arange(len(class_names))
    return {
        "metrics": {
            "test_videos": len(y_test),
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "inference_seconds_total": elapsed,
            "inference_ms_per_video": elapsed / max(1, len(y_test)) * 1000,
        },
        "predictions": predictions,
        "probabilities": probabilities,
        "classification_report": classification_report(y_test, predictions, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
    }
