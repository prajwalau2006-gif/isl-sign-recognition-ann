"""Validation-driven training loop matching the master notebook settings."""
from copy import deepcopy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from .checkpoint import atomic_torch_save


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_run(model, X, y, train_indices, validation_indices, *,
                  checkpoint_path, class_names, frame_count, seed,
                  batch_size=64, epochs=40, patience=7,
                  learning_rate=1e-3, weight_decay=1e-4,
                  num_workers=0, device=None):
    """Train one seed/configuration; early-stop on validation loss.

    This function deliberately receives development indices explicitly and
    never accesses test data.
    """
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if set(train_indices.tolist()) & set(validation_indices.tolist()):
        raise ValueError("Training and validation indices overlap")
    train_ds = TensorDataset(torch.from_numpy(X[train_indices]), torch.from_numpy(y[train_indices]))
    val_ds = TensorDataset(torch.from_numpy(X[validation_indices]), torch.from_numpy(y[validation_indices]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += len(yb)
        model.eval()
        val_loss_sum = val_correct = val_total = 0
        val_pred, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)
                val_pred.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(yb.cpu().numpy())
        val_loss = val_loss_sum / max(1, val_total)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, train_total),
            "train_accuracy": train_correct / max(1, train_total),
            "val_loss": val_loss,
            "val_accuracy": val_correct / max(1, val_total),
            "val_macro_f1": f1_score(val_true, val_pred, average="macro", zero_division=0),
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch, wait = epoch + 1, 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    payload = {
        "model_state_dict": model.state_dict(),
        "frame_count": int(frame_count), "input_dim": 128,
        "hidden_size": 128, "num_layers": 2, "bidirectional": True,
        "num_classes": len(class_names), "class_names": list(class_names),
        "seed": int(seed), "best_epoch": int(best_epoch),
    }
    atomic_torch_save(payload, checkpoint_path)
    return {"history": history, "best_epoch": best_epoch,
            "best_val_loss": best_loss, "checkpoint": str(checkpoint_path),
            "model": model}
