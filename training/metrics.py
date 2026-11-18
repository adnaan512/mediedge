"""
MediEdge — Evaluation metrics.

Provides:
  - compute_metrics   : accuracy, AUC, F1, confusion matrix
  - measure_latency   : CPU inference time per image (ms)
  - get_model_size_mb : on-disk model size in MB
"""

import os
import time
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run model on data_loader and return:
      accuracy, auc (macro OvR), f1 (macro), confusion_matrix.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.squeeze().long()

            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred   = np.concatenate(all_preds)
    y_prob   = np.concatenate(all_probs, axis=0)
    y_true   = np.concatenate(all_labels)

    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(
            y_true, y_prob,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "f1_macro": float(f1),
        "confusion_matrix": cm,
    }


# ── Latency ───────────────────────────────────────────────────────────────────

def measure_latency(
    model: nn.Module,
    image_size: int = 28,
    warmup_runs: int = 10,
    eval_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure CPU inference latency (ms per single image).

    Returns:
        mean_ms, std_ms, min_ms, max_ms
    """
    model.eval()
    model.to(device)

    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(eval_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms":   float(np.mean(times)),
        "std_ms":    float(np.std(times)),
        "min_ms":    float(np.min(times)),
        "max_ms":    float(np.max(times)),
        "p95_ms":    float(np.percentile(times, 95)),
    }


# ── Model size ────────────────────────────────────────────────────────────────

def get_model_size_mb(model: nn.Module, save_path: Optional[str] = None) -> float:
    """
    Return the on-disk size of model weights in MB.
    Uses a temporary file if save_path is not provided.
    """
    if save_path is None:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            save_path = tmp.name

    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 ** 2)

    if save_path.startswith(tempfile.gettempdir()):
        os.remove(save_path)

    return round(size_mb, 3)
