"""
MediEdge — Training loop.

Supports:
  - Standard cross-entropy training
  - Cosine / step LR scheduling
  - Early stopping
  - Checkpointing best model by val accuracy
"""

import os
import time
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, measure_latency, get_model_size_mb


class Trainer:
    """Generic trainer for MediEdge classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        scheduler_type: str = "cosine",
        early_stopping_patience: int = 7,
        checkpoint_dir: str = "./experiments",
        model_name: str = "model",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = None

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
        }

    # ── Train epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.train_loader, desc="  train", leave=False):
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return {"loss": total_loss / total, "acc": correct / total}

    # ── Val epoch ────────────────────────────────────────────────────────────

    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return {"loss": total_loss / total, "acc": correct / total}

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """Run full training loop. Returns best checkpoint metrics."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name}  |  device={self.device}")
        print(f"  Epochs={self.epochs}  |  patience={self.patience}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_stats = self._train_epoch()
            val_stats   = self._val_epoch()

            if self.scheduler:
                self.scheduler.step()

            # Record
            self.history["train_loss"].append(train_stats["loss"])
            self.history["train_acc"].append(train_stats["acc"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["val_acc"].append(val_stats["acc"])

            print(
                f"  Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_stats['loss']:.4f}  train_acc={train_stats['acc']:.4f} | "
                f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['acc']:.4f}"
            )

            # Checkpoint best
            if val_stats["acc"] > self.best_val_acc:
                self.best_val_acc = val_stats["acc"]
                self.epochs_no_improve = 0
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_best.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed/60:.1f} min")
        print(f"  Best val accuracy: {self.best_val_acc:.4f}")

        # Save training history
        history_path = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return {"best_val_acc": self.best_val_acc, "history": self.history}

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_loader: DataLoader,
        load_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Load best checkpoint (if requested) and run full evaluation.
        Returns accuracy, AUC, F1, confusion matrix, latency, size.
        """
        if load_best:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"{self.model_name}_best.pth"
            )
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device)
                )

        metrics = compute_metrics(
            self.model, test_loader, self.num_classes, self.device
        )
        latency = measure_latency(self.model, device=self.device)
        size_mb = get_model_size_mb(self.model)

        result = {
            "model": self.model_name,
            **metrics,
            "size_mb": size_mb,
            "latency_mean_ms": latency["mean_ms"],
            "latency_p95_ms":  latency["p95_ms"],
        }

        print(f"\n  ── Evaluation: {self.model_name} ──")
        print(f"     Accuracy : {result['accuracy']:.4f}")
        print(f"     AUC      : {result['auc']:.4f}")
        print(f"     F1 macro : {result['f1_macro']:.4f}")
        print(f"     Size     : {result['size_mb']} MB")
        print(f"     Latency  : {result['latency_mean_ms']:.2f} ms/img (mean)")

        return result
