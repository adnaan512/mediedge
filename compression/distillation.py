"""
MediEdge — Knowledge Distillation.

Teacher : ResNet-18 (trained baseline)
Student : MobileNetV2 (efficient target)

Loss = alpha * KL(soft_teacher || soft_student) + (1-alpha) * CE(hard_labels)

Where soft labels are temperature-scaled softmax outputs.
Higher temperature → softer probability distributions → more information.
"""

import os
import time
import json
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics, measure_latency, get_model_size_mb


class DistillationLoss(nn.Module):
    """Combined knowledge distillation + hard-label cross-entropy loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Args:
            temperature : Softens teacher/student distributions. Higher = softer.
            alpha       : Weight on soft loss. (1-alpha) goes to hard CE loss.
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Soft loss (KL divergence with temperature scaling)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)

        # Hard loss (standard cross-entropy with true labels)
        hard_loss = self.ce(student_logits, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class DistillationTrainer:
    """Train a student model guided by a frozen teacher model."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        temperature: float = 4.0,
        alpha: float = 0.7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        early_stopping_patience: int = 7,
        checkpoint_dir: str = "./experiments",
        model_name: str = "student_distilled",
        device: str = "cpu",
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.teacher.eval()  # Teacher is always frozen

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.device = device

        self.criterion = DistillationLoss(temperature=temperature, alpha=alpha)
        self.optimizer = torch.optim.Adam(
            student.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_acc": [],
        }

    def _train_epoch(self) -> Dict[str, float]:
        self.student.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.train_loader, desc="  distill", leave=False):
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)
            loss = self.criterion(student_logits, teacher_logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return {"loss": total_loss / total, "acc": correct / total}

    def _val_epoch(self) -> float:
        self.student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                preds = self.student(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
        return correct / total

    def train(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  Knowledge Distillation: {self.model_name}")
        print(f"{'='*60}")

        for epoch in range(1, self.epochs + 1):
            train_stats = self._train_epoch()
            val_acc = self._val_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch:03d}/{self.epochs} | "
                f"loss={train_stats['loss']:.4f}  val_acc={val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_best.pth"
                )
                torch.save(self.student.state_dict(), ckpt_path)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        return {"best_val_acc": self.best_val_acc, "history": self.history}
