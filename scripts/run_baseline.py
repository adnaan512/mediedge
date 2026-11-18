"""
MediEdge — Run baseline models (SmallCNN, ResNet-18).

Usage:
    python scripts/run_baseline.py --dataset dermamnist --model cnn
    python scripts/run_baseline.py --dataset dermamnist --model resnet18
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.dataloader import get_loaders, get_dataset_info
from models import get_model
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="MediEdge Baseline Training")
    p.add_argument("--dataset", default="dermamnist",
                   choices=["dermamnist", "pathmnist"])
    p.add_argument("--model",   default="resnet18",
                   choices=["cnn", "resnet18"])
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override epochs from config")
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Dataset
    info = get_dataset_info(args.dataset)
    num_classes = info["n_classes"]

    train_loader, val_loader, test_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=cfg["dataset"]["batch_size"],
        image_size=cfg["dataset"]["image_size"],
        data_dir=cfg["dataset"]["data_dir"],
        num_workers=cfg["dataset"]["num_workers"],
    )

    # Model
    model = get_model(args.model, num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] {args.model} | params={total_params:,}")

    # Epochs from config or CLI override
    model_cfg = cfg["baseline"].get(args.model, {})
    epochs = args.epochs or model_cfg.get("epochs", cfg["training"]["epochs"])
    lr = model_cfg.get("lr", cfg["training"]["lr"])

    # Train
    model_name = f"{args.model}_{args.dataset}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
        epochs=epochs,
        scheduler_type=cfg["training"]["scheduler"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        checkpoint_dir="./experiments",
        model_name=model_name,
        device=cfg["project"]["device"],
    )

    trainer.train()

    # Evaluate
    result = trainer.evaluate(test_loader)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    result_path = os.path.join(args.results_dir, f"{model_name}_results.json")
    # confusion_matrix is not JSON serializable, convert to list
    result["confusion_matrix"] = result["confusion_matrix"].tolist()
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] Results saved to {result_path}")


if __name__ == "__main__":
    main()
