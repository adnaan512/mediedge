"""
MediEdge — Run compression experiments.

Loads a trained MobileNetV2 checkpoint and applies:
  - int8   : INT8 post-training static quantization
  - prune  : Magnitude pruning (progressive 20→40→60%)
  - distill: Knowledge distillation (teacher=ResNet-18)
  - combined: prune + distill + int8

Usage:
    python scripts/run_compression.py --model mobilenetv2 --method int8
    python scripts/run_compression.py --model mobilenetv2 --method prune
    python scripts/run_compression.py --model mobilenetv2 --method distill
    python scripts/run_compression.py --model mobilenetv2 --method combined
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
from training.metrics import compute_metrics, measure_latency, get_model_size_mb
from compression.quantization import quantize_model_int8
from compression.pruning import prune_model, iterative_prune
from compression.distillation import DistillationTrainer


def parse_args():
    p = argparse.ArgumentParser(description="MediEdge Compression")
    p.add_argument("--dataset", default="dermamnist",
                   choices=["dermamnist", "pathmnist"])
    p.add_argument("--model",   default="mobilenetv2",
                   choices=["mobilenetv2", "efficientnet_lite"])
    p.add_argument("--method",  default="int8",
                   choices=["int8", "prune", "distill", "combined"])
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def load_checkpoint(model, model_name, dataset, device):
    ckpt_path = os.path.join("experiments", f"{model_name}_{dataset}_best.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[checkpoint] Loaded: {ckpt_path}")
    else:
        print(f"[WARNING] No checkpoint found at {ckpt_path}. Using random weights.")
    return model


def evaluate_and_save(model, test_loader, num_classes, model_name, results_dir, device):
    metrics = compute_metrics(model, test_loader, num_classes, device)
    latency = measure_latency(model, device=device)
    size_mb = get_model_size_mb(model)

    result = {
        "model": model_name,
        **metrics,
        "size_mb": size_mb,
        "latency_mean_ms": latency["mean_ms"],
        "latency_p95_ms":  latency["p95_ms"],
    }

    print(f"\n  ── Results: {model_name} ──")
    print(f"     Accuracy : {result['accuracy']:.4f}")
    print(f"     AUC      : {result['auc']:.4f}")
    print(f"     F1 macro : {result['f1_macro']:.4f}")
    print(f"     Size     : {result['size_mb']} MB")
    print(f"     Latency  : {result['latency_mean_ms']:.2f} ms/img (mean)")

    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{model_name}_results.json")
    result["confusion_matrix"] = result["confusion_matrix"].tolist()
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] Results saved to {result_path}")
    return result


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["project"]["device"]
    dataset = args.dataset
    base_model_name = args.model

    info = get_dataset_info(dataset)
    num_classes = info["n_classes"]

    train_loader, val_loader, test_loader = get_loaders(
        dataset_name=dataset,
        batch_size=cfg["dataset"]["batch_size"],
        image_size=cfg["dataset"]["image_size"],
        data_dir=cfg["dataset"]["data_dir"],
        num_workers=cfg["dataset"]["num_workers"],
    )

    # ── INT8 Quantization ─────────────────────────────────────────────────────
    if args.method == "int8":
        model = get_model(base_model_name, num_classes=num_classes)
        model = load_checkpoint(model, base_model_name, dataset, device)

        cal_batches = cfg["compression"]["quantization"]["calibration_batches"]
        q_model = quantize_model_int8(model, train_loader, cal_batches, device="cpu")

        result_name = f"{base_model_name}_{dataset}_int8"
        evaluate_and_save(q_model, test_loader, num_classes, result_name, args.results_dir, "cpu")

    # ── Magnitude Pruning ─────────────────────────────────────────────────────
    elif args.method == "prune":
        model = get_model(base_model_name, num_classes=num_classes)
        model = load_checkpoint(model, base_model_name, dataset, device)

        sparsity_schedule = cfg["compression"]["pruning"]["sparsity_schedule"]
        pruned_model = iterative_prune(model, sparsity_schedule)

        result_name = f"{base_model_name}_{dataset}_pruned"
        evaluate_and_save(pruned_model, test_loader, num_classes, result_name, args.results_dir, device)

    # ── Knowledge Distillation ────────────────────────────────────────────────
    elif args.method == "distill":
        dcfg = cfg["compression"]["distillation"]

        teacher = get_model("resnet18", num_classes=num_classes)
        teacher = load_checkpoint(teacher, "resnet18", dataset, device)

        student = get_model(base_model_name, num_classes=num_classes)

        distill_trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            temperature=dcfg["temperature"],
            alpha=dcfg["alpha"],
            lr=dcfg["lr"],
            epochs=dcfg["epochs"],
            checkpoint_dir="./experiments",
            model_name=f"{base_model_name}_{dataset}_distilled",
            device=device,
        )
        distill_trainer.train()

        result_name = f"{base_model_name}_{dataset}_distilled"
        evaluate_and_save(distill_trainer.student, test_loader, num_classes,
                          result_name, args.results_dir, device)

    # ── Combined: distill + prune + int8 ──────────────────────────────────────
    elif args.method == "combined":
        dcfg = cfg["compression"]["distillation"]
        distilled_ckpt = os.path.join("experiments", f"{base_model_name}_{dataset}_distilled_best.pth")

        student = get_model(base_model_name, num_classes=num_classes)
        if os.path.exists(distilled_ckpt):
            print(f"[distillation] Loading pre-trained distilled checkpoint: {distilled_ckpt}")
            student.load_state_dict(torch.load(distilled_ckpt, map_location=device))
        else:
            print("[distillation] Pre-trained distilled checkpoint not found, running distillation...")
            teacher = get_model("resnet18", num_classes=num_classes)
            teacher = load_checkpoint(teacher, "resnet18", dataset, device)

            distill_trainer = DistillationTrainer(
                teacher=teacher,
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                temperature=dcfg["temperature"],
                alpha=dcfg["alpha"],
                lr=dcfg["lr"],
                epochs=dcfg["epochs"],
                checkpoint_dir="./experiments",
                model_name=f"{base_model_name}_{dataset}_distilled",
                device=device,
            )
            distill_trainer.train()
            student = distill_trainer.student

        # 2. Prune the distilled student (20% sparsity preserves depthwise features)
        sparsity_schedule = cfg["compression"]["pruning"]["sparsity_schedule"]
        pruned = iterative_prune(student, sparsity_schedule)

        # 3. INT8 quantize the pruned student
        cal_batches = cfg["compression"]["quantization"]["calibration_batches"]
        final_model = quantize_model_int8(pruned, train_loader, cal_batches, device="cpu")

        result_name = f"{base_model_name}_{dataset}_combined"
        evaluate_and_save(final_model, test_loader, num_classes, result_name,
                          args.results_dir, "cpu")


if __name__ == "__main__":
    main()
