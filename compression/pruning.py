"""
MediEdge — Magnitude-based Unstructured Pruning.

Uses torch.nn.utils.prune with L1 magnitude criterion.
Progressive sparsity levels: 20% → 40% → 60%.

After pruning, fine-tune for a few epochs to recover accuracy.
"""

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_model(
    model: nn.Module,
    sparsity: float = 0.4,
    make_permanent: bool = True,
) -> nn.Module:
    """
    Apply global unstructured L1 magnitude pruning.

    Args:
        model         : Model to prune (modified in-place, then returned).
        sparsity      : Fraction of weights to zero out (0.0–1.0).
        make_permanent: If True, remove pruning masks and make zeros permanent.

    Returns:
        Pruned model.
    """
    model = copy.deepcopy(model)

    # Collect all Conv2d and Linear layers
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        print("[pruning] No Conv2d/Linear layers found — skipping.")
        return model

    # Global unstructured L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    if make_permanent:
        for module, param_name in parameters_to_prune:
            try:
                prune.remove(module, param_name)
            except ValueError:
                pass  # Already removed

    actual_sparsity = get_sparsity(model)
    print(f"[pruning] Target sparsity={sparsity:.0%} | "
          f"Actual sparsity={actual_sparsity:.4%}")

    return model


def get_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero weights across all parameters."""
    total, zeros = 0, 0
    for param in model.parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def iterative_prune(
    model: nn.Module,
    sparsity_schedule: List[float],
    finetune_fn: Optional[callable] = None,
) -> nn.Module:
    """
    Progressive pruning: apply increasing sparsity levels,
    optionally fine-tuning between steps.

    Args:
        model            : Starting model.
        sparsity_schedule: List of cumulative sparsity targets, e.g. [0.2, 0.4, 0.6].
        finetune_fn      : Optional callable(model) → model for fine-tuning between steps.

    Returns:
        Final pruned (and optionally fine-tuned) model.
    """
    for sparsity in sparsity_schedule:
        print(f"\n[pruning] Progressive step: target sparsity={sparsity:.0%}")
        model = prune_model(model, sparsity=sparsity, make_permanent=True)
        if finetune_fn is not None:
            print(f"[pruning] Fine-tuning after {sparsity:.0%} pruning...")
            model = finetune_fn(model)
    return model
