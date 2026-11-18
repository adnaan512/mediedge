"""
MediEdge — Unit tests for model definitions.

Tests:
  - Each model instantiates without error
  - Forward pass produces correct output shape
  - Model size is within expected range
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from models import get_model, MODEL_REGISTRY
from training.metrics import get_model_size_mb


BATCH_SIZE = 2
NUM_CLASSES = 7
IMAGE_SIZE = 28
DUMMY_INPUT = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_model_forward(model_name):
    """Each model should produce output of shape (batch, num_classes)."""
    model = get_model(model_name, num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        output = model(DUMMY_INPUT)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"{model_name}: expected output ({BATCH_SIZE}, {NUM_CLASSES}), got {output.shape}"
    )


@pytest.mark.parametrize("model_name,max_mb", [
    ("cnn",              2.0),
    ("resnet18",        50.0),
    ("mobilenetv2",     15.0),
    ("efficientnet_lite", 25.0),
])
def test_model_size(model_name, max_mb):
    """Models should be within reasonable size bounds."""
    model = get_model(model_name, num_classes=NUM_CLASSES)
    size_mb = get_model_size_mb(model)
    assert size_mb < max_mb, (
        f"{model_name}: size {size_mb:.1f} MB exceeds limit {max_mb} MB"
    )


def test_model_registry_keys():
    """MODEL_REGISTRY should contain expected entries."""
    expected = {"cnn", "resnet18", "mobilenetv2", "efficientnet_lite"}
    assert set(MODEL_REGISTRY.keys()) == expected


def test_get_model_unknown():
    """get_model should raise ValueError for unknown model names."""
    with pytest.raises(ValueError):
        get_model("unknown_model", num_classes=7)


if __name__ == "__main__":
    # Run without pytest for quick manual check
    for name in MODEL_REGISTRY:
        model = get_model(name, num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        size = get_model_size_mb(model)
        print(f"  {name:25s} | output={tuple(out.shape)} | size={size:.2f} MB  ✓")
