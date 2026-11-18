from .cnn_baseline import SmallCNN
from .resnet18 import get_resnet18
from .mobilenetv2 import get_mobilenetv2
from .efficientnet_lite import get_efficientnet_lite

MODEL_REGISTRY = {
    "cnn": SmallCNN,
    "resnet18": get_resnet18,
    "mobilenetv2": get_mobilenetv2,
    "efficientnet_lite": get_efficientnet_lite,
}


def get_model(name: str, num_classes: int, **kwargs):
    """Factory: return a model by name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


__all__ = [
    "SmallCNN",
    "get_resnet18",
    "get_mobilenetv2",
    "get_efficientnet_lite",
    "get_model",
    "MODEL_REGISTRY",
]
