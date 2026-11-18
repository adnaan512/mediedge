"""
ResNet-18 — standard baseline.

We modify the first conv layer (kernel 3×3 instead of 7×7, stride 1)
to suit 28×28 inputs — the standard ImageNet stem is too aggressive
for small images.

Optionally load ImageNet pretrained weights (torchvision).
~11 MB — this is the "heavy" reference point.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Small(nn.Module):
    """
    ResNet-18 adapted for small (28×28) medical images.

    Changes from stock ResNet-18:
      - conv1: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution)
      - maxpool: removed (input too small)
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt stem for small images
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()  # Skip maxpool — 28×28 is already small

        # Replace classifier head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_resnet18(num_classes: int = 7, pretrained: bool = False, **kwargs) -> ResNet18Small:
    return ResNet18Small(num_classes=num_classes, pretrained=pretrained)
