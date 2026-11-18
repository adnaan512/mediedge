"""
MobileNetV2 — primary efficient model.

Designed for mobile/edge devices via depthwise separable convolutions
and inverted residuals.

We adapt it for 28×28 inputs by modifying the stem stride
and removing the initial aggressive downsampling.

~3.4 MB — the main subject of compression experiments.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Small(nn.Module):
    """
    MobileNetV2 adapted for 28×28 medical images.

    Changes:
      - First conv stride 2 → 1 (preserve spatial resolution)
      - Classifier head replaced with num_classes output
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # Adjust first conv stride for small inputs
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            first_conv.in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,            # Was 2, reduce for 28×28
            padding=first_conv.padding,
            bias=False,
        )

        # Replace classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_mobilenetv2(num_classes: int = 7, pretrained: bool = False, **kwargs) -> MobileNetV2Small:
    return MobileNetV2Small(num_classes=num_classes, pretrained=pretrained)
