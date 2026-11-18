"""
EfficientNet-B0 as a proxy for EfficientNet-Lite.

EfficientNet-Lite is not in torchvision; we use EfficientNet-B0
(the lightest EfficientNet, closest to the Lite variant) and adapt
it for 28×28 inputs. In the report this will be labeled
'EfficientNet-Lite (B0 proxy)'.

~5.3 MB — second efficient model for comparison.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetLiteSmall(nn.Module):
    """
    EfficientNet-B0 adapted for 28×28 medical images.

    Changes:
      - First conv stride 2 → 1 (preserve spatial resolution at 28×28)
      - Classifier head replaced with num_classes output
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Adjust first conv stride for small inputs
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            first_conv.in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=(1, 1),       # Was (2, 2), reduce for 28×28
            padding=first_conv.padding,
            bias=False,
        )

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_efficientnet_lite(num_classes: int = 7, pretrained: bool = False, **kwargs) -> EfficientNetLiteSmall:
    return EfficientNetLiteSmall(num_classes=num_classes, pretrained=pretrained)
