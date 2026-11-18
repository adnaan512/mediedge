"""
Small CNN baseline — intentionally lightweight for comparison.

Architecture:
  Conv(3→32) → BN → ReLU → MaxPool
  Conv(32→64) → BN → ReLU → MaxPool
  Conv(64→128) → BN → ReLU → AdaptiveAvgPool
  FC(128→num_classes)

~0.3 MB, designed to be a floor reference.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """3-block small CNN for 28×28 medical images."""

    def __init__(self, num_classes: int = 7, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 28 → 14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 14 → 7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),        # 7 → 1×1
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_cnn_baseline(num_classes: int = 7, **kwargs) -> SmallCNN:
    return SmallCNN(num_classes=num_classes, **kwargs)
