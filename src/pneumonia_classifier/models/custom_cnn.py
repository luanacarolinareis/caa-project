"""Custom CNN baseline for pediatric chest X-ray classification."""

from __future__ import annotations
import torch
from torch import nn


class CustomCNN(nn.Module):
    """Three-block CNN baseline for chest X-ray classification."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.3, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def build_custom_cnn(num_classes: int = 1) -> CustomCNN:
    return CustomCNN(num_classes=num_classes)
