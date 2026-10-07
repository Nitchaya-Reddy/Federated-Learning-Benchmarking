"""
model.py
--------
Two architectures for CIFAR-10:
  - CIFAR10CNN:     lightweight 3-block ConvNet (~1.3M params, ~86% centralised)
  - CIFAR10ResNet18: ResNet-18 adapted for 32×32 (~11.2M params, ~93-95% centralised)

Use get_model(arch) to select between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class CIFAR10CNN(nn.Module):
    """
    3-block ConvNet: good enough to reach ~85% centralised accuracy on
    CIFAR-10 while being fast to train per FL round.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 -> 16
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 -> 8
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8 -> 4
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CIFAR10ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images).

    Key differences from the ImageNet ResNet-18:
      - First conv: 7×7 stride-2 → 3×3 stride-1  (preserves the 32×32 spatial res)
      - Initial MaxPool removed                   (would shrink 32→8 too aggressively)
      - FC head: 512 → 10 classes

    ~11.2M parameters.  Achieves ~93-95% centralised accuracy on CIFAR-10
    vs ~86% for CIFAR10CNN, giving more headroom to study FL degradation.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Linear(512, num_classes)
        self._model  = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def get_model(arch: str = "cnn", num_classes: int = 10) -> nn.Module:
    """Factory: arch='cnn' returns CIFAR10CNN, arch='resnet18' returns CIFAR10ResNet18."""
    if arch == "resnet18":
        return CIFAR10ResNet18(num_classes)
    return CIFAR10CNN(num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
