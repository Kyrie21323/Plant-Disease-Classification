"""
cnn_baseline.py
Small custom CNN for 8-class plant disease classification.

Architecture overview:
    Input:  3 × 224 × 224 (RGB, ImageNet-normalised)
    Block 1: Conv(3→32)   → BN → ReLU → MaxPool  →  32 × 112 × 112
    Block 2: Conv(32→64)  → BN → ReLU → MaxPool  →  64 ×  56 ×  56
    Block 3: Conv(64→128) → BN → ReLU → MaxPool  → 128 ×  28 ×  28
    Pool:    AdaptiveAvgPool(4×4)                 → 128 ×   4 ×   4
    Head:    Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→8)

Design choices:
    - Batch normalisation after each conv stabilises training without tuning LR carefully.
    - AdaptiveAvgPool lets the spatial path stay flexible if IMAGE_SIZE ever changes.
    - Dropout(0.5) in the head is the primary regulariser; no weight-decay tuning needed yet.
    - 8 raw logits out — loss function (CrossEntropyLoss) applies softmax internally.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool(2×2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Baseline CNN for 8-class plant disease classification.

    Args:
        num_classes: Number of output classes. Default is 8.
        dropout:     Dropout probability in the classifier head. Default 0.5.
    """

    def __init__(self, num_classes: int = 8, dropout: float = 0.5) -> None:
        super().__init__()

        # Three progressive conv blocks; channel depth doubles each time.
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # 224 → 112
            ConvBlock(32,  64),   # 112 →  56
            ConvBlock(64, 128),   #  56 →  28
        )

        # Collapse spatial dims to a fixed 4×4 grid regardless of input size.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 28 → 4  (128 × 4 × 4 = 2048)

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = BaselineCNN(num_classes=8)
    dummy = torch.zeros(4, 3, 224, 224)
    out = model(dummy)
    print(f"BaselineCNN output shape: {out.shape}")   # expected: torch.Size([4, 8])
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
