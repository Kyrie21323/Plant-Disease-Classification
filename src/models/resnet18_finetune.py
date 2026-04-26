"""
resnet18_finetune.py
Pretrained ResNet-18 fine-tuned for 8-class plant disease classification.

Fine-tuning approach:
    - Start from ImageNet-pretrained weights (torchvision IMAGENET1K_V1).
    - Replace the final fully connected layer (fc) with a new Linear(512→8).
      Only the replaced layer has randomly initialised weights; everything else
      starts from ImageNet features.
    - All layers are trainable - full fine-tuning, not feature extraction.
      This is appropriate for our dataset size and keeps the script simple.

Preprocessing alignment:
    The existing transform pipeline (transforms.py) already uses ImageNet
    normalisation statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    and resizes to 224×224. This is identical to what the pretrained ResNet-18
    expects, so no changes to the data pipeline are required. This also means
    baseline and ResNet-18 results are directly comparable - both models see
    exactly the same image tensors.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18_finetune(num_classes: int = 8) -> nn.Module:
    """
    Build a pretrained ResNet-18 with its classifier head replaced for
    `num_classes` outputs.

    The final fc layer (originally Linear(512, 1000)) is swapped for a new
    Linear(512, num_classes). All other layers retain their ImageNet weights.

    Args:
        num_classes: Number of output classes. Default is 8.

    Returns:
        nn.Module ready for training.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # ResNet-18's classifier head is a single fc layer with in_features=512.
    in_features = model.fc.in_features          # 512
    model.fc    = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    model = build_resnet18_finetune(num_classes=8)
    dummy = torch.zeros(4, 3, 224, 224)
    out   = model(dummy)
    print(f"ResNet-18 finetune output shape: {out.shape}")   # expected: [4, 8]

    total_params    = sum(p.numel() for p in model.parameters())
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params     = sum(p.numel() for p in model.fc.parameters())
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"New head params : {head_params:,}")
