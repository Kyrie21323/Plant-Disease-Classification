"""
transforms.py
Image preprocessing and augmentation pipelines for Plant Disease Classification.

Design principles:
- A single IMAGE_SIZE and NORMALIZE_MEAN/STD are defined here and used by
  all transforms. This ensures that the custom CNN and the pretrained model
  receive identically preprocessed inputs, making any performance difference
  attributable to the model rather than inconsistent preprocessing.
- The evaluation transform is fully deterministic (no random operations) so
  that validation and test results are reproducible across runs.
- Training augmentations are mild and ecologically plausible for leaf images:
  flips and small rotations reflect real-world variation; colour jitter accounts
  for lighting differences between PlantVillage and PlantDoc.
- Augmentations are intentionally not applied during evaluation because they
  would introduce randomness into metric computation and make results harder
  to compare across experiments.
"""

import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Shared image settings
# All transforms use these constants.
# IMAGE_SIZE: 224 is standard for both custom CNNs and pretrained models
#   (ImageNet pretrained backbones such as ResNet-18 expect 224×224 input).
# NORMALIZE_MEAN / NORMALIZE_STD: ImageNet statistics are used because
#   pretrained models were trained on those values. Using them consistently
#   for the custom CNN as well keeps preprocessing identical across models.
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_baseline_transform() -> T.Compose:
    """
    Baseline transform: resize, tensor conversion, normalization only.
    No augmentation. Used as the minimal preprocessing reference.

    Returns:
        torchvision Compose transform.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_train_transform() -> T.Compose:
    """
    Training augmentation transform.
    Applied only during training to improve generalization.

    Augmentations chosen for plant leaf images:
    - RandomResizedCrop: simulates slight zoom and framing variation.
      Scale range (0.8, 1.0) keeps the leaf largely visible.
    - RandomHorizontalFlip: leaves are symmetric; flip is ecologically valid.
    - RandomRotation(±15°): accounts for variation in how leaves are presented.
    - ColorJitter: mild brightness, contrast, saturation variation to simulate
      different lighting conditions across PlantVillage and PlantDoc.

    Augmentations intentionally excluded:
    - Vertical flip (less natural for field images)
    - Large rotations / distortions (may remove diagnostic disease features)
    - Cutout / erasing (too aggressive for small datasets)

    Returns:
        torchvision Compose transform.
    """
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=0.4,   # Experiment D: strengthened from 0.2
            contrast=0.4,     # Experiment D: strengthened from 0.2
            saturation=0.4,   # Experiment D: strengthened from 0.2
            hue=0.1,          # Experiment D: strengthened from 0.05
        ),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_eval_transform() -> T.Compose:
    """
    Evaluation transform: deterministic, no randomness.
    Used for validation, test, and cross-dataset (PlantDoc) evaluation.

    Identical resize / normalize behaviour to get_baseline_transform()
    to ensure that evaluation results are reproducible and comparable
    across all models and experiments.

    Returns:
        torchvision Compose transform.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
