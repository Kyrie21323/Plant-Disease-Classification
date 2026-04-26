"""
metrics.py
Lightweight metric helpers for classification evaluation.

Functions:
    compute_accuracy        - fraction of correctly predicted samples
    compute_macro_f1        - unweighted mean F1 across all classes
    compute_confusion_matrix - num_classes × num_classes count matrix

Both functions accept plain Python lists or PyTorch tensors of integer labels.
No external dependencies beyond the standard library and (optionally) PyTorch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Type alias accepted by both functions
# ---------------------------------------------------------------------------
Labels = Sequence[int] | torch.Tensor


def _to_lists(preds: Labels, targets: Labels) -> tuple[list[int], list[int]]:
    """Convert tensors or sequences to plain Python int lists."""
    if isinstance(preds, torch.Tensor):
        preds = preds.tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return list(preds), list(targets)


def compute_accuracy(preds: Labels, targets: Labels) -> float:
    """
    Fraction of predictions that match the ground-truth labels.

    Args:
        preds:   Predicted class indices, shape (N,).
        targets: Ground-truth class indices, shape (N,).

    Returns:
        Accuracy in [0.0, 1.0].
    """
    p, t = _to_lists(preds, targets)
    if len(p) == 0:
        return 0.0
    correct = sum(pi == ti for pi, ti in zip(p, t))
    return correct / len(p)


def compute_macro_f1(preds: Labels, targets: Labels, num_classes: int = 8) -> float:
    """
    Macro-averaged F1 score across all classes.

    Macro averaging computes F1 per class then takes the unweighted mean.
    Classes with no true or predicted samples contribute 0 to the mean,
    which penalises the model if it never predicts a certain class.

    Args:
        preds:       Predicted class indices, shape (N,).
        targets:     Ground-truth class indices, shape (N,).
        num_classes: Total number of classes. Default 8.

    Returns:
        Macro F1 in [0.0, 1.0].
    """
    p, t = _to_lists(preds, targets)

    # Per-class counts.
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for pred, true in zip(p, t):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    f1_scores: list[float] = []
    for cls in range(num_classes):
        precision_denom = tp[cls] + fp[cls]
        recall_denom    = tp[cls] + fn[cls]
        precision = tp[cls] / precision_denom if precision_denom > 0 else 0.0
        recall    = tp[cls] / recall_denom    if recall_denom    > 0 else 0.0
        pr_sum = precision + recall
        f1 = (2 * precision * recall / pr_sum) if pr_sum > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def compute_confusion_matrix(
    preds: Labels, targets: Labels, num_classes: int = 8
) -> list[list[int]]:
    """
    Build a num_classes × num_classes confusion matrix.

    Entry [true][pred] contains the number of samples whose true label is
    `true` and whose predicted label is `pred`. The diagonal holds correct
    predictions; off-diagonal entries are misclassifications.

    Args:
        preds:       Predicted class indices, shape (N,).
        targets:     Ground-truth class indices, shape (N,).
        num_classes: Total number of classes. Default 8.

    Returns:
        A list-of-lists matrix of shape (num_classes, num_classes).
    """
    p, t = _to_lists(preds, targets)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, true in zip(p, t):
        matrix[true][pred] += 1
    return matrix
