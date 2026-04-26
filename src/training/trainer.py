"""
trainer.py
Shared training and evaluation utilities for plant disease classification.

Both train_baseline.py and train_resnet18.py import from here.
Centralising these functions eliminates duplicated loop logic and ensures
that both models are evaluated identically - a prerequisite for fair comparison.

Public API:
    get_next_run_number - find the next unused run index for a given model prefix
    train_one_epoch     - one gradient-update pass over the training loader
    evaluate            - one inference pass; returns loss, accuracy, F1, and raw predictions
    run_training_loop   - full epoch loop with checkpoint saving and console output
    save_checkpoint     - save model weights and metadata to disk

Class labels (label_id → short name) are defined here as CLASS_NAMES so both
training scripts share the same ordering when producing confusion matrices.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Run-number helper
# ---------------------------------------------------------------------------

def get_next_run_number(prefix: str, directory: Path) -> int:
    """
    Return the next unused run number for files matching ``<prefix>_run<N>``.

    Scans *directory* for any file whose stem matches ``<prefix>_run<N>``
    (where N is a positive integer), finds the highest N already present,
    and returns N+1.  Returns 1 if no matching file exists yet.

    This is intentionally model-specific: a ``prefix`` of ``"baseline_results"``
    only counts baseline JSON files; ``"resnet18_best"`` only counts ResNet
    checkpoints.  The two counters are completely independent.

    Args:
        prefix:    The fixed part of the filename before ``_run<N>``.
                   E.g. ``"baseline_results"``, ``"resnet18_best"``.
        directory: Directory to scan (does not need to exist yet).

    Returns:
        Next available integer run number (1-based).

    Examples:
        No files present              → 1
        baseline_results_run1.json    → 2
        baseline_results_run1.json,
        baseline_results_run2.json    → 3
    """
    if not directory.exists():
        return 1

    pattern = re.compile(rf"^{re.escape(prefix)}_run(\d+)\..+$")
    existing = [
        int(m.group(1))
        for f in directory.iterdir()
        if (m := pattern.match(f.name))
    ]
    return max(existing) + 1 if existing else 1


# ---------------------------------------------------------------------------
# Shared class metadata (label_id 0–7, matches class_subset_v1.json)
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "corn_n_leaf_blight",       # 0
    "tomato_septoria",          # 1
    "squash_powdery_mildew",    # 2
    "potato_early_blight",      # 3
    "corn_common_rust",         # 4
    "tomato_bacterial_spot",    # 5
    "tomato_late_blight",       # 6
    "tomato_early_blight",      # 7
]


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """
    One full gradient-update pass over the training loader.

    Args:
        model:     The network being trained.
        loader:    Training DataLoader (shuffled).
        criterion: Loss function (CrossEntropyLoss).
        optimiser: Optimiser (Adam, SGD, …).
        device:    CPU or CUDA device.

    Returns:
        Mean training loss over all samples.
    """
    model.train()
    running_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation pass (val or test)
# ---------------------------------------------------------------------------

def evaluate(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    num_classes: int = 8,
) -> tuple[float, float, float, list[int], list[int]]:
    """
    One full inference pass over a dataloader (no gradient updates).

    Args:
        model:       The network being evaluated.
        loader:      Val or test DataLoader (unshuffled).
        criterion:   Loss function.
        device:      CPU or CUDA device.
        num_classes: Number of output classes. Default 8.

    Returns:
        (mean_loss, accuracy, macro_f1, all_preds, all_targets)
        - all_preds and all_targets are plain int lists, useful for
          computing confusion matrices after the call.
    """
    # Import here to avoid circular paths when trainer is imported first.
    from metrics import compute_accuracy, compute_macro_f1  # src/utils/metrics.py

    model.eval()
    running_loss = 0.0
    all_preds:   list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    mean_loss = running_loss / len(loader.dataset)
    accuracy  = compute_accuracy(all_preds, all_targets)
    macro_f1  = compute_macro_f1(all_preds, all_targets, num_classes=num_classes)

    return mean_loss, accuracy, macro_f1, all_preds, all_targets


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     nn.Module,
    ckpt_path: Path,
    epoch:     int,
    val_loss:  float,
    val_acc:   float,
    val_f1:    float,
) -> None:
    """
    Save model weights plus training metadata to a .pt file.

    Args:
        model:     Model whose state_dict is saved.
        ckpt_path: Destination file path (parent dir must exist).
        epoch:     Epoch number this checkpoint corresponds to.
        val_loss:  Validation loss at this epoch.
        val_acc:   Validation accuracy at this epoch.
        val_f1:    Validation macro-F1 at this epoch.
    """
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    val_loss,
            "val_acc":     val_acc,
            "val_f1":      val_f1,
        },
        ckpt_path,
    )


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training_loop(
    model:       nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:  torch.utils.data.DataLoader,
    criterion:   nn.Module,
    optimiser:   torch.optim.Optimizer,
    device:      torch.device,
    num_epochs:  int,
    num_classes: int,
    ckpt_path:   Path,
) -> dict[str, Any]:
    """
    Run the full training loop and return a history dictionary.

    Saves a checkpoint to `ckpt_path` each time validation loss improves.
    Prints a concise per-epoch summary table to stdout.

    Args:
        model:        Network to train.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        criterion:    Loss function.
        optimiser:    Optimiser.
        device:       CPU or CUDA device.
        num_epochs:   Number of epochs to train.
        num_classes:  Number of output classes.
        ckpt_path:    File path to write the best checkpoint.

    Returns:
        History dict with keys:
            train_losses, val_losses, val_accs, val_f1s,
            best_val_loss, best_epoch
    """
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_accs:     list[float] = []
    val_f1s:      list[float] = []

    best_val_loss = float("inf")
    best_epoch    = -1

    print(
        f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}"
        f"  {'Val Acc':>8}  {'Val F1':>8}  {'Time':>6}"
    )
    print("-" * 62)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        elapsed = time.time() - t0
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}"
            f"  {val_acc:>7.3f}  {val_f1:>7.3f}  {elapsed:>5.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            save_checkpoint(model, ckpt_path, epoch, val_loss, val_acc, val_f1)

    print("-" * 62)
    print(f"\nBest val loss : {best_val_loss:.4f}  (epoch {best_epoch}, checkpoint: {ckpt_path})\n")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "val_accs":     val_accs,
        "val_f1s":      val_f1s,
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
    }
