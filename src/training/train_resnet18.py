"""
train_resnet18.py
Training script for the fine-tuned ResNet-18 model.

This script is responsible for:
    - selecting the model (ResNet-18 + replaced head)
    - setting hyperparameters
    - building dataloaders
    - delegating the training loop to trainer.py
    - saving model-specific outputs

Every run is saved with an auto-incrementing run number so previous results
are never overwritten.  Run numbers are determined independently for each
model prefix, so ResNet and baseline counters do not interfere.

Run from the repository root:
    cd /path/to/Plant-Disease-Classification
    python src/training/train_resnet18.py

Outputs (written automatically, N = next available run number):
    outputs/checkpoints/resnet18_best_run<N>.pt
    outputs/figures/resnet18_training_curves_run<N>.png
    outputs/figures/resnet18_confusion_matrix_run<N>.png
    outputs/results/resnet18_results_run<N>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup - allow imports from src/data/, src/models/, src/utils/,
#              and src/training/ (for trainer.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

for _sub in ("src/data", "src/models", "src/utils", "src/training"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloaders import build_dataloaders                         # src/data/
from resnet18_finetune import build_resnet18_finetune             # src/models/
from metrics import compute_confusion_matrix                      # src/utils/
from plotting import save_training_curves, save_confusion_matrix  # src/utils/
from trainer import (                                             # src/training/
    run_training_loop, evaluate, CLASS_NAMES, get_next_run_number,
)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_DIR     = REPO_ROOT / "outputs" / "results"
FIGURES_DIR     = REPO_ROOT / "outputs" / "figures"

for _d in (CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters - edit here for quick experiments
# ---------------------------------------------------------------------------
NUM_EPOCHS    = 15
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4    # Lower than baseline - pretrained weights need gentle updates
NUM_WORKERS   = 0       # 0 is safest in WSL
NUM_CLASSES   = 8
SEED          = 42


def main() -> None:
    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # Determine this run's number before touching any output files.
    # The checkpoint directory is used as the authoritative source because
    # it is always written first and every run produces exactly one .pt file.
    # -----------------------------------------------------------------------
    run_num = get_next_run_number("resnet18_best", CHECKPOINTS_DIR)
    run_tag = f"run{run_num}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ResNet-18 Fine-tune - Plant Disease Classification  [{run_tag}]")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Data - identical loaders to the baseline for a fair comparison
    # -----------------------------------------------------------------------
    print("Loading datasets...")
    _, loaders = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train : {len(loaders.pv_train.dataset):>6} images")
    print(f"  Val   : {len(loaders.pv_val.dataset):>6} images")
    print(f"  Test  : {len(loaders.pv_test.dataset):>6} images\n")

    # -----------------------------------------------------------------------
    # Model, loss, optimiser
    # -----------------------------------------------------------------------
    model     = build_resnet18_finetune(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    # Full fine-tuning: all parameters updated, but at a low LR to preserve
    # the pretrained features learned on ImageNet.
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model : ResNet-18 (fine-tune)  ({total_params:,} parameters)\n")

    # -----------------------------------------------------------------------
    # Training loop (delegated to trainer.py)
    # -----------------------------------------------------------------------
    ckpt_path = CHECKPOINTS_DIR / f"resnet18_best_{run_tag}.pt"

    history = run_training_loop(
        model=model,
        train_loader=loaders.pv_train,
        val_loader=loaders.pv_val,
        criterion=criterion,
        optimiser=optimiser,
        device=device,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        ckpt_path=ckpt_path,
    )

    # -----------------------------------------------------------------------
    # Test evaluation using best checkpoint weights
    # -----------------------------------------------------------------------
    print("Evaluating on PlantVillage test split...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(
        model, loaders.pv_test, criterion, device, NUM_CLASSES
    )

    print(f"\n  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  Test F1   : {test_f1:.4f}\n")

    # -----------------------------------------------------------------------
    # Save training curve
    # -----------------------------------------------------------------------
    curve_path = save_training_curves(
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        filename=f"resnet18_training_curves_{run_tag}.png",
        figures_dir=FIGURES_DIR,
        title=f"ResNet-18 Fine-tune - Training Curves ({run_tag})",
    )
    print(f"Training curve saved  → {curve_path}")

    # -----------------------------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------------------------
    cm = compute_confusion_matrix(test_preds, test_targets, num_classes=NUM_CLASSES)
    cm_path = save_confusion_matrix(
        matrix=cm,
        class_names=CLASS_NAMES,
        filename=f"resnet18_confusion_matrix_{run_tag}.png",
        figures_dir=FIGURES_DIR,
    )
    print(f"Confusion matrix saved → {cm_path}")

    # -----------------------------------------------------------------------
    # Save results JSON  (run_num stored inside for traceability)
    # -----------------------------------------------------------------------
    results = {
        "model":         "ResNet18_finetune",
        "run":           run_num,
        "num_epochs":    NUM_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed":          SEED,
        "per_epoch": {
            "train_loss": history["train_losses"],
            "val_loss":   history["val_losses"],
            "val_acc":    history["val_accs"],
            "val_f1":     history["val_f1s"],
        },
        "best_val_loss": history["best_val_loss"],
        "best_epoch":    history["best_epoch"],
        "test_loss":     test_loss,
        "test_acc":      test_acc,
        "test_f1":       test_f1,
    }

    results_path = RESULTS_DIR / f"resnet18_results_{run_tag}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved    → {results_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete  [{run_tag}].")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
