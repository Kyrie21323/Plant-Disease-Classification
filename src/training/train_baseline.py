"""
train_baseline.py
Training script for the custom BaselineCNN model.

This script is responsible for:
    - selecting the model (BaselineCNN)
    - setting hyperparameters
    - building dataloaders
    - delegating the training loop to trainer.py
    - saving model-specific outputs

Every run is saved with an auto-incrementing run number so previous results
are never overwritten.  Run numbers are determined independently for each
model prefix, so baseline and ResNet counters do not interfere.

Run from the repo root inside WSL:
    cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
    python src/training/train_baseline.py

Outputs (written automatically, N = next available run number):
    outputs/checkpoints/baseline_cnn_best_run<N>.pt
    outputs/figures/baseline_training_curves_run<N>.png
    outputs/figures/baseline_confusion_matrix_run<N>.png
    outputs/results/baseline_results_run<N>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/data/, src/models/, src/utils/,
#              and src/training/ (for trainer.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification")

for _sub in ("src/data", "src/models", "src/utils", "src/training"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloaders import build_dataloaders                         # type: ignore[import]  # src/data/
from cnn_baseline import BaselineCNN                              # type: ignore[import]  # src/models/
from metrics import compute_confusion_matrix                      # type: ignore[import]  # src/utils/
from plotting import save_training_curves, save_confusion_matrix  # type: ignore[import]  # src/utils/
from trainer import (                                             # type: ignore[import]  # src/training/
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
# Hyperparameters — edit here for quick experiments
# ---------------------------------------------------------------------------
NUM_EPOCHS    = 25      # Experiment A: extended from 15 (model had not converged)
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4   # Experiment A: reduced from 1e-3 to smooth oscillating val loss
WEIGHT_DECAY  = 1e-4   # Experiment B: L2 regularisation added to Adam
NUM_WORKERS   = 0       # 0 is safest in WSL
NUM_CLASSES   = 8
DROPOUT       = 0.3    # Experiment C: reduced from 0.5 to give head more capacity
SEED          = 42


def main() -> None:
    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # Determine this run's number before touching any output files.
    # The checkpoint directory is used as the authoritative source because
    # it is always written first and every run produces exactly one .pt file.
    # -----------------------------------------------------------------------
    run_num = get_next_run_number("baseline_cnn_best", CHECKPOINTS_DIR)
    run_tag = f"run{run_num}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BaselineCNN — Plant Disease Classification  [{run_tag}]")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Data
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
    model     = BaselineCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model : BaselineCNN  ({total_params:,} parameters)\n")

    # -----------------------------------------------------------------------
    # Training loop (delegated to trainer.py)
    # -----------------------------------------------------------------------
    ckpt_path = CHECKPOINTS_DIR / f"baseline_cnn_best_{run_tag}.pt"

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
        filename=f"baseline_training_curves_{run_tag}.png",
        figures_dir=FIGURES_DIR,
        title=f"BaselineCNN — Training Curves ({run_tag})",
    )
    print(f"Training curve saved  → {curve_path}")

    # -----------------------------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------------------------
    cm = compute_confusion_matrix(test_preds, test_targets, num_classes=NUM_CLASSES)
    cm_path = save_confusion_matrix(
        matrix=cm,
        class_names=CLASS_NAMES,
        filename=f"baseline_confusion_matrix_{run_tag}.png",
        figures_dir=FIGURES_DIR,
    )
    print(f"Confusion matrix saved → {cm_path}")

    # -----------------------------------------------------------------------
    # Save results JSON  (run_num stored inside for traceability)
    # -----------------------------------------------------------------------
    results = {
        "model":         "BaselineCNN",
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

    results_path = RESULTS_DIR / f"baseline_results_{run_tag}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved    → {results_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete  [{run_tag}].")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
