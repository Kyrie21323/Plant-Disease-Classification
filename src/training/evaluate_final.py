"""
evaluate_final.py
Final evaluation of selected models on PlantVillage test and PlantDoc.

This script is read-only with respect to model weights:
    - No training, no optimizer, no backpropagation.
    - Models are loaded from saved checkpoints and immediately set to eval mode.
    - PlantDoc is used here for the first time as the external generalization test.

Selected checkpoints:
    BaselineCNN  → outputs/checkpoints/baseline_cnn_best_run3.pt  (run3, epoch 22)
    ResNet-18    → outputs/checkpoints/resnet18_best.pt            (original run, epoch 12)

Run from the repository root (any path / WSL or native Linux):
    cd /path/to/Plant-Disease-Classification
    python src/training/evaluate_final.py

Outputs:
    outputs/figures/baseline_pv_test_confusion_matrix.png
    outputs/figures/baseline_plantdoc_confusion_matrix.png
    outputs/figures/resnet18_pv_test_confusion_matrix.png
    outputs/figures/resnet18_plantdoc_confusion_matrix.png
    outputs/results/baseline_final_eval.json
    outputs/results/resnet18_final_eval.json
    outputs/results/final_comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup (repo root = parent of src/; works on WSL, Linux, any clone location)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

for _sub in ("src/data", "src/models", "src/utils", "src/training"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloaders import build_dataloaders                         # type: ignore[import]  # src/data/
from cnn_baseline import BaselineCNN                              # type: ignore[import]  # src/models/
from resnet18_finetune import build_resnet18_finetune             # type: ignore[import]  # src/models/
from metrics import compute_confusion_matrix                      # type: ignore[import]  # src/utils/
from plotting import save_confusion_matrix                        # type: ignore[import]  # src/utils/
from trainer import evaluate, CLASS_NAMES                         # type: ignore[import]  # src/training/

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_DIR     = REPO_ROOT / "outputs" / "results"
FIGURES_DIR     = REPO_ROOT / "outputs" / "figures"

BASELINE_CKPT  = CHECKPOINTS_DIR / "baseline_cnn_best_run3.pt"
RESNET18_CKPT  = CHECKPOINTS_DIR / "resnet18_best.pt"

NUM_CLASSES = 8
NUM_WORKERS = 0


def load_model(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load state_dict from checkpoint, move model to device, set eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)   # weights must be on the same device as the input tensors
    model.eval()
    print(f"  Loaded {ckpt_path.name}  (saved at epoch {ckpt['epoch']}"
          f", val_loss={ckpt['val_loss']:.4f}, val_f1={ckpt['val_f1']:.4f})")
    return model


def eval_one_model(
    model:       nn.Module,
    model_tag:   str,
    loaders:     object,
    criterion:   nn.Module,
    device:      torch.device,
) -> dict:
    """
    Evaluate a model on PlantVillage test and PlantDoc.

    Saves confusion matrices for both datasets.
    Returns a result dict ready for JSON serialisation.
    """
    results: dict = {"model": model_tag}

    for split_name, loader, cm_filename in [
        ("pv_test",   loaders.pv_test,  f"{model_tag}_pv_test_confusion_matrix.png"),
        ("plantdoc",  loaders.pd_eval,  f"{model_tag}_plantdoc_confusion_matrix.png"),
    ]:
        loss, acc, f1, preds, targets = evaluate(
            model, loader, criterion, device, NUM_CLASSES
        )

        cm       = compute_confusion_matrix(preds, targets, num_classes=NUM_CLASSES)
        cm_path  = save_confusion_matrix(
            matrix=cm,
            class_names=CLASS_NAMES,
            filename=cm_filename,
            figures_dir=FIGURES_DIR,
        )

        results[split_name] = {
            "loss": loss,
            "acc":  acc,
            "f1":   f1,
        }
        print(f"    {split_name:<10}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}"
              f"  → {cm_path.name}")

    # Generalization gap: positive means PV test > PlantDoc (expected).
    results["generalization_gap_acc"] = (
        results["pv_test"]["acc"] - results["plantdoc"]["acc"]
    )
    results["generalization_gap_f1"] = (
        results["pv_test"]["f1"] - results["plantdoc"]["f1"]
    )
    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Final Evaluation - PlantVillage Test + PlantDoc")
    print(f"  Device : {device}")
    print(f"{'='*65}\n")

    # -----------------------------------------------------------------------
    # Dataloaders (build once, share between both models)
    # -----------------------------------------------------------------------
    print("Building dataloaders...")
    _, loaders = build_dataloaders(num_workers=NUM_WORKERS)
    print(f"  PV test  : {len(loaders.pv_test.dataset):>5} images")
    print(f"  PlantDoc : {len(loaders.pd_eval.dataset):>5} images\n")

    criterion = nn.CrossEntropyLoss()

    all_results: list[dict] = []

    # -----------------------------------------------------------------------
    # BaselineCNN - run3
    # -----------------------------------------------------------------------
    print("[ BaselineCNN - run3 ]")
    model_cnn = load_model(
        BaselineCNN(num_classes=NUM_CLASSES, dropout=0.3),
        BASELINE_CKPT,
        device,
    )
    cnn_results = eval_one_model(model_cnn, "baseline", loaders, criterion, device)
    all_results.append(cnn_results)

    result_path = RESULTS_DIR / "baseline_final_eval.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(cnn_results, f, indent=2)
    print(f"  Saved → {result_path.name}\n")

    # -----------------------------------------------------------------------
    # ResNet-18 - original run
    # -----------------------------------------------------------------------
    print("[ ResNet-18 - original run ]")
    model_rn = load_model(
        build_resnet18_finetune(num_classes=NUM_CLASSES),
        RESNET18_CKPT,
        device,
    )
    rn_results = eval_one_model(model_rn, "resnet18", loaders, criterion, device)
    all_results.append(rn_results)

    result_path = RESULTS_DIR / "resnet18_final_eval.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(rn_results, f, indent=2)
    print(f"  Saved → {result_path.name}\n")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Comparison: PlantVillage Test vs PlantDoc")
    print(f"{'='*65}")
    header = f"{'Model':<12}  {'PV Test Acc':>11}  {'PD Acc':>8}  {'Gap':>7}  {'PV F1':>7}  {'PD F1':>7}"
    print(header)
    print("-" * len(header))

    comparison = []
    for r in all_results:
        pv_acc = r["pv_test"]["acc"]
        pd_acc = r["plantdoc"]["acc"]
        gap    = r["generalization_gap_acc"]
        pv_f1  = r["pv_test"]["f1"]
        pd_f1  = r["plantdoc"]["f1"]
        print(
            f"{r['model']:<12}  {pv_acc:>11.4f}  {pd_acc:>8.4f}  {gap:>+7.4f}"
            f"  {pv_f1:>7.4f}  {pd_f1:>7.4f}"
        )
        comparison.append({
            "model":          r["model"],
            "pv_test_acc":    pv_acc,
            "plantdoc_acc":   pd_acc,
            "gap_acc":        gap,
            "pv_test_f1":     pv_f1,
            "plantdoc_f1":    pd_f1,
            "gap_f1":         r["generalization_gap_f1"],
        })
    print(f"{'='*65}\n")

    comparison_path = RESULTS_DIR / "final_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison table saved → {comparison_path.name}")
    print("Final evaluation complete.\n")


if __name__ == "__main__":
    main()
