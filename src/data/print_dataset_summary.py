#!/usr/bin/env python3
"""
Print a presentation-friendly dataset summary.

Reads only configs and CSV metadata/split files under the repo (no training code).
Run from the repository root:

    python src/data/print_dataset_summary.py

If split/metadata CSVs are missing locally (e.g. only configs tracked), totals fall back to
configs/split_settings.json for PlantVillage and omit PlantDoc counts unless metadata CSV exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def count_csv_rows(path: Path) -> int | None:
    """Return number of data rows (excluding header); None if file missing."""
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    return len(rows) - 1  # header


def load_split_settings(root: Path) -> dict | None:
    p = root / "configs" / "split_settings.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_class_subset(root: Path) -> list[dict]:
    p = root / "configs" / "class_subset_v1.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = repo_root()
    train_csv = root / "data" / "splits" / "plantvillage_train_split.csv"
    val_csv = root / "data" / "splits" / "plantvillage_val_split.csv"
    test_csv = root / "data" / "splits" / "plantvillage_test_split.csv"
    pd_meta = root / "data" / "metadata" / "plantdoc_subset_metadata.csv"

    classes = load_class_subset(root)
    n_classes = len(classes)
    ss = load_split_settings(root)

    n_train = count_csv_rows(train_csv)
    n_val = count_csv_rows(val_csv)
    n_test = count_csv_rows(test_csv)
    n_pd = count_csv_rows(pd_meta)

    # Prefer CSV row counts; fall back to split_settings for PV if splits missing
    source_pv = "CSV row counts (data/splits/*.csv)"
    if n_train is None or n_val is None or n_test is None:
        if ss:
            n_train = ss.get("train_count")
            n_val = ss.get("val_count")
            n_test = ss.get("test_count")
            total_pv = ss.get("total_images")
            source_pv = "configs/split_settings.json (CSV splits not found on disk)"
        else:
            total_pv = None
    else:
        total_pv = (n_train or 0) + (n_val or 0) + (n_test or 0)

    if total_pv is None:
        total_pv_err = "unclear (missing CSVs and split_settings.json)"
    else:
        total_pv_err = None

    width = 62
    bar = "=" * width
    thin = "-" * width

    print(bar)
    print(" DATASET SETUP SUMMARY (project configuration + on-disk CSVs)")
    print(bar)
    print()

    print("1) PLANTVILLAGE SUBSET")
    print(thin)
    print(f"   Source for counts: {source_pv}")
    if total_pv_err:
        print(f"   Total images:     {total_pv_err}")
    else:
        print(f"   Total images:     {total_pv}")
        print(f"   Train:            {n_train}")
        print(f"   Validation:       {n_val}")
        print(f"   Test:             {n_test}")
        if ss and n_train is not None:
            tr, va, te = ss.get("train_ratio"), ss.get("val_ratio"), ss.get("test_ratio")
            if tr is not None:
                print(f"   Split ratios:     {tr:.0%} / {va:.0%} / {te:.0%} (train/val/test)")
            print(f"   Random seed:      {ss.get('random_seed', '-')}")
    print(f"   Number of classes:{n_classes}")
    print()

    print("2) PLANTDOC SUBSET (external evaluation)")
    print(thin)
    if n_pd is None:
        print("   Total images:     (file not found)")
        print(f"   Expected path:    data/metadata/plantdoc_subset_metadata.csv")
        print("                     Run build_subset_metadata.py after placing raw PlantDoc.")
    else:
        print(f"   Total images:     {n_pd}")
    print(
        f"   Number of classes:{n_classes} (same unified label ids 0-{n_classes - 1} as PlantVillage)"
    )
    print()

    print("3) CLASS NAMES / LABEL IDS (configs/class_subset_v1.json)")
    print(thin)
    print(f"   {'id':>3}  {'unified_label':<32}  plant / disease")
    print(thin)
    for e in sorted(classes, key=lambda x: x["label_id"]):
        lbl = e.get("unified_label", "")
        plant = e.get("plant", "")
        disease = e.get("disease", "")
        print(f"   {e['label_id']:>3}  {lbl:<32}  {plant} - {disease}")
    print()

    print("4) DATA USAGE (confirmed by training/eval design in this repository)")
    print(thin)
    lines = [
        "   - PlantVillage TRAIN:  used for gradient updates (train_one_epoch on pv_train).",
        "   - PlantVillage VAL:    each epoch for loss/metrics and BEST checkpoint selection",
        "                          (lowest validation loss in trainer.run_training_loop).",
        "   - PlantVillage TEST:  held-out in-domain evaluation after training (evaluate on",
        "                          pv_test in train_*.py; also in evaluate_final.py).",
        "   - PlantDoc:            NOT used in training or validation; loaded only for the",
        "                          external pass in src/training/evaluate_final.py (pd_eval).",
    ]
    print("\n".join(lines))
    print()
    print(bar)
    print(" End of summary")
    print(bar)


if __name__ == "__main__":
    main()
