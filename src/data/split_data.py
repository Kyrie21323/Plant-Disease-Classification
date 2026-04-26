"""
split_data.py
Create reproducible train / validation / test splits for the PlantVillage subset.

Reads:   data/metadata/plantvillage_subset_metadata.csv
Writes:  data/splits/plantvillage_train_split.csv
         data/splits/plantvillage_val_split.csv
         data/splits/plantvillage_test_split.csv
         configs/split_settings.json

PlantDoc is NOT split here. It is used as a separate cross-dataset
evaluation set and remains untouched by this script.

Usage (from Ubuntu WSL):
    python /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/split_data.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
    "Split ratios must sum to 1.0"

REPO_ROOT = Path("/mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification")

INPUT_CSV     = REPO_ROOT / "data" / "metadata" / "plantvillage_subset_metadata.csv"
TRAIN_CSV     = REPO_ROOT / "data" / "splits"   / "plantvillage_train_split.csv"
VAL_CSV       = REPO_ROOT / "data" / "splits"   / "plantvillage_val_split.csv"
TEST_CSV      = REPO_ROOT / "data" / "splits"   / "plantvillage_test_split.csv"
SETTINGS_JSON = REPO_ROOT / "configs"           / "split_settings.json"
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Load a CSV file and return (fieldnames, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def save_csv(fieldnames: list[str], rows: list[dict], path: Path) -> None:
    """Write rows to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(
    rows: list[dict],
    label_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Perform a stratified train / val / test split.

    Splits are done per-class so each class is proportionally represented
    in every split. The random seed ensures reproducibility.

    Args:
        rows:        All metadata rows.
        label_key:   Column name to stratify by (e.g. "label_id").
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation.
        seed:        Random seed.

    Returns:
        (train_rows, val_rows, test_rows)
    """
    rng = random.Random(seed)

    # Group rows by class label.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[label_key]].append(row)

    train, val, test = [], [], []

    for label in sorted(groups.keys()):
        class_rows = groups[label][:]
        rng.shuffle(class_rows)
        n = len(class_rows)

        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        # Test gets the remainder to avoid rounding loss.
        n_test  = n - n_train - n_val

        train.extend(class_rows[:n_train])
        val.extend(class_rows[n_train : n_train + n_val])
        test.extend(class_rows[n_train + n_val :])

    return train, val, test


def count_per_class(rows: list[dict]) -> dict[str, int]:
    """Return a dict of unified_label -> image count."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["unified_label"]] += 1
    return dict(sorted(counts.items()))


def print_split_summary(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    total_input: int,
) -> None:
    """Print a concise split summary to stdout."""
    all_labels = sorted(
        set(r["unified_label"] for r in train + val + test)
    )

    print(f"\n  {'Split':<10} {'Images':>8}")
    print(f"  {'-' * 20}")
    print(f"  {'Train':<10} {len(train):>8}")
    print(f"  {'Val':<10} {len(val):>8}")
    print(f"  {'Test':<10} {len(test):>8}")
    print(f"  {'TOTAL':<10} {len(train)+len(val)+len(test):>8}")

    # Verify no rows are lost or duplicated.
    total_out = len(train) + len(val) + len(test)
    if total_out == total_input:
        print(f"\n  ✓ Row count verified: {total_out} in == {total_out} out")
    else:
        print(f"\n  [!] Row count mismatch: {total_input} in, {total_out} out")

    print(f"\n  Per-class breakdown:")
    train_c = count_per_class(train)
    val_c   = count_per_class(val)
    test_c  = count_per_class(test)

    print(f"\n  {'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print(f"  {'-' * 60}")
    for label in all_labels:
        t = train_c.get(label, 0)
        v = val_c.get(label, 0)
        s = test_c.get(label, 0)
        print(f"  {label:<35} {t:>6} {v:>6} {s:>6} {t+v+s:>7}")
    print()


def main() -> None:
    print("\n=== PlantVillage Subset — Train/Val/Test Split ===\n")
    print(f"  Seed        : {RANDOM_SEED}")
    print(f"  Train ratio : {TRAIN_RATIO:.0%}")
    print(f"  Val ratio   : {VAL_RATIO:.0%}")
    print(f"  Test ratio  : {TEST_RATIO:.0%}")
    print(f"  Input CSV   : {INPUT_CSV}\n")

    if not INPUT_CSV.exists():
        print(f"[!] Input file not found: {INPUT_CSV}")
        print("    Run build_subset_metadata.py first.")
        return

    fieldnames, rows = load_csv(INPUT_CSV)
    print(f"  Loaded {len(rows)} rows from plantvillage_subset_metadata.csv")

    train, val, test = stratified_split(
        rows=rows,
        label_key="label_id",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )

    print("\n============================================================")
    print("  Split Summary")
    print("============================================================")
    print_split_summary(train, val, test, total_input=len(rows))

    # Save split CSVs.
    save_csv(fieldnames, train, TRAIN_CSV)
    save_csv(fieldnames, val,   VAL_CSV)
    save_csv(fieldnames, test,  TEST_CSV)
    print(f"  Saved: {TRAIN_CSV}")
    print(f"  Saved: {VAL_CSV}")
    print(f"  Saved: {TEST_CSV}")

    # Save split settings.
    settings = {
        "random_seed":   RANDOM_SEED,
        "train_ratio":   TRAIN_RATIO,
        "val_ratio":     VAL_RATIO,
        "test_ratio":    TEST_RATIO,
        "total_images":  len(rows),
        "train_count":   len(train),
        "val_count":     len(val),
        "test_count":    len(test),
        "per_class_train": count_per_class(train),
        "per_class_val":   count_per_class(val),
        "per_class_test":  count_per_class(test),
        "note": (
            "PlantDoc is kept as a separate cross-dataset evaluation set "
            "and is not split here."
        ),
    }
    with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"  Saved: {SETTINGS_JSON}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
