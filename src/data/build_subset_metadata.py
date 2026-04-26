"""
build_subset_metadata.py
Build filtered metadata CSVs for the selected class subset.

Loads configs/class_subset_v1.json, collects image paths from both datasets
for only the selected 8 classes, checks readability, and saves one CSV per
dataset under data/metadata/.

Usage (from Ubuntu WSL):
    python /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/build_subset_metadata.py

Output CSVs are saved to:
    /mnt/c/.../Plant-Disease-Classification/data/metadata/plantvillage_subset_metadata.csv
    /mnt/c/.../Plant-Disease-Classification/data/metadata/plantdoc_subset_metadata.csv
"""

import csv
from pathlib import Path

from data_utils import (
    load_class_subset,
    validate_class_folders,
    collect_subset_metadata,
)

# ---------------------------------------------------------------------------
# Paths - update if your local layout differs.
# ---------------------------------------------------------------------------
PLANTVILLAGE_DIR = Path.home() / "plantvillage" / "plantvillage_dataset" / "color"
PLANTDOC_DIR     = Path.home() / "plantdoc" / "train"

# Repo root is accessible via /mnt/c/... in Ubuntu WSL.
REPO_ROOT     = Path("/mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification")
SUBSET_CONFIG = REPO_ROOT / "configs" / "class_subset_v1.json"
OUTPUT_DIR    = REPO_ROOT / "data" / "metadata"

# If the repo root is not accessible, fall back to home directory copies.
if not SUBSET_CONFIG.exists():
    SUBSET_CONFIG = Path.home() / "class_subset_v1.json"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path.home()
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "image_path",
    "source_dataset",
    "unified_label",
    "label_id",
    "plant",
    "disease",
    "raw_class_name",
    "is_readable",
]


def save_csv(records: list[dict], output_path: Path) -> None:
    """Write a list of metadata dicts to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def print_summary(
    name: str,
    dataset_dir: Path,
    records: list[dict],
    missing: list[str],
) -> None:
    """Print a concise per-dataset summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Path : {dataset_dir}")
    print(f"{'=' * 60}")

    if missing:
        print(f"  [!] Missing class folders ({len(missing)}):")
        for m in missing:
            print(f"      - {m}")

    if not records:
        print("  [!] No images collected.\n")
        return

    unreadable = [r for r in records if r["is_readable"] is False]
    if unreadable:
        print(f"  [!] Unreadable images: {len(unreadable)}")

    print(f"\n  Total images : {len(records)}")
    print(f"\n  {'Unified label':<35} {'Raw class':<45} {'Images':>6}")
    print(f"  {'-' * 88}")

    counts: dict[str, dict] = {}
    for r in records:
        key = r["unified_label"]
        if key not in counts:
            counts[key] = {"raw": r["raw_class_name"], "n": 0}
        counts[key]["n"] += 1

    for label, info in sorted(counts.items()):
        print(f"  {label:<35} {info['raw']:<45} {info['n']:>6}")

    print()


def main() -> None:
    print("\n=== Build Subset Metadata ===\n")

    if not SUBSET_CONFIG.exists():
        print(f"[!] class_subset_v1.json not found at: {SUBSET_CONFIG}")
        print("    Copy it to ~/class_subset_v1.json and retry.")
        return

    class_entries = load_class_subset(SUBSET_CONFIG)
    print(f"Loaded {len(class_entries)} classes from {SUBSET_CONFIG.name}")

    pv_class_names = [e["plantvillage_class"] for e in class_entries]
    pd_class_names = [e["plantdoc_class"]     for e in class_entries]

    datasets = [
        ("PlantVillage", "plantvillage", PLANTVILLAGE_DIR, pv_class_names),
        ("PlantDoc",     "plantdoc",     PLANTDOC_DIR,     pd_class_names),
    ]

    output_files = {
        "plantvillage": OUTPUT_DIR / "plantvillage_subset_metadata.csv",
        "plantdoc":     OUTPUT_DIR / "plantdoc_subset_metadata.csv",
    }

    for display_name, key, dataset_dir, class_names in datasets:
        found, missing = validate_class_folders(dataset_dir, class_names)

        print(f"\nCollecting images for {display_name} "
              f"({len(found)}/{len(class_names)} folders found)...")

        records = collect_subset_metadata(
            dataset_dir=dataset_dir,
            dataset_key=key,
            class_entries=class_entries,
            check_readable=True,
        )

        print_summary(display_name, dataset_dir, records, missing)

        out_path = output_files[key]
        save_csv(records, out_path)
        print(f"  Saved: {out_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
