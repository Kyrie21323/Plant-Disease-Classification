"""
inspect_data.py
Runnable script for basic dataset inspection and statistics.

Usage:
    Run from within WSL so that both the project code (via /mnt/...) and the
    WSL-hosted datasets are accessible in the same session:

        cd /mnt/c/Users/<your-username>/Documents/GitHub/Plant-Disease-Classification
        python src/inspect_data.py

Dataset path notes:
    - PlantDoc is stored in the WSL Linux filesystem because its filenames
      contain characters forbidden on Windows NTFS (?, %, +).
      Set PLANTDOC_DIR to the WSL path where you cloned it, e.g. ~/plantdoc.
    - PlantVillage may be stored in WSL for the same reason, or inside the
      repository data/ folder if its filenames are Windows-safe.
    - Do not hardcode personal usernames or machine-specific absolute paths
      in shared code. Update the constants below locally as needed.
"""

from pathlib import Path

from data_utils import list_class_folders, count_images_per_class

# ---------------------------------------------------------------------------
# Dataset root paths.
# Both datasets are stored in the WSL Linux filesystem.
# Path.home() resolves to the WSL home directory (e.g. /root) at runtime.
# Update these constants if your local paths differ.
# ---------------------------------------------------------------------------
PLANTVILLAGE_DIR = Path.home() / "plantvillage" / "plantvillage_dataset" / "color"
PLANTDOC_DIR     = Path.home() / "plantdoc" / "train"
# ---------------------------------------------------------------------------


def inspect_dataset(name: str, dataset_dir: Path) -> None:
    """
    Print class names, per-class image counts, and totals for one dataset.

    Args:
        name:        Human-readable label for the dataset (used in output).
        dataset_dir: Root directory where each subdirectory is one class.
    """
    print(f"{'=' * 60}")
    print(f"  Dataset : {name}")
    print(f"  Path    : {dataset_dir}")
    print(f"{'=' * 60}")

    if not dataset_dir.exists():
        print(f"  [!] Directory not found. Skipping.\n")
        return

    classes = list_class_folders(dataset_dir)
    if not classes:
        print(f"  [!] No class subfolders found in {dataset_dir}.\n")
        return

    counts = count_images_per_class(dataset_dir)
    total  = sum(counts.values())

    print(f"  Classes : {len(classes)}")
    print(f"  Total images : {total}\n")

    print(f"  {'Class':<50} {'Images':>7}")
    print(f"  {'-' * 58}")
    for cls in classes:
        print(f"  {cls:<50} {counts.get(cls, 0):>7}")

    print()


def main() -> None:
    """Inspect PlantVillage and PlantDoc and print basic statistics."""

    print("\n=== Plant Disease Classification - Dataset Inspection ===\n")

    inspect_dataset("PlantVillage (training / in-domain test)", PLANTVILLAGE_DIR)
    inspect_dataset("PlantDoc (cross-dataset generalization test)", PLANTDOC_DIR)

    print("Inspection complete.\n")


if __name__ == "__main__":
    main()
