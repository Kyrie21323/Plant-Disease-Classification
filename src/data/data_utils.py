"""
data_utils.py
Utility functions for dataset discovery, inspection, and subset filtering.
"""

import json
from pathlib import Path

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def is_valid_image(path: Path) -> bool:
    """Return True if the file has a recognised image extension."""
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def list_class_folders(dataset_dir: str | Path) -> list[str]:
    """
    Return a sorted list of class folder names found directly inside
    dataset_dir. Only immediate subdirectories are considered.

    Args:
        dataset_dir: Path to the root dataset directory.

    Returns:
        Sorted list of class folder names.
    """
    dataset_dir = Path(dataset_dir)
    return sorted(
        entry.name
        for entry in dataset_dir.iterdir()
        if entry.is_dir()
    )


def count_images_per_class(dataset_dir: str | Path) -> dict[str, int]:
    """
    Count the number of valid image files in each class subfolder.

    Args:
        dataset_dir: Path to the root dataset directory.

    Returns:
        Dictionary mapping class name -> image count.
    """
    dataset_dir = Path(dataset_dir)
    counts: dict[str, int] = {}
    for class_name in list_class_folders(dataset_dir):
        class_path = dataset_dir / class_name
        counts[class_name] = sum(
            1 for f in class_path.iterdir() if f.is_file() and is_valid_image(f)
        )
    return counts


def collect_image_paths(dataset_dir: str | Path) -> list[Path]:
    """
    Recursively collect all valid image file paths under dataset_dir.

    Args:
        dataset_dir: Path to the root dataset directory.

    Returns:
        List of Path objects pointing to valid image files.
    """
    dataset_dir = Path(dataset_dir)
    return [
        f
        for f in dataset_dir.rglob("*")
        if f.is_file() and is_valid_image(f)
    ]


# ---------------------------------------------------------------------------
# Subset config loading
# ---------------------------------------------------------------------------

def load_class_subset(config_path: str | Path) -> list[dict]:
    """
    Load the class subset config from a JSON file.

    Args:
        config_path: Path to class_subset_v1.json or equivalent.

    Returns:
        List of class entry dicts, each with keys:
            label_id, unified_label, plant, disease,
            plantvillage_class, plantdoc_class
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Folder validation
# ---------------------------------------------------------------------------

def validate_class_folders(
    dataset_dir: str | Path,
    class_names: list[str],
) -> tuple[list[str], list[str]]:
    """
    Check which expected class folders exist and which are missing.

    Args:
        dataset_dir: Root directory of the dataset.
        class_names: Expected class folder names.

    Returns:
        (found, missing) — two lists of class names.
    """
    dataset_dir = Path(dataset_dir)
    found, missing = [], []
    for name in class_names:
        if (dataset_dir / name).is_dir():
            found.append(name)
        else:
            missing.append(name)
    return found, missing


# ---------------------------------------------------------------------------
# Image readability check
# ---------------------------------------------------------------------------

def try_open_image(path: Path) -> bool:
    """
    Attempt to open and verify an image file using Pillow.
    Returns True if the image is readable, False otherwise.

    Args:
        path: Path to the image file.

    Returns:
        True if readable, False if corrupt or unreadable.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Subset metadata collection
# ---------------------------------------------------------------------------

def collect_subset_metadata(
    dataset_dir: str | Path,
    dataset_key: str,
    class_entries: list[dict],
    check_readable: bool = True,
) -> list[dict]:
    """
    Collect image metadata for the selected class subset from one dataset.

    Each returned record contains:
        image_path, source_dataset, unified_label, label_id,
        plant, disease, raw_class_name, is_readable

    Args:
        dataset_dir:   Root directory of the dataset.
        dataset_key:   Either "plantvillage" or "plantdoc".
        class_entries: List of class dicts from load_class_subset().
        check_readable: If True, attempt to open each image to verify it.

    Returns:
        List of metadata dicts, one per image.
    """
    dataset_dir = Path(dataset_dir)
    records: list[dict] = []

    class_field = (
        "plantvillage_class" if dataset_key == "plantvillage" else "plantdoc_class"
    )

    for entry in class_entries:
        raw_class = entry[class_field]
        class_dir = dataset_dir / raw_class

        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file() or not is_valid_image(img_path):
                continue

            is_readable = try_open_image(img_path) if check_readable else None

            records.append({
                "image_path":     str(img_path),
                "source_dataset": dataset_key,
                "unified_label":  entry["unified_label"],
                "label_id":       entry["label_id"],
                "plant":          entry["plant"],
                "disease":        entry["disease"],
                "raw_class_name": raw_class,
                "is_readable":    is_readable,
            })

    return records
