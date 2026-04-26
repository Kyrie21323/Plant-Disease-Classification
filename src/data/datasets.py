"""
datasets.py
PyTorch Dataset classes for Plant Disease Classification.

Two dataset classes are provided:
- PlantDiseaseDataset: general-purpose, loads from any metadata/split CSV.
- Used for both PlantVillage splits and the PlantDoc evaluation set.

Each __getitem__ returns:
    image   : transformed PIL image (tensor if transform is applied)
    label   : integer label_id (0-7)
    meta    : dict with unified_label, source_dataset, image_path

Images are loaded as RGB via Pillow. If an image fails to load, a black
tensor of the correct shape is returned and a warning is printed.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    """
    Dataset for the 8-class plant disease subset.

    Loads image paths and labels from a pre-built CSV file
    (either a full metadata CSV or a split CSV). Applies an optional
    torchvision transform to each image.

    Args:
        csv_path:  Path to the CSV file (must contain image_path, label_id,
                   unified_label, source_dataset columns).
        transform: Optional torchvision transform to apply to each image.
                   If None, raw PIL Images are returned.
    """

    def __init__(self, csv_path: str | Path, transform=None) -> None:
        self.csv_path  = Path(csv_path)
        self.transform = transform
        self.records   = self._load_csv()

    def _load_csv(self) -> list[dict]:
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        row   = self.records[idx]
        path  = row["image_path"]
        label = int(row["label_id"])

        meta = {
            "unified_label":  row["unified_label"],
            "source_dataset": row["source_dataset"],
            "image_path":     path,
        }

        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, meta

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """
        Load an image as RGB. Returns a black 224×224 PIL image if loading fails.
        """
        try:
            img = Image.open(path).convert("RGB")
            img.load()
            return img
        except Exception as e:
            print(f"  [!] Failed to load image: {path} — {e}")
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def class_labels(self) -> list[str]:
        """Return a sorted list of unique unified_label values in this dataset."""
        return sorted({r["unified_label"] for r in self.records})

    def label_ids(self) -> list[int]:
        """Return a sorted list of unique label_id integers in this dataset."""
        return sorted({int(r["label_id"]) for r in self.records})
