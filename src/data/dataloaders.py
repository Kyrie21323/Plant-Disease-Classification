"""
dataloaders.py
DataLoader factory functions for Plant Disease Classification.

Build order:
    1. build_plantvillage_datasets() - train / val / test from split CSVs
    2. build_plantdoc_dataset()      - full PlantDoc metadata as eval set
    3. build_dataloaders()           - wraps all four datasets into DataLoaders

Transform assignment:
    - PlantVillage train : get_train_transform()   (augmented)
    - PlantVillage val   : get_eval_transform()    (deterministic)
    - PlantVillage test  : get_eval_transform()    (deterministic)
    - PlantDoc eval      : get_eval_transform()    (deterministic)

Using the same eval transform for all non-training sets ensures that
validation, in-domain test, and cross-dataset results are directly comparable
regardless of which model is being evaluated.
"""

import json
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from datasets import PlantDiseaseDataset
from transforms import get_train_transform, get_eval_transform

# ---------------------------------------------------------------------------
# Default paths - all relative to the repo root via /mnt/host/c/...
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification")

PV_TRAIN_CSV  = REPO_ROOT / "data" / "splits"   / "plantvillage_train_split.csv"
PV_VAL_CSV    = REPO_ROOT / "data" / "splits"   / "plantvillage_val_split.csv"
PV_TEST_CSV   = REPO_ROOT / "data" / "splits"   / "plantvillage_test_split.csv"
PD_META_CSV   = REPO_ROOT / "data" / "metadata" / "plantdoc_subset_metadata.csv"
SUBSET_JSON   = REPO_ROOT / "configs"           / "class_subset_v1.json"

# ---------------------------------------------------------------------------
# Default dataloader settings
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE  = 32
DEFAULT_NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Named container for the four datasets
# ---------------------------------------------------------------------------

class AllDatasets(NamedTuple):
    pv_train: PlantDiseaseDataset
    pv_val:   PlantDiseaseDataset
    pv_test:  PlantDiseaseDataset
    pd_eval:  PlantDiseaseDataset


class AllDataLoaders(NamedTuple):
    pv_train: DataLoader
    pv_val:   DataLoader
    pv_test:  DataLoader
    pd_eval:  DataLoader


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_plantvillage_datasets(
    train_csv: Path = PV_TRAIN_CSV,
    val_csv:   Path = PV_VAL_CSV,
    test_csv:  Path = PV_TEST_CSV,
) -> tuple[PlantDiseaseDataset, PlantDiseaseDataset, PlantDiseaseDataset]:
    """
    Build PlantVillage train / val / test datasets.

    - Train dataset uses the augmented training transform.
    - Val and test datasets use the deterministic eval transform.
      This ensures that metric values are reproducible and directly
      comparable across different model runs.

    Args:
        train_csv: Path to the train split CSV.
        val_csv:   Path to the val split CSV.
        test_csv:  Path to the test split CSV.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_ds = PlantDiseaseDataset(train_csv, transform=get_train_transform())
    val_ds   = PlantDiseaseDataset(val_csv,   transform=get_eval_transform())
    test_ds  = PlantDiseaseDataset(test_csv,  transform=get_eval_transform())
    return train_ds, val_ds, test_ds


def build_plantdoc_dataset(
    meta_csv: Path = PD_META_CSV,
) -> PlantDiseaseDataset:
    """
    Build the PlantDoc cross-dataset evaluation set.

    Uses the deterministic eval transform - identical to PV val/test -
    so that in-domain vs cross-dataset performance comparisons are fair.

    Args:
        meta_csv: Path to the PlantDoc metadata CSV.

    Returns:
        PlantDoc dataset.
    """
    return PlantDiseaseDataset(meta_csv, transform=get_eval_transform())


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size:  int  = DEFAULT_BATCH_SIZE,
    num_workers: int  = DEFAULT_NUM_WORKERS,
    pin_memory:  bool = False,
    train_csv:   Path = PV_TRAIN_CSV,
    val_csv:     Path = PV_VAL_CSV,
    test_csv:    Path = PV_TEST_CSV,
    pd_csv:      Path = PD_META_CSV,
) -> tuple[AllDatasets, AllDataLoaders]:
    """
    Build all four datasets and their DataLoaders in one call.

    Args:
        batch_size:  Images per batch (same for all loaders).
        num_workers: Parallel workers for data loading.
        pin_memory:  Set True when training on GPU.
        train_csv:   Path to PV train split CSV.
        val_csv:     Path to PV val split CSV.
        test_csv:    Path to PV test split CSV.
        pd_csv:      Path to PlantDoc metadata CSV.

    Returns:
        (AllDatasets, AllDataLoaders) named tuples.
    """
    pv_train_ds, pv_val_ds, pv_test_ds = build_plantvillage_datasets(
        train_csv, val_csv, test_csv
    )
    pd_eval_ds = build_plantdoc_dataset(pd_csv)

    datasets = AllDatasets(
        pv_train=pv_train_ds,
        pv_val=pv_val_ds,
        pv_test=pv_test_ds,
        pd_eval=pd_eval_ds,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loaders = AllDataLoaders(
        pv_train=DataLoader(pv_train_ds, shuffle=True,  **common),
        pv_val=  DataLoader(pv_val_ds,   shuffle=False, **common),
        pv_test= DataLoader(pv_test_ds,  shuffle=False, **common),
        pd_eval= DataLoader(pd_eval_ds,  shuffle=False, **common),
    )

    return datasets, loaders


# ---------------------------------------------------------------------------
# Verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== DataLoader Verification ===\n")

    # Load expected class order from config.
    with open(SUBSET_JSON, encoding="utf-8") as f:
        subset = json.load(f)
    expected_labels = {e["label_id"]: e["unified_label"] for e in subset}
    print(f"Expected classes from config ({len(expected_labels)}):")
    for lid, lname in sorted(expected_labels.items()):
        print(f"  {lid}: {lname}")

    print("\nBuilding datasets and dataloaders...")
    datasets, loaders = build_dataloaders(
        batch_size=32,
        num_workers=0,   # 0 is safest for WSL / Alpine
        pin_memory=False,
    )

    print(f"\nDataset sizes:")
    print(f"  PlantVillage train : {len(datasets.pv_train):>6}")
    print(f"  PlantVillage val   : {len(datasets.pv_val):>6}")
    print(f"  PlantVillage test  : {len(datasets.pv_test):>6}")
    print(f"  PlantDoc eval      : {len(datasets.pd_eval):>6}")

    # Pull one batch from the train loader.
    print("\nFetching one batch from PlantVillage train loader...")
    images, labels, meta = next(iter(loaders.pv_train))

    print(f"\nBatch results:")
    print(f"  Image tensor shape : {list(images.shape)}")
    print(f"  Label tensor shape : {list(labels.shape)}")
    print(f"  Label min / max    : {labels.min().item()} / {labels.max().item()}")
    print(f"  Unique label ids   : {sorted(labels.unique().tolist())}")

    # Verify all label ids are within the expected 0–7 range.
    all_valid = all(0 <= lid <= 7 for lid in labels.tolist())
    print(f"  All label ids in 0–7 : {'✓ Yes' if all_valid else '[!] NO - unexpected ids found'}")

    # Verify observed labels are a subset of expected.
    observed = set(labels.unique().tolist())
    expected_set = set(expected_labels.keys())
    unexpected = observed - expected_set
    if unexpected:
        print(f"  [!] Unexpected label ids in batch: {unexpected}")
    else:
        print(f"  Label ids consistent with config : ✓ Yes")

    # Check image value range (should be roughly [-2.5, 2.5] after normalize).
    print(f"\n  Image tensor value range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Image dtype              : {images.dtype}")
    print(f"  Label dtype              : {labels.dtype}")

    print("\nVerification complete.\n")
