# Plant Disease Classification
## A Study of Shortcut Learning

This project uses deep learning to classify plant diseases from leaf images and examines whether models learn true disease patterns or shortcut cues such as background, lighting, and image style. Two models (a custom CNN and a fine-tuned pretrained model) are compared on in-domain and cross-dataset generalization.

## Dataset Location Note

Some datasets cannot be stored inside the repository's `data/` folder on Windows. The PlantDoc dataset contains filenames with characters (`?`, `%`, `+`) that are forbidden on Windows NTFS. Both datasets are stored in the WSL (Windows Subsystem for Linux) filesystem. Scripts that access them must be run from within WSL. Dataset paths are defined as configurable constants at the top of each script.

| Dataset | WSL Path |
|---|---|
| PlantVillage | `~/plantvillage/plantvillage_dataset/color` |
| PlantDoc | `~/plantdoc/train` |

## Repository Layout

```
configs/              Machine-readable configs (class subset, split settings)
data/
  metadata/           Filtered image metadata CSVs (one per dataset)
  splits/             PlantVillage train/val/test split CSVs
docs/                 Project planning and class mapping documents
notebooks/            Jupyter notebooks for reporting and EDA
outputs/
  checkpoints/        Saved model weights
  figures/            Saved plots and visualisations
  results/            Evaluation result CSVs and summaries
src/
  data/               Data pipeline scripts (metadata, splits, utils)
  models/             Model definitions (CNN, pretrained)
  training/           Training and evaluation loops
  utils/              Shared utilities (metrics, plotting)
```

## Running the Data Pipeline (from WSL)

Copy scripts from the Windows repo to WSL home, then run:

```bash
# 1. Build subset metadata CSVs
cp /mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/build_subset_metadata.py ~/
cp /mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/data_utils.py ~/
cp /mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/configs/class_subset_v1.json ~/
python ~/build_subset_metadata.py

# 2. Generate train/val/test splits
cp /mnt/host/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification/src/data/split_data.py ~/
python ~/split_data.py
```
