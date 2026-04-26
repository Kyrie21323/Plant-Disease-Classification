# Plant Disease Classification

## A Study of Generalization, Dataset Shift, and Possible Shortcut Learning

This repository trains and evaluates **plant disease** classifiers to answer more than in-domain
accuracy: **do models that perform well on clean PlantVillage images still work on real-world,
field-style images from a different source (PlantDoc)?**

The work compares a **custom CNN (BaselineCNN)** trained from scratch to a **fine-tuned
ResNet-18** (ImageNet pretrained) under the same PlantVillage splits, then measures
**out-of-distribution** behavior on PlantDoc.

Results are **consistent with** strong **dataset shift** and **possible** shortcut learning
(e.g. background, lighting, style)—this is **evidence and interpretation**, not proof of a
specific spurious cue for every error.

**Detailed write-up and figures:** [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)

---

## Dataset roles (experimental protocol)

| Data split / set | Use |
| --- | --- |
| **PlantVillage — train** | Supervised training for both models. |
| **PlantVillage — validation** | **Only** basis for **hyperparameter tuning and model selection** (Baseline CNN numbered runs: LR, weight decay, dropout, augmentation trials). ResNet-18 was trained with fixed initial settings. |
| **PlantVillage — test** | **Final in-domain** held-out evaluation **after** model selection. |
| **PlantDoc** | **After** all selection: **external** generalization and shortcut-learning **analysis only**. **Not** used for tuning, checkpoint selection, or label decisions. |

**Why this order:** If PlantDoc were used to pick hyperparameters, the external set would be
contaminated. Keeping PlantDoc for **one-shot** final evaluation gives a **cleaner** test of
whether PlantVillage-trained models transfer.

---

## Storing data (WSL)

Some files in PlantDoc use characters (`?`, `%`, `+`) that **Windows NTFS cannot store**.
Datasets are kept on the **WSL (Linux) filesystem**; run scripts that read image paths from
there **inside WSL**. Paths in code are configuration constants (see `src/data/dataloaders.py`
and related scripts).

| Dataset | WSL path (typical) |
| --- | --- |
| **PlantVillage** | `~/plantvillage/plantvillage_dataset/color` |
| **PlantDoc** | `~/plantdoc/train` |

The repository’s `data/metadata/` and `data/splits/` hold **CSV** metadata and splits
generated from these trees (see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)).

---

## Models compared

| Model | Description | Final checkpoint |
| --- | --- | --- |
| **BaselineCNN** | Small custom CNN, trained from scratch on PlantVillage. | `outputs/checkpoints/baseline_cnn_best_run3.pt` |
| **ResNet-18** | `torchvision` ResNet-18, ImageNet weights, 8-class head, full fine-tune on PlantVillage. | `outputs/checkpoints/resnet18_best.pt` |

**Selected BaselineCNN (run3)** was chosen by **lowest PlantVillage validation loss** among
tuning runs, using only PV val + training augmentations in `src/data/transforms.py`
(mild/“original” augmentation at selection time: no strong run4 `ColorJitter`).

- Learning rate: `3e-4`
- Epochs: `25`
- Batch size: `32`
- Weight decay: `1e-4`
- Dropout (classifier): `0.3`
- See [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) and `outputs/results/baseline_results_run3.json` for full metrics.

---

## Final aggregate results (from `src/training/evaluate_final.py` JSON outputs)

| Model | PV test acc | PV test macro-F1 | PlantDoc acc | PlantDoc macro-F1 | Acc gap* | F1 gap* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BaselineCNN (run3)** | 0.9718 | 0.9698 | 0.2330 | 0.1865 | 0.7388 | 0.7833 |
| **ResNet-18** | 0.9989 | 0.9987 | 0.4000 | 0.3202 | 0.5989 | 0.6785 |

\*Gap = PlantVillage test metric minus PlantDoc metric (larger = larger drop on PlantDoc).

**Takeaway:** ResNet-18 **transfers** better to PlantDoc (higher external accuracy / F1,
smaller gaps), but **both** models show a **large** generalization gap. Do **not** equate high
PlantVillage test scores with guaranteed field reliability.

**Sources:** `outputs/results/baseline_final_eval.json`, `outputs/results/resnet18_final_eval.json`, `outputs/results/final_comparison.json`

---

## How to run final evaluation (no training)

From WSL, at the **repository root** (adjust `/mnt/c/Users/...` to match your user):

```bash
cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
python src/training/evaluate_final.py
```

This loads the **final** checkpoints, evaluates on **PlantVillage test** and **PlantDoc**,
writes JSON and confusion-matrix figures under `outputs/`. It does **not** train or change
weights.

---

## Training (optional reference)

Tuning the Baseline uses numbered run outputs; see [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) Section 4.

```bash
cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
python src/training/train_baseline.py
python src/training/train_resnet18.py
```

Requires the same WSL dataset paths, Python env with PyTorch, etc.

**Regenerating metadata / splits (historical / advanced):** if you need to rebuild CSVs, run
the scripts from the repo in WSL with `PYTHONPATH` or `cd` to `src/data` as your workflow
requires—see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the completed pipeline. The
project no longer requires copying files to `~/` unless you prefer that layout.

---

## Class subset and documentation

- **8 classes (V1):** 11,819 PlantVillage images and 940 PlantDoc images in the subset (see
  [docs/FINAL_CLASS_SUBSET.md](docs/FINAL_CLASS_SUBSET.md) and
  [configs/class_subset_v1.json](configs/class_subset_v1.json)).
- **Broader name alignment:** [docs/CLASS_MAPPING.md](docs/CLASS_MAPPING.md) (all candidate
  overlaps).
- **Current status and completed steps:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

---

## Repository layout

```text
configs/              # Class subset V1, split settings, etc.
data/
  metadata/           # Filtered metadata CSVs for PV / PlantDoc subset
  splits/              # PlantVillage train / val / test CSVs
docs/                 # Planning, class mapping, final subset
notebooks/              # EDA and reporting notebooks
outputs/
  checkpoints/        # model .pt files
  figures/            # training curves, confusion matrices, EDA figures
  results/            # JSON and markdown result summaries
src/
  data/               # Datasets, dataloaders, transforms, split scripts
  models/             # BaselineCNN, ResNet-18
  training/            # train_*.py, trainer.py, evaluate_final.py
  utils/              # metrics, plotting
FINAL_ANALYSIS.md      # Step 15 final report (figures + interpretation)
```

---

## Dataset citations and licenses

This project uses PlantVillage and PlantDoc. Please see [DATASET_LICENSES.md](DATASET_LICENSES.md) for dataset citations, license notes, and attribution.

---

## Citation / academic use

If you use this work, cite the **datasets** (PlantVillage, PlantDoc) and describe the
**protocol** (PV val for selection, PlantDoc for external eval only) when reporting results.
