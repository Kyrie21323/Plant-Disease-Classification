# Plant Disease Classification — Project plan (current status)

## A study of generalization, dataset shift, and possible shortcut learning

This document describes the **completed** end-to-end workflow: what was done, in what order, and
how each dataset split was used. The **code** may evolve; the **protocol** below is the one the
reported results follow.

**Interpretation:** Findings on PlantDoc are **consistent with** dataset shift and **possible**
shortcut learning. They do **not** prove that a specific cue (e.g. background only) caused
each error.

---

## 1. Project goal (completed)

- Train **plant disease** classifiers on **PlantVillage** and compare **in-domain** vs
  **external** performance on **PlantDoc**.
- Compare **BaselineCNN** (from scratch) vs **ResNet-18** (ImageNet-pretrained, fine-tuned on
  PlantVillage).
- Ask whether good PlantVillage scores **translate** to field-like images and whether large drops
  on PlantDoc are **compatible with** reliance on **dataset-specific** regularities
  (shortcut-style hypotheses), without claiming a single mechanistic proof per prediction.

---

## 2. Environment and data location (current)

| Item | Status |
| --- | --- |
| Project code | Git repository (Windows or any clone); run data scripts in **WSL** when using Linux-stored images. |
| **PlantVillage** | WSL: `~/plantvillage/plantvillage_dataset/color` (or equivalent; metadata CSVs point to absolute paths used at build time). |
| **PlantDoc** | WSL: `~/plantdoc/train` (many raw filenames use characters **not** allowed on Windows NTFS; data lives on Linux). |

**Why WSL?** Web-scraped PlantDoc files retain characters valid on Linux but invalid on some
Windows paths. Storing data under WSL avoids renaming the corpus.

**Repository data products:** `data/metadata/*.csv` and `data/splits/*.csv` — generated from
the above trees, not a copy of the full image corpus inside the repo.

---

## 3. Experimental protocol (must stay consistent in write-ups)

| Stage | Data | Role |
| --- | --- | --- |
| **Training** | PlantVillage **train** | Learn parameters for BaselineCNN and ResNet-18. |
| **Tuning / model selection** | PlantVillage **validation** | **Only** this split was used to select BaselineCNN hyperparameters (numbered runs, best val loss). **PlantDoc not used.** |
| **In-domain test** | PlantVillage **test** | Final accuracy / F1 on held-out **same-distribution** images. **Not** used to pick hyperparameters. |
| **External evaluation** | **PlantDoc** (full V1 metadata) | **After** all selection: one-shot **generalization** and **shortcut-style** **analysis**. **Not** used for tuning. |

**PlantDoc** is a **final** stress test, not a validation set for model selection.

---

## 4. Agenda actually executed (chronological, completed)

1. **Setup & class subset** — Download / place datasets in WSL; build **8-class** V1 subset with
   unambiguous cross-dataset names; document in
   [FINAL_CLASS_SUBSET.md](FINAL_CLASS_SUBSET.md) and
   [CLASS_MAPPING.md](CLASS_MAPPING.md); config: [class_subset_v1.json](../configs/class_subset_v1.json).
2. **Metadata & splits** — `build_subset_metadata` → CSVs; `split_data` → 70/15/15 stratified
   **PlantVillage** train/val/test; PlantDoc as **separate** eval list (no split in `split_data`’s
   test role for PD training).
3. **EDA** — Distributions, sample grids, size stats → `outputs/` and
   [final_subset_eda_summary.md](../outputs/results/final_subset_eda_summary.md).
4. **Preprocessing** — `src/data/transforms.py`: train vs eval transforms; shared ImageNet
   normalization.
5. **Dataloaders** — `build_dataloaders()`: PV train/val/test + PlantDoc eval loader.
6. **Initial model training** — Baseline + ResNet training scripts, shared
   [trainer.py](../src/training/trainer.py).
7. **Baseline improvement experiments (PV val only)** — Numbered runs: e.g. lower LR + longer
   training, weight decay, dropout 0.3, (optional) strong ColorJitter run rejected. **Run3**
   selected: `outputs/checkpoints/baseline_cnn_best_run3.pt` (best **PV val loss**). ResNet
   **final** checkpoint: `resnet18_best.pt`.
8. **Final evaluation** — [evaluate_final.py](../src/training/evaluate_final.py): load **final**
   checkpoints, evaluate on **PV test** + **PlantDoc**; JSON + confusion matrices. **No
   training.**
9. **Shortcut / generalization analysis** — Compare gaps, read confusion matrices; document in
   [FINAL_ANALYSIS.md](../FINAL_ANALYSIS.md). **Framing: evidence, not proof** of a specific
   spurious feature per error.
10. **Final reporting** — [FINAL_ANALYSIS.md](../FINAL_ANALYSIS.md) as the long-form report;
    this file as **status/plan** reference; [README.md](../README.md) as the **public
    overview**; and the **notebook** [plant_disease_shift_report.ipynb](../notebooks/plant_disease_shift_report.ipynb)
    as a **runnable** walk-through of the same protocol and metrics using **saved** JSON/PNGs
    (by default, no retraining in the notebook).

**Notebook** — The report notebook in `notebooks/plant_disease_shift_report.ipynb` replaces
the older “data inspection” draft and mirrors the final write-ups. The **source of truth** for
numbers remains [FINAL_ANALYSIS.md](../FINAL_ANALYSIS.md) and `outputs/results/*.json`.

---

## 5. Final selected checkpoints and key hyperparameters (Baseline run3)

| Item | Value |
| --- | --- |
| **Checkpoint** | `outputs/checkpoints/baseline_cnn_best_run3.pt` |
| Learning rate | `3e-4` |
| Epochs | `25` |
| Batch size | `32` |
| Weight decay | `1e-4` |
| Dropout (head) | `0.3` |
| Augmentation (train) | Mild / “original” pipeline at time of selection (not the rejected strong ColorJitter experiment) |
| ResNet-18 | `outputs/checkpoints/resnet18_best.pt` |

---

## 6. Final aggregate results (from evaluation JSON; for convenience)

| Model | PV test acc | PlantDoc acc | Acc gap | PV F1 | PlantDoc F1 | F1 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline (run3)** | 0.9718 | 0.2330 | 0.7388 | 0.9698 | 0.1865 | 0.7833 |
| **ResNet-18** | 0.9989 | 0.4000 | 0.5989 | 0.9987 | 0.3202 | 0.6785 |

**Conclusion (high level):** ResNet-18 **transfers** better; **both** show a **large**
PlantVillage→PlantDoc gap. See [FINAL_ANALYSIS.md](../FINAL_ANALYSIS.md) for full narrative.

---

## 7. Future work (optional extensions)

- More field or mixed-domain **training** data.
- **Domain adaptation** or self-supervised pretraining on unlabeled field leaves.
- **Tighter** leaf crops or segmentation to **reduce** background.
- **Additional** external benchmarks beyond PlantDoc.
- V2 class expansion (candidates in [FINAL_CLASS_SUBSET.md](FINAL_CLASS_SUBSET.md)).

---

*Last aligned with: completed V1 pipeline, WSL paths above, and
[FINAL_ANALYSIS.md](../FINAL_ANALYSIS.md).*
