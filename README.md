# Plant Disease Classification

## A Study of Generalization, Dataset Shift, and Possible Shortcut Learning

This repository trains and evaluates **plant disease** classifiers to answer more than in-domain
accuracy: **do models that perform well on clean PlantVillage images still work on real-world,
field-style images from a different source (PlantDoc)?**

The work compares a **custom CNN (BaselineCNN)** trained from scratch to a **fine-tuned
ResNet-18** (ImageNet pretrained) under the same PlantVillage splits, then measures
**out-of-distribution** behavior on PlantDoc.

Results are **consistent with** strong **dataset shift** and **possible** shortcut learning
(e.g. background, lighting, style)-this is **evidence and interpretation**, not proof of a
specific spurious cue for every error.

**What to read**

| Document | What it is |
| --- | --- |
| **README** (this file) | Quick protocol, results table, and how to run evaluation |
| **[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)** | Full narrative, all figures, and interpretation |
| **[notebooks/plant_disease_shift_report.ipynb](notebooks/plant_disease_shift_report.ipynb)** | Notebook report: protocol, EDA, tuning table, and final metrics from **saved** JSON/PNGs (no training by default) |
| **[DATASET_LICENSES.md](DATASET_LICENSES.md)** | Dataset citations, CC BY 4.0 (PlantDoc) notes, and ownership |
| **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)** | End-to-end workflow and what was run in order |

---

## Reproducibility / running the project

1. **Install dependencies** (from the repository root, in a Python 3 environment with a suitable
   PyTorch install for your machine):

   ```bash
   pip install -r requirements.txt
   ```

2. **Raw image datasets** are **not** included in this repository. Download **PlantVillage** and
   **PlantDoc** from the original sources and follow
   [DATASET_LICENSES.md](DATASET_LICENSES.md). The training/evaluation code expects typical
   **WSL** paths as in the table under **Storing data (WSL)** below:
   - **PlantVillage:** `~/plantvillage/plantvillage_dataset/color`
   - **PlantDoc:** `~/plantdoc/train`  
   If your paths differ, adjust the constants in `src/data/dataloaders.py` (and rebuild
   `data/metadata/` and `data/splits/` if needed-see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)).

3. **Result JSONs and figure PNGs** in `outputs/results/` and `outputs/figures/` are **versioned
   in git** (see `.gitignore`) so you can read **final numbers** and static plots **without**
   training and **without** local image folders.

4. **Model checkpoints** (`outputs/checkpoints/*.pt`) are **intentionally not** tracked. To
   **re-run** `src/training/evaluate_final.py` on real data, you must have the final weight files
   locally (e.g. produce them with `src/training/train_baseline.py` and
   `src/training/train_resnet18.py`, or obtain copies from a zip/instructor if your course
   requires that). The evaluation script **loads** those `.pt` files; it does **not** download
   them for you.

5. **Run final evaluation** (only after **checkpoints** and **dataset paths** are in place, from
   the repo root-adjust the `cd` path to match your machine, e.g. WSL):

   ```bash
   cd /path/to/Plant-Disease-Classification
   python src/training/evaluate_final.py
   ```

6. **Report notebook:** open and run
   [notebooks/plant_disease_shift_report.ipynb](notebooks/plant_disease_shift_report.ipynb). With
   only a clone, you can execute it using **saved** JSON, figures, and `configs/`; cells that
   build DataLoaders **warn and skip** if split/metadata CSVs or images are missing (see
   notebook). That path does **not** replace full evaluation on disk.

7. **What you can do without raw images or local checkpoints:** Read this README and
   [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md), browse tracked **JSON** and **PNGs**, and use the
   notebook in **saved-results** mode to review the same tables and figures.

8. **What actually requires** both **raw image folders** the loaders can read, **and** local
   **`.pt` checkpoints** (and the usual `data/` CSV layout the project uses): **retraining** and a
   **full** run of `evaluate_final.py` (it loads saved weights, scores real images, and writes
   outputs under `outputs/`).

**Protocol (unchanged):** PlantVillage **validation** is the **only** basis for Baseline
tuning/selection; **PlantDoc** is for **final external** evaluation only-not for tuning. See
the table in **Dataset roles** below.

---

## Dataset roles (experimental protocol)

| Data split / set | Use |
| --- | --- |
| **PlantVillage - train** | Supervised training for both models. |
| **PlantVillage - validation** | **Only** basis for **hyperparameter tuning and model selection** (Baseline CNN numbered runs: LR, weight decay, dropout, augmentation trials). ResNet-18 was trained with fixed initial settings. |
| **PlantVillage - test** | **Final in-domain** held-out evaluation **after** model selection. |
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
requires-see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the completed pipeline. The
project no longer requires copying files to `~/` unless you prefer that layout.

---

## Class subset and documentation

- **8 classes (V1):** 11,819 PlantVillage images and 940 PlantDoc images in the subset (see
  [docs/FINAL_CLASS_SUBSET.md](docs/FINAL_CLASS_SUBSET.md) and
  [configs/class_subset_v1.json](configs/class_subset_v1.json)).
- **Broader name alignment:** [docs/CLASS_MAPPING.md](docs/CLASS_MAPPING.md) (all candidate
  overlaps).
- **Workflow and history:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) (chronology of the
  completed pipeline).

---

## Repository layout

At the **root** of the repository you will find the main **Markdown reports**; **code** lives
under `src/`. **Raw image datasets** are not stored in git (place them under WSL paths as
described above); **`data/`** only holds **generated** metadata and split CSVs when you build
them. **`outputs/`** contains checkpoints, training curves, and result JSON. Checkpoints
(`.pt`) are **not** version-controlled by default; `outputs/results/*.json` and
`outputs/figures/*.png` are tracked so the docs and GitHub can reference them (see
`.gitignore`).

```text
.
├── README.md                 # This overview
├── FINAL_ANALYSIS.md         # Long-form analysis + all figure links
├── DATASET_LICENSES.md        # Citations, licenses, usage table
├── requirements.txt
├── configs/                  # class_subset_v1.json, split_settings.json, …
│   ├── class_subset_v1.json
│   └── split_settings.json
├── data/                     # Built locally: CSV metadata + PV splits (not raw images)
│   ├── metadata/
│   └── splits/
├── docs/                      # Class mapping, subset rationale, project plan
│   ├── CLASS_MAPPING.md
│   ├── FINAL_CLASS_SUBSET.md
│   └── PROJECT_PLAN.md
├── notebooks/                 # Jupyter
│   └── plant_disease_shift_report.ipynb
├── outputs/                   # By-products of training and evaluation
│   ├── checkpoints/          # *.pt (ignored; keep local or LFS for sharing weights)
│   ├── figures/              # Curves, confusion matrices, EDA (PNGs; tracked in git)
│   └── results/              # JSON + optional EDA summary .md
└── src/                       # All Python for data, models, training, metrics
    ├── data/                 # Datasets, dataloaders, transforms, split & EDA scripts
    ├── models/               # cnn_baseline.py, resnet18_finetune.py
    ├── training/            # train_*.py, trainer.py, evaluate_final.py
    └── utils/                # metrics.py, plotting.py
```

**Where to look**

| Path | Role |
| --- | --- |
| `src/training/evaluate_final.py` | Final eval on **PV test** + **PlantDoc** (after training) |
| `src/data/dataloaders.py` | Loaders; override CSV paths in code if your data layout differs |
| `outputs/results/*_final_eval.json` | Step-13 aggregate numbers used in the README table |
| `notebooks/plant_disease_shift_report.ipynb` | Same story as the reports, in a runnable notebook form |

---

## Dataset citations and licenses

This project uses PlantVillage and PlantDoc. Please see [DATASET_LICENSES.md](DATASET_LICENSES.md) for dataset citations, license notes, and attribution.

---

## Citation / academic use

If you use this work, cite the **datasets** (PlantVillage, PlantDoc) and describe the
**protocol** (PV val for selection, PlantDoc for external eval only) when reporting results.
