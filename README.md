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

### Run everything in WSL2 Ubuntu (or another Linux), not in Windows PowerShell

This project was developed and run in **WSL2 with Ubuntu** on Windows, with Python and the
image datasets on the **Linux** side. You should do the same for **training, evaluation, and
pip-install**:

- All command lines below are **Bash** for **WSL2 Ubuntu** (or native Linux), not `cmd` or
  PowerShell.
- Open **Ubuntu** (or your WSL distro) from the Start menu, or run `wsl` from a Windows
  terminal, then work only in that environment for `python`, `pip`, and the scripts.
- The paths in `src/data/dataloaders.py` and the **Storing data (WSL)** table assume **Linux**
  paths. PlantDoc in particular is easier on the WSL ext4 filesystem; see that section for NTFS
  filename limits.
- Reading **Markdown / JSON** in a Windows editor is fine, but use **WSL** for executing the
  steps in this section.

### Three reproducibility modes (choose your path)

This project can be used in **three** ways. Pick the one that matches your goal.

#### Mode 1: Saved-results review (no raw data, no checkpoints)

- **You need:** this repository (clone) only.
- **Read:** this README, [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md), tracked
  [outputs/results/*.json](outputs/results/) and [outputs/figures/*.png](outputs/figures/).
- **Notebook:** open [notebooks/plant_disease_shift_report.ipynb](notebooks/plant_disease_shift_report.ipynb) in
  **saved-results** mode (saved JSON/PNGs; no local images or weights required by default). You
  cannot re-run a live `evaluate_final.py` from this mode alone.
- This is the lightest way to **inspect** the reported numbers, figures, and narrative.

#### Mode 2: Reproduce final evaluation *exactly* (recommended) - checkpoint-based

To match the **reported final PlantVillage test vs PlantDoc** metrics without retraining, use the
**same** `evaluate_final.py` path the project used, with the **final** model weights and the
**same** data layout (images + `data/` CSVs). **Do not** expect matching final-eval numbers
from retraining instead of loading these checkpoints, unless you accept small numerical drift
(Mode 3).

**Why checkpoints are not in Git**

- They are **large binary** model artifacts; keeping them in the main branch would **bloat** the
  repo and slow clones.
- **GitHub** repositories stay smaller and easier to share when code + tracked JSON/PNGs are
  separated from multi-megabyte **`.pt` files**.
- Checkpoints are **intentionally excluded** by [`.gitignore`](.gitignore) (and policy).
- If a course or reviewer requires **bit-for-bit** final-evaluation reproduction, the **two**
  checkpoint files should be **provided separately** (e.g. a **zip** download, a **release
  asset**, an **LMS** attachment, or a **shared drive** link) - not committed to the default git
  tree.

**Why checkpoint-based final evaluation is the recommended way to “re-run the numbers”**

- It does **not** force every user to **retrain** two models (expensive and often impractical on
  CPU-only or shared machines).
- **Training time** and **GPU** availability differ widely; retraining is not a fair bar for
  *verification* of the reported Step-13 table.
- **CUDA**, **PyTorch**, and **hardware** differ across systems; retraining can yield **small**
  numerical differences even with **fixed random seeds** (dataloader order, non-deterministic
  ops, etc.).
- With **fixed checkpoint weights** and **fixed** `data/metadata` + `data/splits` CSVs, running
  `src/training/evaluate_final.py` is the **cleanest** way to recompute the **same** final
  PlantVillage / PlantDoc aggregates the write-up is based on (subject to the same images and
  paths the loaders use).

**What you need for Mode 2**

- **Raw** PlantVillage and PlantDoc in the **expected WSL** paths (see **§3** and **Storing
  data (WSL)**).
- **Generated** `data/metadata/*.csv` and `data/splits/*.csv` (from **§3** or a matching build).
- The **final** checkpoint files (obtained separately from the git repo; names must match what
  the training/evaluation code expects), typically:
  - `outputs/checkpoints/baseline_cnn_best_run3.pt`
  - `outputs/checkpoints/resnet18_best.pt`

**Place provided checkpoints** (if you unzipped a release into `/path/to/checkpoint_release`):

```bash
cd /path/to/Plant-Disease-Classification
mkdir -p outputs/checkpoints
cp /path/to/checkpoint_release/baseline_cnn_best_run3.pt outputs/checkpoints/
cp /path/to/checkpoint_release/resnet18_best.pt outputs/checkpoints/
ls -lh outputs/checkpoints
source "$HOME/.venvs/plant-disease-classification/bin/activate"   # or your venv
python src/training/evaluate_final.py
```

(Adjust `cd`, venv, and `cp` source paths. **Protocol:** PlantDoc was used **only** for this
**final** external pass - **not** for training or hyperparameter choice; see **Dataset
roles**.)

**Packaging checkpoints for submission (project owner / maintainer)**

If you need to hand off weights for **Mode 2**, from a machine that already has the final
`outputs/checkpoints/` files, you can build a small zip (checkpoint files **only** - **no** raw
datasets inside the zip):

```bash
cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
mkdir -p checkpoint_release
cp outputs/checkpoints/baseline_cnn_best_run3.pt checkpoint_release/
cp outputs/checkpoints/resnet18_best.pt checkpoint_release/
zip -r checkpoints.zip checkpoint_release
ls -lh checkpoints.zip
```

- **Do not** add `checkpoints.zip` to git unless your instructor explicitly says to; upload it
  **separately** when exact final-evaluation reproduction is required.
- The zip should contain **only** the `.pt` files (or a single folder of them), **not** images.

#### Mode 3: Full training reproduction (retrain, then evaluate)

- **You need:** raw datasets, `data/` CSVs (see **§3**), the venv, and time (often **GPU**).
- **Run** (from the repo root, venv **active**; adjust venv path if needed):

  ```bash
  cd /path/to/Plant-Disease-Classification
  source "$HOME/.venvs/plant-disease-classification/bin/activate"
  python src/training/train_baseline.py
  python src/training/train_resnet18.py
  python src/training/evaluate_final.py
  ```

- **Expect:** long runtimes, hardware-dependent performance, and the possibility that final
  metrics **differ slightly** from the **tracked** JSON/PNGs (CUDA/PyTorch versions, ordering,
  float behavior). The project’s **published** headline numbers in git are the ones in
  `outputs/results/*.json` and the figures, unless you re-verify after your own run.

---

### 1. Clone this repository and `cd` into it (WSL)

**If you do not already have a local copy**, get the project with **git** in WSL (install `git`
in Ubuntu if needed: `sudo apt update && sudo apt install git`). Use the **HTTPS** or **SSH** URL
from the green **Code** button on the GitHub page for *this* project (or your fork if you
forked it). The canonical remote for this work is often:

```bash
cd ~
git clone https://github.com/Kyrie21323/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

If you use a different fork, clone that URL instead.

**If the repo is already on disk** (e.g. you unzipped a release or you keep the project on
**Windows** and open it from WSL), you only need to **`cd` to the project root** in a WSL
shell:

- From a clone on a Windows drive, the WSL path is usually under **`/mnt/c/...`**:
  ```bash
  cd /mnt/c/Users/YourName/Documents/GitHub/Plant-Disease-Classification
  ```
- If you used **`git clone` inside** WSL (Linux home only), a typical path is
  `~/Documents/Plant-Disease-Classification` or `~/Plant-Disease-Classification` - `cd` there.

All later commands in this section assume you are in the **repository root** (the directory that
contains `README.md` and `requirements.txt`).

### 2. Create a virtual environment, activate it, and install dependencies (recommended)

Do this **in the same WSL session** you use for training and evaluation. On **Ubuntu/Debian**,
install the pieces that ship a working `venv`+`pip` **once** (names may be `python3.12-venv` on
your distro):

```bash
sudo apt update
sudo apt install -y python3-venv
```

**Put the venv on Linux’s ext4 home, not on `/mnt/c/`.** If your clone lives on a Windows
drive (path like `/mnt/c/Users/...`), creating `python3 -m venv .venv` **inside** that tree can
leave `pip` **half-broken** on NTFS (you may see `No module named 'pip._internal...'` and odd
`pip` behavior). **Safer default flow** (WSL, venv in `~`, then install from the project root).
Start in your clone (or `cd` there first), then:

```bash
cd /path/to/Plant-Disease-Classification
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/plant-disease-classification"
source "$HOME/.venvs/plant-disease-classification/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`cd` should be the directory that contains `requirements.txt` (your clone). The prompt
should show the venv as active. **Use only the venv’s `python` + `python -m pip`**, never a
bare `pip`/`pip3` (that can hit **system** Python and trigger PEP 668
“externally-managed-environment” on Ubuntu 24.04+). In a **new terminal**, activate
again before any `python` / `pip` work:  
`source "$HOME/.venvs/plant-disease-classification/bin/activate"`.

**If the repo is already under `~` in WSL** (native ext4, not `/mnt/c/`), a project-local
`.venv` is usually fine:

```bash
cd /path/to/Plant-Disease-Classification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`requirements.txt` pulls in `torch` and `torchvision`. If you need a **CPU-only** or
**GPU-specific (CUDA)** build, follow the official **PyTorch** install selector at
[pytorch.org](https://pytorch.org) in **this** environment, then keep using the same
activated venv for all commands below. Jupyter (for the report notebook) is included in
`requirements.txt` once installed.

**If `pip` still errors:** delete the broken venv folder, confirm `apt install python3-venv`
succeeded, recreate the venv on **`$HOME/.venvs/...`**, and run **`python -m pip`** (not
`pip3` without the venv active).

### 3. Download raw image datasets, place them in WSL, and build `data/` CSVs

**Raw images are not in the repository.** You must **download** the corpora yourself, follow
[DATASET_LICENSES.md](DATASET_LICENSES.md) for **citation and license** text, and keep files on
the **WSL (Linux) filesystem** (see **Storing data (WSL)** for NTFS / filename issues).

**Target layout** (Bash, under your WSL home):

- **PlantVillage:** `~/plantvillage/plantvillage_dataset/color`
- **PlantDoc:** `~/plantdoc/train`

**Where to download**

| Dataset | Project role | Upstream (official) |
| --- | --- | --- |
| **PlantVillage** | Training, val selection, in-domain test | [github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Mohanty *et al.*, 2016) |
| **PlantDoc** | Final external / shift evaluation only (not for tuning) | [github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh *et al.*, 2020, CC BY 4.0) |

**Suggested steps (WSL, illustrative)** - adjust paths to match what you unzipped:

1. `mkdir -p ~/plantvillage ~/plantdoc`
2. **PlantVillage:** from the upstream repo, **clone** or use **Code > Download ZIP** on GitHub.
   Locate the `color` directory whose **immediate** subfolders are **per-class** folders (e.g.
   `Tomato___Early_blight`). The archive layout can vary (extra folders like `raw/` or
   `segmented/`). The code expects the **single** `color` level that directly contains those
   classes. Create `~/plantvillage/plantvillage_dataset/` and **copy** or **symlink** that
   `color` tree so that `ls ~/plantvillage/plantvillage_dataset/color` lists class
   subdirectories.
3. **PlantDoc:** from the upstream repo, clone or download ZIP, then make **training** images
   available as `~/plantdoc/train` (per-class subfolders), e.g. with a symlink if the download
   lives elsewhere: `ln -s /path/to/.../train ~/plantdoc/train`
4. If you use different absolute paths, set `PLANTVILLAGE_DIR` / `PLANTDOC_DIR` in
   `src/data/build_subset_metadata.py` and the matching settings in
   `src/data/dataloaders.py` (and rebuild metadata/splits after any change).

**“Data cleaning / prep” in this project (not manual image retouching)**

The **8-class** label list, cross-dataset name alignment, and **CSV** metadata (then the
**70/15/15** PlantVillage split) are produced by the scripts below; see
[docs/FINAL_CLASS_SUBSET.md](docs/FINAL_CLASS_SUBSET.md) and
[docs/CLASS_MAPPING.md](docs/CLASS_MAPPING.md) for what was chosen and
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full order of operations.

1. If loaders cannot find your `data/splits` and `data/metadata` CSVs, set **`REPO_ROOT`** in
   `src/data/dataloaders.py` to this clone’s path. The **`build_subset_metadata.py`** and
   **`split_data.py`** scripts resolve the repo root from the script file location; run them from
   the same clone after building datasets (see their docstrings).
2. With the venv **activated**:
   ```bash
   cd /path/to/Plant-Disease-Classification/src/data
   source ../../.venv/bin/activate
   python build_subset_metadata.py
   python split_data.py
   ```
   Run from **`src/data/`** so `data_utils` imports match the scripts’ module layout. Check each
   script’s module docstring if anything fails. Outputs land under `data/metadata/` and
   `data/splits/`.

**Tracked results vs local checkpoints:** `outputs/results/*.json` and `outputs/figures/*.png`
are **in git** for reading without training. **`outputs/checkpoints/*.pt` are not**; see
**Three reproducibility modes** above (Modes 1-3).

### 4. Report notebook (Jupyter; overlaps Mode 1)

With the venv **activated** when you use live kernels, or with the right interpreter, open
[notebooks/plant_disease_shift_report.ipynb](notebooks/plant_disease_shift_report.ipynb). You can
re-run on **saved** JSON, figures, and `configs/`; cells that build DataLoaders **warn and
skip** if split/metadata CSVs or images are missing. For a full **on-disk** final evaluation,
use **Mode 2** and `evaluate_final.py`.

### 5. Protocol note

**Unchanged protocol:** **PlantVillage validation** is the **only** basis for Baseline
tuning/selection. **ResNet-18** used fixed initial settings. **PlantDoc** is for **final
external** evaluation **only** - it was **not** used to train, tune, or select checkpoints. See
**Dataset roles** below.

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

This is **Mode 2** in **Three reproducibility modes** (checkpoint-based final evaluation). You
need local **checkpoints**, **raw** images in the expected WSL paths, and built **`data/`
CSVs** - see that section and **Storing data (WSL)**.

From **WSL2 Ubuntu** (Bash), at the **repository root** (adjust `/mnt/c/Users/...` to match
your user). **Activate the venv first** if you use one (see **Reproducibility** above):

```bash
cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
source .venv/bin/activate
python src/training/evaluate_final.py
```

This loads the **final** checkpoints, evaluates on **PlantVillage test** and **PlantDoc**,
writes JSON and confusion-matrix figures under `outputs/`. It does **not** train or change
weights. **Protocol:** PlantDoc is **only** used here as the external eval set, **not** for
tuning.

---

## Training (optional reference)

This corresponds to **Mode 3** in **Three reproducibility modes** (full training
reproduction). Tuning the Baseline uses numbered run outputs; see
[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) Section 4. **Selection** uses **PlantVillage validation
only** - **not** PlantDoc.

```bash
cd /mnt/c/Users/2028e/Documents/GitHub/Plant-Disease-Classification
source .venv/bin/activate
python src/training/train_baseline.py
python src/training/train_resnet18.py
```

Requires the same WSL paths for datasets, an **activated** venv (or equivalent) with
PyTorch installed, etc. For exact headline numbers, prefer **Mode 2** (checkpoint
evaluation) after obtaining or generating the final `.pt` files.

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
