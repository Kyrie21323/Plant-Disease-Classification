# Plant Disease Classification

## A Study of Generalization, Dataset Shift, and Possible Shortcut Learning

This repository trains and evaluates **plant disease** classifiers to answer more than in-domain
accuracy: **do models that perform well on clean PlantVillage images still work on real-world,
field-style images from a different source (PlantDoc)?**

The work compares a **custom CNN (BaselineCNN)** trained from scratch to a **fine-tuned
ResNet-18** (ImageNet pretrained) under the same PlantVillage splits, then measures
**out-of-distribution** behavior on PlantDoc.

Results are **consistent with** strong **dataset shift** and **possible** shortcut learning
(e.g. background, lighting, style). This is **evidence and interpretation**, not proof of a
specific spurious cue for every error.

**What to read**

| Document | What it is |
| --- | --- |
| **README** (this file) | Quick protocol, results table, and how to run evaluation |
| **[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)** | Long-form report: narrative, all figures, and interpretation (aligned with `outputs/results/`) |
| **[outputs/results/*.json](outputs/results/)** | Tracked metrics (final eval, per-run training, comparison JSON) |
| **[outputs/figures/*.png](outputs/figures/)** | Tracked curves, confusion matrices, and EDA figures |
| **Python in [`src/`](src/)** | Data prep, training, and `evaluate_final.py` for reproducible runs |
| **[DATASET_LICENSES.md](DATASET_LICENSES.md)** | Dataset citations, CC BY 4.0 (PlantDoc) notes, and ownership |

---

## Reproducibility / running the project

### Environment

Run everything in **WSL2** or **native Linux** using **Bash** (not PowerShell). Use Linux paths for
data; see **Storing data (WSL)** for NTFS.

**Not in this repo:** raw images, generated `data/*.csv` ([`.gitignore`](.gitignore)), and `.pt`
checkpoints. **Clone + `pip` alone** does not reproduce evaluation—you need **§3** and **§4**.

**Recommended way to match the reported final evaluation numbers:** **Mode 2** (checkpoint-based
`evaluate_final.py`), not full retraining—see **Training scripts and environment-dependent
retraining** below.

**Usual “recompute final eval” order:** §1 → §2 → §3 → §4 → §5. To **read** numbers only: **Mode 1**
+ §6.

### Two reproducibility modes

#### Mode 1: Saved-results review

- **You need:** this repo (clone) only—no raw images, no local `.pt` files.
- **Not in git:** `data/metadata/*.csv` and `data/splits/*.csv` (ignored; see `.gitignore`). A
  clone has `data/.gitkeep` only.
- **Read:** this README, [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md), [outputs/results/](outputs/results/),
  [outputs/figures/](outputs/figures/) (inspect published metrics and figures; no recompute on disk).

#### Mode 2: Final evaluation reproduction with provided checkpoints

- **You need:** **§3** (images in the expected paths + built `data/` CSVs) and **§4** (the two
  checkpoint files below). Then **§5** runs `src/training/evaluate_final.py` with those weights.
- **On-disk names** (from the Release or your own run placed to match the scripts):

`outputs/checkpoints/baseline_cnn_best_run3.pt` · `outputs/checkpoints/resnet18_best.pt`

**Weights:** [`.gitignore`](.gitignore) excludes `*.pt`. Get
**[checkpoints.zip](https://github.com/Kyrie21323/Plant-Disease-Classification/releases/download/v1.0-checkpoints/checkpoints.zip)**
([Release v1.0-checkpoints](https://github.com/Kyrie21323/Plant-Disease-Classification/releases/tag/v1.0-checkpoints); weights only), then **§4**.

**Maintainer — zip to share** (`.pt` only):

```bash
cd /path/to/Plant-Disease-Classification
mkdir -p checkpoint_release
cp outputs/checkpoints/baseline_cnn_best_run3.pt outputs/checkpoints/resnet18_best.pt checkpoint_release/
zip -r checkpoints.zip checkpoint_release
```

Do not commit the zip; use a Release or separate upload.

### Training scripts and environment-dependent retraining

Training code lives under [`src/training/`](src/training/) (e.g. `train_baseline.py`, `train_resnet18.py`,
`trainer.py`, `evaluate_final.py`). The **original** training was run in the **author’s local**
environment (hardware, PyTorch/CUDA, memory, dataloader `num_workers`, wall time).

**On another machine, retraining may require** editing batch size, workers, or other settings to
fit CPU/GPU memory, available CUDA, and runtime limits. Even with **fixed random seeds**,
reported metrics can **differ slightly** from the tracked JSON (different BLAS/CUDA, nondeterministic
ops, dataloader order).

**Therefore:** to **reproduce the published final evaluation numbers** in this repo, use **Mode
2** (Release checkpoints + **§3–§5**). The training scripts remain useful for **transparency**,
**extensions**, and **future** experiments—not as the primary reproducibility path.

**Optional, environment-dependent** (activate your venv from **§2**; adjust as needed for your box):

```bash
cd /path/to/Plant-Disease-Classification
source "$HOME/.venvs/plant-disease-classification/bin/activate"
python src/training/train_baseline.py
python src/training/train_resnet18.py
python src/training/evaluate_final.py
```

---

### 1. Clone and `cd` to the repo root

```bash
cd ~
git clone https://github.com/Kyrie21323/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

Use your fork’s URL if needed. On Windows + WSL the repo is often under `/mnt/c/...`; if you
cloned inside Linux home, `cd ~/Plant-Disease-Classification` (or similar). Following steps use
**repository root** = directory with `README.md` and `requirements.txt`.

### 2. Virtual environment and `requirements.txt`

```bash
sudo apt update
sudo apt install -y python3-venv
```

**WSL + clone on `/mnt/c/...`:** prefer a venv under **`$HOME`**, not `.venv` on NTFS (avoids
broken `pip`). **Clone under `~` in WSL:** `python3 -m venv .venv` in the repo is usually fine.

```bash
cd /path/to/Plant-Disease-Classification
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/plant-disease-classification"
source "$HOME/.venvs/plant-disease-classification/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Use **`python -m pip`** from the venv only (avoids system Python / PEP 668 on Ubuntu 24+). New
shell: `source "$HOME/.venvs/plant-disease-classification/bin/activate"` again.

**Local `.venv` in repo** (when clone is on ext4):

```bash
cd /path/to/Plant-Disease-Classification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For CPU- vs GPU-specific **torch**, use [pytorch.org](https://pytorch.org) in this venv. If
`pip` keeps failing, delete the venv, fix `apt install python3-venv`, and recreate.

### 3. Raw datasets (WSL/Linux) and build `data/` CSVs

Images are **not** in this repo. Fetch them from **upstream**; cite and comply with
[DATASET_LICENSES.md](DATASET_LICENSES.md). This project **does not** ship or re-host the data.

| Path | Source |
| --- | --- |
| `~/plantvillage/plantvillage_dataset/color` | [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Mohanty *et al.*) — sparse checkout of **`raw/color`** only |
| `~/plantdoc/train` | [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh *et al.*) — full clone, symlink to `train` |

**Auto setup** — requires `git` (`sudo apt install -y git`). **Subversion: not used.**

**PlantVillage (sparse checkout — only `raw/color`)**

```bash
cd ~

sudo apt update
sudo apt install -y git

rm -rf ~/plantvillage
mkdir -p ~/plantvillage

git clone --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git ~/plantvillage/PlantVillage-Dataset
cd ~/plantvillage/PlantVillage-Dataset
git sparse-checkout set raw/color

mkdir -p ~/plantvillage/plantvillage_dataset
ln -sfn ~/plantvillage/PlantVillage-Dataset/raw/color ~/plantvillage/plantvillage_dataset/color

ls ~/plantvillage/plantvillage_dataset/color | head
```

**PlantDoc (full clone + symlink to `train`)**

```bash
cd ~

rm -rf ~/plantdoc
mkdir -p ~/plantdoc

git clone https://github.com/pratikkayal/PlantDoc-Dataset.git ~/plantdoc/PlantDoc-Dataset
ln -sfn ~/plantdoc/PlantDoc-Dataset/train ~/plantdoc/train

ls ~/plantdoc/train | head
```

**Sanity check:** PlantVillage folders look like `Tomato___Early_blight`; PlantDoc like
`Tomato Early blight leaf`. Elsewhere on disk: copy or symlink to these paths, or set
`PLANTVILLAGE_DIR` / `PLANTDOC_DIR` in `src/data/build_subset_metadata.py` and rebuild.

**Build metadata + 70/15/15 split** (class list: [docs/FINAL_CLASS_SUBSET.md](docs/FINAL_CLASS_SUBSET.md)):

```bash
cd /path/to/Plant-Disease-Classification/src/data
source "$HOME/.venvs/plant-disease-classification/bin/activate"   # or: source ../../.venv/bin/activate
python build_subset_metadata.py
python split_data.py
```

Writes `data/metadata/*.csv` and `data/splits/*.csv`. Without real images, CSVs are empty;
training/eval will fail. Checkpoints: **§4** before **§5**.

### 4. Checkpoints → `outputs/checkpoints/`

Not in git. Download
**[checkpoints.zip](https://github.com/Kyrie21323/Plant-Disease-Classification/releases/download/v1.0-checkpoints/checkpoints.zip)**
(weights only; [Release v1.0-checkpoints](https://github.com/Kyrie21323/Plant-Disease-Classification/releases/tag/v1.0-checkpoints)).
Unzip so you have:

- `outputs/checkpoints/baseline_cnn_best_run3.pt`
- `outputs/checkpoints/resnet18_best.pt`

If files land in `$HOME` or `$HOME/checkpoint_release/`, copy them into `outputs/checkpoints/`
(or use `find "$HOME" -maxdepth 3 -name "*.pt"`).

**From repo root:**

```bash
cd /path/to/Plant-Disease-Classification

wget -O "$HOME/checkpoints.zip" "https://github.com/Kyrie21323/Plant-Disease-Classification/releases/download/v1.0-checkpoints/checkpoints.zip"
unzip -o "$HOME/checkpoints.zip" -d "$HOME"

mkdir -p outputs/checkpoints

# Case A: zip extracted .pt files directly into $HOME
if [ -f "$HOME/baseline_cnn_best_run3.pt" ] && [ -f "$HOME/resnet18_best.pt" ]; then
    cp "$HOME/baseline_cnn_best_run3.pt" outputs/checkpoints/
    cp "$HOME/resnet18_best.pt" outputs/checkpoints/

# Case B: zip extracted into $HOME/checkpoint_release/
elif [ -f "$HOME/checkpoint_release/baseline_cnn_best_run3.pt" ] && [ -f "$HOME/checkpoint_release/resnet18_best.pt" ]; then
    cp "$HOME/checkpoint_release/baseline_cnn_best_run3.pt" outputs/checkpoints/
    cp "$HOME/checkpoint_release/resnet18_best.pt" outputs/checkpoints/

else
    echo "Could not find checkpoint files after unzipping."
    echo "Run: find \"$HOME\" -maxdepth 3 -name '*.pt'"
fi

ls -lh outputs/checkpoints
```

(`sudo apt install -y wget unzip` or use a browser to fetch the zip.)

### 5. Final evaluation (no retraining)

Needs **§3** + **§4**. Loads the two saved weights; evaluates **PV test** and **PlantDoc** (external
set only; not used for training/tuning in this project).

```bash
cd /path/to/Plant-Disease-Classification
source "$HOME/.venvs/plant-disease-classification/bin/activate"   # or: source .venv/bin/activate
python src/training/evaluate_final.py
```

Approximate headline **accuracy** (see JSON under `outputs/results/` for exact runs):

| Model | PlantVillage test acc (approx.) | PlantDoc acc (approx.) |
| --- | ---: | ---: |
| **BaselineCNN (run3)** | 0.9718 | 0.2330 |
| **ResNet-18** | 0.9989 | 0.4000 |

### 6. Reading results without recompute

Narrative + figures: [**FINAL_ANALYSIS.md**](FINAL_ANALYSIS.md). Numbers/PNGs: [outputs/results/](outputs/results/),
[outputs/figures/](outputs/figures/), [configs/](configs/). Recomputing headline metrics on disk
needs §3–§5, not the JSON/PNGs alone.

### 7. Protocol (short)

**BaselineCNN:** run-sweep; **run3** chosen by best PlantVillage **val** (not PlantDoc). **ResNet-18:** fixed
hyperparameters, one training recipe. **PlantDoc:** external eval only—never for tuning. See
**Dataset roles** and **Models compared**.

---

## Dataset roles (experimental protocol)

| Data split / set | Use |
| --- | --- |
| **PlantVillage - train** | Supervised training for both models. |
| **PlantVillage - validation** | **BaselineCNN:** tuning + run choice (e.g. run3 by best val). **ResNet-18:** fixed recipe, no run-sweep like the baseline. |
| **PlantVillage - test** | **Final in-domain** held-out evaluation **after** model selection. |
| **PlantDoc** | **After** all selection: **external** generalization and shortcut-learning **analysis only**. **Not** used for tuning, checkpoint selection, or label decisions. |

**Why:** PlantDoc must stay clean for a one-shot OOD test—do not use it to tune.

---

## Storing data (WSL)

Some PlantDoc names use `?` `%` `+` (bad on **NTFS**). Keep data on **Linux** (WSL ext4), run readers in WSL.

| Dataset | Path |
| --- | --- |
| **PlantVillage** | `~/plantvillage/plantvillage_dataset/color` |
| **PlantDoc** | `~/plantdoc/train` |

`data/metadata/` and `data/splits/` are built in **§3** from these trees.

---

## Models compared

| Model | Description | Final checkpoint |
| --- | --- | --- |
| **BaselineCNN** | Small custom CNN, trained from scratch on PlantVillage. | `outputs/checkpoints/baseline_cnn_best_run3.pt` |
| **ResNet-18** | `torchvision` ResNet-18, ImageNet weights, 8-class head, full fine-tune on PlantVillage. | `outputs/checkpoints/resnet18_best.pt` |

**Baseline run3:** chosen by best PV val loss among runs. Hyperparams: LR `3e-4`, 25 epochs, batch 32,
weight decay `1e-4`, dropout `0.3` — [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md), `outputs/results/baseline_results_run3.json`.

---

## Final aggregate results (from `src/training/evaluate_final.py` JSON outputs)

| Model | PV test acc | PV test macro-F1 | PlantDoc acc | PlantDoc macro-F1 | Acc gap* | F1 gap* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BaselineCNN (run3)** | 0.9718 | 0.9698 | 0.2330 | 0.1865 | 0.7388 | 0.7833 |
| **ResNet-18** | 0.9989 | 0.9987 | 0.4000 | 0.3202 | 0.5989 | 0.6785 |

\*Gap = PV test minus PlantDoc (higher = bigger drop on PlantDoc).

**Takeaway:** ResNet-18 generalizes better to PlantDoc than the baseline; both show a large PV→PD gap. High PV test ≠ field reliability.

**JSON:** `outputs/results/*_final_eval.json`, `final_comparison.json`

---

## Optional: retrain + eval

To train from scratch then evaluate, use the **optional** block under **Training scripts and
environment-dependent retraining** (same `python` lines as there). Tuning/selection background:
[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) §4. To match **published** headline final-eval numbers, prefer
**Mode 2** (§3 + §4 + §5) with the Release checkpoints.

---

## Class subset and documentation

- **8-class V1** (11,819 PV / 940 PD): [docs/FINAL_CLASS_SUBSET.md](docs/FINAL_CLASS_SUBSET.md), [configs/class_subset_v1.json](configs/class_subset_v1.json)
- **Name alignment:** [docs/CLASS_MAPPING.md](docs/CLASS_MAPPING.md)
- **Narrative:** [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)

---

## Repository layout

Code: `src/`. **Raw** images: not in git. **`data/`:** local CSVs from §3. **`outputs/`:** local
`checkpoints/`, `figures/`, `results/`; `.pt` gitignored, key JSON+PNGs often tracked
(see `.gitignore`).

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
├── docs/                      # Class mapping, subset rationale
│   ├── CLASS_MAPPING.md
│   └── FINAL_CLASS_SUBSET.md
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
| `outputs/results/*_final_eval.json` | Final-evaluation JSON aggregates (same numbers as the summary table in this file) |
| `FINAL_ANALYSIS.md` | Long-form report (narrative + figure references) |

---

## Dataset citations and licenses

This project uses PlantVillage and PlantDoc. Please see [DATASET_LICENSES.md](DATASET_LICENSES.md) for dataset citations, license notes, and attribution.

---

## Citation / academic use

If you use this work, cite the **datasets** (PlantVillage, PlantDoc) and describe the
**protocol** (PV val for selection, PlantDoc for external eval only) when reporting results.
