# Project Completion Checklist - Remaining Items Only

*This file tracks **only** what still matters for the **repository** as a self-contained
artifact. It does **not** list **presentation** items (e.g. opening framing, reflection,
batch-size talk track)-those are handled in the **slides / oral**. Low-priority polish (pinning
`requirements.txt`, exporting confusion matrices to CSV, Colab badge, ad-hoc checkpoint uploads)
is **out of scope** here unless a syllabus explicitly requires it. The **duplicate** notebook
was removed; the canonical report is `notebooks/plant_disease_shift_report.ipynb`.*

**Guardrails (unchanged in intent):** Standard **plant disease** classification on public
benchmarks, plus **data-centric** **generalization** and what is *consistent with* **possible**
shortcut learning-not a claim of full **field** deployment. **PlantVillage validation** = **only**
tuning/selection; **PlantDoc** = **final external** eval only. **Dataset** licenses and citations
live in [DATASET_LICENSES.md](DATASET_LICENSES.md). **JSON/PNGs** are tracked; **raw images** and
**checkpoints** are **not** in git (by design).

---

## 1. Final verdict

**Project repo complete; no blocking items remain** after the **Reproducibility / running the
project** section was added to [README.md](README.md) (see Task 2). Everything else in this
checklist is either de-scoped (presentation / optional polish) or **already done** (duplicate
notebook removed).

---

## 2. Final repo-level item (reproducibility) - **complete**

**Instruction:** One pass was made to ensure [README.md](README.md) explains how to reproduce
or **inspect** the work without overclaiming. The following are now covered in **README §
Reproducibility / running the project**:

- [x] **Clone the project** - README §1: `git clone` with GitHub Code URL, or `cd` to an
  existing tree under `/mnt/c/...` or `~/...`.
- [x] **Where to run (WSL2 Ubuntu + Bash)** and **venv** - README requires Linux/WSL; `apt install
  python3-venv`; for clones on `/mnt/c/` use venv under `$HOME/.venvs/...` (ext4) to avoid
  broken `pip` on NTFS; use `source .../bin/activate` and **`python -m pip`** (not system
  `pip3`); optional PyTorch note; training/eval can `source` project `.venv` or the same home
  venv if documented.
- [x] **Install dependencies** - `python -m pip install -r requirements.txt` after venv is
  active, from repo root in WSL.
- [x] **Download / layout / metadata pipeline** - README §3: official GitHub links for
  PlantVillage and PlantDoc, target `~/...` tree, `build_subset_metadata.py` and `split_data.py`
  from `src/data/`, and pointers to `DATASET_LICENSES` / `PROJECT_PLAN` / class-subset docs.
- [x] **Expected dataset locations** - PlantVillage `~/plantvillage/plantvillage_dataset/color`,
  PlantDoc `~/plantdoc/train`, with pointer to adjust `dataloaders.py` and `DATASET_LICENSES.md`.
- [x] **Raw datasets not in repo** - stated explicitly.
- [x] **Result JSONs and figures tracked** - stated with path pointers.
- [x] **Checkpoints not tracked** - stated; not implied to ship with the clone.
- [x] **If checkpoints are needed** - train with `train_*.py` and/or obtain copies per instructor
  / zip; `evaluate_final.py` **loads** local `.pt` files.
- [x] **Three reproducibility modes + checkpoint path** - README: Mode 1 saved-results; Mode 2
  **recommended** for exact final eval (WSL + data + CSVs + two `.pt` under
  `outputs/checkpoints/`, then `evaluate_final.py`); why **`.pt` files stay out of git**; why
  checkpoint eval is recommended vs mandatory retrain; **GitHub Release** asset
  [v1.0-checkpoints](https://github.com/Kyrie21323/Plant-Disease-Classification/releases/tag/v1.0-checkpoints)
  for **`checkpoints.zip`**; unzip + `cp` + `ls` for placing weights; **Packaging checkpoints for
  submission** (`checkpoint_release`, `checkpoints.zip`); Mode 3 full training + caveats; **checkpoint
  files** stay outside the main tree (release asset, not the clone); **Protocol** (PV val =
  selection, PlantDoc = external only) preserved.
- [x] **Checkpoint release / external weight sharing** - **Complete.** Release
  <https://github.com/Kyrie21323/Plant-Disease-Classification/releases/tag/v1.0-checkpoints>
  hosts `checkpoints.zip` (weights only, not raw datasets). Checkpoints **intentionally** not in
  git; no repo-level blocker.
- [x] **How to run final evaluation** - `python src/training/evaluate_final.py` with `cd` example
  and note that data + weights must exist first (does not imply you can re-evaluate with **no**
  checkpoints or **no** images); cross-link to **Mode 2**.
- [x] **How to open/run the report notebook** - path and behavior when data are missing
  (saved-results mode).
- [x] **What works without data/checkpoints** - read docs, JSON, figures, notebook in saved form.
- [x] **What requires** raw data and checkpoints - retraining; full `evaluate_final.py` run.
- [x] **Protocol** restated (PV val = selection; PlantDoc = external only).

**Status:** **No blocking repo-level checklist items remain.** **Checkpoint-based** final
evaluation is documented, with the **v1.0-checkpoints** GitHub Release as the **documented
download** for `checkpoints.zip` (release asset, not committed). Checkpoint files are
**intentionally** outside the default git tree. If a syllabus demands exact final-eval
reproduction, use that Release (or the equivalent your instructor provides). Optional narrative
polish is left to the presentation, not this file.

---

## 3. De-scoped (not repo to-do list items)

| Topic | Disposition |
| --- | --- |
| Reflection / “biggest challenge” | **Presentation** (not required in the repo) |
| Extra opening framing of the task | **Presentation**; README is already clear on goals |
| One-line batch-size rationale | **Presentation**; README states `32` as a fact |
| Pin requirements, export CMs, release checkpoints, Colab badge | **Disregarded for now** (not expected to affect grading) |
| Duplicate `01_data_inspection.ipynb` | **Removed** - fixed |

---

## 4. Already complete (short)

Code and docs for problem definition, protocol, EDA, models, training/tuning, final evaluation,
generalization / careful shortcut wording, tracked results, ignored data/checkpoints, and a single
report notebook-**no** further repo work required for those areas beyond what README now states.

---

*Presentation slides, oral Q&A, and LMS-specific uploads are out of scope for this checklist.*
