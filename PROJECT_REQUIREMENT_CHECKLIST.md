# Project Requirement Checklist

*Audit date: based on a full read of the repository layout and `git ls-files` (tracked paths). This document does not modify code, results, or training artifacts. Course rubric point weights below follow the user-provided template; if your course sheet differs, adjust the table in §11.*

**Interpretation rules used here:** The project text correctly treats **domain shift** and **possible** shortcut learning as *consistent with* evidence, not as proof of a specific cue. **PlantDoc** is described as **not** used for tuning or model selection, consistent with the protocol in `README.md` and `FINAL_ANALYSIS.md`.

---

## 1. Overall readiness summary

| | |
| --- | --- |
| **Overall status** | **Almost ready** |
| **Main completed strengths** | Clear protocol and final metrics in `README.md` and `FINAL_ANALYSIS.md`; both models trained and compared; PlantDoc reserved for external eval; ablation on Baseline via numbered runs; `evaluate_final.py` and full JSON/figure suite tracked; dataset licenses in `DATASET_LICENSES.md`; two report-style notebooks in git; source code for data, models, and training. |
| **Main remaining concerns** | **Presentation slides** are not present in the repository (no `.pptx`/`.pdf`/`.key` found). **`requirements.txt`** has no version pins (reproducibility risk). **Raw images** and **checkpoints** are intentionally absent from git—fine for policy, but graders must be told to obtain data locally and/or expect separate artifact upload for weights. A dedicated **“reflection / biggest challenge”** write-up is only partially implicit (could be expanded for presentation). |

---

## 2. Core requirements checklist

| Requirement | Status | Evidence in repo | Notes |
| --- | --- | --- | --- |
| Deep learning project | **Complete** | `src/models/*.py`, `src/training/*.py`, `torch` in `requirements.txt` | Image classification with CNNs |
| PyTorch implementation | **Complete** | `cnn_baseline.py`, `resnet18_finetune.py`, `trainer.py` | — |
| Public dataset usage | **Complete** | `DATASET_LICENSES.md`, `README.md`, Mohanty *et al.*, Singh *et al.* | PlantVillage + PlantDoc cited |
| Appropriate dataset size | **Complete** | `docs/FINAL_CLASS_SUBSET.md`, `README.md` (11,819 / 940 etc.) | 8-class V1 subset documented |
| Clear problem definition | **Complete** | `README.md`, `FINAL_ANALYSIS.md` §1 | PV → PlantDoc generalization |
| Data preprocessing | **Complete** | `src/data/transforms.py`, `dataloaders.py` | ImageNet norm, eval/train split |
| EDA | **Mostly complete** | `outputs/figures/*distribution*`, `*sample_grid*`, `final_subset_eda_summary.md` (tracked) | EDA in notebook + report |
| Model architecture | **Complete** | `BaselineCNN`, `build_resnet18_finetune`, docs | Two architectures |
| Training with validation | **Complete** | `trainer.py`, `train_*.json` with per-epoch val metrics | Val-based selection for Baseline |
| Model improvement / tuning | **Complete** | `baseline_results_run1..4.json`, `FINAL_ANALYSIS` §4 | Run3 selected; run4 rejected |
| Final test evaluation | **Complete** | `evaluate_final.py`, `*_final_eval.json` | PV test + PlantDoc |
| Generalization analysis | **Complete** | `FINAL_ANALYSIS` §7–8, `final_comparison.json` | Gaps + careful wording |
| Metric justification | **Mostly complete** | `README`, `FINAL_ANALYSIS` acc / macro-F1, gaps | Cross-entropy in eval JSON |
| Training/validation loss graphs | **Complete** | `outputs/figures/baseline_training_curves*.png`, `resnet18_training_curves.png` | Tracked in git |
| Source code | **Complete** | `src/data`, `src/models`, `src/training`, `src/utils` | All in `git ls-files` |
| Documentation | **Complete** | `README`, `FINAL_ANALYSIS`, `docs/*`, `DATASET_LICENSES` | — |
| Dependencies | **Mostly complete** | `requirements.txt` | Unpinned versions |
| Reproducibility instructions | **Mostly complete** | `README` (WSL paths, `evaluate_final` cmd) | Data + checkpoints not in repo by design |
| Fixed random seeds | **Mostly complete** | `SEED=42` in training scripts, notebook `set_seed` in setup cell | Documented in JSON |

---

## 3. Problem definition checklist

| Item | Status | Where it appears |
| --- | --- | --- |
| **Objective** | **Complete** | `README.md` (classify disease; compare models; OOD on PlantDoc) |
| **Motivation** | **Complete** | `README`, `FINAL_ANALYSIS` §1 (field vs clean images) |
| **Data type** | **Complete** | Images (RGB leaves), stated throughout |
| **Task type** | **Complete** | Multi-class (8) image classification |
| **Dataset size** | **Complete** | `README`, `docs/FINAL_CLASS_SUBSET.md` (11,819 / 940; split 8273/1773/1773) |
| **Number of classes** | **Complete** | 8, `configs/class_subset_v1.json` |
| **Example input / output** | **Partial** | “Input: 224×224 RGB” implied in code; class names in config—no single diagram of “one image → logit vector” in README, but `FINAL_ANALYSIS` and figures show outputs |
| **Why the problem matters** | **Complete** | Deployment / real-world vs lab imaging (`FINAL_ANALYSIS` §9) |
| **Why generalization matters** | **Complete** | Core narrative: PV test ≠ PlantDoc performance |

---

## 4. Data preprocessing and EDA checklist

| Item | Status | Evidence |
| --- | --- | --- |
| Dataset source descriptions | **Complete** | `README`, `DATASET_LICENSES`, `docs/PROJECT_PLAN` |
| Dataset links (repos / DOIs) | **Complete** | `DATASET_LICENSES.md` (GitHub + DOI) |
| Citation & license notes | **Complete** | `DATASET_LICENSES.md` (CC BY 4.0 for PlantDoc; PV “check source”) |
| Class mapping | **Complete** | `docs/CLASS_MAPPING.md` |
| Final class subset | **Complete** | `docs/FINAL_CLASS_SUBSET.md`, `class_subset_v1.json` |
| Selected class / counts | **Complete** | Subset doc + `README` |
| PlantVillage train/val/test sizes | **Complete** | `docs/FINAL_CLASS_SUBSET`, `README` (8273 / 1773 / 1773) |
| PlantDoc external size | **Complete** | 940 images (subset) in docs |
| Preprocessing / transforms | **Complete** | `src/data/transforms.py`, explained in `FINAL_ANALYSIS`, notebook |
| Normalization (ImageNet) | **Complete** | `transforms.py` constants |
| Training augmentation | **Complete** | `get_train_transform()`; run4 ColorJitter ablation in analysis |
| Evaluation transforms | **Complete** | `get_eval_transform()`; same for PV test and PlantDoc |
| Class distribution plot | **Complete** | `outputs/figures/final_subset_class_distribution.png` (tracked) |
| Sample image grids | **Complete** | `plantvillage_sample_grid.png`, `plantdoc_sample_grid.png` (tracked) |
| Style / domain comparison | **Complete** | `FINAL_ANALYSIS` §2.3, §5–6, notebook text |
| Notes on imbalance | **Partial** | Class distribution figure + counts; not a long formal imbalance “chapter” (acceptable for 8 classes) |
| **PlantVillage vs PlantDoc** visual difference | **Complete** | Grids + narrative in `FINAL_ANALYSIS` and notebook |

---

## 5. Model design checklist

| Item | Status | Evidence |
| --- | --- | --- |
| BaselineCNN architecture | **Complete** | `src/models/cnn_baseline.py`, `FINAL_ANALYSIS` §3.1, figures |
| ResNet-18 fine-tune | **Complete** | `resnet18_finetune.py`, `FINAL_ANALYSIS` §3.2 |
| Why CNNs / deep models | **Complete** | Implicit in task; both models image-classify leaves |
| Why from-scratch vs pretrained | **Complete** | `README`, `FINAL_ANALYSIS` §3.3 (transfer prior vs no ImageNet) |
| Loss function | **Complete** | Cross-entropy in `trainer` / eval (documented in analysis) |
| Metric choices (acc, macro-F1) | **Complete** | JSON + `README` table, gaps |
| Parameter count | **Partial** | Notebook can print param counts; not stressed in `README` (acceptable) |
| Architecture justification | **Mostly complete** | Docstrings in code + narrative in `FINAL_ANALYSIS` | 

---

## 6. Training checklist

| Item | Status | Evidence |
| --- | --- | --- |
| Baseline training script | **Complete** | `src/training/train_baseline.py` |
| ResNet-18 training script | **Complete** | `src/training/train_resnet18.py` |
| Optimizer | **Complete** | Adam in Baseline (per script); consistent with JSON metadata |
| Learning rate (final Baseline) | **Complete** | `3e-4` run3, in JSON + docs |
| Batch size | **Complete** | 32 in JSON and docs |
| Epochs (Baseline run3) | **Complete** | 25 |
| Dropout (head) | **Complete** | 0.3 run3 |
| Weight decay | **Complete** | `1e-4` run2+ |
| Validation loop | **Complete** | `trainer.py` `evaluate` |
| Checkpointing | **Complete** | Best val loss, paths in `README` / `FINAL_ANALYSIS` |
| Training curves (PNG) | **Complete** | `outputs/figures/*training_curves*` (tracked) |
| Validation (loss) on curves | **Complete** | Curves in JSON `per_epoch` and figures |
| Seed | **Complete** | 42 in training JSON, scripts |

*Note: ResNet-18 uses its own hyperparameters in `train_resnet18.py` (e.g. LR as per saved run); all documented in `resnet18_results.json` and narrative.*

---

## 7. Model improvement / ablation checklist

| Item | Status | Evidence |
| --- | --- | --- |
| Learning rate / longer training | **Complete** | Original vs run1 in `FINAL_ANALYSIS` §4, JSON |
| Weight decay | **Complete** | Run2 |
| Dropout | **Complete** | Run3 selected |
| Augmentation (strong jitter) | **Complete** | Run4 rejected; `transforms` |
| Batch size | **Partial** | Fixed 32; little “why not sweep batch” text—optional to add one sentence |
| Tuning summary table | **Complete** | `README` / `FINAL_ANALYSIS` + JSON |
| **Selection only on PV val** | **Complete** | Stated in `README`, `DATASET_LICENSES`, `FINAL_ANALYSIS` |
| **PlantDoc not used for tuning** | **Complete** | Same sources |
| Rejected settings explained | **Complete** | Run4 narrative |

---

## 8. Final evaluation checklist

| Item | Status | Evidence |
| --- | --- | --- |
| Final evaluation script | **Complete** | `src/training/evaluate_final.py` (in git) |
| Baseline PV test | **Complete** | `baseline_final_eval.json` (0.9718, F1 0.9698) |
| Baseline PlantDoc | **Complete** | Same file (0.2330, 0.1865) |
| ResNet-18 PV test | **Complete** | `resnet18_final_eval.json` |
| ResNet-18 PlantDoc | **Complete** | Same |
| Accuracy gap | **Complete** | `final_comparison.json`, tables in README/analysis |
| Macro-F1 gap | **Complete** | Same |
| Confusion matrices (Step 13) | **Complete** | `baseline_*`, `resnet18_*` PV + PlantDoc PNGs (tracked) |
| Final comparison table | **Complete** | README, `FINAL_ANALYSIS` §5.3, JSON |
| Generalization analysis | **Complete** | `FINAL_ANALYSIS` §7–8 |
| Shortcut-learning narrative | **Complete** | Careful “consistent with” / “possible” wording in §8 |
| Not proven | **Complete** | Explicit in `FINAL_ANALYSIS` §1, §8 |

---

## 9. Presentation content checklist

*Slides themselves are not in the repo; this table judges whether **material** exists to build each slide.*

| Slide topic | Content readiness | Primary sources |
| --- | --- | --- |
| **Problem** | **Complete** | `README` intro, `FINAL_ANALYSIS` §1 |
| **Dataset** | **Complete** | `DATASET_LICENSES`, subset docs, sample figures |
| **Preprocessing & EDA** | **Complete** | `transforms.py` description, EDA figures, notebook |
| **Model design** | **Complete** | `FINAL_ANALYSIS` §3, code docstrings |
| **Training** | **Complete** | Curves, JSON, `PROJECT_PLAN` |
| **Model improvement** | **Complete** | Tuning table, run4 story |
| **Final evaluation** | **Complete** | Tables, confusion matrices, gaps |
| **Conclusion** | **Complete** | `FINAL_ANALYSIS` §9, `README` takeaway |
| **Future work** | **Complete** | `FINAL_ANALYSIS` §9.3, `README`-style bullets |
| **Reflections / biggest challenge** | **Partial** | Implied in discussion of domain shift; **no** standalone “My biggest challenge was…” section in repo—add 1–2 oral/written lines for the rubric if required |
| **At least two optional components** | **Complete** | See **§12**; multiple are demonstrable from repo |

---

## 10. Deliverables checklist

| Deliverable | Status | Evidence / notes |
| --- | --- | --- |
| **Presentation slides** | **Missing (in repo)** | No `.pptx` / `.pdf` / `.key` found under the project root; prepare offline or upload to LMS separately |
| **Source code** | **Complete** | Entire `src/` tracked |
| **Jupyter notebook** | **Complete** | `notebooks/plant_disease_shift_report.ipynb` and `01_data_inspection.ipynb` both in `git ls-files` (parallel copies) |
| **Documentation** | **Complete** | `README`, `FINAL_ANALYSIS`, `docs/*`, `DATASET_LICENSES` |
| **Dependencies** | **Mostly complete** | `requirements.txt` present; **unpinned** |
| **Reproduction instructions** | **Mostly complete** | `README` (paths, `evaluate_final`); data/checkpoints not in git |
| **Dataset citations & licenses** | **Complete** | `DATASET_LICENSES.md` |
| **Results JSON + figures** | **Complete** (tracked) | `outputs/results/*.json`, `outputs/figures/*.png` in `git ls-files` |
| **Checkpoints** | **Intentionally not in git** | `.gitignore` has `outputs/checkpoints/**`—document in submission if course asks for weights separately |

*Raw image datasets: **not** tracked—`data/*` ignored except `.gitkeep` (correct policy).*

---

## 11. Rubric alignment

*Point values follow the user-supplied template; replace with the official course sheet if different.*

| Rubric category | Points | Status | Evidence | Risk / gap |
| --- | ---: | --- | --- | --- |
| Problem Definition & Motivation | 10 | **Strong** | `README` + `FINAL_ANALYSIS` | Low risk if slides restate the same story |
| Data Handling & EDA | 10 | **Strong** | Figures, subset docs, licenses | “Reflection” on data cleaning could be oral only |
| Model Design & Justification | 10 | **Strong** | Two models, clear comparison | Ensure slides name loss + metrics |
| Experimental Design | 10 | **Strong** | Protocol (PV val only; PlantDoc after) | Must **not** misstate protocol in talk |
| Training & Optimization | 10 | **Strong** | Curves, JSON, ablations | If asked: explain Adam, LR, dropout choice |
| Evaluation & Analysis | 10 | **Strong** | Gaps, CMs, shortcut narrative | Phrase as *consistent with* shift, not “proved background” |
| Contribution / Originality | 5 | **Strong** | Cross-dataset, curated 8-class, ablation | Frame clearly in presentation |
| Reproducibility & Code Quality | 5 | **Mostly strong** | Organized `src/`, `trainer` | Unpinned `requirements.txt`; no data in git |
| Presentation Quality | 30 | **Not assessable from repo** | *Slides not in repository* | Make slides; rehearse story from `FINAL_ANALYSIS` |
| **Bonus** | up to 5 | **TBD** | e.g. extra ablation, extra dataset | Optional |

---

## 12. Optional components completed

*Pick **two** of these for a concise “optional components” slide; all are well supported by the repo:*

1. **Fine-tuning a pretrained ResNet-18** on the 8 classes (`resnet18_finetune.py`, `resnet18_results.json`, checkpoint path in docs; weights not in git).
2. **Architecture comparison** — custom CNN *vs.* transfer learning (`README`, `FINAL_ANALYSIS` §3, comparison JSON).
3. **Data augmentation** and **rejection of harmful augmentation** (run4 `ColorJitter` vs run3, `transforms.py`).
4. **Hyperparameter / ablation study** on the Baseline (runs 0–4, val-loss selection).
5. **External generalization** on **PlantDoc** after *frozen* model selection (protocol in `DATASET_LICENSES`, `README`).
6. **Curated 8-class cross-dataset mapping** (`CLASS_MAPPING`, `FINAL_CLASS_SUBSET`, `class_subset_v1.json`).
7. **Shortcut learning / spurious-cue** discussion (careful, non-definitive) in `FINAL_ANALYSIS` §8.

**Safest two to name in a presentation (clear + examinable): (5) external generalization on PlantDoc** and **(2) from-scratch vs ImageNet-pretrained ResNet-18**—both have numbers and figures.

---

## 13. Remaining action items

### Must do before submission

- **Produce and submit presentation slides** (or follow course upload process)—they are **not** in this git repo.
- **Confirm** with the instructor whether **checkpoints** must be submitted separately (LFS, zip, or not required); they are **gitignored** by design.
- **Skim** `README` and `DATASET_LICENSES` in the final PDF/export so **attribution** is visible in any written report or appendix.

### Should do if time remains

- **Pin** key versions in `requirements.txt` (e.g. `torch==…`, `torchvision==…`) for stricter reproducibility.
- Add **1–2 sentences** in slides or a short `REFLECTION.md` (optional new file) on the **largest project challenge** if the rubric explicitly asks.
- If the course allows: **one slide** of **failure modes** with confusion matrices (already in `outputs/figures/`).

### Nice to have

- Export **per-class** confusion to CSV/JSON if the rubric wants numeric tables (currently qualitative from PNGs in docs).
- **Optional:** link the report notebook in the first slide of the deck (path `notebooks/plant_disease_shift_report.ipynb`).

### Not necessary (unless the syllabus demands)

- **Committing** raw image folders (large; not recommended).
- **Committing** `.pt` files (large; only if the instructor requires them in the repo or as an artifact).
- Rerunning training or re-evaluating (already have final JSON/figures in git).

---

## 14. Final verdict

| | |
| --- | --- |
| **Verdict** | **Almost ready** |

**Explanation:** The codebase, documentation, final metrics, EDA and training figures, ablation record, and external evaluation on PlantDoc are in **strong** shape and largely align with a rigorous deep-learning project rubric, including careful wording around **domain shift** and *possible* shortcut learning without overclaiming. The repository **does** track the main **result JSONs** and **figure** PNGs needed for a self-contained read. Gaps for a **full** “submission-ready” package are mostly **process** items: **presentation slides** are not in the repository, **dependency** versions are not pinned, and **checkpoints** are intentionally **excluded** from version control, which may need a separate handoff step depending on the course. Addressing the **must do** list above should move the project to **ready** for grading from a content perspective.

---

## Post-audit report (for the maintainer)

1. **File path:** `PROJECT_REQUIREMENT_CHECKLIST.md` (repository root).  
2. **Overall readiness verdict in this file:** **Almost ready** (§1, §14).  
3. **Missing or partial (high level):** Slides (missing in repo), unpinned `requirements.txt` (partial), optional explicit “reflection” block (partial), batch-size justification (nice-to-have).  
4. **Presentation slides in repo?** **No** (no presentation files found by pattern search).  
5. **Final notebook(s) in git?** **Yes** — `notebooks/plant_disease_shift_report.ipynb` and `notebooks/01_data_inspection.ipynb` are in `git ls-files`.  
6. **Result JSONs and figures tracked?** **Yes** — all listed under `outputs/results/*.json` and `outputs/figures/*.png` in `git ls-files` (as allowed by `.gitignore`).  
7. **Raw datasets or checkpoints accidentally tracked?** **No** — `data/*` and `outputs/checkpoints/**` are ignored; `git ls-files` contains **no** `.pt` and **no** image dataset tree under `data/` (only `data/.gitkeep`).
