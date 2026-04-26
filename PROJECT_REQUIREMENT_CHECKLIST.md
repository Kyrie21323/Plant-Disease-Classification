# Project Completion Checklist — Remaining Items Only

*This file lists **only** what may still be worth doing in the **repository and documentation** before you consider the project “done.” It does **not** include presentation slides, oral prep, or slide decks. A short **§5** states what is already in good shape at a high level.*

**Guardrails (already reflected across `README.md`, `FINAL_ANALYSIS.md`, and `DATASET_LICENSES.md`):** The work is a standard **plant disease image classification** setup on a public benchmark, extended with a **data-centric** question—**generalization** under shift and what is *consistent with* **possible** shortcut learning—not a claim of solving **field deployment** everywhere. **PlantVillage validation** is the **only** basis for Baseline **tuning/selection**; **PlantDoc** is **only** for **final external** analysis after selection. **Shortcut learning** is phrased as *consistent with* / *suggestive of*, not proven. **Dataset** citations and licenses are present. **Result JSONs** and key **PNGs** are **tracked**; **raw data** and **checkpoints** stay **out of git** by design (`.gitignore`).

---

## 1. Final verdict

**Complete except for minor documentation / reproducibility polish**

The repository already contains the expected code, final metrics, tracked results and figures, dataset-attribution file, and long-form write-ups. Nothing in the **repo** is a hard blocker for considering the project **substantively complete**; remaining items are **polish, clarity, and optional** strengthening for graders or for your own report appendices. **Presentation** deliverables (slides, talk) are **out of scope** for this checklist.

---

## 2. Must fix before project can be considered complete

`No blocking project/repo issues found.`

*(No code, data, or documentation gap was identified that must be fixed for the **repository** to be internally consistent and submission-ready, given the course-style assumptions above. If your instructor mandates **submitted weight files** or a **specific report filename**, that is a **course policy** item—handle outside this file.)*

---

## 3. Should fix if time remains

Worth doing only if you want cleaner narrative or stricter reproducibility. **Not** required for a coherent repo.

| # | Requirement area | Current status | Why it matters | File(s) involved | Suggested action |
| --- | --- | --- | --- | --- | --- |
| 1 | **Framing: well-known task + your angle** | **Adequate** | Readers should see PlantVillage/PlantDoc disease classification as the **backbone**, and your **contribution** as **protocol + generalization/shortcut** analysis. | `README.md` (first sections) | Optional: add **one** opening sentence that names the task as **mult-class leaf disease ID** on a standard benchmark, then your **shift** question. |
| 2 | **Reflection / “biggest challenge”** | **Implicit only** | Some rubrics ask for a short **reflection**; it is not a dedicated section in the repo. | N/A in repo | Optional: 3–5 sentences in **`FINAL_ANALYSIS.md`** (new short subsection) or a minimal **`REFLECTION.md`** (or course report only). |
| 3 | **Batch size: why not sweep** | **Stated as fixed `32`**, not justified | A single sentence can preempt “why not ablate batch size.” | `README.md` (hyperparam list) or `FINAL_ANALYSIS.md` (tuning) | Optional: e.g. “Batch size 32 for GPU memory; selection focused on LR/WD/dropout/augmentation.” |
| 4 | **Checkpoints: not in git** | **Documented in README** (ignored `.pt`, tree text) | Graders who expect weights in thezip need to see that **intentional** omission once. | `README.md` (Repository layout) | Optional: one line in **this** file or **README** if your syllabus requires: “Regenerate with `train_*.py` or supply checkpoints per instructor.” *(README already says checkpoints not version-controlled by default.)* — **low priority** |
| 5 | **Reproducibility: pinned deps** | **`requirements.txt` unpinned** | Same `pip install` on another machine can drift. | `requirements.txt` | Optional: pin `torch`, `torchvision`, and core libs to versions you used. |
| 6 | **Duplicate noteboooks** | **Two** full copies in git: `notebooks/plant_disease_shift_report.ipynb` and `notebooks/01_data_inspection.ipynb` | **README** only links the **shift_report** name; duplicate can confuse. | `notebooks/`, `README.md` | Optional: **remove** the duplicate, **or** add one line in **README** that `01_data_inspection` is a **duplicate** for a legacy course name. |

---

## 4. Nice-to-have / optional improvements

- **Pin** all important packages in `requirements.txt` (see §3 item 5).
- **Export** per-class **confusion matrices** to **CSV/JSON** if a rubric wants numeric cells (today they live in **PNGs** and aggregate metrics in **JSON**).
- **`REFLECTION.md`** (see §3 item 2) or a final-report paragraph only.
- **GitHub Release** or **Drive link** for **checkpoints** if the course requires them but they stay out of git.
- **Colab** one-liner in **README** or notebook intro (the notebook already has Colab + **auto** `REPO_ROOT`).
- **Remove** or **relabel** duplicate notebook (§3 item 6).

---

## 5. Already complete enough — no action needed

**Broad areas in good shape (no expanded checklist here):** problem definition and protocol; **dataset** citations and **licenses**; **preprocessing** and **EDA** (figures + summary); **two-model** design; **training** and **tuning** (incl. run4 story); **final evaluation** and **gaps**; **generalization** and careful **shortcut** wording; **JSON/figures** in git; **raw data** and **checkpoints** not accidentally committed; final **report notebook** present. **No** change needed for these in the **repo** unless you are chasing optional polish above.

---

## 6. Final remaining checklist

Copy this into your own task list and check items off as you do them. **All are optional** unless your syllabus says otherwise.

- [ ] Optional: one-sentence **task framing** (benchmark disease classification + your generalization/shortcut focus) in `README.md` if you want it even clearer.
- [ ] Optional: short **reflection** (biggest challenge) in `FINAL_ANALYSIS.md` or `REFLECTION.md` (or only in the course PDF).
- [ ] Optional: **one sentence** on why **batch size** was not part of the tuning search.
- [ ] Optional: **pin** versions in `requirements.txt`.
- [ ] Optional: **resolve** duplicate `notebooks/01_data_inspection.ipynb` vs `plant_disease_shift_report.ipynb` (delete duplicate or document in `README.md`).
- [ ] Optional: export **confusion matrices** to **CSV/JSON** if required.
- [ ] Optional: provide **checkpoints** via release/zip if the **instructor** requires them (they remain **gitignored** here).

**Required by this repo checklist:** *none* — there are no `[ ] Required:` items at this time.

---

*Exclusions: **Presentation slides**, **oral presentation** prep, and **LMS upload** steps are not listed here. **Course-specific** file names or grade thresholds are for your syllabus, not this file.*
