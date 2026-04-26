# Final Class Subset - Version 1 (confirmed)

This document defines the **8-class** Version 1 subset used for **all** training, **PlantVillage
validation** (tuning), **in-domain test** evaluation, and **final PlantDoc** external analysis
in this project. It was derived from the full reference table in
[CLASS_MAPPING.md](CLASS_MAPPING.md) by applying the selection criteria there.

The subset is small and conservative: every retained class has an **unambiguous** cross-dataset
name mapping and **enough** PlantDoc images for external evaluation. Machine-readable list:
[configs/class_subset_v1.json](../configs/class_subset_v1.json).

---

## Selected classes

| Use in V1? | PlantVillage Class | PlantDoc Class | Plant | Disease | PV Count | PD Count | Reason selected |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| ✓ | `Corn_(maize)___Northern_Leaf_Blight` | Corn leaf blight | Corn | Northern leaf blight | 985 | 180 | Highest PlantDoc count of any clear match; unambiguous mapping |
| ✓ | `Tomato___Septoria_leaf_spot` | Tomato Septoria leaf spot | Tomato | Septoria leaf spot | 1771 | 140 | Exact name match; strong PlantDoc support |
| ✓ | `Squash___Powdery_mildew` | Squash Powdery mildew leaf | Squash | Powdery mildew | 1835 | 124 | Clear match; good PlantDoc count; visually distinct disease |
| ✓ | `Potato___Early_blight` | Potato leaf early blight | Potato | Early blight | 1000 | 109 | Clear match; well-balanced counts |
| ✓ | `Corn_(maize)___Common_rust_` | Corn rust leaf | Corn | Common rust | 1192 | 106 | Clear match; good PlantDoc support (PlantVillage name is exactly `Corn_(maize)___Common_rust_` in `class_subset_v1.json`) |
| ✓ | `Tomato___Bacterial_spot` | Tomato leaf bacterial spot | Tomato | Bacterial spot | 2127 | 101 | Clear match; large PlantVillage count |
| ✓ | `Tomato___Late_blight` | Tomato leaf late blight | Tomato | Late blight | 1909 | 101 | Clear match; important disease for generalization study |
| ✓ | `Tomato___Early_blight` | Tomato Early blight leaf | Tomato | Early blight | 1000 | 79 | Clear match; pairs well with Tomato Late blight for comparison |

**Total V1 classes: 8**

**Total PlantVillage images (V1 classes): 11,819**

**Total PlantDoc images (V1 classes): 940**

---

## Why this subset was chosen

All eight classes meet every selection criterion in [CLASS_MAPPING.md](CLASS_MAPPING.md):

- Unambiguous name mapping (no healthy/diseased label mixing in these pairs).
- At least 79 PlantDoc images per class (≥ 50 as in mapping rules; threshold met).
- Four plant species (Corn, Tomato, Squash, Potato) for variety.
- Excludes ambiguous healthy-only PlantDoc labels and the Tomato spider mites class (only 2
  PlantDoc images in the reference mapping).

The same eight classes are used for **all** of **BaselineCNN (run3)**, **ResNet-18**, and
**final** evaluation. **PlantDoc** was **not** used to choose this list; the list was fixed
**before** any model selection.

---

## Optional future expansion (V2+)

V1 experiments, tuning, and final reporting are **complete**. A future **V2** could add more
overlapping classes if desired, for example:

- `Tomato___Leaf_Mold` ↔ `Tomato mold leaf` (85 PlantDoc images - was held back only to keep V1
  small)
- `Potato___Late_blight` ↔ `Potato leaf late blight` (97 PlantDoc images)
- `Apple___Apple_scab` ↔ `Apple Scab Leaf` (83 PlantDoc images)
- `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` ↔ `Corn Gray leaf spot` (64 PlantDoc
  images - lower priority)
- Revisit ambiguous healthy-class pairs only with manual content verification of PlantDoc
  images.

Any expansion would require new metadata, possible re-split, and a clear **re-run** of the
training / evaluation **protocol** - it is **optional future work**, not a prerequisite for the
current project deliverables.
