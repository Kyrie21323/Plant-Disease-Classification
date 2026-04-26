# Final Class Subset — Version 1

This document defines the first-version class subset used for all experiments in this project. It was derived from the full class mapping in `CLASS_MAPPING.md` by applying the selection criteria documented there.

This subset is intentionally small and conservative. The goal is to establish a clean, defensible baseline before expanding to more classes. All selected classes have unambiguous cross-dataset name mappings and sufficient PlantDoc images for meaningful generalization testing.

---

## Selected Classes

| Use in V1? | PlantVillage Class | PlantDoc Class | Plant | Disease | PV Count | PD Count | Reason selected |
|---|---|---|---|---|---|---|---|
| ✓ | Corn_(maize)___Northern_Leaf_Blight | Corn leaf blight | Corn | Northern leaf blight | 985 | 180 | Highest PlantDoc count of any clear match; unambiguous mapping |
| ✓ | Tomato___Septoria_leaf_spot | Tomato Septoria leaf spot | Tomato | Septoria leaf spot | 1771 | 140 | Exact name match; strong PlantDoc support |
| ✓ | Squash___Powdery_mildew | Squash Powdery mildew leaf | Squash | Powdery mildew | 1835 | 124 | Clear match; good PlantDoc count; visually distinct disease |
| ✓ | Potato___Early_blight | Potato leaf early blight | Potato | Early blight | 1000 | 109 | Clear match; well-balanced counts |
| ✓ | Corn_(maize)___Common_rust_ | Corn rust leaf | Corn | Common rust | 1192 | 106 | Clear match; good PlantDoc support |
| ✓ | Tomato___Bacterial_spot | Tomato leaf bacterial spot | Tomato | Bacterial spot | 2127 | 101 | Clear match; large PlantVillage count |
| ✓ | Tomato___Late_blight | Tomato leaf late blight | Tomato | Late blight | 1909 | 101 | Clear match; important disease for generalization study |
| ✓ | Tomato___Early_blight | Tomato Early blight leaf | Tomato | Early blight | 1000 | 79 | Clear match; pairs well with Tomato Late blight for comparison |

**Total V1 classes: 8**
**Total PlantVillage images (V1 classes): 11,819**
**Total PlantDoc images (V1 classes): 940**

---

## Why This Subset Was Chosen

All eight classes were selected because they meet every selection criterion from `CLASS_MAPPING.md`:

- The name mapping between the two datasets is unambiguous in every case — there is no risk of mixing healthy and diseased images under the same label.
- Every class has at least 79 PlantDoc images, which is enough to produce meaningful cross-dataset evaluation results.
- The subset spans four different plant species (Corn, Tomato, Squash, Potato), which adds variety and makes the generalization experiment more interesting.
- The subset avoids all ambiguous healthy-class mappings and excludes the Tomato spider mites class, which had only 2 PlantDoc images.

Starting with 8 clean classes allows the full training, evaluation, and analysis pipeline to be built and validated before any ambiguous or borderline classes are added.

---

## Future Expansion

Once the pipeline is stable and results have been recorded for the V1 subset, additional overlapping classes may be added. Candidates for a second version include:

- `Tomato___Leaf_Mold` ↔ `Tomato mold leaf` (85 PlantDoc images — clear match, held back only to keep V1 small)
- `Potato___Late_blight` ↔ `Potato leaf late blight` (97 PlantDoc images — strong candidate)
- `Apple___Apple_scab` ↔ `Apple Scab Leaf` (83 PlantDoc images — clear match)
- `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` ↔ `Corn Gray leaf spot` (64 PlantDoc images — acceptable but lower priority)
- Ambiguous healthy-class mappings may be revisited if the PlantDoc label content can be verified manually.

Expansion should only happen after the V1 experiments are complete and documented.
