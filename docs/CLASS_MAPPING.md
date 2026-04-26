# Class Mapping: PlantVillage ↔ PlantDoc

This document is the **broad reference** of possible PlantVillage ↔ PlantDoc name alignments. The
**final 8-class V1 subset** actually used in training, validation, test, and evaluation is
**fixed** in [FINAL_CLASS_SUBSET.md](FINAL_CLASS_SUBSET.md) and
[configs/class_subset_v1.json](../configs/class_subset_v1.json). Use that subset document for
the exact experimental classes.

This table maps class names between the two datasets to identify overlapping classes for
cross-dataset generalization experiments. Both collections use different naming conventions for
the same plant diseases. Rows are aligned manually by plant species and disease type.

---

## Class Mapping Table

| Keep? | PlantVillage Class | PlantDoc Class | Plant | Disease | PV Count | PD Count | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| ✓ | Apple___Apple_scab | Apple Scab Leaf | Apple | Apple scab | 630 | 83 | Clear match |
| ✗ | Apple___Black_rot | — | Apple | Black rot | 621 | 0 | No PlantDoc equivalent found |
| ✓ | Apple___Cedar_apple_rust | Apple rust leaf | Apple | Cedar apple rust | 275 | 79 | Clear match |
| ? | Apple___healthy | Apple leaf | Apple | Healthy | 1645 | 82 | PlantDoc "Apple leaf" label may include diseased images — uncertain |
| ? | Blueberry___healthy | Blueberry leaf | Blueberry | Healthy | 1502 | 106 | Same uncertainty as above |
| ? | Cherry_(including_sour)___Powdery_mildew | Cherry leaf | Cherry | Powdery mildew | 1052 | 47 | PlantDoc "Cherry leaf" may be healthy only — ambiguous |
| ? | Cherry_(including_sour)___healthy | Cherry leaf | Cherry | Healthy | 854 | 47 | Same label as above — overlapping ambiguity |
| ✓ | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn Gray leaf spot | Corn | Gray leaf spot | 513 | 64 | Clear match |
| ✓ | Corn_(maize)___Common_rust_ | Corn rust leaf | Corn | Common rust | 1192 | 106 | Clear match |
| ✓ | Corn_(maize)___Northern_Leaf_Blight | Corn leaf blight | Corn | Northern leaf blight | 985 | 180 | Clear match |
| ✗ | Corn_(maize)___healthy | — | Corn | Healthy | 1162 | 0 | No PlantDoc equivalent found |
| ✓ | Grape___Black_rot | grape leaf black rot | Grape | Black rot | 1180 | 56 | Clear match |
| ✗ | Grape___Esca_(Black_Measles) | — | Grape | Esca / Black measles | 1383 | 0 | No PlantDoc equivalent found |
| ✗ | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | — | Grape | Leaf blight | 1076 | 0 | No PlantDoc equivalent found |
| ? | Grape___healthy | grape leaf | Grape | Healthy | 423 | 57 | PlantDoc "grape leaf" label uncertain |
| ✗ | Orange___Haunglongbing_(Citrus_greening) | — | Orange | Citrus greening | 5507 | 0 | No PlantDoc equivalent found |
| ? | Peach___Bacterial_spot | Peach leaf | Peach | Bacterial spot | 2297 | 103 | PlantDoc "Peach leaf" may be healthy only — ambiguous |
| ? | Peach___healthy | Peach leaf | Peach | Healthy | 360 | 103 | Same label as above — ambiguous |
| ✓ | Pepper,_bell___Bacterial_spot | Bell_pepper leaf spot | Bell pepper | Bacterial spot | 997 | 62 | Clear match |
| ? | Pepper,_bell___healthy | Bell_pepper leaf | Bell pepper | Healthy | 1478 | 53 | PlantDoc label uncertain |
| ✓ | Potato___Early_blight | Potato leaf early blight | Potato | Early blight | 1000 | 109 | Clear match |
| ✓ | Potato___Late_blight | Potato leaf late blight | Potato | Late blight | 1000 | 97 | Clear match |
| ✗ | Potato___healthy | — | Potato | Healthy | 152 | 0 | No PlantDoc equivalent and very small PV count |
| ? | Raspberry___healthy | Raspberry leaf | Raspberry | Healthy | 371 | 112 | PlantDoc label uncertain |
| ? | Soybean___healthy | Soyabean leaf | Soybean | Healthy | 5090 | 57 | PlantDoc label uncertain |
| ✓ | Squash___Powdery_mildew | Squash Powdery mildew leaf | Squash | Powdery mildew | 1835 | 124 | Clear match |
| ✗ | Strawberry___Leaf_scorch | Strawberry leaf | Strawberry | Leaf scorch | 1109 | 88 | PlantDoc "Strawberry leaf" likely healthy — ambiguous, skip |
| ? | Strawberry___healthy | Strawberry leaf | Strawberry | Healthy | 456 | 88 | Same label as above |
| ✓ | Tomato___Bacterial_spot | Tomato leaf bacterial spot | Tomato | Bacterial spot | 2127 | 101 | Clear match |
| ✓ | Tomato___Early_blight | Tomato Early blight leaf | Tomato | Early blight | 1000 | 79 | Clear match |
| ✓ | Tomato___Late_blight | Tomato leaf late blight | Tomato | Late blight | 1909 | 101 | Clear match |
| ✓ | Tomato___Leaf_Mold | Tomato mold leaf | Tomato | Leaf mold | 952 | 85 | Clear match |
| ✓ | Tomato___Septoria_leaf_spot | Tomato Septoria leaf spot | Tomato | Septoria leaf spot | 1771 | 140 | Clear match |
| ✗ | Tomato___Spider_mites Two-spotted_spider_mite | Tomato two spotted spider mites leaf | Tomato | Spider mites | 1676 | 2 | Only 2 PlantDoc images — unusable for testing |
| ✗ | Tomato___Target_Spot | — | Tomato | Target spot | 1404 | 0 | No PlantDoc equivalent found |
| ? | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato leaf yellow virus | Tomato | Yellow leaf curl virus | 5357 | 70 | Likely match but naming differs slightly |
| ? | Tomato___Tomato_mosaic_virus | Tomato leaf mosaic virus | Tomato | Mosaic virus | 373 | 44 | Likely match — small counts in both |
| ? | Tomato___healthy | Tomato leaf | Tomato | Healthy | 1591 | 55 | PlantDoc "Tomato leaf" label uncertain |

---

## Selection Criteria

When deciding which overlapping classes to keep for the project, apply the following rules:

- **Same plant and same disease** — the mapping must be unambiguous. If the PlantDoc class name
  could refer to either a diseased or healthy leaf, exclude it.
- **Enough PlantDoc images** — aim for at least 50 PlantDoc images per class to make
  cross-dataset evaluation statistically meaningful. Classes with fewer than 50 should be
  avoided or flagged.
- **Avoid ambiguous mappings** — PlantDoc uses generic labels like "Apple leaf" or "Peach
  leaf" that may contain mixed content. These should not be paired with a specific disease
  class from PlantVillage unless confirmed otherwise.
- **Avoid extremely small classes** — classes with very few images in either dataset (e.g. Tomato
  spider mites with 2 PlantDoc images) should be excluded entirely.
- **Prefer disease classes over healthy classes** — healthy class labels in PlantDoc are
  unreliable since the dataset was scraped from web images and class boundaries may not be
  clean.

---

## V1 subset (confirmed — in use)

The **8-class** Version 1 disease subset was **confirmed** before metadata and split generation.
It uses exactly the candidates below (all marked ✓ in the table) with **unambiguous** mappings
and **≥ 50** PlantDoc images per class. Full rationale and counts:
[FINAL_CLASS_SUBSET.md](FINAL_CLASS_SUBSET.md).

| # | PlantVillage | PlantDoc | PD Count |
| --- | --- | --- | ---: |
| 1 | `Tomato___Septoria_leaf_spot` | Tomato Septoria leaf spot | 140 |
| 2 | `Corn_(maize)___Northern_Leaf_Blight` | Corn leaf blight | 180 |
| 3 | `Squash___Powdery_mildew` | Squash Powdery mildew leaf | 124 |
| 4 | `Corn_(maize)___Common_rust_` | Corn rust leaf | 106 |
| 5 | `Tomato___Bacterial_spot` | Tomato leaf bacterial spot | 101 |
| 6 | `Tomato___Late_blight` | Tomato leaf late blight | 101 |
| 7 | `Potato___Early_blight` | Potato leaf early blight | 109 |
| 8 | `Tomato___Early_blight` | Tomato Early blight leaf | 79 |
