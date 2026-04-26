# Final Subset EDA Summary

**Datasets:** PlantVillage (in-domain) vs PlantDoc (cross-dataset generalization test)
**Classes in subset:** 8

---

## Class Distribution

| Class | PlantVillage | PlantDoc | PV / PD ratio |
|---|---|---|---|
| corn_northern_leaf_blight | 985 | 180 | 5.5x |
| tomato_septoria_leaf_spot | 1771 | 140 | 12.7x |
| squash_powdery_mildew | 1835 | 124 | 14.8x |
| potato_early_blight | 1000 | 109 | 9.2x |
| corn_common_rust | 1192 | 106 | 11.2x |
| tomato_bacterial_spot | 2127 | 101 | 21.1x |
| tomato_late_blight | 1909 | 101 | 18.9x |
| tomato_early_blight | 1000 | 79 | 12.7x |

**Total PlantVillage images:** 11819
**Total PlantDoc images:** 940
**Overall PV / PD ratio:** 12.6x

PlantVillage is substantially larger for every class. PlantDoc counts range from ~79 to ~180 images per class, which is sufficient for evaluation but not for training.

---

## Image Size Statistics
*(based on first 200 PlantVillage and 200 PlantDoc images sampled)*

| Property | PlantVillage | PlantDoc |
|---|---|---|
| Width min (px) | 256 | 194 |
| Width max (px) | 256 | 4608 |
| Width mean (px) | 256 | 854 |
| Height min (px) | 256 | 185 |
| Height max (px) | 256 | 3456 |
| Height mean (px) | 256 | 789 |

---

## File Extensions

**PlantVillage:** `.jpeg` (1), `.jpg` (11818)

**PlantDoc:** `.jpeg` (1), `.jpg` (938), `.png` (1)

---

## Visual Style Differences

- **PlantVillage** images are lab-collected against uniform or plain backgrounds. Lighting is controlled and consistent. Images are typically close-up leaf shots with minimal background clutter.
- **PlantDoc** images are field-collected with natural backgrounds (soil, other plants, sky). Lighting varies significantly. Images often include multiple leaves, plant stems, or distracting background elements.
- These differences make PlantDoc a realistic generalization test: a model that over-relies on background or lighting cues from PlantVillage will likely degrade on PlantDoc.

---

## Class Balance Notes

- Within PlantVillage, classes are reasonably balanced for this subset (range: ~985 to ~2,127 images). No class is severely under-represented.
- Within PlantDoc, class sizes are smaller and less balanced (range: ~79 to ~180 images). `tomato_early_blight` has the fewest images and should be monitored in evaluation.
- The large PV/PD ratio (~12.6x overall) is expected: PlantVillage was purpose-built for training while PlantDoc was scraped for generalization testing.