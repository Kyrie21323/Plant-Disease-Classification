# Dataset Citations and Licenses

This project uses PlantVillage and PlantDoc for academic plant disease classification experiments. The datasets themselves are not owned by this repository. Please cite the original dataset papers and follow their respective licenses when using the data.

## PlantVillage

**Usage in this project:** PlantVillage was used for **training**, **validation** (model selection and hyperparameter tuning), and **in-domain test** evaluation.

**Dataset and paper:** The dataset was introduced in Mohanty, Hughes, and Salathé (2016), *Using deep learning for image-based plant disease detection.*

**Original repository:** <https://github.com/spMohanty/PlantVillage-Dataset>

**Suggested citation (APA-style):**

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7. <https://doi.org/10.3389/fpls.2016.01419>

**BibTeX:**

```bibtex
@article{Mohanty_Hughes_Salathe_2016,
  title   = {Using deep learning for image-based plant disease detection},
  volume  = {7},
  DOI     = {10.3389/fpls.2016.01419},
  journal = {Frontiers in Plant Science},
  author  = {Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
  year    = {2016},
  month   = {Sep}
}
```

**License note:** This repository does **not** state a specific license for PlantVillage from third-hand sources alone. License and redistribution terms can change; **users should check the [original repository](https://github.com/spMohanty/PlantVillage-Dataset) or upstream source** for the **current** licensing, attribution requirements, and any restrictions on use or redistribution.

## PlantDoc

**Usage in this project:** PlantDoc was used **only after** model selection, as the **final** external **generalization** and **dataset-shift** evaluation set. It was not used to tune models or to pick checkpoints.

**Dataset and paper:** PlantDoc was introduced in Singh *et al.* (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection.*

**Original repository:** <https://github.com/pratikkayal/PlantDoc-Dataset>

**Suggested citation (APA-style):**

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253. <https://doi.org/10.1145/3371158.3371196>

**BibTeX:**

```bibtex
@inproceedings{10.1145/3371158.3371196,
  author = {Singh, Davinder and Jain, Naman and Jain, Pranjali and Kayal, Pratik and Kumawat, Sudhakar and Batra, Nipun},
  title = {PlantDoc: A Dataset for Visual Plant Disease Detection},
  year = {2020},
  isbn = {9781450377386},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3371158.3371196},
  doi = {10.1145/3371158.3371196},
  booktitle = {Proceedings of the 7th ACM IKDD CoDS and 25th COMAD},
  pages = {249--253},
  numpages = {5},
  keywords = {Deep Learning, Object Detection, Image Classification},
  location = {Hyderabad, India},
  series = {CoDS COMAD 2020}
}
```

**License note:** The [PlantDoc dataset repository](https://github.com/pratikkayal/PlantDoc-Dataset) lists the data under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Full license text: <https://creativecommons.org/licenses/by/4.0/>

CC BY 4.0 generally requires, among other things, **attribution** to the rights holder, a **link to the license** (or a copy of the license notice), and **indication of changes** if you modify the material, subject to the exact terms in the license.

This project repository is **citation- and documentation-focused**; it does **not** claim ownership of PlantDoc, and **as configured it does not redistribute** PlantDoc image files. If you obtain PlantDoc for your own work, follow the terms at the **original** source, including the CC BY 4.0 requirements where they apply.

## Dataset usage in this project

| Dataset      | Used for                                                                 | Used for model selection?           |
| ------------ | ------------------------------------------------------------------------ | ----------------------------------- |
| PlantVillage | Training, validation, in-domain test evaluation                        | **Yes** - **validation split only** |
| PlantDoc     | Final external generalization / dataset-shift evaluation                 | **No**                              |

PlantDoc was intentionally **not** used for tuning, checkpoint selection, or label-decision changes. That keeps PlantDoc a **genuine** external stress test for generalization and dataset shift, consistent with the project protocol described in the README and in `docs/PROJECT_PLAN.md`.

## Ownership / redistribution

- This repository **does not** claim ownership of PlantVillage or PlantDoc.
- **Users** should download datasets from the **original** sources and follow each dataset’s terms.
- If dataset images are **ever** redistributed in this repository, the **corresponding** original license terms, notices, and any required attributions from those sources should be **included and followed**.
