# Plant Disease Classification
## A Study of Shortcut Learning

This project uses deep learning to classify plant diseases from leaf images while examining whether models rely on true disease patterns or shortcut cues such as background and image style. The main experimental focus is how model type and training choices affect generalization across changing visual conditions.

---

## Current Environment Setup

| Item | Location |
|---|---|
| Project code | Windows repository — edited in Cursor, committed with Git as normal |
| PlantDoc dataset | WSL Linux filesystem (`~/plantdoc`) — stored here because many PlantDoc filenames contain characters (`?`, `%`, `+`) that Windows NTFS forbids |
| Scripts accessing PlantDoc | Must be run from inside WSL, where the Linux filesystem is accessible |
| PlantVillage (pending) | Will also be stored in WSL for consistency, unless a clean zip without problematic filenames is used |

**Why WSL?**
The PlantDoc repository was scraped from the web and retains raw URL-style filenames. These are valid on Linux but cannot be written to a Windows filesystem. Cloning into WSL resolves this cleanly without renaming any files.

**Path convention used in this project:**
- Dataset paths are configurable constants defined at the top of each script and notebook.
- They are not assumed to live inside the repository `data/` folder.
- The actual paths used locally should be documented here and updated as datasets are confirmed.

**Current confirmed dataset paths (local machine):**

| Dataset | Path |
|---|---|
| PlantDoc | `~/plantdoc` (WSL) |
| PlantVillage | Not yet downloaded |

---

# Project Build Tasks

## 1. Decide the exact datasets and classes
- Download PlantVillage
- Download PlantDoc
- Inspect class names in both datasets
- Identify only the overlapping disease classes to use
- Keep the class subset small and realistic at first
- Define dataset roles clearly:
  - **PlantVillage** = main training / validation / in-domain test dataset
  - **PlantDoc** = cross-dataset generalization test dataset

## 2. Collect and organize the data
- Store raw datasets in a stable location (WSL filesystem or Windows `data/` folder, depending on filename compatibility)
- Document the actual storage path for each dataset in `PROJECT_PLAN.md` under Current Environment Setup
- Create a class-name mapping file
- Document which classes were kept and which were excluded
- Remove unusable or corrupted images if needed
- Define dataset paths as configurable constants in scripts and notebooks — do not hardcode absolute paths in logic

## 3. Do initial data inspection
- Count number of images per class
- View sample images from each class
- Check image sizes and formats
- Check whether classes are imbalanced
- Compare visual differences between PlantVillage and PlantDoc
- Summarize why PlantDoc is a reasonable generalization test setting

## 4. Define the exact experiment question
- Lock the main question:
  - How do model type and training choices affect generalization in plant disease classification?
- Lock the interpretation question:
  - Do cross-dataset failures suggest reliance on shortcut cues such as background, lighting, or image style?
- Decide the main comparison:
  - Custom CNN vs fine-tuned pretrained model
- Decide the main tuning comparisons:
  - Baseline vs augmentation
  - Baseline vs regularization
- Decide the analysis angle:
  - In-domain performance vs cross-dataset performance

## 5. Prepare preprocessing pipeline
- Resize images
- Normalize images
- Create train / validation / test splits for PlantVillage
- Prepare PlantDoc as a separate evaluation dataset
- Create PyTorch dataset classes and dataloaders
- Define augmentations for controlled experiments

## 6. Build baseline model
- Implement a small custom CNN
- Make sure one full training run works end to end
- Verify metrics, saving, and plotting all work

## 7. Build second model
- Fine-tune a pretrained model such as ResNet-18
- Use the same PlantVillage splits for fair comparison
- Keep the evaluation pipeline identical between models

## 8. Run initial experiments
- Train both models with initial hyperparameters
- Save training and validation loss curves
- Record accuracy, macro-F1, and confusion matrix
- Compare in-domain PlantVillage test performance first

## 9. Run generalization experiments
- Test both trained models on PlantDoc
- Compare how much performance drops from PlantVillage to PlantDoc
- Identify which model generalizes better under changed visual conditions

## 10. Run improvement experiments
- Vary learning rate
- Vary batch size
- Try augmentation
- Try dropout / weight decay
- Compare results systematically
- Focus especially on which changes improve cross-dataset generalization, not just in-domain accuracy

## 11. Run shortcut-learning analysis
- Compare which models or training settings degrade more on PlantDoc
- Inspect whether improvements in augmentation or regularization reduce that degradation
- Analyze failure cases visually
- Use the results to discuss whether the model may be relying on shortcut cues

## 12. Do final evaluation
- Choose best settings for each model
- Report final PlantVillage test results
- Report final PlantDoc generalization results
- Summarize strengths, weaknesses, and failure cases
- Explain what the results suggest about generalization and shortcut learning

## 13. Prepare presentation materials
- Problem slide
- Dataset slide
- EDA visuals
- Preprocessing summary
- Architecture comparison
- Training curves
- In-domain vs cross-dataset evaluation
- Tuning / ablation results
- Shortcut-learning interpretation
- Conclusion and future work

---

# Notebook Creation Steps

## Cell 1: Project title and brief problem description
- Title
- Short paragraph explaining plant disease classification, model comparison, generalization focus, and shortcut-learning interpretation

## Cell 2: Imports and setup
- Import libraries
- Set random seeds
- Define dataset paths as configurable path variables (not assumed to be inside `data/`)
- Define output paths

## Cell 3: Dataset loading
- Load PlantVillage metadata
- Load PlantDoc metadata
- Print dataset sizes
- Print discovered class names

## Cell 4: Class mapping and final class selection
- Show overlapping classes
- Show final selected classes
- Explain why only those classes are used

## Cell 5: EDA
- Class distribution plots
- Sample images per class
- Image size statistics
- Basic comparison of dataset appearance
- Notes on class imbalance and visual differences

## Cell 6: Preprocessing
- Transforms
- Normalization
- Training augmentations
- Evaluation transforms

## Cell 7: DataLoaders
- PlantVillage train / val / test loaders
- PlantDoc evaluation loader

## Cell 8: Baseline model
- Define custom CNN
- Briefly explain architecture

## Cell 9: Training function
- Training loop
- Validation loop
- Metric tracking
- Model checkpointing

## Cell 10: Train baseline model
- Run training
- Save training curves
- Report PlantVillage validation / test results

## Cell 11: Second model
- Define / load pretrained model
- Briefly explain fine-tuning setup

## Cell 12: Train second model
- Run training
- Save training curves
- Report PlantVillage validation / test results
- Compare with baseline

## Cell 13: Generalization evaluation
- Evaluate both models on PlantDoc
- Compare PlantVillage vs PlantDoc performance
- Summarize generalization gap

## Cell 14: Tuning and ablation experiments
- Compare augmentation settings
- Compare regularization settings
- Compare learning rates / batch sizes if included
- Show which choices help generalization most

## Cell 15: Shortcut-learning analysis
- Inspect failure examples
- Discuss whether performance drops suggest reliance on shortcut cues
- Relate results back to background, lighting, and image style differences

## Cell 16: Final conclusion
- What each model learned
- Which model generalized better
- What tuning helped most
- Whether shortcut reliance seems likely
- Future work
