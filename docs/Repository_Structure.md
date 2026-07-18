# Repository Structure

This document describes the organization of the PLGA Nanoparticle Drug Delivery Simulation repository after completion of Phase 1 (Computational Simulation), Phase 2 (Machine Learning Workflow), and the establishment of the `paper/` research manuscript workspace.

---

## 1. Top-Level Structure

![Figure 4: Repository Structure](../paper/assets/repository/repository_structure.jpg)
*Figure 4: Structure map showing top-level files, source codes, datasets, models, docs, results study directories, and paper templates.*

---

## 2. Research Paper Workspace (`paper/`)

Dedicated workspace for manuscript drafting and assets:
- **`IEEE_Manuscript.md`**: Main draft outline skeleton divided into standard IEEE sections.
- **`equations/`**: Contains `Equations_Summary.md` cataloging Darcy equations, mass transport equations, PLGA release kinetics, and optimization metrics.
- **`figures/`**: Publication-ready plots copied from `results/` and renamed according to manuscript figure IDs.
- **`tables/`**: CSV tables summarizing simulation parameters, hyperparameters, and evaluation outcomes.
- **`references/`**: BibTeX citation database (`references.bib`), checklists, and review notes.

---

## 3. Computational Simulation (`src/`)

Validated numerical simulation modules:
- `parameters.py`: Grid space constants, physical properties, and bounds.
- `solver.py`: Core convection-diffusion-reaction time-stepping loop.
- `pressure_model.py`: Darcy fluid pressure solver.
- `transport_model.py`: Advective-diffusive mass transport solver.
- `drug_release.py`: Controlled drug release kinetic solver.
- `tumor_growth.py`: Logistic tumor volume response solver.
- `visualization.py`: Matplotlib plotting scripts.
- `result1_*` through `result6_*`: Result studies 1 to 6 runners.

---

## 4. Machine Learning & AI Surrogate Pipeline (`ml/`)

Chronological scripts for fitting and explaining the surrogate model:
1. `generate_dataset.py`: Sequentially simulates random parameter sets to yield 1,000 accepted samples.
2. `preprocess.py`: Cleans raw dataset and verifies no NaN/Inf metrics.
3. `split_dataset.py`: Partitions dataset into reproducible 80/20 train/test sets and fits standard scaling pipeline.
4. `eda.py`: Performs exploratory data plotting (histograms, heatmaps).
5. `train_model.py`: RandomizedSearchCV tuning and XGBoost model training.
6. `evaluate_model.py`: held-out test predictions and residual analyses.
7. `shap_analysis.py`: Tree SHAP feature explainability.
8. `optimize_parameters.py`: Global Differential Evolution optimization.
9. `validate_optimization.py`: Bounded feasibility checking and column renaming.
10. `generate_result7.py`: Generates the parameter distribution histogram panel.

---

## 5. Documentation (`docs/`)

Technical guides and indexes:
- `Architecture.md`: Modular code organization guide.
- `Execution_Flow.md`: Detailed runner script execution commands.
- `Machine_Learning_Pipeline.md`: Data processing and surrogate regression modeling guide.
- `Optimization_Workflow.md`: Evolutionary global optimization framework details.
- `Validation_Workflow.md`: Surrogate model validation checks.
- `Image_Index.md`: Auto-generated catalog mapping figure numbers to local files.
- `Repository_Structure.md`: This file.
