# System Architecture

## Overview

The PLGA Nanoparticle Drug Delivery Simulation project is organized into three major phases:

1. **Phase 1: Physics-Based Computational Simulation**: Simulates mass transport and cellular decay dynamics over grids.
2. **Phase 2: Machine Learning Surrogate Pipeline**: Learns a leakage-free surrogate map from design inputs to efficacy.
3. **Phase 3: Bounded Parameter Optimization**: Explores parameter configurations using Differential Evolution and evaluates their structural feasibility.

---

## High-Level Architecture

![Figure 3: System Architecture](../paper/assets/architecture/system_architecture.jpg)
*Figure 3: Modular boundaries and visual flowchart separating Phase 1 grids, Phase 2 XGBoost proxy regression, and Phase 3 global optimizations.*

---

## Phase 1 Architecture (Numerical Solver)

The computational model is implemented in `src/`.
- `pressure_model.py`: Darcy flow pressure distributions.
- `transport_model.py`: Convective transport.
- `drug_release.py`: Shell hydrolysis controlled release.
- `tumor_growth.py`: Treated tumor clearance.
- `solver.py`: Discretized finite difference loop.
- `parameters.py`: Grid parameters and constants.

---

## Phase 2 Architecture (Surrogate Regressor)

The machine learning workflow is implemented in `ml/`.
- `generate_dataset.py`: Raw dataset sampling.
- `preprocess.py`: Column standardization and quality validation.
- `split_dataset.py`: Train/test splits and scaler serialization.
- `train_model.py`: Hyperparameter optimization.
- `shap_analysis.py`: SHAP attribution explanations.
- `optimize_parameters.py` & `validate_optimization.py`: Bounded search and feasibility check.

---

## Data Flow

```text
Phase 1 Simulation Grid
         │
         ▼
simulation_dataset.csv
         │
         ▼
processed_dataset.csv (Standardized)
         │
         ▼
StandardScaler fit (Training Partition)
         │
         ▼
XGBoost Surrogate Model
         │
 ┌───────┼────────────────┐
 ▼       ▼                ▼
Test R²  Tree SHAP  Differential Evolution
                    (results/result10/)
```

---

## Design Principles

- **No Data Leakage**: The surrogate model utilizes only the 8 independent simulation input parameters. Derived simulation-specific results are excluded from training features, enabling valid forward predictions.
- **Reproducibility**: Set standard random seeds (e.g. 42) for splits and optimization.
- **Modularity**: Separation between physical simulation grids, preprocessing scripts, model estimators, and optimization loops.
