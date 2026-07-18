# Validation Workflow

## Overview

The validation workflow represents the final stage of the machine learning pipeline. Its objective is to determine whether optimized nanoparticle formulations produced by the machine learning surrogate can be rigorously compared with the original Phase 1 computational simulation.

Unlike many optimization workflows, this project performs a **validation feasibility assessment** before attempting any computational comparison. This prevents scientifically invalid conclusions and ensures that the surrogate model is free from data leakage (i.e. it does not require derived simulation outputs as input features).

---

## Workflow

```text
Optimized Formulations
          │
          ▼
Read Trained XGBoost Model
          │
          ▼
Inspect Required Feature Set
          │
          ▼
Compare With Independent Simulation Inputs
          │
          ▼
Feasibility Assessment
          │
    ┌─────┴─────┐
    │           │
Feasible     Not Feasible
    │           │
Run Phase 1   Generate Validation
Simulation    Limitation Report
```

---

## Validation Procedure

The validation module (`ml/validate_optimization.py`) performs the following steps:

1. Loads the optimized formulations from `results/result10/10.4 Optimized Parameters.csv`.
2. Loads the trained XGBoost surrogate model from `models/xgboost_model.pkl`.
3. Inspects the feature names expected by the trained model.
4. Compares those features with the independent simulation input parameters (reconstructed from `PARAMETER_RANGES` in `ml/generate_dataset.py`).
5. Determines whether direct computational validation is scientifically valid.
6. Writes feasibility reports summarizing the outcome.

---

## Project Rebuild Outcome

In the current rebuild of the project:
- **Feasibility Status**: **Feasible / Passed**.
- **Model Details**: The tuned XGBoost surrogate model was retrained using **only the eight independent simulation input parameters**:
  - `particle_size_nm`
  - `drug_diffusion`
  - `np_diffusion`
  - `release_rate`
  - `uptake_rate`
  - `drug_loading`
  - `tumor_growth_rate`
  - `drug_efficacy`
- **Leakage Status**: **None**. All derived simulation outputs (`final_tumor_volume`, `average_drug_concentration`, `penetration_depth`) were successfully excluded from model training features.
- **Validation Feasibility Conclusion**: The model is leakage-free and can be directly validated.

---

## Generated Outputs

The validation stage produces the following files inside `results/validation/`:

- **`validation_report.txt`**: Documents the feasibility check, metrics, and confirmation that the model features align with independent inputs.
- **`validation_limitations.txt`**: Confirms that full validation is feasible and there are no derived-output dependencies.

---

## Key Scientific Note

By enforcing that the model relies strictly on independent design parameters, we ensure that:
1. The optimization algorithm can query the surrogate regressor across the entire parameter space.
2. The optimized formulation designs can be directly mapped back to physical simulation grids for independent numerical verification.
3. The overall workflow remains scientifically consistent, reproducible, and ready for publication alongside clinical and numerical experiments.
