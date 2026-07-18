# Project Report and Manuscript Writing Guide

This guide serves as a handbook for preparing the final academic report or IEEE research manuscript. It maps the project components to their respective folders and results.

---

## 1. Title Page and Abstract
- **Proposed Title**: A Machine Learning-Accelerated physics-based surrogate model for bounded optimization of PLGA drug delivery systems.
- **Key Abstract Elements**: Frame the physical transport barrier (high Interstitial Fluid Pressure) in tumors, describe the 2D finite difference grid solver, outline the Latin Hypercube Sampling ($N=1,000$) dataset generation, highlight the tuned XGBoost surrogate performance ($R^2 = 0.951075$), explain the global search via Differential Evolution, and state the best optimized design ($114.1\text{ nm}$ size, $0.126\text{ h}^{-1}$ release constant).

---

## 2. Structure Map

| Paper Section | Source Directories | Relevant Files / Figures |
| :--- | :--- | :--- |
| **Introduction** | `paper/references/` | `reference_notes.md` |
| **Methodology: Math** | `paper/equations/` | `Equations_Summary.md` |
| **Methodology: Code** | `src/`, `ml/` | `docs/Architecture.md`, `docs/Execution_Flow.md` |
| **Results: Physics** | `results/result1/` to `results/result6/` | `Figure_1` to `Figure_7` inside `paper/figures/simulation/` |
| **Results: Model fit** | `results/result7/`, `results/result8/` | `Figure_8` to `Figure_10` inside `paper/figures/machine_learning/` |
| **Results: SHAP** | `results/result9/` | `Figure_11` to `Figure_12` inside `paper/figures/machine_learning/` |
| **Results: Optimization**| `results/result10/`, `results/validation/` | `Figure_13` inside `paper/figures/optimization/`, `Table_5` |

---

## 3. Results Section Guidelines

### A. Phase 1 (Computational Baseline)
Present the baseline Darcy pressure contours and advective nanoparticle profiles. 
*   **Key Numbers**: High core fluid pressure of 10 mmHg ($1.33\text{ kPa}$) creates outward velocities at boundaries. Smaller particles ($50\text{ nm}$) achieve deeper penetration compared to larger particles ($200\text{ nm}$).
*   **Figures**: Insert Figures 1, 2, 3, 4, 5.

### B. Machine Learning Performance
Discuss the XGBoost training logs and metrics.
*   **Key Numbers**: Tune hyperparameters via 5-fold CV to select a model achieving test $R^2 = 0.951075$.
*   **Figures**: Insert Figures 8, 9, 10.

### C. SHAP Feature Interpretations
Illustrate how Tree SHAP attributions rank parameters.
*   **Key Numbers**: Show that drug efficacy ($E_d$) and tumor growth rate ($r_g$) dominate response, while nanoparticle release rate ($k_{rel}$) is the most sensitive design parameter.
*   **Figures**: Insert Figures 11, 12.

### D. Bounded Global Optimization
Report the Differential Evolution global search outputs.
*   **Key Numbers**: List the top 10 formulations. Report that the best design ($114.1\text{ nm}$ particle size, $0.126\text{ h}^{-1}$ release rate) achieves a predicted tumor volume reduction of **`94.99%`**.
*   **Figures**: Insert Figure 13 and Table 5.

---

## 4. Verification and Leakage Checklist
- **No Data Leakage**: Confirm that the final model is trained only on independent physical inputs.
- **Physical Boundedness**: Confirm that predicted tumor reduction percentages do not exceed $95\%$ (carrying limit).
- **Temporal Consistency**: Ensure all figures and metrics match the latest execution logs.
