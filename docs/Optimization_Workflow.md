# Optimization Workflow

## Overview

The optimization workflow uses the trained XGBoost surrogate model to identify promising PLGA nanoparticle formulations without repeatedly executing the computational simulation. 

The optimizer searches only within the validated simulation parameter space and treats the machine learning model as a computationally efficient surrogate.

---

## Workflow

![Figure 6: Optimization Framework](../paper/assets/optimization/optimization_framework.jpg)
*Figure 6: Optimization Framework. Showcases the step-by-step search methodology utilizing Differential Evolution guided by the surrogate model.*

---

## Optimization Objective

The objective is to maximize the predicted value of `tumor_reduction_percent` while remaining inside the validated parameter ranges defined by the computational model.

No computational simulation is executed during optimization. The XGBoost surrogate regressor acts as a high-fidelity proxy.

---

## Optimization Variables

Only independent formulation parameters are optimized.

These include:
- `particle_size_nm`
- `drug_diffusion`
- `np_diffusion`
- `release_rate`
- `uptake_rate`
- `drug_loading`
- `tumor_growth_rate`
- `drug_efficacy`

All bounds are read directly from the sampling boundary definitions to ensure consistency with Phase 1.

---

## Optimization Algorithm

The project uses **Differential Evolution**, a stochastic global optimization algorithm suitable for continuous multi-dimensional spaces.

Configuration:
- Random seed: 42
- Maximum iterations: 80
- Population multiplier: 15
- Tolerance: $10^{-6}$

---

## Derived Input Policy

Unlike older versions of this pipeline, **the surrogate model contains no data leakage**. It does not expect derived simulation outputs (`final_tumor_volume`, `average_drug_concentration`, `penetration_depth`) as input features.

This removes the need for artificial fixed-reference value policies, ensuring that predictions remain physically and mathematically consistent across the entire search space.

---

## Generated Outputs

The optimization stage generates the following files in `results/result10/`:
- **`10.4 Optimized Parameters.csv`**: Contains the top 10 diverse, near-optimal formulation parameter configurations.
- **`result10_summary.txt`**: Detailed text summary reporting the best candidates and sweep statistics.
- **`10.1 Tumor Reduction vs Particle Size.png`**: Bivariate sweep curve showing optimized sizing trends.
- **`10.2 Tumor Reduction vs Drug Release Rate.png`**: Bivariate sweep curve showing optimized release constant trends.
- **`10.3 Optimization Heatmap.png`**: Bivariate parameter optimization search space.

---

## Scientific Interpretation

The ranked formulations represent optimized designs predicted by the surrogate model. Because the surrogate model is leakage-free, these formulations can be directly validated in subsequent clinical or numerical mass-transport simulations.
