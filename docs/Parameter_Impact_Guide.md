# Parameter and Workflow Guide

This guide summarizes the key design parameters, physical units, and scientific interpretation rules for the PLGA Nanoparticle Drug Delivery Simulation project.

---

## 1. Independent Formulation Parameters

These are the 8 variables sampled via Latin Hypercube Sampling (LHS) and used as inputs for surrogate model training:

1. **`particle_size_nm`**
   - **Range**: $20 - 200\text{ nm}$
   - **Significance**: Controls spatial diffusion (via the Stokes-Einstein relation) and cellular uptake rates. Smaller core sizes permit deeper penetration but might suffer from fast clearance.
2. **`drug_diffusion`**
   - **Range**: $10^{-10} - 10^{-8}\text{ m}^2/\text{s}$
   - **Significance**: Governing coefficient for local drug molecule movement once released from the polymer core.
3. **`np_diffusion`** (Nanoparticle Diffusion)
   - **Range**: $10^{-10} - 10^{-8}\text{ m}^2/\text{s}$
   - **Significance**: Governs Brownian motion of the PLGA nanosphere carriers in tumor tissue space.
4. **`release_rate`** ($k_{rel}$)
   - **Range**: $0.01 - 0.20\text{ h}^{-1}$
   - **Significance**: Degradation kinetic constant of the PLGA shell controlling chemotherapeutic discharge rate.
5. **`uptake_rate`** ($k_{up}$)
   - **Range**: $0.01 - 0.10\text{ h}^{-1}$
   - **Significance**: Cellular internalization rate of particles by target cancer cells.
6. **`drug_loading`**
   - **Range**: $0.20 - 1.00$ (fraction)
   - **Significance**: Determines the therapeutic payload capacity of the formulation.
7. **`tumor_growth_rate`** ($r_g$)
   - **Range**: $0.01 - 0.05\text{ h}^{-1}$
   - **Significance**: Proliferation kinetics representing patient-specific malignancy levels.
8. **`drug_efficacy`** ($E_d$)
   - **Range**: $0.30 - 1.00$ (dimensionless)
   - **Significance**: Scaling factor of the chemotherapeutic clearance rate on tumor cell volume.

---

## 2. Leakage-Free Surrogate Status

In the latest model training run:
- Derived simulation outputs (like `penetration_depth`, `average_drug_concentration`, and `final_tumor_volume`) **have been completely excluded from model inputs**.
- This makes the surrogate model **leakage-free**, allowing direct optimization of formulation variables without hardcoding references.
- Feasibility checks pass successfully, enabling direct numerical comparison.

---

## 3. Interpreting Results

- **Pressure & Transport Profiles**: Describe the baseline fluid boundary resistance and mass transfer characteristics resolved by Phase 1 grids.
- **Surrogate Regression Scores**: Evaluate the approximation quality of the tree ensemble surrogate regressor.
- **SHAP values**: Quantify the local and global contributions of input features to predictions, indicating which parameters are candidates for optimization.
- **Differential Evolution Outputs**: Identify diverse, near-optimal formulation designs for direct experimental validation.
