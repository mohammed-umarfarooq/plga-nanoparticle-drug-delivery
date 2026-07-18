# Reference Notes & Literature Review

This document aggregates research notes for framing the introduction, methodology, and discussion sections of the manuscript.

## Key Themes

### 1. Interstitial Fluid Pressure (IFP) and Transport Barriers
- *Context*: High IFP is a hallmark of solid tumors caused by blood vessel abnormalities and poor lymphatic drainage. This inhibits transport, leading to poor drug distribution.
- *Notes*: Cite Jain (2007) for characterization of fluid barriers. Our computational model simulates radial velocity scaling under elevated boundary pressures, mapping directly to these physiological features.

### 2. PLGA Nanoparticles for Controlled Release
- *Context*: PLGA degrades through bulk hydrolysis, permitting biphasic release (diffusion followed by degradation/burst).
- *Notes*: Core parameters are `particle_size_nm` (influencing diffusion and cellular uptake) and `release_rate` (degradation kinetics). Our simulation demonstrates the significance of balancing size and release rate to avoid premature clearance.

### 3. Tree-based Surrogates for PDE Models
- *Context*: PDE simulations are computationally intensive. Tree boosting regressor (XGBoost) provides a high-fidelity surrogate model to bypass grid calculations in global parameter search.
- *Notes*: Discuss why tree ensembles are robust to boundary step functions but require density constraints (SHAP explainability) to avoid extrapolation errors outside sampling bounds.
