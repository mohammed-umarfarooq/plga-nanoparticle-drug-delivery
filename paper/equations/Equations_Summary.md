# Mathematical Catalog of Implemented Equations

This document catalog lists all mathematical equations implemented in the computational simulation and machine learning pipeline, along with their parameters, units, implementation details, and biological contexts.

---

## 1. Interstitial Fluid Pressure (IFP)
* **Equation Name**: Fluid Flow in Porous Media (Modified Darcy Equation)
* **Mathematical Formulation**:
  $$\nabla^2 P_i - \frac{L_p S_V}{K} (P_i - P_e) = 0$$
  *Where:*
  - $P_i$: Interstitial fluid pressure [mmHg] (or converted to Pa)
  - $K$: Hydraulic conductivity [$\text{m}^2/\text{Pa}\cdot\text{s}$]
  - $L_p$: Microvascular filtration coefficient [$\text{m}/\text{Pa}\cdot\text{s}$]
  - $S_V$: Surface area per unit volume [$\text{m}^{-1}$]
  - $P_e$: Effective microvascular pressure [mmHg]
* **Purpose**: Simulates elevated fluid pressure distributions across the tumor mass.
* **Source Code Location**: [src/solver.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/solver.py) in function `solve_pressure`.
* **Literature Reference**: *Jain, R. K. (2007). Interstitial fluid pressure in tumors: therapeutic barriers and opportunities. Cancer Research.*

---

## 2. Darcy Velocity Field
* **Equation Name**: Darcy's Law for Velocity
* **Mathematical Formulation**:
  $$\mathbf{v} = -K \nabla P_i$$
  *Where:*
  - $\mathbf{v}$: Fluid velocity vector [m/s]
  - $K$: Hydraulic conductivity [$\text{m}^2/\text{Pa}\cdot\text{s}$]
  - $\nabla P_i$: Pressure gradient [Pa/m]
* **Purpose**: Calculates the convective velocity field driving advective transport.
* **Source Code Location**: [src/solver.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/solver.py) in function `compute_velocity`.
* **Literature Reference**: *Darcy, H. (1856). Les fontaines publiques de la ville de Dijon.*

---

## 3. Nanoparticle Transport
* **Equation Name**: Convection-Diffusion-Reaction Equation for Nanoparticles
* **Mathematical Formulation**:
  $$\frac{\partial C_{np}}{\partial t} + \nabla \cdot (\mathbf{v} C_{np}) = D_{np} \nabla^2 C_{np} - k_{up} C_{np}$$
  *Where:*
  - $C_{np}$: Local nanoparticle concentration [fraction or arbitrary units]
  - $\mathbf{v}$: Fluid velocity field [m/s]
  - $D_{np}$: Nanoparticle diffusion coefficient in tissue [$\text{m}^2/\text{s}$]
  - $k_{up}$: Cellular uptake coefficient [$\text{h}^{-1}$]
* **Purpose**: Simulates radial diffusion, convective transport, and cellular ingestion of particles inside tissue.
* **Source Code Location**: [src/solver.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/solver.py) in function `solve_transport`.
* **Literature Reference**: *Bird, R. B., Stewart, W. E., & Lightfoot, E. N. (2007). Transport Phenomena.*

---

## 4. Drug Release and Dispersion
* **Equation Name**: First-Order Controlled Release and Diffusion
* **Mathematical Formulation**:
  $$\frac{\partial C_{drug}}{\partial t} = k_{rel} C_{np} + D_d \nabla^2 C_{drug} - \lambda_d C_{drug}$$
  *Where:*
  - $C_{drug}$: Local drug concentration [arbitrary units]
  - $k_{rel}$: Drug release constant [$\text{h}^{-1}$] (PLGA degradation kinetic)
  - $D_d$: Small-molecule drug diffusion coefficient [$\text{m}^2/\text{s}$]
  - $\lambda_d$: Drug elimination/decay coefficient [$\text{h}^{-1}$]
* **Purpose**: Governs the sustained release of drug molecules from PLGA cores and their subsequently rapid diffusion in tissue.
* **Source Code Location**: [src/solver.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/solver.py) in function `solve_drug_release`.
* **Literature Reference**: *Fredenberg, S., et al. (2011). Unraveling the mechanisms of drug release from metal-and polymer-based drug-delivering systems.*

---

## 5. Tumor Volume Response
* **Equation Name**: Logistic Growth with Drug-Induced Clearance
* **Mathematical Formulation**:
  $$\frac{d V}{d t} = r_g V \left(1 - \frac{V}{V_{max}}\right) - E_d \bar{C}_{drug} V$$
  *Where:*
  - $V$: Tumor cell volume or count
  - $r_g$: Baseline tumor cell proliferation rate [$\text{h}^{-1}$]
  - $V_{max}$: Tumor carrying capacity or limit
  - $E_d$: Drug efficacy coefficient
  - $\bar{C}_{drug}$: Average drug concentration in the tumor domain
* **Purpose**: Simulates the dynamic volume response of the tumor under the influence of released chemotherapeutic molecules.
* **Source Code Location**: [src/solver.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/solver.py) in function `solve_tumor_growth`.
* **Literature Reference**: *Simeoni, M., et al. (2004). Predictive pharmacokinetic-pharmacodynamic modeling of tumor growth kinetics in xenograft models after administration of anticancer agents. Cancer Research.*

---

## 6. Optimization Objective Function
* **Equation Name**: Custom Weighted Multi-Objective Formulation
* **Mathematical Formulation**:
  $$\text{Objective Score} = w_1 \left( \frac{\bar{C}_{drug}}{\max(\bar{C}_{drug}) + \epsilon} \right) - w_2 \left( \frac{V_{final}}{\max(V_{final}) + \epsilon} \right)$$
  *Where:*
  - $w_1 = 0.7$ (weight maximizing therapeutic exposure)
  - $w_2 = 0.3$ (weight penalizing residual tumor volume)
  - $\epsilon = 10^{-6}$ (numerical stabilization factor)
* **Purpose**: Guides hierarchical grid searches and global evolutionary search algorithms toward designs offering maximal efficacy.
* **Source Code Location**: [src/result6_optimization.py](file:///c:/Users/moham/Downloads/plga-nanoparticle-drug-delivery-main/src/result6_optimization.py) in function `_evaluate_candidate`.
* **Literature Reference**: *Coello, C. A. C. (2006). Evolutionary algorithms for solving multi-objective problems.*
