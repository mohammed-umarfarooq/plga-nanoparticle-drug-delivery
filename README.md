# PLGA Nanoparticle Drug Delivery Simulation

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

A research-oriented computational framework for simulating **PLGA (Poly Lactic-co-Glycolic Acid) nanoparticle-based drug delivery** in a tumor microenvironment using the **Finite Difference Method (FDM)**.

The framework models pressure-driven nanoparticle transport, controlled drug release, tumor growth dynamics, and multiple parameter studies to evaluate factors influencing therapeutic efficacy.

---

# Features

The framework includes:

- Interstitial Fluid Pressure (IFP) simulation
- Interstitial velocity field computation
- Nanoparticle diffusion and convection
- Cellular uptake
- Controlled PLGA drug release
- Tumor growth simulation
- Automatic visualization
- CSV and TXT report generation

Implemented studies:

- Result 1 – Particle Size Study
- Result 2 – Drug Release Rate Study
- Result 3 – Interstitial Fluid Pressure Study
- Result 4 – Diffusion Coefficient Study
- Result 5 – Sensitivity Analysis
- Result 6 – Nanoparticle Design Optimization

---

# Mathematical Model

## Pressure

\[
\nabla^2P = 0
\]

Outputs

- Pressure contour
- Pressure distribution

---

## Velocity

Darcy's Law

\[
\mathbf{v}=-K\nabla P
\]

Outputs

- Velocity field
- Velocity magnitude

---

## Nanoparticle Transport

\[
\frac{\partial C}{\partial t}
=
D\nabla^2C
-
\mathbf{v}\cdot\nabla C
-
k_uC
\]

Transport mechanisms

- Diffusion
- Convection
- Cellular uptake

Outputs

- Concentration contour
- Penetration depth
- Concentration history

---

## Drug Release

Finite controlled release from PLGA nanoparticles.

Outputs

- Drug concentration
- Drug release curve

---

## Tumor Growth

\[
\frac{dN}{dt}
=
rN
\left(1-\frac{N}{K}\right)
-
\alpha DN
\]

Outputs

- Tumor growth
- Tumor reduction
- Treatment response

---

# Computational Workflow

```
Tumor Geometry
      │
      ▼
Pressure Solver
      │
      ▼
Velocity Solver
      │
      ▼
Nanoparticle Transport
      │
      ▼
Drug Release
      │
      ▼
Tumor Growth
      │
      ▼
Results 1–6
      │
      ▼
Visualization
      │
      ▼
CSV / TXT Reports
```

---

# Project Structure

```text
PLGA-Nanoparticle-Simulation/
│
├── main.py
├── debug_run.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_simulation.log
│
├── docs/
│
├── src/
│   ├── __init__.py
│   ├── parameters.py
│   ├── utils.py
│   ├── solver.py
│   ├── pressure_model.py
│   ├── transport_model.py
│   ├── drug_release.py
│   ├── tumor_growth.py
│   ├── visualization.py
│   ├── result1_particle_size.py
│   ├── result2_release_rate.py
│   ├── result3_ifp.py
│   ├── result4_diffusion.py
│   ├── result5_sensitivity.py
│   └── result6_optimization.py
│
└── results/
```

---

# Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- SciPy
- pandas

---

# Installation

Clone the repository

```bash
git clone https://github.com/mohammed-umarfarooq/plga-nanoparticle-drug-delivery.git

cd sim

pip install -r requirements.txt
```

---

# Running the Simulation

```bash
python main.py
```

Expected runtime

Approximately **2–5 minutes** (depending on system configuration).

---

# Simulation Studies

## Result 1 — Particle Size Study

Investigates the influence of nanoparticle size on penetration depth.

Outputs

- Concentration contours
- Penetration depth comparison

---

## Result 2 — Drug Release Rate Study

Investigates different PLGA release profiles.

Outputs

- Drug release curves
- Tumor response
- Tumor reduction

---

## Result 3 — Interstitial Fluid Pressure Study

Investigates the effect of elevated tumor pressure.

Outputs

- Drug Concentration vs IFP
- Penetration Depth vs IFP
- Nanoparticle contour maps

---

## Result 4 — Diffusion Coefficient Study

Evaluates diffusion-controlled transport.

Outputs

- Penetration depth
- Average concentration
- Radial concentration profiles

---

## Result 5 — Sensitivity Analysis

Determines the influence of model parameters.

Outputs

- Tornado chart
- Spider plot
- Tumor reduction chart

---

## Result 6 — Nanoparticle Optimization

Determines an optimal nanoparticle configuration.

Outputs

- Optimization heatmap
- Optimization surface
- Objective score plots
- Ranked nanoparticle designs

---

# Validation

Automatic validation includes

- Penetration depth
- Drug release characteristics
- Tumor reduction
- Sensitivity ranking
- Optimization ranking

Validation summaries are printed to the console and exported with simulation results.

---

# Generated Outputs

The framework automatically generates

- Pressure contours
- Velocity fields
- Nanoparticle concentration maps
- Drug release curves
- Tumor growth curves
- Result 1–6 figures
- CSV summary tables
- TXT summary reports

All outputs are stored in the **results/** directory.

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- SciPy
- pandas
- Finite Difference Method (FDM)

---

# Limitations

- Two-dimensional tumor model
- Homogeneous tissue assumptions
- Deterministic simulation
- No experimental validation
- Simplified tumor microenvironment

---

# Future Improvements

- Three-dimensional tumor modeling
- Multi-drug delivery
- Patient-specific simulations
- Adaptive mesh refinement
- GPU acceleration
- PK/PD coupling
- Experimental validation

---

## Authors

| Name | Course / Specialization |
| :--- | :--- |
| **Mohammed Umar Farooq** | B.Tech AIML (Minor: Robotics & Automation) |

---

# License

This project is licensed under the MIT License.

---

# Citation

If you use this project for academic or research purposes, please cite the repository and acknowledge the author.

---

⭐ If you find this project useful, consider giving the repository a star.
