# PLGA Nanoparticle Simulation

## Overview

This project presents a computational simulation of PLGA (Poly Lactic-co-Glycolic Acid) nanoparticle-based drug delivery in a tumor microenvironment.

The model simulates:

* Interstitial fluid pressure
* Velocity field generation
* Nanoparticle transport
* PLGA drug release
* Tumor growth response
* Nanoparticle size comparison
* Drug release rate comparison

The simulation is implemented in Python using finite difference methods on a 2D tumor domain.

---

## Mathematical Model

### Equation 1: Interstitial Fluid Pressure

The pressure distribution inside the tumor is computed by solving:

∇²P = 0

Outputs:

* Pressure contour map
* Velocity field

---

### Equation 2: Nanoparticle Transport

Nanoparticle concentration is governed by:

∂C/∂t = D∇²C − v·∇C − kuC

where:

* C = Nanoparticle concentration
* D = Diffusion coefficient
* v = Velocity field
* ku = Cellular uptake coefficient

Outputs:

* Concentration contour map
* Concentration vs time

---

### Equation 3: Drug Release

Drug is released from PLGA nanoparticles using a finite-release model.

Outputs:

* Drug concentration contour
* Drug release curve

---

### Equation 4: Tumor Growth

Tumor growth follows logistic growth with drug-induced cell death:

dN/dt = rN(1 − N/K) − αDN

where:

* N = Tumor cell population
* r = Growth rate
* K = Carrying capacity
* α = Drug efficacy

Outputs:

* Tumor growth curve
* Tumor reduction percentage

---

## Computational Workflow

Tumor Geometry

↓

Pressure Model

↓

Velocity Field

↓

Nanoparticle Transport

↓

Drug Release

↓

Tumor Growth

↓

Result Generation

---

## Project Structure

```text
PLGA-Nanoparticle-Simulation/

├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── docs/
│   ├── Mathematical_Model.pdf
│   └── Parameter_Impact_Guide.txt
│
├── src/
│   ├── parameters.py
│   ├── pressure_model.py
│   ├── transport_model.py
│   ├── drug_release.py
│   ├── tumor_growth.py
│   ├── result1_particle_size.py
│   ├── result2_release_rate.py
│   ├── visualization.py
│   └── solver.py
│
└── results/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mohammed-umarfarooq/plga-nanoparticle-drug-delivery.git
cd PLGA-Nanoparticle-Simulation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python main.py
```

---

## Main Results

### Result 1: Nanoparticle Penetration

Comparison of:

* 50 nm particles
* 100 nm particles
* 200 nm particles

Outputs:

* Concentration contour maps
* Penetration depth vs particle size

Research Question:

How does nanoparticle size influence penetration into tumor tissue?

---

### Result 2: Drug Release and Tumor Response

Comparison of:

* Slow release
* Medium release
* Fast release

Outputs:

* Drug concentration vs time
* Tumor cell population vs time

Research Question:

How does drug release rate affect tumor suppression?

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Finite Difference Methods

---

## Future Improvements

* 3D tumor modeling
* Multi-drug delivery systems
* Patient-specific simulations
* Experimental validation
* Machine learning optimization

---

## Author

*Mohammed Umar Farooq
*B.Tech AIML (Minor: Robotics & Automation)
*Shreya (BCS)
*Shreya (BCS)
---

## License

This project is licensed under the MIT License.
