# Machine Learning-Accelerated Optimization of PLGA Nanoparticles for Tumoral Drug Delivery

![Repository Banner](paper/assets/banner/repository_banner.jpg)
*Figure 1: Title Banner - Machine Learning-Accelerated physics-based framework for targeted chemotherapeutic nanoparticle drug delivery systems.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic Publication](https://img.shields.io/badge/Paper-IEEE--Manuscript-orange.svg)](paper/IEEE_Manuscript.md)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

This repository houses a hybrid **numerical physics-based mass transport simulation and machine learning surrogate pipeline** to optimize **PLGA (Poly Lactic-co-glycolic Acid) nanoparticle formulations** for targeted chemotherapeutic delivery in solid tumors.

---

## 1. Project Overview & Research Motivation

In solid tumors, elevated fluid pressure at the core restricts chemotherapeutic entry. Delivering drugs using polymer nanoparticles (PLGA) allows for convective transport and cellular internalization, bypassing vascular barriers. However, searching the parameter space (particle size, degradation kinetics, diffusion) requires solving coupled partial differential equations (PDEs) over long grids, which is computationally expensive.

This framework introduces a **leakage-free machine learning surrogate regressor (XGBoost)** trained on a quality-controlled simulation dataset ($N=1,000$). The surrogate model accelerates design evaluations from hours to milliseconds, enabling global optimization via **Differential Evolution**.

---

## 2. Complete Research Workflow

![Complete Research Workflow](paper/assets/workflow/complete_research_workflow.jpg)
*Figure 2: Complete Research Workflow. The workflow is divided into Phase 1 (generating physical convection-diffusion grids), Phase 2 (surrogate regression, training, and explainability), and Phase 3 (global optimization and validation checks).*

---

## 3. System Architecture

![System Architecture](paper/assets/architecture/system_architecture.jpg)
*Figure 3: System Architecture. Illustrates the modular boundaries separating Darcy fluid solvers, advective transport loops, XGBoost estimators, and Differential Evolution optimization.*

---

## 4. Repository Structure

![Repository Structure](paper/assets/repository/repository_structure.jpg)
*Figure 4: Repository structure map showing data directories, ML solvers, result plots, docs, and manuscript drafting folders.*

---

## 5. Machine Learning Pipeline

![Machine Learning Pipeline](paper/assets/machine_learning/machine_learning_pipeline.jpg)
*Figure 5: Machine Learning Pipeline. Outlines dataset preparation steps, train/test splitting, scaler mapping, cross-validation tuning, and model evaluations.*

---

## 6. Optimization Framework

![Optimization Framework](paper/assets/optimization/optimization_framework.jpg)
*Figure 6: Optimization Framework. Bounded global search leverages the leakage-free XGBoost model to evaluate candidates within Differential Evolution iterations.*

---

## 7. Example Simulation Results

### Phase 1: Fluid Pressure and Nanoparticle Transport Solver
Elevated pressure at the tumor core restricts standard molecular entry. Our Darcy solver resolves pressure distributions, showing convective velocity vectors pointing outward at boundaries.

![Figure 7: Interstitial Fluid Pressure Contour](paper/figures/simulation/Figure_1_Pressure_Distribution.png)
*Figure 7: Interstitial fluid pressure (IFP) contours. Fluid pressure peaks at 10 mmHg ($1.33\text{ kPa}$) in the core tissue boundary, preventing passive filtration.*

![Figure 8: Interstitial Velocity Field](paper/figures/simulation/Figure_2_Velocity_Field.png)
*Figure 8: Interstitial velocity vector fields computed via Darcy's Law ($v = -K \nabla P$). Vector directions highlight convective resistance at outer boundaries.*

Our transport solver simulates the penetration of different particle sizes (50 nm, 100 nm, 200 nm), indicating that smaller particles achieve superior spatial penetration depths despite rapid clearance.

![Figure 9: Nanoparticle Penetration Profiles](paper/figures/simulation/Figure_3_Nanoparticle_Penetration.png)
*Figure 9: Nanoparticle radial concentration profiles comparison. Smaller particles ($50\text{ nm}$) show significantly deeper tissue penetration compared to larger ($200\text{ nm}$) alternatives.*

![Figure 10: Drug Release and Tumor Volume Response](paper/figures/simulation/Figure_4_Drug_Release_Tumor_Response.png)
*Figure 10: Sustained release of chemotherapeutics under varied PLGA degradation constants. Faster release rates trigger earlier tumor regression but are bounded by drug elimination half-lives.*

### Phase 2: Sensitivity Analysis & ML Dataset Generation
Tornado and spider plots identify that baseline tumor proliferation rate ($r_g$) and drug efficacy ($E_d$) dominate responses.

![Figure 11: One-Way Sensitivity Analysis Tornado Plot](paper/figures/simulation/Figure_6_Sensitivity_Tornado.png)
*Figure 11: Tornado plot showing final tumor volume sensitivity to $\pm 20\%$ variations in parameter values. Physiological indicators dominate mechanical transport variables.*

The 1,000 accepted configurations generated using LHS sampling represent physically consistent boundaries.

![Figure 12: Parameter Sampling Distribution Panel](paper/figures/machine_learning/Figure_8_Dataset_Distribution.png)
*Figure 12: Latin Hypercube Sampling (LHS) parameter distributions for the independent simulation inputs ($N=1,000$).*

### Phase 3: Machine Learning Surrogate Evaluation & SHAP Interpretations
The hyperparameter-tuned XGBoost surrogate regressor trained on independent variables predicts tumor reduction percentages with high generalization accuracy.

![Figure 13: Model Actual vs Predicted Curve](paper/figures/machine_learning/Figure_9_Model_Performance.png)
*Figure 13: Actual vs. predicted tumor reduction percentages on held-out test data. The surrogate achieves $R^2 = 0.951075$ with tight variance.*

![Figure 14: Regression Residual Diagnostics](paper/figures/machine_learning/Figure_10_Residual_Analysis.png)
*Figure 14: Residual vs. predicted plot showing homoscedastic distribution with zero systematic bias.*

Tree SHAP explainability ranks drug efficacy ($E_d$), tumor growth rate ($r_g$), and drug release rate ($k_{rel}$) as the primary drivers of efficacy.

![Figure 15: Tree SHAP Global Summary Plot](paper/figures/machine_learning/Figure_11_SHAP_Summary.png)
*Figure 15: Global Tree SHAP feature attributions showing feature impacts across the testing cohort.*

![Figure 16: Tree SHAP Mean Absolute Importance](paper/figures/machine_learning/Figure_12_SHAP_Importance.png)
*Figure 16: Feature importance ranking based on mean absolute SHAP values.*

### Phase 4: Parameter Optimization Results
Using the leakage-free XGBoost model, Differential Evolution global search identified formulation designs that maximize therapeutic exposure while minimizing tumor volume.

![Figure 17: Joint Optimization Sweep Heatmap](paper/figures/optimization/Figure_13_Joint_Optimization_Heatmap.png)
*Figure 17: Joint parameter sweep heatmap illustrating predicted tumor reduction over varied particle sizes and release rates.*

The top-ranked optimal design offers a predicted tumor reduction of **`94.99%`** with parameters:
- Core Particle Size: **`114.1 nm`**
- Sustained Release Constant: **`0.126 h^-1`**
- Nanoparticle Diffusion: **`5.46e-9 m^2/s`**

---

## 8. Graphical Abstract

![Graphical Abstract](paper/assets/graphical_abstract/graphical_abstract.jpg)
*Figure 18: Graphical Abstract. Summary illustration showing the complete physics-to-AI formulation search loop.*

---

## 9. Installation & Quick Start

### Prerequisites
- Python `3.10` or later
- Git

### Installation Commands
```bash
# Clone the repository
git clone https://github.com/mohammed-umarfarooq/plga-nanoparticle-drug-delivery.git
cd plga-nanoparticle-drug-delivery

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Reproducing the Pipeline
Run the following scripts sequentially from the repository root:
```bash
# 1. Execute computational simulation and create Results 1-6
python main.py

# 2. Sample 1,000 accepted simulation cases using LHS
python ml/generate_dataset.py

# 3. Process, split, and analyze the dataset
python ml/preprocess.py
python ml/split_dataset.py
python ml/eda.py

# 4. Train, evaluate, and interpret the XGBoost surrogate
python ml/train_model.py
python ml/evaluate_model.py
python ml/shap_analysis.py

# 5. Search for optimal formulations and validate feasibility
python ml/optimize_parameters.py
python ml/validate_optimization.py
python ml/generate_result7.py

# 6. Run Pearson correlation standalone analysis
$env:MPLBACKEND="Agg"; python analysis.py
```

---

## 10. Future Work
- Verify top Differential Evolution candidates through in vitro and in vivo tumor transport models.
- Support 3D tissue geometry scans using CT imaging slices.
- Incorporate multi-layered shell structures (e.g. lipid-covered polymers) inside numerical kinetic models.
- Integrate active targeting ligands (e.g., folate-decorated shells) inside transport models.

---

## 11. Citations & Academic References

If you utilize this computational framework or surrogate optimization models in academic work, please cite:

```text
@misc{farooq2026plga,
  title={A Machine Learning-Accelerated physics-based surrogate model for bounded optimization of PLGA drug delivery systems},
  author={Farooq, Mohammed Umar},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/mohammed-umarfarooq/plga-nanoparticle-drug-delivery}},
  year={2026}
}
```

---

## 12. License
This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 13. Author and Acknowledgement
- **Mohammed Umar Farooq** - B.Tech AIML (Minor: Robotics & Automation)
- Framework developed under the supervision of the Department of Nanotechnology.
