# PLGA Nanoparticle Simulation Project Report

## 1. Title

**Computational Modeling of PLGA Nanoparticle Transport, Drug Release,
and Tumor Response in a Tumor Microenvironment**

------------------------------------------------------------------------

## 2. Abstract

This project presents a computational framework for simulating PLGA
(Poly Lactic-co-Glycolic Acid) nanoparticle-based drug delivery in a
tumor microenvironment. The framework models interstitial fluid
pressure, velocity-driven transport, nanoparticle diffusion, controlled
drug release, and tumor growth dynamics using finite difference methods.
Six computational studies are performed to investigate the effects of
nanoparticle size, drug release rate, interstitial fluid pressure,
diffusion coefficient, parameter sensitivity, and nanoparticle
optimization. The framework automatically generates figures, CSV files,
and summary reports to support computational analysis.

------------------------------------------------------------------------

## 3. Introduction

Cancer remains one of the leading causes of mortality worldwide. PLGA
nanoparticles have become an important platform for targeted drug
delivery because of their biodegradability, biocompatibility, and
controlled drug release properties.

The objective of this work is to develop a computational model capable
of predicting nanoparticle transport, drug release, and tumor response
under different physiological and design conditions.

------------------------------------------------------------------------

## 4. Objectives

-   Simulate interstitial fluid pressure.
-   Compute interstitial velocity fields.
-   Model nanoparticle transport.
-   Simulate PLGA drug release.
-   Model tumor growth under treatment.
-   Study nanoparticle size effects.
-   Study drug release rate effects.
-   Investigate interstitial fluid pressure effects.
-   Investigate diffusion coefficient effects.
-   Perform parameter sensitivity analysis.
-   Determine optimal nanoparticle design.

------------------------------------------------------------------------

## 5. Mathematical Model

### Pressure

∇²P = 0

### Velocity

v = -K∇P

### Nanoparticle Transport

∂C/∂t = D∇²C − v·∇C − kuC

Transport mechanisms:

-   Diffusion
-   Convection
-   Cellular uptake

### Drug Release

Finite controlled release from PLGA nanoparticles.

### Tumor Growth

dN/dt = rN(1 − N/K) − αDN

------------------------------------------------------------------------

## 6. Computational Workflow

Tumor Geometry

↓

Pressure Solver

↓

Velocity Solver

↓

Nanoparticle Transport

↓

Drug Release

↓

Tumor Growth

↓

Results 1--6

↓

Visualization

↓

CSV / TXT Reports

------------------------------------------------------------------------

## 7. Simulation Parameters

  Parameter                Example Value
  ------------------------ -----------------
  Tumor Size               10 × 10 mm
  Diffusion Coefficient    5×10⁻⁸ m²/s
  Hydraulic Conductivity   Model Parameter
  Drug Release Rate        Variable
  Uptake Rate              Model Parameter
  Tumor Growth Rate        Model Parameter
  Drug Efficacy            Model Parameter

------------------------------------------------------------------------

# 8. Result 1 -- Particle Size Study

Particle sizes:

-   50 nm
-   100 nm
-   200 nm

Observation:

Smaller nanoparticles penetrate deeper because of larger diffusion
coefficients.

Conclusion:

Particle size strongly affects nanoparticle penetration.

------------------------------------------------------------------------

# 9. Result 2 -- Drug Release Study

Cases:

-   Slow
-   Medium
-   Fast

Observation:

Higher release rates improve tumor suppression.

Conclusion:

Drug release kinetics significantly influence treatment effectiveness.

------------------------------------------------------------------------

# 10. Result 3 -- Interstitial Fluid Pressure Study

Research Question:

How does elevated interstitial fluid pressure affect nanoparticle
transport?

Outputs

-   Drug Concentration vs IFP
-   Penetration Depth vs IFP
-   Nanoparticle Contours

Observation

Higher IFP reduced penetration depth and average drug concentration.

Conclusion

Elevated interstitial pressure acts as a barrier to drug delivery.

------------------------------------------------------------------------

# 11. Result 4 -- Diffusion Coefficient Study

Outputs

-   Penetration Depth vs Diffusion
-   Average Concentration vs Diffusion
-   Radial Profiles

Observation

Higher diffusion coefficients improved nanoparticle penetration.

Conclusion

Diffusion is a key factor governing transport efficiency.

------------------------------------------------------------------------

# 12. Result 5 -- Sensitivity Analysis

Outputs

-   Tornado Chart
-   Spider Plot
-   Tumor Reduction Chart

Observation

Drug efficacy showed the greatest influence, while release rate had
comparatively lower influence within the tested range.

Conclusion

Sensitivity analysis identifies the parameters that most strongly affect
treatment performance.

------------------------------------------------------------------------

# 13. Result 6 -- Nanoparticle Optimization

Outputs

-   Optimization Heatmap
-   Optimization Surface
-   Objective Score Curves
-   Top-Ranked Designs

Observation

Adaptive optimization identified efficient nanoparticle designs while
reducing computational cost.

Conclusion

Optimization provides a practical method for selecting improved
nanoparticle configurations.

------------------------------------------------------------------------

# 14. Advantages

-   Modular architecture
-   Automated result generation
-   Parameter studies
-   Sensitivity analysis
-   Optimization framework
-   CSV and TXT export
-   Easy to extend

------------------------------------------------------------------------

# 15. Limitations

-   Two-dimensional model
-   Simplified tumor physiology
-   Homogeneous tissue assumptions
-   No experimental validation
-   Deterministic simulation

------------------------------------------------------------------------

# 16. Future Scope

-   Three-dimensional tumor modeling
-   Patient-specific simulations
-   Multi-drug delivery
-   GPU acceleration
-   Adaptive mesh refinement
-   Experimental validation
-   PK/PD coupling

------------------------------------------------------------------------

# 17. Technologies Used

-   Python
-   NumPy
-   Matplotlib
-   Finite Difference Method (FDM)

------------------------------------------------------------------------

# 18. Conclusion

The developed framework successfully models pressure-driven nanoparticle
transport, controlled drug release, and tumor response. Six
computational studies were performed to evaluate particle size, drug
release rate, interstitial fluid pressure, diffusion coefficient,
parameter sensitivity, and nanoparticle optimization. The results
demonstrate the capability of the framework to investigate factors
influencing nanoparticle-mediated drug delivery and provide a foundation
for future computational and experimental research.

------------------------------------------------------------------------

# 19. References

1.  PLGA-Based Nanoparticles for Cancer Therapy.
2.  Nanoparticle Drug Delivery Systems.
3.  Mathematical Modeling of Tumor Growth.
4.  Drug Transport in Tumor Microenvironments.
5.  Controlled Drug Release from PLGA Nanoparticles.
