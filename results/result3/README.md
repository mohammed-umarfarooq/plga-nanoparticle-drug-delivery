# Result 3: Interstitial Fluid Pressure Study

## Objective
Evaluate how varying boundaries of elevated Interstitial Fluid Pressure (IFP) affect nanoparticle transport barriers, convective velocity magnitudes, and treatment outcomes.

## Methodology
A parameter sweep over boundary Interstitial Fluid Pressures (5, 10, 15, 20, 25, and 30 mmHg) is executed, resolving fluid velocities and local concentrations on a finite difference grid.

## Generated Outputs
- `ifp_5mmHg.png` to `ifp_30mmHg.png`: Spatial concentration contours under varied pressures.
- `drug_vs_ifp.png`: Local drug concentration as a function of pressure.
- `penetration_vs_ifp.png`: Penetration depth curve as a function of pressure.
- `summary.csv`: Grid data listing pressure boundary, velocity scale, effective diffusion, and final tumor volume.
- `summary.txt`: Descriptive summary of parameters.

## Key Observations
- Increasing boundary pressure from 5 mmHg to 30 mmHg reduces average drug concentration in tissue by over **`70%`**.
- Higher pressure boundaries restrict nanoparticle penetration depth, demonstrating why physical transport barriers are a major cause of chemotherapy failure.

## Related Figures
- [Concentration vs Pressure sweep](drug_vs_ifp.png)
- [Penetration Depth vs Pressure Curve](penetration_vs_ifp.png)
- [IFP 10 mmHg concentration contour](ifp_10mmHg.png)

## Links to Summary Files
- Summary report: [summary.txt](summary.txt)
- Summary table: [summary.csv](summary.csv)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
