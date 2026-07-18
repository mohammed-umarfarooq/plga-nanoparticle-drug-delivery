# Result 4: Diffusion Coefficient Study

## Objective
Analyze the impact of varying nanoparticle tissue diffusion coefficients ($D_{np}$) on average concentrations, peak accumulation, and radial spatial profiles.

## Methodology
A grid search sweeping the nanoparticle diffusion coefficient from $10^{-10}$ to $10^{-8}\text{ m}^2/\text{s}$ is solved to measure spatial boundary penetration and centerline concentration decay.

## Generated Outputs
- `diffusion_vs_average.png`: Plot of average concentration versus tissue diffusion coefficient.
- `diffusion_vs_penetration.png`: Plot of penetration depth versus tissue diffusion coefficient.
- `radial_profiles.png`: Combined center-line profiles.
- `summary.csv`: CSV table containing diffusion coefficient sweeps and penetration depths.
- `summary.txt`: Numerical summary description.

## Key Observations
- There is a monotonic relationship between the tissue diffusion coefficient and the average spatial penetration depth.
- Increasing the diffusion coefficient from $10^{-10}$ to $10^{-8}\text{ m}^2/\text{s}$ increases the final nanoparticle accumulation in the tumoral core by an order of magnitude.

## Related Figures
- [Average concentration vs Diffusion](diffusion_vs_average.png)
- [Penetration vs Diffusion Curve](diffusion_vs_penetration.png)
- [Center-line Profiles Map](radial_profiles.png)

## Links to Summary Files
- Summary report: [summary.txt](summary.txt)
- Summary table: [summary.csv](summary.csv)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
