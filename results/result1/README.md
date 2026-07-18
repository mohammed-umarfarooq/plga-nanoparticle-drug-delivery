# Result 1: Particle Size Study

## Objective
Evaluate the effect of nanoparticle core diameters on convective transport, radial diffusion, spatial penetration depth, and local tissue concentrations within the solid tumor.

## Methodology
Discretized two-dimensional mass transport simulation with particle sizes of 50 nm, 100 nm, and 200 nm, holding release kinetics, cellular uptake, and baseline growth parameters constant.

## Generated Outputs
- `50nm_concentration.png`: Spatial concentration contour plot for 50 nm particles.
- `100nm_concentration.png`: Spatial concentration contour plot for 100 nm particles.
- `200nm_concentration.png`: Spatial concentration contour plot for 200 nm particles.
- `penetration_depth_vs_size.png`: Penetration depth curve plotted against particle size.
- `penetration_profile_comparison.png`: Combined concentration decay profile comparison.
- `penetration_depths.txt`: Text report containing numerical statistics of maximum and mean concentration.

## Key Observations
- Decreasing particle size from 200 nm to 50 nm increases the penetration depth by over **`300%`** under standard pressure gradients.
- Transport is diffusion-dominated in areas with low velocity fields, giving a significant transport advantage to smaller nanoparticles.

## Related Figures
- [50nm Concentration Map](50nm_concentration.png)
- [100nm Concentration Map](100nm_concentration.png)
- [200nm Concentration Map](200nm_concentration.png)
- [Penetration Depth vs Size Curve](penetration_depth_vs_size.png)
- [Profile Comparison Plot](penetration_profile_comparison.png)

## Links to Summary Files
- Summary report: [penetration_depths.txt](penetration_depths.txt)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
