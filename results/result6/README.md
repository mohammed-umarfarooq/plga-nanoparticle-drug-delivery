# Result 6: Adaptive Hierarchical Optimization Study

## Objective
Identify optimal nanoparticle size and release rate configurations that maximize therapeutic exposure while minimizing tumor size using adaptive grid search.

## Methodology
Coarse grids are swept across particle sizes ($20-100\text{ nm}$) and release rates ($0.01-0.09\text{ h}^{-1}$) and subsequently refined around identified local peaks to optimize a composite carrying objective score.

## Generated Outputs
- `surface3d.png`: 3D response surface plot of objective score vs. particle size and release rate.
- `heatmap.png`: 2D projection heatmap of the search space.
- `objective_vs_size.png`: Bivariate curve showing size impact.
- `objective_vs_release.png`: Bivariate curve showing release rate impact.
- `summary.txt`: Statistics of coarse and refined solves.
- `optimal_design.txt`: Detailed parameters of the top candidate designs.

## Key Observations
- The objective function balances average concentration (70% weight) and tumor volume clearance (30% weight). 
- The optimization reveals that the best configuration leverages smaller particle sizes ($\sim 20-30\text{ nm}$) to maximize penetration, combined with an intermediate-to-fast release rate ($\sim 0.09\text{ h}^{-1}$) to avoid drug elimination boundaries.

## Related Figures
- [3D objective Response Surface](surface3d.png)
- [2D parameter Sweep Heatmap](heatmap.png)
- [Objective vs Size curve](objective_vs_size.png)

## Links to Summary Files
- Summary report: [summary.txt](summary.txt)
- Optimal design parameters: [optimal_design.txt](optimal_design.txt)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
