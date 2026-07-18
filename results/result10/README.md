# Result 10: Leakage-Free Parameter Optimization

## Objective
Identify optimal nanoparticle size and release rate configurations that maximize predicted tumor reduction percentage using bounded global Differential Evolution search.

## Methodology
Differential Evolution global search optimization using the leakage-free XGBoost model as the objective evaluator, followed by joint parameter sweeps.

## Generated Outputs
- `10.1 Tumor Reduction vs Particle Size.png`: Bivariate curve showing optimized size trends.
- `10.2 Tumor Reduction vs Drug Release Rate.png`: Bivariate curve showing optimized release constant trends.
- `10.3 Optimization Heatmap.png`: Joint parameter optimization space.
- `10.4 Optimized Parameters.csv`: Table containing the top 10 diverse, near-optimal formulation parameter configurations.
- `result10_summary.txt`: Optimization text summary report.

## Key Observations
- Using the leakage-free XGBoost model, global optimization identifies an optimal particle size of $\sim 114.1\text{ nm}$ and release rate of $\sim 0.126\text{ h}^{-1}$.
- This configuration balances transport and cellular uptake limits, yielding a predicted tumor reduction of **`94.99%`**.

## Related Figures
- [Tumor reduction vs Particle size trend](10.1 Tumor Reduction vs Particle Size.png)
- [Tumor reduction vs Release rate trend](10.2 Tumor Reduction vs Drug Release Rate.png)
- [Optimization Heatmap](10.3 Optimization Heatmap.png)

## Links to Summary Files
- Summary report: [result10_summary.txt](result10_summary.txt)
- CSV candidates: [10.4 Optimized Parameters.csv](10.4 Optimized Parameters.csv)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
