# Result 5: Sensitivity Analysis

## Objective
Evaluate the sensitivity of the final tumor volume to $\pm 20\%$ variations in independent physical parameters, identifying key therapeutic drivers.

## Methodology
One-way sensitivity analysis executed by perturbing eight physical parameters to $80\%$, $100\%$, and $120\%$ of their baseline values and mapping the outputs onto tornado and spider plots.

## Generated Outputs
- `tornado.png`: Tornado plot showing final tumor volume sensitivity to individual parameter variations.
- `spider.png`: Spider plot mapping response curves.
- `tumor_reduction.png`: Volume reduction plot.
- `summary.csv`: Table summarizing baseline, modified volumes, and sensitivity indexes.
- `summary.txt`: Detailed text report.

## Key Observations
- The analysis demonstrates that tumor proliferation rates ($r_g$) and drug efficacy constants ($E_d$) dominate final response metrics.
- Among the transport variables, the nanoparticle release rate ($k_{rel}$) has the highest sensitivity, highlighting it as the primary target for formulation optimization.

## Related Figures
- [Tornado Sensitivity Plot](tornado.png)
- [Spider Parameter Swing Plot](spider.png)
- [Tumor Volume Reduction comparison](tumor_reduction.png)

## Links to Summary Files
- Summary report: [summary.txt](summary.txt)
- Summary table: [summary.csv](summary.csv)

## Links to Datasets
- Core dataset: [simulation_dataset.csv](../../data/simulation_dataset.csv)
