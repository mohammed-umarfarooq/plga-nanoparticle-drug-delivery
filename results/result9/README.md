# Result 9: SHAP Explainability Study

## Objective
Explain the trained XGBoost model predictions using coalition game theory (Tree SHAP), mapping feature contributions globally and locally.

## Methodology
Tree SHAP is computed over the 200 testing samples to calculate feature attributions, global summaries, and dependencies.

## Generated Outputs
- `9.1 SHAP Summary Plot.png`: Global Tree SHAP dot plot.
- `9.2 SHAP Feature Importance.png`: Feature ranking based on mean absolute SHAP value.
- `9.3 SHAP Dependence Plot - Particle Size.png`: Interaction scatter plot for particle size.
- `9.4 SHAP Dependence Plot - Drug Release Rate.png`: Interaction scatter plot for drug release rate.
- `result9_summary.txt`: Numerical summary of feature rankings and outlier profiles.

## Key Observations
- SHAP values show that drug efficacy ($E_d$) and tumor growth rate ($r_g$) are the primary drivers of final tumor volume reduction.
- Particle size exhibits an optimal range near $100-120\text{ nm}$ because smaller sizes allow deep transport but can be limited by low cellular uptake rates.

## Related Figures
- [SHAP summary dot plot](9.1 SHAP Summary Plot.png)
- [SHAP importance bar chart](9.2 SHAP Feature Importance.png)
- [SHAP particle size dependency](9.3 SHAP Dependence Plot - Particle Size.png)

## Links to Summary Files
- Summary report: [result9_summary.txt](result9_summary.txt)

## Links to Datasets
- Test dataset: [test.csv](../../data/test.csv)
