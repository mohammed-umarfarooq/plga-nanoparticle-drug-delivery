# Result 8: Model Performance & Cross-Validation Study

## Objective
Evaluate the predictive performance of the hyperparameter-tuned XGBoost surrogate regressor, examining cross-validation scores and testing error diagnostics.

## Methodology
Evaluates the surrogate on an 80/20 train/test split. Cross-validation distributions and residuals are checked for homoscedasticity.

## Generated Outputs
- `8.1 Actual vs Predicted.png`: Scatter plot mapping predictions against true simulation values.
- `8.2 Residual Plot.png`: Residual vs. predicted diagnostic.
- `8.3 Cross-Validation Candidate Scores.png`: Score distributions across CV candidates.
- `8.4 Performance Metrics.txt`: Summary metrics ($R^2$, MSE, RMSE, MAE, Max Error) for train and test cohorts.

## Key Observations
- The tuned XGBoost regressor achieves high generalization capability ($R^2 = 0.951$ on testing data).
- Residual diagnostics confirm homoscedasticity, showing that errors do not depend on the predicted magnitude.

## Related Figures
- [Actual vs Predicted scatter plot](8.1 Actual vs Predicted.png)
- [Residual plot](8.2 Residual Plot.png)
- [Cross validation scores distribution](8.3 Cross-Validation Candidate Scores.png)

## Links to Summary Files
- Performance report: [8.4 Performance Metrics.txt](8.4 Performance Metrics.txt)
- CV table: [cross_validation_results.csv](../../results/cross_validation_results.csv)

## Links to Datasets
- Train dataset: [train.csv](../../data/train.csv)
- Test dataset: [test.csv](../../data/test.csv)
