# Manuscript Table Index

This index logs all formatted tables generated under `paper/tables/` for insertion into the research paper.

| Table Number | Table Name | Source Path | Purpose |
| :--- | :--- | :--- | :--- |
| **Table 1** | Simulation Parameters | `paper/tables/Table_1_Simulation_Parameters.csv` | Lists the 8 independent parameters, their mathematical symbols, sampling bounds, physical units, and biological descriptions. |
| **Table 2** | Model Hyperparameters | `paper/tables/Table_2_Model_Hyperparameters.csv` | Lists the optimal hyperparameters (depth, estimators, learning rate, subsample) for the XGBoost model selected via randomized search CV. |
| **Table 3** | Model Evaluation Metrics | `paper/tables/Table_3_Model_Evaluation_Metrics.csv` | Summary regression metrics ($R^2$, MSE, RMSE, MAE, Max Error) for both the training (800 rows) and held-out testing (200 rows) partitions. |
| **Table 4** | Cross-Validation Scores | `paper/tables/Table_4_Cross_Validation_Scores.csv` | Folds performance breakdown from the baseline and hyperparameter tuning 5-fold cross-validations. |
| **Table 5** | Optimization Results | `paper/tables/Table_5_Optimization_Results.csv` | The top 10 diverse, near-optimal formulation parameter designs identified using the differential evolution global search. |
