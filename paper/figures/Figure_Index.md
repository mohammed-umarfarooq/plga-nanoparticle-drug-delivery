# Manuscript Figure Index

This index logs all figures prepared in `paper/figures/` for inclusion in the research manuscript.

| Figure Number | Filename | Source Path | Description | Result Section |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | `Figure_1_Pressure_Distribution.png` | `results/pressure/pressure_contour.png` | Interstitial Fluid Pressure (IFP) radial contour plots showing high pressure at the tumor core. | Baseline Fluid Model |
| **Figure 2** | `Figure_2_Velocity_Field.png` | `results/pressure/velocity_field.png` | Darcy velocity vector field inside and surrounding the tumor boundary. | Baseline Fluid Model |
| **Figure 3** | `Figure_3_Nanoparticle_Penetration.png` | `results/result1/penetration_profile_comparison.png` | Radial concentration comparison curves for 50nm, 100nm, and 200nm particles. | Result 1 (Size Study) |
| **Figure 4** | `Figure_4_Drug_Release_Tumor_Response.png` | `results/result2/tumor_response_comparison.png` | Comparison of tumor volume decay curves under slow, medium, and fast drug release profiles. | Result 2 (Release kinetics) |
| **Figure 5** | `Figure_5_IFP_Impact.png` | `results/result3/penetration_vs_ifp.png` | Plot of nanoparticle penetration depth as a function of elevated interstitial boundary fluid pressures. | Result 3 (IFP Study) |
| **Figure 6** | `Figure_6_Sensitivity_Tornado.png` | `results/result5/tornado.png` | Tornado plot illustrating sensitivity of final tumor volume to $\pm 20\%$ variations in parameters. | Result 5 (Sensitivity) |
| **Figure 7** | `Figure_7_Sensitivity_Spider.png` | `results/result5/spider.png` | Spider plot displaying parameter response trends on tumor volume reduction. | Result 5 (Sensitivity) |
| **Figure 8** | `Figure_8_Dataset_Distribution.png` | `results/result7/7.2 Parameter Distribution.png` | Multi-panel histogram showing distributions of the 1,000 accepted Latin Hypercube samples. | ML Dataset Generation |
| **Figure 9** | `Figure_9_Model_Performance.png` | `results/result8/8.1 Actual vs Predicted.png` | Scatter plot of actual vs. XGBoost-predicted tumor volume reduction percentages on held-out test data. | ML Evaluation |
| **Figure 10**| `Figure_10_Residual_Analysis.png` | `results/result8/8.2 Residual Plot.png` | Residual values plotted against predicted targets to check for homoscedasticity. | ML Evaluation |
| **Figure 11**| `Figure_11_SHAP_Summary.png` | `results/result9/9.1 SHAP Summary Plot.png` | Global Tree SHAP feature attribution summary plot ranking features. | SHAP Explainability |
| **Figure 12**| `Figure_12_SHAP_Importance.png` | `results/result9/9.2 SHAP Feature Importance.png` | Bar chart of mean absolute SHAP values for the independent inputs. | SHAP Explainability |
| **Figure 13**| `Figure_13_Joint_Optimization_Heatmap.png` | `results/result10/10.3 Optimization Heatmap.png` | Heatmap sweep showing predicted tumor reduction over particle sizes and release rates. | Parameter Optimization |
