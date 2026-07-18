"""Standalone parameter impact and weight analysis script.

This module loads the completed simulation dataset, filters the top 20% most effective 
formulations, computes their optimal parameter bounds (95% CI), ranks inputs using 
Pearson Correlation coefficients, and writes the analytical results to Excel format.
"""

# ============================================================
# PARAMETER ANALYSIS USING PEARSON CORRELATION ONLY
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/simulation_dataset.csv")

# Remove missing values
df = df.dropna()

# ============================================================
# DEFINE TARGET VARIABLE
# ============================================================

target = "tumor_reduction_percent"

# ============================================================
# SELECT ALL NUMERIC PARAMETERS EXCEPT TARGET
# ============================================================

input_parameters = df.select_dtypes(include=np.number).columns.tolist()

# Remove target
input_parameters.remove(target)

# Remove constant columns
input_parameters = [
    col for col in input_parameters
    if df[col].nunique() > 1
]

print("\nParameters Included in Analysis:\n")

for col in input_parameters:
    print(col)

# ============================================================
# PART 1 : OPTIMAL BIOLOGICAL RANGE
# ============================================================

print("\nCalculating Optimal Parameter Ranges...")

# Top 20% simulations
threshold = df[target].quantile(0.80)

best = df[df[target] >= threshold]

range_results = []

for param in input_parameters:

    range_results.append({

        "Parameter": param,

        "Minimum": df[param].min(),

        "Maximum": df[param].max(),

        "Mean": df[param].mean(),

        "Median": df[param].median(),

        "Standard Deviation": df[param].std(),

        "Optimal Lower Limit (5%)":
            best[param].quantile(0.05),

        "Optimal Upper Limit (95%)":
            best[param].quantile(0.95)

    })

range_df = pd.DataFrame(range_results)

# ============================================================
# PART 2 : PEARSON CORRELATION
# ============================================================

print("\nCalculating Pearson Correlation...")

impact = []

for param in input_parameters:

    r, p = pearsonr(df[param], df[target])

    impact.append({

        "Parameter": param,

        "Pearson Correlation": r,

        "Absolute Correlation": abs(r),

        "P-value": p

    })

impact_df = pd.DataFrame(impact)

# ============================================================
# PART 3 : PARAMETER WEIGHTS
# ============================================================

total = impact_df["Absolute Correlation"].sum()

impact_df["Weight"] = (
    impact_df["Absolute Correlation"] / total
)

impact_df["Weight (%)"] = (
    impact_df["Weight"] * 100
)

# ============================================================
# PART 4 : FINAL RANKING
# ============================================================

impact_df = impact_df.sort_values(

    by="Weight",

    ascending=False

)

impact_df["Rank"] = range(1, len(impact_df)+1)

final_df = pd.merge(

    range_df,

    impact_df,

    on="Parameter"

)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("OPTIMAL PARAMETER RANGES")
print("==============================")

print(range_df)

print("\n==============================")
print("PARAMETER IMPACT & WEIGHTS")
print("==============================")

print(impact_df)

print("\n==============================")
print("FINAL ANALYSIS")
print("==============================")

print(final_df)

# ============================================================
# SAVE RESULTS
# ============================================================

with pd.ExcelWriter("Analysis_Results.xlsx") as writer:

    range_df.to_excel(

        writer,

        sheet_name="Optimal Ranges",

        index=False

    )

    impact_df.to_excel(

        writer,

        sheet_name="Impact and Weights",

        index=False

    )

    final_df.to_excel(

        writer,

        sheet_name="Final Analysis",

        index=False

    )

print("\nResults saved as Analysis_Results.xlsx")

# ============================================================
# PLOT PARAMETER WEIGHTS
# ============================================================

plt.figure(figsize=(12,6))

plt.bar(

    impact_df["Parameter"],

    impact_df["Weight (%)"]

)

plt.xticks(rotation=90)

plt.xlabel("Parameters")

plt.ylabel("Weight (%)")

plt.title("Parameter Weights")

plt.tight_layout()

plt.show()

# ============================================================
# PLOT PEARSON CORRELATION
# ============================================================

plt.figure(figsize=(12,6))

plt.bar(

    impact_df["Parameter"],

    impact_df["Pearson Correlation"]

)

plt.xticks(rotation=90)

plt.xlabel("Parameters")

plt.ylabel("Pearson Correlation")

plt.title("Impact of Parameters on Tumor Reduction")

plt.tight_layout()

plt.show()

print("\nAnalysis Completed Successfully.")

# Target column
target = "tumor_reduction_percent"

print("\nTarget Statistics")
print("-" * 40)
print(f"Minimum               : {df[target].min():.6f}")
print(f"Maximum               : {df[target].max():.6f}")
print(f"Mean                  : {df[target].mean():.6f}")
print(f"Standard Deviation    : {df[target].std():.6f}")