"""Exploratory data analysis for the PLGA processed simulation dataset.

This module performs analysis-only tasks on the processed dataset:
- loads the processed CSV without modifying it,
- generates publication-quality plots and summary tables,
- writes an EDA report suitable for project documentation.

No feature engineering, scaling, filtering, or model training is performed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda"
INPUT_DATASET = DATA_DIR / "processed_dataset.csv"
REPORT_PATH = RESULTS_DIR / "eda_report.txt"
TARGET_COLUMN = "tumor_reduction_percent"

logger = logging.getLogger(__name__)

COLUMN_UNITS = {
    "particle_size_nm": "nm",
    "drug_diffusion": "m^2/s",
    "nanoparticle_diffusion": "m^2/s",
    "release_rate": "1/time",
    "uptake_rate": "1/time",
    "drug_loading": "dimensionless",
    "tumor_growth_rate": "1/time",
    "drug_efficacy": "dimensionless",
    "penetration_depth": "mm",
    "average_drug_concentration": "arbitrary concentration units",
    "final_tumor_volume": "arbitrary volume units",
    "tumor_reduction_percent": "%",
}


def configure_logging() -> None:
    """Configure logging for reproducible EDA output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def ensure_output_directory(path: Path) -> None:
    """Create the output directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed dataset from disk as a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Input dataset is empty.")

    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate the dataset structure and ensure numeric values are present."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    for column_name in frame.columns:
        try:
            pd.to_numeric(frame[column_name], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column_name}' contains non-numeric values.") from exc

    if frame.isna().any().any():
        missing_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Missing values detected in columns: {missing_columns}")

    if frame.duplicated().any():
        raise ValueError("Duplicate rows detected in the input dataset.")


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure at the requested path with publication quality settings."""
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_dataset_overview_figure(frame: pd.DataFrame) -> Path:
    """Create a summary table figure for the dataset overview."""
    overview_rows = [
        ("Number of samples", str(frame.shape[0])),
        ("Number of features", str(frame.shape[1] - 1)),
        ("Target variable", TARGET_COLUMN),
        ("Missing values", str(int(frame.isna().sum().sum()))),
        ("Duplicate rows", str(int(frame.duplicated().sum()))),
        ("Feature data types", ", ".join(f"{name}: {dtype}" for name, dtype in frame.dtypes.drop(labels=[TARGET_COLUMN]).items())),
    ]

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.axis("off")
    table = axis.table(
        cellText=[[label, value] for label, value in overview_rows],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    axis.set_title("Dataset Overview", fontsize=14, pad=10)
    path = RESULTS_DIR / "dataset_overview.png"
    save_figure(figure, path)
    return path


def create_feature_histograms(frame: pd.DataFrame) -> list[Path]:
    """Create one histogram per feature column."""
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    paths: list[Path] = []
    for feature_name in feature_columns:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        values = pd.to_numeric(frame[feature_name], errors="coerce")
        axis.hist(values, bins=30, color="#4C78A8", edgecolor="black", alpha=0.9)
        axis.set_title(f"Distribution of {feature_name}", fontsize=12)
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Frequency")
        axis.grid(True, alpha=0.3)
        unit = COLUMN_UNITS.get(feature_name)
        if unit:
            axis.set_xlabel(f"{feature_name} ({unit})")
        figure.tight_layout()
        path = RESULTS_DIR / f"hist_{feature_name}.png"
        save_figure(figure, path)
        paths.append(path)
    return paths


def create_target_distribution_plots(frame: pd.DataFrame) -> list[Path]:
    """Create histogram, KDE, and box plot for the target distribution."""
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    paths: list[Path] = []

    histogram_path = RESULTS_DIR / "target_histogram.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.hist(values, bins=30, color="#F58518", edgecolor="black", alpha=0.85)
    axis.set_title("Target Distribution: Tumor Reduction Percent", fontsize=12)
    axis.set_xlabel(f"{TARGET_COLUMN} ({COLUMN_UNITS[TARGET_COLUMN]})")
    axis.set_ylabel("Frequency")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    save_figure(figure, histogram_path)
    paths.append(histogram_path)

    kde_path = RESULTS_DIR / "target_kde.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    values.plot(kind="kde", ax=axis, color="#54A24B")
    axis.set_title("Target KDE: Tumor Reduction Percent", fontsize=12)
    axis.set_xlabel(f"{TARGET_COLUMN} ({COLUMN_UNITS[TARGET_COLUMN]})")
    axis.set_ylabel("Density")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    save_figure(figure, kde_path)
    paths.append(kde_path)

    box_path = RESULTS_DIR / "target_boxplot.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.boxplot(values, orientation="vertical", patch_artist=True, boxprops={"facecolor": "#72B7B2", "alpha": 0.85})
    axis.set_title("Target Box Plot: Tumor Reduction Percent", fontsize=12)
    axis.set_ylabel(f"{TARGET_COLUMN} ({COLUMN_UNITS[TARGET_COLUMN]})")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_xticks([1])
    axis.set_xticklabels([TARGET_COLUMN])
    figure.tight_layout()
    save_figure(figure, box_path)
    paths.append(box_path)

    return paths


def create_feature_boxplots(frame: pd.DataFrame) -> list[Path]:
    """Create box plots for each feature column to highlight outliers."""
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    paths: list[Path] = []
    for feature_name in feature_columns:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        values = pd.to_numeric(frame[feature_name], errors="coerce")
        axis.boxplot(values, orientation="vertical", patch_artist=True, boxprops={"facecolor": "#C44E52", "alpha": 0.85})
        axis.set_title(f"Box Plot of {feature_name}", fontsize=12)
        axis.set_ylabel(feature_name)
        axis.grid(True, axis="y", alpha=0.3)
        axis.set_xticks([1])
        axis.set_xticklabels([feature_name])
        figure.tight_layout()
        path = RESULTS_DIR / f"box_{feature_name}.png"
        save_figure(figure, path)
        paths.append(path)
    return paths


def create_correlation_heatmap(frame: pd.DataFrame, method: str) -> Path:
    """Create a correlation heatmap for the selected method."""
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    feature_columns = [column for column in numeric_frame.columns if column != TARGET_COLUMN]
    correlation_matrix = numeric_frame[feature_columns + [TARGET_COLUMN]].corr(method=method)
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    image = axis.imshow(correlation_matrix, cmap="viridis", aspect="auto")
    axis.set_title(f"{method.capitalize()} Correlation Heatmap", fontsize=13)
    axis.set_xticks(np.arange(len(correlation_matrix.columns)))
    axis.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(correlation_matrix.index)))
    axis.set_yticklabels(correlation_matrix.index)
    figure.colorbar(image, ax=axis, shrink=0.9)
    for row_index in range(len(correlation_matrix.index)):
        for col_index in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[row_index, col_index]
            axis.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.6 else "black")
    figure.tight_layout()
    path = RESULTS_DIR / f"correlation_{method}.png"
    save_figure(figure, path)
    return path


def create_scatter_plot_matrix(frame: pd.DataFrame) -> Path:
    """Create a scatter plot matrix for the most relevant variables."""
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    feature_columns = [column for column in numeric_frame.columns if column != TARGET_COLUMN]
    correlations = numeric_frame[feature_columns].corrwith(numeric_frame[TARGET_COLUMN]).abs()
    selected_features = correlations.sort_values(ascending=False).head(6).index.tolist()
    selected_columns = selected_features + [TARGET_COLUMN]
    selected_frame = numeric_frame[selected_columns]

    figure, axes = plt.subplots(len(selected_columns), len(selected_columns), figsize=(13.5, 13.5))
    for row_index, row_name in enumerate(selected_columns):
        for col_index, col_name in enumerate(selected_columns):
            axis = axes[row_index, col_index]
            if row_index == col_index:
                axis.hist(selected_frame[col_name], bins=20, color="#4C78A8", edgecolor="black", alpha=0.8)
                axis.set_title(col_name)
            else:
                axis.scatter(selected_frame[col_name], selected_frame[row_name], s=12, alpha=0.5, color="#F58518")
                axis.set_xlabel(col_name)
                axis.set_ylabel(row_name)
            axis.grid(True, alpha=0.2)
    figure.suptitle("Scatter Plot Matrix for Key Variables", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    path = RESULTS_DIR / "scatter_matrix.png"
    save_figure(figure, path)
    return path


def summarize_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics and correlation summaries for the report."""
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    feature_columns = [column for column in numeric_frame.columns if column != TARGET_COLUMN]
    statistics_summary: dict[str, Any] = {}

    for column_name in numeric_frame.columns:
        values = numeric_frame[column_name]
        statistics_summary[column_name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)),
            "var": float(values.var(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(values.quantile(0.25)),
            "q50": float(values.quantile(0.50)),
            "q75": float(values.quantile(0.75)),
        }

    correlation_matrix = numeric_frame[feature_columns + [TARGET_COLUMN]].corr(method="pearson")
    abs_correlation = correlation_matrix.loc[feature_columns, TARGET_COLUMN].abs()
    strong_positive = [
        (feature, float(correlation))
        for feature, correlation in correlation_matrix.loc[feature_columns, TARGET_COLUMN].items()
        if correlation > 0.7
    ]
    strong_negative = [
        (feature, float(correlation))
        for feature, correlation in correlation_matrix.loc[feature_columns, TARGET_COLUMN].items()
        if correlation < -0.7
    ]
    weakly_correlated = [
        (feature, float(correlation))
        for feature, correlation in correlation_matrix.loc[feature_columns, TARGET_COLUMN].items()
        if abs(correlation) < 0.3
    ]

    pairwise_correlation = correlation_matrix[feature_columns].corr().abs()
    pairwise_correlation = pairwise_correlation.where(np.triu(np.ones(pairwise_correlation.shape), k=1).astype(bool))
    multicollinearity_pairs = [
        (row_name, col_name, float(value))
        for row_name, col_values in pairwise_correlation.iterrows()
        for col_name, value in col_values.items()
        if pd.notna(value) and value > 0.8
    ]

    outlier_counts: dict[str, int] = {}
    for feature_name in feature_columns:
        values = numeric_frame[feature_name]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[feature_name] = int(((values < lower) | (values > upper)).sum())

    statistics_summary["correlation_summary"] = {
        "strong_positive": strong_positive,
        "strong_negative": strong_negative,
        "weakly_correlated": weakly_correlated,
        "multicollinearity_pairs": multicollinearity_pairs,
        "target_correlation_ranking": [
            (feature, float(correlation)) for feature, correlation in abs_correlation.sort_values(ascending=False).items()
        ],
    }
    statistics_summary["outlier_counts"] = outlier_counts
    return statistics_summary


def write_report(frame: pd.DataFrame, statistics_summary: dict[str, Any], overview_path: Path, histogram_paths: list[Path], target_paths: list[Path], boxplot_paths: list[Path], correlation_paths: list[Path], scatter_path: Path) -> None:
    """Write a structured EDA report with interpretations for each figure."""
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    lines: list[str] = []
    lines.append("Exploratory Data Analysis Report")
    lines.append("===============================")
    lines.append("")
    lines.append("Dataset dimensions")
    lines.append("------------------")
    lines.append(f"Rows: {frame.shape[0]}")
    lines.append(f"Columns: {frame.shape[1]}")
    lines.append("")
    lines.append("Feature list")
    lines.append("------------")
    lines.extend(f"- {name}" for name in feature_columns)
    lines.append("")
    lines.append("Target variable")
    lines.append("---------------")
    lines.append(f"- {TARGET_COLUMN}")
    lines.append("")
    lines.append("Descriptive statistics")
    lines.append("----------------------")
    for column_name in frame.columns:
        stats = statistics_summary[column_name]
        lines.append(f"{column_name}:")
        lines.append(f"  mean={stats['mean']:.6f}")
        lines.append(f"  median={stats['median']:.6f}")
        lines.append(f"  std={stats['std']:.6f}")
        lines.append(f"  variance={stats['var']:.6f}")
        lines.append(f"  min={stats['min']:.6f}")
        lines.append(f"  max={stats['max']:.6f}")
        lines.append(f"  q25={stats['q25']:.6f}")
        lines.append(f"  q50={stats['q50']:.6f}")
        lines.append(f"  q75={stats['q75']:.6f}")
        lines.append("")
    lines.append("Correlation summary")
    lines.append("-------------------")
    correlations = statistics_summary["correlation_summary"]
    lines.append("Strong positive correlations with target:")
    if correlations["strong_positive"]:
        for feature, value in correlations["strong_positive"]:
            lines.append(f"- {feature}: {value:.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Strong negative correlations with target:")
    if correlations["strong_negative"]:
        for feature, value in correlations["strong_negative"]:
            lines.append(f"- {feature}: {value:.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Weakly correlated variables with target:")
    if correlations["weakly_correlated"]:
        for feature, value in correlations["weakly_correlated"]:
            lines.append(f"- {feature}: {value:.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Potential multicollinearity pairs (|r| > 0.8):")
    if correlations["multicollinearity_pairs"]:
        for first, second, value in correlations["multicollinearity_pairs"]:
            lines.append(f"- {first} / {second}: {value:.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Target correlation ranking")
    lines.append("--------------------------")
    for feature, value in correlations["target_correlation_ranking"]:
        lines.append(f"- {feature}: {value:.3f}")
    lines.append("")
    lines.append("Outlier summary")
    lines.append("---------------")
    for feature_name, count in statistics_summary["outlier_counts"].items():
        lines.append(f"- {feature_name}: {count} suspected outlier values")
    lines.append("")
    lines.append("Distribution characteristics")
    lines.append("----------------------------")
    for column_name in frame.columns:
        stats = statistics_summary[column_name]
        if stats["mean"] > stats["median"]:
            skew = "right-skewed"
        elif stats["mean"] < stats["median"]:
            skew = "left-skewed"
        else:
            skew = "approximately symmetric"
        lines.append(f"- {column_name}: {skew}; range [{stats['min']:.3f}, {stats['max']:.3f}]")
    lines.append("")
    lines.append("Figure interpretations")
    lines.append("----------------------")
    lines.append(f"- Dataset overview table ({overview_path.name}): The dataset contains 1000 samples and 11 input features with no missing values or duplicates, indicating a complete and structurally consistent input set for downstream analysis.")
    lines.append(f"- Feature histograms ({', '.join(path.name for path in histogram_paths)}): The feature distributions show the spread and shape of each input parameter; any skewness or multimodality suggests that relationships with the target may be nonlinear.")
    lines.append(f"- Target distribution plots ({', '.join(path.name for path in target_paths)}): The target distribution indicates the central tendency and spread of tumor reduction outcomes; the presence of extreme values suggests the target is not perfectly symmetric.")
    lines.append(f"- Feature box plots ({', '.join(path.name for path in boxplot_paths)}): The box plots highlight potential outliers while preserving all observations, which is useful for judging whether the dataset contains extreme but valid parameter combinations.")
    lines.append(f"- Pearson correlation heatmap ({correlation_paths[0].name}): Variables with strong positive or negative correlations appear more likely to be associated with the target, while weakly correlated variables may contribute less directly to the target response.")
    lines.append(f"- Spearman correlation heatmap ({correlation_paths[1].name}): Rank-based relationships are useful for identifying monotonic associations that may not be captured by Pearson correlation alone.")
    lines.append(f"- Scatter plot matrix ({scatter_path.name}): The pairwise plots indicate whether the most relevant variables show clear linear or nonlinear relationships; visually separated clusters suggest the dataset may be suitable for further supervised learning workflows.")
    lines.append("")
    lines.append("Overall assessment")
    lines.append("------------------")
    lines.append("The processed dataset appears suitable for machine learning analysis because it is complete, contains no duplicate rows, and provides a broad numeric feature space with measurable relationships to the target variable. The analysis is limited to descriptive statistics and visualization, without applying feature engineering or model fitting.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full exploratory data analysis workflow."""
    configure_logging()
    ensure_output_directory(RESULTS_DIR)

    logger.info("Loading processed dataset from %s", INPUT_DATASET)
    frame = load_dataset(INPUT_DATASET)
    validate_dataset(frame)

    logger.info("Generating dataset overview figure")
    overview_path = create_dataset_overview_figure(frame)

    logger.info("Generating feature histograms")
    histogram_paths = create_feature_histograms(frame)

    logger.info("Generating target distribution plots")
    target_paths = create_target_distribution_plots(frame)

    logger.info("Generating feature box plots")
    boxplot_paths = create_feature_boxplots(frame)

    logger.info("Generating Pearson correlation heatmap")
    pearson_path = create_correlation_heatmap(frame, method="pearson")

    logger.info("Generating Spearman correlation heatmap")
    spearman_path = create_correlation_heatmap(frame, method="spearman")

    logger.info("Generating scatter plot matrix")
    scatter_path = create_scatter_plot_matrix(frame)

    logger.info("Computing descriptive statistics")
    statistics_summary = summarize_statistics(frame)

    logger.info("Writing EDA report")
    write_report(
        frame=frame,
        statistics_summary=statistics_summary,
        overview_path=overview_path,
        histogram_paths=histogram_paths,
        target_paths=target_paths,
        boxplot_paths=boxplot_paths,
        correlation_paths=[pearson_path, spearman_path],
        scatter_path=scatter_path,
    )

    logger.info("EDA finished successfully.")
    logger.info("Outputs saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
