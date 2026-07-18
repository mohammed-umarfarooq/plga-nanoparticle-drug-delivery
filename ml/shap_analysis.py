"""Create global and local SHAP explanations for the trained PLGA XGBoost model.

This module is strictly interpretive: it loads the persisted model and held-out
features, computes SHAP values, and writes explanation artifacts without model
training, hyperparameter changes, or input-data modification.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SHAP_DIR = PROJECT_ROOT / "results" / "shap"
RESULT9_DIR = PROJECT_ROOT / "results" / "result9"
MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
X_TEST_PATH = DATA_DIR / "X_test.csv"
IMPORTANCE_PATH = SHAP_DIR / "global_feature_importance.csv"
SHAP_VALUES_PATH = SHAP_DIR / "shap_values.csv"
REPORT_PATH = SHAP_DIR / "shap_report.txt"
TARGET_COLUMN = "tumor_reduction_percent"
LOCAL_SAMPLE_COUNT = 5

MODEL_FEATURES = [
    "particle_size_nm",
    "drug_diffusion",
    "nanoparticle_diffusion",
    "release_rate",
    "uptake_rate",
    "drug_loading",
    "tumor_growth_rate",
    "drug_efficacy",
]
DEPENDENCE_FEATURES = {
    "particle_size_nm": "drug_loading",
    "release_rate": "drug_loading",
}
RESULT9_FIGURES = {
    "summary": RESULT9_DIR / "9.1 SHAP Summary Plot.png",
    "importance": RESULT9_DIR / "9.2 SHAP Feature Importance.png",
    "particle_size_nm": RESULT9_DIR / "9.3 SHAP Dependence Plot - Particle Size.png",
    "release_rate": RESULT9_DIR / "9.4 SHAP Dependence Plot - Drug Release Rate.png",
}
RESULT9_SUMMARY_PATH = RESULT9_DIR / "result9_summary.txt"

logger = logging.getLogger(__name__)


class ShapDataValidationError(ValueError):
    """Raised when SHAP input artifacts fail integrity validation."""


def configure_logging() -> None:
    """Configure concise progress logging for the SHAP workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def ensure_output_directory() -> None:
    """Create the dedicated SHAP result directory when necessary."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT9_DIR.mkdir(parents=True, exist_ok=True)


def load_model() -> Any:
    """Load the pre-trained model without fitting or modifying it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not provide a predict method.")
    return model


def load_preprocessing_pipeline() -> dict[str, Any]:
    """Load the training-fitted preprocessing metadata used for unit restoration."""
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline not found: {PIPELINE_PATH}")
    with PIPELINE_PATH.open("rb") as handle:
        pipeline = pickle.load(handle)
    if pipeline.get("feature_names") != MODEL_FEATURES:
        raise ShapDataValidationError("Preprocessing pipeline does not match approved model features.")
    return pipeline


def load_test_features() -> pd.DataFrame:
    """Load the fixed held-out feature matrix used for SHAP explanations."""
    if not X_TEST_PATH.is_file():
        raise FileNotFoundError(f"Held-out feature dataset not found: {X_TEST_PATH}")
    features = pd.read_csv(X_TEST_PATH)
    if features.empty:
        raise ShapDataValidationError("Held-out feature dataset is empty.")
    return features


def validate_features(model: Any, features: pd.DataFrame) -> None:
    """Validate finite numeric features and exact model feature-name alignment."""
    if TARGET_COLUMN in features.columns:
        raise ShapDataValidationError(
            f"Target column '{TARGET_COLUMN}' must not be present in X_test.csv."
        )
    try:
        numeric_features = features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ShapDataValidationError("X_test.csv contains non-numeric values.") from exc
    if numeric_features.isna().any().any():
        raise ShapDataValidationError("X_test.csv contains missing or NaN values.")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ShapDataValidationError("X_test.csv contains NaN or Inf values.")

    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        raise ShapDataValidationError(
            "The loaded model does not retain feature names for alignment validation."
        )
    if list(features.columns) != expected_features:
        raise ShapDataValidationError(
            "X_test feature names or ordering do not match the trained model."
        )
    if list(features.columns) != MODEL_FEATURES:
        raise ShapDataValidationError(
            "SHAP may only explain the approved independent model input parameters."
        )


def restore_physical_units(features: pd.DataFrame, pipeline: dict[str, Any]) -> pd.DataFrame:
    """Invert training-fitted scaling so SHAP plots show physical parameter units."""
    if not pipeline["scaling_applied"]:
        return features.copy()
    means = pd.Series(pipeline["means"], index=MODEL_FEATURES, dtype=float)
    scales = pd.Series(pipeline["scales"], index=MODEL_FEATURES, dtype=float)
    restored = features * scales + means
    restored.columns = MODEL_FEATURES
    return restored


def compute_shap_values(
    model: Any, features: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Compute Tree SHAP values and the scalar expected model output."""
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(features), dtype=float)
    if values.shape != features.shape:
        raise RuntimeError(
            "Unexpected SHAP value dimensions; expected one value per sample and feature."
        )
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return values, expected_value


def build_importance_table(features: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Rank global features by mean absolute SHAP contribution."""
    importance = pd.DataFrame(
        {
            "Feature": features.columns,
            "Mean Absolute SHAP Value": np.mean(np.abs(shap_values), axis=0),
        }
    )
    importance = importance.sort_values(
        "Mean Absolute SHAP Value", ascending=False, kind="stable"
    ).reset_index(drop=True)
    importance["Rank"] = np.arange(1, len(importance) + 1)
    return importance


def save_shap_values(features: pd.DataFrame, shap_values: np.ndarray) -> None:
    """Save a long-form SHAP value table for every held-out sample and feature."""
    sample_ids = np.repeat(features.index.to_numpy(), features.shape[1])
    feature_names = np.tile(features.columns.to_numpy(), len(features))
    values = shap_values.reshape(-1)
    pd.DataFrame(
        {
            "Sample ID": sample_ids,
            "Feature Name": feature_names,
            "SHAP Value": values,
        }
    ).to_csv(SHAP_VALUES_PATH, index=False)


def _save_current_figure(path: Path, aliases: tuple[Path, ...] = ()) -> None:
    """Save the current matplotlib figure with consistent publication settings."""
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    for alias in aliases:
        figure.savefig(alias, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_global_figures(display_features: pd.DataFrame, shap_values: np.ndarray) -> None:
    """Create SHAP summary and global bar-importance figures."""
    shap.summary_plot(shap_values, display_features, show=False, plot_size=(9, 6))
    plt.gca().set_title("SHAP Summary Plot: Independent Inputs (Held-Out Test Set)")
    _save_current_figure(
        SHAP_DIR / "SHAP_Summary_Plot.png",
        aliases=(RESULT9_FIGURES["summary"],),
    )

    shap.summary_plot(
        shap_values,
        display_features,
        plot_type="bar",
        show=False,
        plot_size=(9, 6),
    )
    plt.gca().set_title("SHAP Global Importance: Independent Inputs")
    _save_current_figure(
        SHAP_DIR / "SHAP_Feature_Importance_Bar_Plot.png",
        aliases=(RESULT9_FIGURES["importance"],),
    )


def select_representative_samples(predictions: np.ndarray) -> list[int]:
    """Select highest, lowest, median, and two intermediate prediction cases."""
    ordered = np.argsort(predictions, kind="stable")
    positions = [0, len(ordered) - 1, len(ordered) // 2, len(ordered) // 4, (3 * len(ordered)) // 4]
    selected: list[int] = []
    for position in positions:
        index = int(ordered[position])
        if index not in selected:
            selected.append(index)
    for index in ordered:
        if len(selected) == LOCAL_SAMPLE_COUNT:
            break
        if int(index) not in selected:
            selected.append(int(index))
    if len(selected) != LOCAL_SAMPLE_COUNT:
        raise RuntimeError("Unable to select five unique representative test samples.")
    return selected


def create_local_figures(
    display_features: pd.DataFrame,
    shap_values: np.ndarray,
    expected_value: float,
    predictions: np.ndarray,
) -> list[int]:
    """Create waterfall plots for five representative held-out predictions."""
    selected = select_representative_samples(predictions)
    for sample_index in selected:
        explanation = shap.Explanation(
            values=shap_values[sample_index],
            base_values=expected_value,
            data=display_features.iloc[sample_index].to_numpy(),
            feature_names=display_features.columns.tolist(),
        )
        shap.plots.waterfall(explanation, show=False, max_display=len(display_features.columns))
        plt.gca().set_title(
            "SHAP Waterfall Plot: Test Sample "
            f"{display_features.index[sample_index]} (prediction={predictions[sample_index]:.3f})"
        )
        _save_current_figure(
            SHAP_DIR / f"SHAP_Waterfall_Sample_{display_features.index[sample_index]}.png"
        )
    return selected


def create_dependence_figures(
    display_features: pd.DataFrame, shap_values: np.ndarray
) -> list[str]:
    """Create physical-unit dependence plots for the requested scientific parameters."""
    for feature_name, interaction_feature in DEPENDENCE_FEATURES.items():
        shap.dependence_plot(
            feature_name,
            shap_values,
            display_features,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.gca().set_title(
            f"SHAP Dependence: {feature_name} (colour: {interaction_feature})"
        )
        _save_current_figure(
            SHAP_DIR / f"SHAP_Dependence_{feature_name}.png",
            aliases=(RESULT9_FIGURES[feature_name],),
        )
    return list(DEPENDENCE_FEATURES)


def write_release_rate_outliers(
    display_features: pd.DataFrame, shap_values: np.ndarray
) -> pd.DataFrame:
    """Record unusually large release-rate SHAP values for transparent review."""
    column = display_features.columns.get_loc("release_rate")
    values = shap_values[:, column]
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = pd.DataFrame(
        {
            "Test Sample": display_features.index,
            "Release Rate (1/h)": display_features["release_rate"],
            "Drug Loading (fraction)": display_features["drug_loading"],
            "Release-Rate SHAP Value": values,
        }
    )
    outliers = outliers.loc[
        (outliers["Release-Rate SHAP Value"] < lower)
        | (outliers["Release-Rate SHAP Value"] > upper)
    ]
    outliers.to_csv(SHAP_DIR / "release_rate_shap_outliers.csv", index=False)
    return outliers


def feature_interpretations(
    features: pd.DataFrame, shap_values: np.ndarray, top_features: list[str]
) -> list[str]:
    """Describe model associations for leading features without causal claims."""
    lines: list[str] = []
    for feature_name in top_features:
        column = features.columns.get_loc(feature_name)
        association = float(np.corrcoef(features[feature_name], shap_values[:, column])[0, 1])
        direction = "higher" if association >= 0 else "lower"
        lines.append(
            f"- {feature_name}: {direction} feature values are associated with higher "
            f"model contributions on this test set (feature-SHAP correlation={association:.3f})."
        )
    return lines


def write_report(
    display_features: pd.DataFrame,
    importance: pd.DataFrame,
    top_features: list[str],
    selected_samples: list[int],
    predictions: np.ndarray,
    shap_values: np.ndarray,
    release_rate_outliers: pd.DataFrame,
) -> None:
    """Write a scientifically cautious report of global and local model explanations."""
    ranking = [
        f"- {feature}: {value:.6f}"
        for feature, value in zip(
            importance["Feature"], importance["Mean Absolute SHAP Value"]
        )
    ]
    local_lines = [
        f"- Test sample {display_features.index[index]}: prediction={predictions[index]:.6f}, "
        f"sum of SHAP contributions={np.sum(shap_values[index]):.6f}"
        for index in selected_samples
    ]
    lines = [
        "SHAP Explainability Report",
        "==========================",
        "",
        "Model and dataset",
        "-----------------",
        "Model: XGBRegressor loaded from models/xgboost_model.pkl",
        "Dataset: data/X_test.csv (held-out test features)",
        f"Explained samples: {len(display_features)}",
        f"Explained features: {display_features.shape[1]}",
        "Target: tumor_reduction_percent",
        "Features: independent formulation and model inputs only; target-derived simulation outputs are excluded.",
        "Dependence-plot axes use restored physical units from the training preprocessing pipeline.",
        "",
        "Global feature ranking (mean absolute SHAP value)",
        "---------------------------------------------------",
        *ranking,
        "",
        "Interpretation of requested dependence features",
        "----------------------------------------------",
        *feature_interpretations(display_features, shap_values, top_features),
        "",
        "Local explanations",
        "------------------",
        "Waterfall plots were generated for lowest, highest, median, and two intermediate predictions:",
        *local_lines,
        "Each waterfall plot decomposes one model prediction into its baseline output and feature contributions.",
        "",
        "Release-rate SHAP outlier review",
        "-------------------------------",
        f"IQR-rule outliers recorded: {len(release_rate_outliers)}",
        "The outlier table records original release-rate and drug-loading values so unusual contributions can be checked against modeled parameter bounds.",
        "",
        "Interpretation limits and optimization readiness",
        "-----------------------------------------------",
        "SHAP explains the trained model's learned associations, not the underlying biological system.",
        "These feature attributions do not establish causality or validate biological mechanisms.",
        "The global ranking and local contribution patterns are suitable inputs for the subsequent optimization module, subject to domain review and constraint handling.",
        "",
        "Generated figures",
        "-----------------",
        "- SHAP_Summary_Plot.png: distribution, direction, and magnitude of feature contributions across the test set.",
        "- SHAP_Feature_Importance_Bar_Plot.png: mean absolute SHAP ranking of global feature importance.",
        "- SHAP_Waterfall_Sample_*.png: feature-by-feature decomposition of five representative predictions.",
        "- SHAP_Dependence_particle_size_nm.png: particle-size dependence in nm, coloured by drug loading.",
        "- SHAP_Dependence_release_rate.png: release-rate dependence in 1/h, coloured by drug loading.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result9_summary(
    importance: pd.DataFrame, release_rate_outliers: pd.DataFrame
) -> None:
    """Write a concise summary for the four Result 9 presentation figures."""
    ranking = [
        f"- {feature}: {value:.6f}"
        for feature, value in zip(
            importance["Feature"], importance["Mean Absolute SHAP Value"]
        )
    ]
    lines = [
        "Result 9: SHAP Analysis Summary",
        "===============================",
        "",
        "SHAP values were calculated for the 200 held-out test samples using the retrained XGBoost model.",
        "The model contains only eight independent formulation and model parameters; final_tumor_volume and all target-derived simulation outputs are excluded.",
        "",
        "Global ranking by mean absolute SHAP value",
        "-------------------------------------------",
        *ranking,
        "",
        "Results 9.3 and 9.4 use physical particle-size and release-rate values rather than standardized z-scores.",
        "The dependence plots are coloured by drug loading to show a biologically relevant interaction.",
        f"Release-rate SHAP outliers identified by the IQR rule: {len(release_rate_outliers)}. Details are saved in results/shap/release_rate_shap_outliers.csv.",
        "SHAP values describe learned model associations and do not establish biological causality.",
    ]
    RESULT9_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(top_features: list[str], selected_samples: list[int]) -> None:
    """Confirm all required SHAP tables, report, and figures were generated."""
    required = [
        IMPORTANCE_PATH,
        SHAP_VALUES_PATH,
        SHAP_DIR / "release_rate_shap_outliers.csv",
        RESULT9_SUMMARY_PATH,
        REPORT_PATH,
        SHAP_DIR / "SHAP_Summary_Plot.png",
        SHAP_DIR / "SHAP_Feature_Importance_Bar_Plot.png",
        *RESULT9_FIGURES.values(),
        *[SHAP_DIR / f"SHAP_Dependence_{feature}.png" for feature in top_features],
        *[SHAP_DIR / f"SHAP_Waterfall_Sample_{sample}.png" for sample in selected_samples],
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Required SHAP output(s) were not created: " + ", ".join(missing))


def main() -> None:
    """Run SHAP explainability on the persisted model and held-out feature matrix."""
    configure_logging()
    ensure_output_directory()
    model = load_model()
    pipeline = load_preprocessing_pipeline()
    features = load_test_features()
    validate_features(model, features)
    logger.info("Validated held-out features and model feature alignment.")
    display_features = restore_physical_units(features, pipeline)
    shap_values, expected_value = compute_shap_values(model, features)
    predictions = np.asarray(model.predict(features), dtype=float)
    importance = build_importance_table(features, shap_values)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    save_shap_values(features, shap_values)
    create_global_figures(display_features, shap_values)
    selected_samples = create_local_figures(
        display_features, shap_values, expected_value, predictions
    )
    top_features = create_dependence_figures(display_features, shap_values)
    release_rate_outliers = write_release_rate_outliers(display_features, shap_values)
    write_report(
        display_features,
        importance,
        top_features,
        selected_samples,
        predictions,
        shap_values,
        release_rate_outliers,
    )
    write_result9_summary(importance, release_rate_outliers)
    validate_outputs(top_features, selected_samples)
    logger.info("SHAP analysis completed. Outputs saved to %s.", SHAP_DIR)


if __name__ == "__main__":
    main()
