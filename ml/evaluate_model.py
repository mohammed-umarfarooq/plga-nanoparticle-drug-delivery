"""Evaluate the trained PLGA XGBoost regressor without modifying it.

The module loads the persisted model and fixed train/test artifacts, produces
fresh predictions, calculates regression metrics, and writes reproducible
evaluation figures and a project-ready report.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
RESULT8_DIR = RESULTS_DIR / "result8"
TARGET_COLUMN = "tumor_reduction_percent"

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

MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
INPUT_PATHS = {
    "X_train": DATA_DIR / "X_train.csv",
    "y_train": DATA_DIR / "y_train.csv",
    "X_test": DATA_DIR / "X_test.csv",
    "y_test": DATA_DIR / "y_test.csv",
}
TRAIN_PREDICTIONS_PATH = RESULTS_DIR / "predictions_train.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "predictions_test.csv"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
REPORT_PATH = RESULTS_DIR / "model_evaluation_report.txt"
FIGURE_PATHS = {
    "actual_vs_predicted": EVALUATION_DIR / "Actual_vs_Predicted.png",
    "residual_vs_predicted": EVALUATION_DIR / "Residual_vs_Predicted.png",
    "residual_histogram": EVALUATION_DIR / "Residual_Histogram.png",
    "residual_qq_plot": EVALUATION_DIR / "Residual_QQ_Plot.png",
    "prediction_error_distribution": EVALUATION_DIR / "Prediction_Error_Distribution.png",
}
RESULT8_FIGURE_PATHS = {
    "actual_vs_predicted": RESULT8_DIR / "8.1 Actual vs Predicted.png",
    "residual_vs_predicted": RESULT8_DIR / "8.2 Residual Plot.png",
}
RESULT8_METRICS_PATH = RESULT8_DIR / "8.4 Performance Metrics.txt"

logger = logging.getLogger(__name__)


class EvaluationDataValidationError(ValueError):
    """Raised when model-evaluation input artifacts fail validation."""


def configure_logging() -> None:
    """Configure concise logging for the evaluation workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def ensure_output_directories() -> None:
    """Create the result directories required by the evaluation workflow."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULT8_DIR.mkdir(parents=True, exist_ok=True)


def load_model() -> Any:
    """Load the previously trained model without fitting or mutating it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not expose a predict method.")
    return model


def load_evaluation_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the prescribed train/test matrices and target vectors."""
    missing = [str(path) for path in INPUT_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation file(s) not found: " + ", ".join(missing)
        )
    x_train = pd.read_csv(INPUT_PATHS["X_train"])
    y_train_frame = pd.read_csv(INPUT_PATHS["y_train"])
    x_test = pd.read_csv(INPUT_PATHS["X_test"])
    y_test_frame = pd.read_csv(INPUT_PATHS["y_test"])
    for name, frame in {"y_train": y_train_frame, "y_test": y_test_frame}.items():
        if list(frame.columns) != [TARGET_COLUMN]:
            raise EvaluationDataValidationError(
                f"{name}.csv must contain exactly the '{TARGET_COLUMN}' column."
            )
    return x_train, y_train_frame[TARGET_COLUMN], x_test, y_test_frame[TARGET_COLUMN]


def _as_numeric(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Convert a data frame to numeric values or raise an informative error."""
    try:
        return frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvaluationDataValidationError(
            f"{label} contains non-numeric values."
        ) from exc


def validate_evaluation_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Validate dimensions, numeric types, missing values, NaNs, and infinities."""
    if x_train.empty or x_test.empty or y_train.empty or y_test.empty:
        raise EvaluationDataValidationError("Features and targets must not be empty.")
    if TARGET_COLUMN in x_train.columns or TARGET_COLUMN in x_test.columns:
        raise EvaluationDataValidationError(
            f"Target column '{TARGET_COLUMN}' must not be present in feature data."
        )
    if list(x_train.columns) != list(x_test.columns):
        raise EvaluationDataValidationError(
            "X_train and X_test must have identical ordered feature columns."
        )
    if list(x_train.columns) != MODEL_FEATURES:
        raise EvaluationDataValidationError(
            "Evaluation feature files must contain only the approved independent input parameters."
        )
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise EvaluationDataValidationError(
            "Feature and target row counts must match for both dataset splits."
        )
    for label, frame in {
        "X_train": x_train,
        "y_train": y_train.to_frame(),
        "X_test": x_test,
        "y_test": y_test.to_frame(),
    }.items():
        numeric_values = _as_numeric(frame, label)
        if numeric_values.isna().any().any():
            raise EvaluationDataValidationError(f"{label} contains missing or NaN values.")
        if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
            raise EvaluationDataValidationError(f"{label} contains NaN or Inf values.")


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate all requested regression metrics for one dataset split."""
    return {
        "r2_score": float(r2_score(actual, predicted)),
        "mean_squared_error": float(mean_squared_error(actual, predicted)),
        "root_mean_squared_error": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mean_absolute_error": float(mean_absolute_error(actual, predicted)),
        "explained_variance_score": float(explained_variance_score(actual, predicted)),
        "maximum_absolute_error": float(max_error(actual, predicted)),
    }


def build_prediction_frame(actual: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    """Build a prediction table with the conventional actual-minus-predicted residual."""
    actual_values = actual.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "Actual Value": actual_values,
            "Predicted Value": predicted,
            "Residual": actual_values - predicted,
        }
    )


def analyze_residuals(predictions: pd.DataFrame) -> dict[str, Any]:
    """Quantify residual bias, spread, normality, and error-pattern indicators."""
    residuals = predictions["Residual"].to_numpy(dtype=float)
    predicted = predictions["Predicted Value"].to_numpy(dtype=float)
    absolute_residuals = np.abs(residuals)
    correlation, p_value = stats.pearsonr(predicted, absolute_residuals)
    residual_range = np.ptp(residuals)
    bias_threshold = max(0.05 * float(np.std(residuals, ddof=1)), 1e-12)
    if abs(float(np.mean(residuals))) <= bias_threshold:
        bias_assessment = "Negligible mean residual bias."
    elif float(np.mean(residuals)) > 0:
        bias_assessment = "Positive mean residual bias: predictions tend to be low."
    else:
        bias_assessment = "Negative mean residual bias: predictions tend to be high."
    if abs(correlation) < 0.2 or p_value >= 0.05:
        heteroscedasticity = "No material heteroscedasticity signal from absolute residual correlation."
    else:
        heteroscedasticity = "Potential heteroscedasticity: absolute residuals vary with prediction level."
    systematic_error = (
        "No strong systematic-error signal from residual bias and residual-versus-predicted pattern."
        if abs(correlation) < 0.2 and abs(float(np.mean(residuals))) <= bias_threshold
        else "Residual diagnostics indicate a possible systematic prediction pattern."
    )
    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "residual_standard_deviation": float(np.std(residuals, ddof=1)),
        "residual_variance": float(np.var(residuals, ddof=1)),
        "minimum_residual": float(np.min(residuals)),
        "maximum_residual": float(np.max(residuals)),
        "residual_range": float(residual_range),
        "mean_absolute_residual": float(np.mean(absolute_residuals)),
        "absolute_residual_prediction_correlation": float(correlation),
        "absolute_residual_correlation_p_value": float(p_value),
        "bias_assessment": bias_assessment,
        "heteroscedasticity_assessment": heteroscedasticity,
        "systematic_error_assessment": systematic_error,
    }


def assess_generalization(
    training_metrics: dict[str, float], testing_metrics: dict[str, float]
) -> dict[str, Any]:
    """Assess overfitting, underfitting, and generalization from split metrics."""
    r2_gap = training_metrics["r2_score"] - testing_metrics["r2_score"]
    rmse_ratio = testing_metrics["root_mean_squared_error"] / max(
        training_metrics["root_mean_squared_error"], np.finfo(float).eps
    )
    if training_metrics["r2_score"] < 0.5 and testing_metrics["r2_score"] < 0.5:
        assessment = "Underfitting is indicated: both training and testing R2 values are low."
    elif r2_gap > 0.05 or rmse_ratio > 1.5:
        assessment = "Overfitting is indicated by materially weaker held-out performance."
    elif r2_gap <= 0.02 and rmse_ratio <= 1.15:
        assessment = "Excellent generalization: training and testing performance are closely aligned."
    else:
        assessment = "Acceptable generalization with a modest train-to-test performance gap."
    return {
        "r2_gap_train_minus_test": float(r2_gap),
        "rmse_ratio_test_to_train": float(rmse_ratio),
        "assessment": assessment,
    }


def _save_figure(figure: plt.Figure, path: Path, aliases: tuple[Path, ...] = ()) -> None:
    """Apply consistent publication-ready persistence settings to a figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    for alias in aliases:
        figure.savefig(alias, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_figures(test_predictions: pd.DataFrame) -> None:
    """Create the five required held-out prediction and residual diagnostics."""
    actual = test_predictions["Actual Value"].to_numpy()
    predicted = test_predictions["Predicted Value"].to_numpy()
    residuals = test_predictions["Residual"].to_numpy()
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.scatter(actual, predicted, alpha=0.75, color="#1f77b4", edgecolors="none")
    axis.plot(limits, limits, "--", color="#d62728", label="Ideal prediction")
    axis.set_title("Actual vs Predicted Tumor Reduction")
    axis.set_xlabel("Actual tumor reduction percent")
    axis.set_ylabel("Predicted tumor reduction percent")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _save_figure(
        figure,
        FIGURE_PATHS["actual_vs_predicted"],
        aliases=(RESULT8_FIGURE_PATHS["actual_vs_predicted"],),
    )

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.scatter(predicted, residuals, alpha=0.75, color="#2ca02c", edgecolors="none")
    axis.axhline(0.0, linestyle="--", color="#d62728")
    axis.set_title("Residuals vs Predicted Tumor Reduction")
    axis.set_xlabel("Predicted tumor reduction percent")
    axis.set_ylabel("Residual (actual - predicted)")
    axis.grid(True, alpha=0.3)
    _save_figure(
        figure,
        FIGURE_PATHS["residual_vs_predicted"],
        aliases=(RESULT8_FIGURE_PATHS["residual_vs_predicted"],),
    )

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.hist(residuals, bins=25, color="#9467bd", edgecolor="black", alpha=0.8)
    axis.axvline(0.0, linestyle="--", color="#d62728")
    axis.set_title("Residual Histogram")
    axis.set_xlabel("Residual (actual - predicted)")
    axis.set_ylabel("Frequency")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, FIGURE_PATHS["residual_histogram"])

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    stats.probplot(residuals, dist="norm", plot=axis)
    axis.set_title("Residual Q-Q Plot")
    axis.set_xlabel("Theoretical normal quantiles")
    axis.set_ylabel("Ordered residuals")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, FIGURE_PATHS["residual_qq_plot"])

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.hist(np.abs(residuals), bins=25, color="#ff7f0e", edgecolor="black", alpha=0.8)
    axis.set_title("Prediction Error Distribution")
    axis.set_xlabel("Absolute prediction error")
    axis.set_ylabel("Frequency")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, FIGURE_PATHS["prediction_error_distribution"])


def write_report(
    training_metrics: dict[str, float],
    testing_metrics: dict[str, float],
    residual_analysis: dict[str, Any],
    generalization: dict[str, Any],
    training_size: int,
    testing_size: int,
) -> None:
    """Write the detailed, evidence-based model evaluation report."""
    def metric_lines(title: str, metrics: dict[str, float]) -> list[str]:
        return [
            title,
            "-" * len(title),
            *[f"{key}: {value:.6f}" for key, value in metrics.items()],
            "",
        ]

    lines = [
        "XGBoost Model Evaluation Report",
        "===============================",
        "",
        "Model information",
        "-----------------",
        "Model: XGBRegressor loaded from models/xgboost_model.pkl",
        "Evaluation method: fresh predictions from the persisted model; no fitting, tuning, or cross-validation performed.",
        f"Training samples: {training_size}",
        f"Testing samples: {testing_size}",
        "",
        *metric_lines("Training metrics", training_metrics),
        *metric_lines("Testing metrics", testing_metrics),
        "Generalization assessment",
        "-------------------------",
        f"Training minus testing R2: {generalization['r2_gap_train_minus_test']:.6f}",
        f"Testing/training RMSE ratio: {generalization['rmse_ratio_test_to_train']:.6f}",
        generalization["assessment"],
        "",
        "Residual analysis (test dataset)",
        "---------------------------------",
        f"Mean residual: {residual_analysis['mean_residual']:.6f}",
        f"Residual standard deviation: {residual_analysis['residual_standard_deviation']:.6f}",
        f"Residual variance: {residual_analysis['residual_variance']:.6f}",
        f"Residual range: {residual_analysis['residual_range']:.6f}",
        "Absolute residual/prediction correlation: "
        f"{residual_analysis['absolute_residual_prediction_correlation']:.6f}",
        f"Correlation p-value: {residual_analysis['absolute_residual_correlation_p_value']:.6f}",
        residual_analysis["bias_assessment"],
        residual_analysis["heteroscedasticity_assessment"],
        residual_analysis["systematic_error_assessment"],
        "",
        "Model strengths",
        "---------------",
        "- Uses an independent held-out test set and fresh model predictions.",
        "- Reports complementary variance, squared-error, absolute-error, and worst-case-error metrics.",
        "- Includes graphical residual diagnostics for error-pattern inspection.",
        "",
        "Model limitations",
        "-----------------",
        "- Evaluation is limited to the current simulation data distribution.",
        "- Feature importance and causal interpretation are outside this evaluation; use SHAP next.",
        "",
        "Prediction quality and SHAP readiness",
        "-------------------------------------",
        "The held-out metrics and residual diagnostics provide the performance baseline for interpretation. "
        "The persisted model is ready for the subsequent SHAP explainability module.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result8_metrics(testing_metrics: dict[str, float]) -> None:
    """Write the held-out metrics reported alongside Result 8 figures."""
    lines = [
        "Result 8: XGBoost Held-out Test Performance",
        "=============================================",
        "",
        "All values below use the same independent test set as Results 8.1 and 8.2.",
        f"R²:                 {testing_metrics['r2_score']:.6f}",
        f"MAE:                {testing_metrics['mean_absolute_error']:.6f}",
        f"RMSE:               {testing_metrics['root_mean_squared_error']:.6f}",
        f"MSE:                {testing_metrics['mean_squared_error']:.6f}",
        f"Maximum Error:      {testing_metrics['maximum_absolute_error']:.6f}",
        "",
        "Result 8.3 shows cross-validation scores from hyperparameter search; it is not a training-versus-validation learning curve.",
    ]
    RESULT8_METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    """Confirm all required metrics, predictions, report, and figures were created."""
    required = [
        METRICS_PATH,
        REPORT_PATH,
        TRAIN_PREDICTIONS_PATH,
        TEST_PREDICTIONS_PATH,
        *FIGURE_PATHS.values(),
        *RESULT8_FIGURE_PATHS.values(),
        RESULT8_METRICS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Required evaluation output(s) were not created: " + ", ".join(missing))


def main() -> None:
    """Run the complete non-mutating evaluation workflow."""
    configure_logging()
    ensure_output_directories()
    model = load_model()
    x_train, y_train, x_test, y_test = load_evaluation_data()
    validate_evaluation_data(x_train, y_train, x_test, y_test)
    logger.info("Validated evaluation artifacts and loaded the trained model.")
    train_predictions = build_prediction_frame(y_train, model.predict(x_train))
    test_predictions = build_prediction_frame(y_test, model.predict(x_test))
    training_metrics = calculate_metrics(
        train_predictions["Actual Value"], train_predictions["Predicted Value"].to_numpy()
    )
    testing_metrics = calculate_metrics(
        test_predictions["Actual Value"], test_predictions["Predicted Value"].to_numpy()
    )
    residual_analysis = analyze_residuals(test_predictions)
    generalization = assess_generalization(training_metrics, testing_metrics)
    metrics = {
        "model": "XGBRegressor",
        "target": TARGET_COLUMN,
        "training_samples": len(train_predictions),
        "testing_samples": len(test_predictions),
        "training_metrics": training_metrics,
        "testing_metrics": testing_metrics,
        "test_residual_analysis": residual_analysis,
        "generalization": generalization,
    }
    train_predictions.to_csv(TRAIN_PREDICTIONS_PATH, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    create_figures(test_predictions)
    write_result8_metrics(testing_metrics)
    write_report(
        training_metrics,
        testing_metrics,
        residual_analysis,
        generalization,
        len(train_predictions),
        len(test_predictions),
    )
    validate_outputs()
    logger.info("Model evaluation completed. Outputs saved to %s.", RESULTS_DIR)


if __name__ == "__main__":
    main()
