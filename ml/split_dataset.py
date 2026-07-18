"""Prepare the PLGA simulation dataset for downstream machine learning modules.

This module is intentionally limited to:
- validating the processed dataset,
- separating features from the target variable,
- summarizing feature characteristics,
- building a reusable preprocessing pipeline,
- creating reproducible train/test splits,
- writing the required CSV and text artifacts.

No exploratory analysis, feature selection, dimensionality reduction, or model
training is performed here.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
INPUT_DATASET = DATA_DIR / "processed_dataset.csv"
FEATURE_SUMMARY_PATH = DATA_DIR / "feature_summary.txt"
TRAIN_DATASET_PATH = DATA_DIR / "train.csv"
TEST_DATASET_PATH = DATA_DIR / "test.csv"
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"
PREPROCESSING_REPORT_PATH = DATA_DIR / "preprocessing_report.txt"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
TARGET_COLUMN = "tumor_reduction_percent"
RANDOM_STATE = 42
TEST_SIZE_FRACTION = 0.2

# Only independent formulation and model inputs may be used to predict the
# treatment outcome.  Simulation outputs are excluded to prevent target leakage.
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
FORBIDDEN_FEATURES = {
    TARGET_COLUMN,
    "penetration_depth",
    "average_drug_concentration",
    "final_tumor_volume",
}

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when the processed dataset does not satisfy preprocessing rules."""


def configure_logging() -> None:
    """Configure the module logger for reproducible preprocessing output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed dataset from disk as a pandas DataFrame."""
    if not path.exists():
        raise DatasetValidationError(f"Input dataset not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise DatasetValidationError("Input dataset is empty.")

    return frame


def validate_dataset(frame: pd.DataFrame, target_column: str) -> None:
    """Validate the dataset against the required quality rules."""
    if frame.empty:
        raise DatasetValidationError("Input dataset is empty.")

    if target_column not in frame.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' is missing from the input dataset."
        )

    missing_features = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing_features:
        raise DatasetValidationError(
            "Required independent model feature(s) are missing: "
            + ", ".join(missing_features)
        )

    for column_name in frame.columns:
        try:
            pd.to_numeric(frame[column_name], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                f"Non-numeric value detected in column '{column_name}'."
            ) from exc

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        missing_columns = numeric_frame.columns[numeric_frame.isna().any()].tolist()
        raise DatasetValidationError(
            f"Missing values detected in the input dataset: {missing_columns}"
        )

    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise DatasetValidationError("The dataset contains NaN or Inf values.")

    if numeric_frame.duplicated().any():
        raise DatasetValidationError("Duplicate rows were detected in the input dataset.")

    if numeric_frame[MODEL_FEATURES].duplicated().any():
        raise DatasetValidationError(
            "Duplicate parameter combinations were detected in the feature columns."
        )


def separate_features_target(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target variables."""
    if target_column not in frame.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' is missing from the input dataset."
        )

    features = frame.loc[:, MODEL_FEATURES].copy()
    target = frame[target_column].copy()
    target.name = target_column

    if target_column in features.columns:
        raise DatasetValidationError("The target column must not appear inside the feature matrix.")

    leaked_columns = FORBIDDEN_FEATURES.intersection(features.columns)
    if leaked_columns:
        raise DatasetValidationError(
            "Target-derived feature(s) are not allowed: " + ", ".join(sorted(leaked_columns))
        )

    return features, target


def summarize_features(features: pd.DataFrame) -> str:
    """Generate a text summary describing feature characteristics."""
    lines: list[str] = []
    lines.append("Feature Summary")
    lines.append("===============")
    lines.append(f"Number of features: {features.shape[1]}")
    lines.append("")
    lines.append("Feature names")
    lines.append("-------------")
    for feature_name in features.columns:
        lines.append(f"- {feature_name}")

    lines.append("")
    lines.append("Feature data types")
    lines.append("------------------")
    for feature_name in features.columns:
        lines.append(f"- {feature_name}: {features[feature_name].dtype}")

    lines.append("")
    lines.append("Feature ranges")
    lines.append("--------------")
    for feature_name in features.columns:
        values = pd.to_numeric(features[feature_name], errors="coerce")
        minimum = float(values.min())
        maximum = float(values.max())
        lines.append(f"- {feature_name}: min={minimum:.6g}, max={maximum:.6g}")

    lines.append("")
    lines.append("Constant features")
    lines.append("-----------------")
    constant_features: list[str] = []
    for feature_name in features.columns:
        values = pd.to_numeric(features[feature_name], errors="coerce")
        if np.isclose(values.max(), values.min(), rtol=0.0, atol=1e-12):
            constant_features.append(feature_name)
    if constant_features:
        lines.extend(f"- {name}" for name in constant_features)
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Near-constant features")
    lines.append("-----------------------")
    near_constant_features: list[str] = []
    for feature_name in features.columns:
        values = pd.to_numeric(features[feature_name], errors="coerce")
        minimum = float(values.min())
        maximum = float(values.max())
        span = abs(maximum - minimum)
        reference_scale = max(abs(maximum), abs(minimum), 1.0)
        if span <= 1e-10 * reference_scale:
            near_constant_features.append(feature_name)
    if near_constant_features:
        lines.extend(f"- {name}" for name in near_constant_features)
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def build_preprocessing_pipeline(features: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Fit preprocessing using the training feature matrix only."""
    feature_names = list(features.columns)
    numeric_features = features.apply(pd.to_numeric, errors="coerce")

    if numeric_features.empty:
        raise DatasetValidationError("Feature matrix is empty after separation.")

    std_values = numeric_features.std(axis=0, ddof=0)
    scaling_required = bool(
        (std_values > 1e-8).any() and (numeric_features.nunique(dropna=True) > 1).any()
    )

    if scaling_required:
        means = numeric_features.mean(axis=0)
        scales = numeric_features.std(axis=0, ddof=0)
        scales = scales.replace(0, 1.0)
        transformed = (numeric_features - means) / scales
        scaling_method = "StandardScaler"
        reason = (
            "Feature scales vary substantially across columns, so StandardScaler is "
            "applied to place them on a comparable scale for downstream ML modules."
        )
    else:
        transformed = numeric_features.copy()
        scaling_method = "None"
        reason = (
            "Feature scales are already comparable and do not require scaling for "
            "the current dataset."
        )

    pipeline = {
        "feature_names": feature_names,
        "scaling_applied": scaling_required,
        "scaling_method": scaling_method,
        "reason": reason,
        "means": means.to_dict() if scaling_required else None,
        "scales": scales.to_dict() if scaling_required else None,
    }

    return pipeline, transformed, reason


def transform_features(features: pd.DataFrame, pipeline: dict[str, Any]) -> pd.DataFrame:
    """Apply a training-fitted preprocessing pipeline to a held-out feature matrix."""
    if list(features.columns) != pipeline["feature_names"]:
        raise DatasetValidationError(
            "Held-out feature columns do not match the training preprocessing pipeline."
        )

    numeric_features = features.apply(pd.to_numeric, errors="raise")
    if not pipeline["scaling_applied"]:
        return numeric_features.copy()

    means = pd.Series(pipeline["means"], index=pipeline["feature_names"], dtype=float)
    scales = pd.Series(pipeline["scales"], index=pipeline["feature_names"], dtype=float)
    return (numeric_features - means) / scales


def split_dataset(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create reproducible train/test splits using a fixed random seed."""
    indices = np.arange(len(features))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(indices)

    test_count = int(round(len(features) * TEST_SIZE_FRACTION))
    test_count = max(1, min(test_count, len(features) - 1))
    train_count = len(features) - test_count

    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    x_train = features.iloc[train_indices].copy()
    x_test = features.iloc[test_indices].copy()
    y_train = target.iloc[train_indices].copy()
    y_test = target.iloc[test_indices].copy()

    x_train.index = features.index[train_indices]
    x_test.index = features.index[test_indices]
    y_train.index = target.index[train_indices]
    y_test.index = target.index[test_indices]

    return x_train, y_train, x_test, y_test


def write_outputs(
    features: pd.DataFrame,
    target: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: dict[str, Any],
    scaling_method: str,
    scaling_reason: str,
) -> None:
    """Write all preprocessing outputs, reports, and the serialized pipeline."""
    _ensure_directory(DATA_DIR)
    _ensure_directory(MODELS_DIR)

    train_frame = pd.concat([x_train, y_train], axis=1)
    test_frame = pd.concat([x_test, y_test], axis=1)

    train_frame.to_csv(TRAIN_DATASET_PATH, index=False)
    test_frame.to_csv(TEST_DATASET_PATH, index=False)
    x_train.to_csv(X_TRAIN_PATH, index=False)
    x_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame().to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame().to_csv(Y_TEST_PATH, index=False)

    with PIPELINE_PATH.open("wb") as handle:
        pickle.dump(pipeline, handle)

    feature_summary = summarize_features(features)
    FEATURE_SUMMARY_PATH.write_text(feature_summary, encoding="utf-8")

    report_lines: list[str] = []
    report_lines.append("PLGA Machine Learning Preprocessing Report")
    report_lines.append("========================================")
    report_lines.append(f"Input dataset: {INPUT_DATASET}")
    report_lines.append(f"Number of samples: {len(features)}")
    report_lines.append(f"Number of features: {features.shape[1]}")
    report_lines.append(f"Target variable: {TARGET_COLUMN}")
    report_lines.append(f"Training samples: {len(x_train)}")
    report_lines.append(f"Testing samples: {len(x_test)}")
    report_lines.append(f"Random seed: {RANDOM_STATE}")
    report_lines.append(f"Scaling method: {scaling_method}")
    report_lines.append(f"Reason for scaling decision: {scaling_reason}")
    report_lines.append("Preprocessing fit: training set only; the held-out test set was transformed using training statistics.")
    report_lines.append("")
    report_lines.append("Feature list")
    report_lines.append("-----------")
    report_lines.extend(f"- {column}" for column in features.columns)
    report_lines.append("")
    report_lines.append("Files generated")
    report_lines.append("---------------")
    report_lines.append(f"- {FEATURE_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {PIPELINE_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {TRAIN_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {TEST_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {X_TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {X_TEST_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {Y_TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {Y_TEST_PATH.relative_to(PROJECT_ROOT)}")
    report_lines.append("")
    report_lines.append("Validation status")
    report_lines.append("-----------------")
    report_lines.append("- Dataset validation: passed")
    report_lines.append("- Target separation: passed")
    report_lines.append("- Train/test split: passed")
    report_lines.append("- Missing values: none")
    report_lines.append("- NaN values: none")
    report_lines.append("- Inf values: none")

    PREPROCESSING_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def validate_outputs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Perform a final validation of the generated train/test artifacts."""
    if len(x_train) + len(x_test) != len(features):
        raise DatasetValidationError("Train/test row counts do not match the source dataset size.")

    if len(x_train) != len(y_train):
        raise DatasetValidationError("Training feature and target row counts do not match.")

    if len(x_test) != len(y_test):
        raise DatasetValidationError("Testing feature and target row counts do not match.")

    if not x_train.columns.equals(features.columns):
        raise DatasetValidationError("Training feature ordering does not match the source feature order.")

    if not x_test.columns.equals(features.columns):
        raise DatasetValidationError("Testing feature ordering does not match the source feature order.")

    if not x_train.index.equals(y_train.index):
        raise DatasetValidationError("Training feature and target indices do not match.")

    if not x_test.index.equals(y_test.index):
        raise DatasetValidationError("Testing feature and target indices do not match.")

    if x_train.isna().any().any() or x_test.isna().any().any():
        raise DatasetValidationError("Missing values were found in the split feature matrices.")

    if y_train.isna().any() or y_test.isna().any():
        raise DatasetValidationError("Missing values were found in the split target vectors.")

    if not np.isfinite(pd.concat([x_train, x_test], axis=0).to_numpy(dtype=float)).all():
        raise DatasetValidationError("NaN or Inf values were found in the feature split matrices.")

    if not np.isfinite(pd.concat([y_train, y_test], axis=0).to_numpy(dtype=float)).all():
        raise DatasetValidationError("NaN or Inf values were found in the target split vectors.")

    if target.name != TARGET_COLUMN:
        raise DatasetValidationError("The target vector name does not match the configured target.")

    if TARGET_COLUMN in features.columns:
        raise DatasetValidationError("The target column must not remain in the feature matrix.")

    if list(features.columns) != MODEL_FEATURES:
        raise DatasetValidationError("Feature matrix does not match the approved independent model inputs.")

    if not x_train.index.intersection(x_test.index).empty:
        raise DatasetValidationError("Training and testing splits share one or more samples.")


def main() -> None:
    """Run the end-to-end preprocessing pipeline for the PLGA dataset."""
    configure_logging()
    logger.info("Starting ML dataset preparation pipeline.")

    frame = load_dataset(INPUT_DATASET)
    logger.info("Loaded processed dataset with %s rows and %s columns.", len(frame), len(frame.columns))

    validate_dataset(frame, TARGET_COLUMN)
    logger.info("Dataset validation passed.")

    features, target = separate_features_target(frame, TARGET_COLUMN)
    logger.info("Separated %s features from the target variable.", features.shape[1])

    x_train_raw, y_train, x_test_raw, y_test = split_dataset(features, target)
    logger.info("Created independent train/test splits of %s and %s rows.", len(x_train_raw), len(x_test_raw))

    pipeline, x_train, scaling_reason = build_preprocessing_pipeline(x_train_raw)
    logger.info("Preprocessing decision: %s", pipeline["scaling_method"])
    x_test = transform_features(x_test_raw, pipeline)

    write_outputs(
        features=features,
        target=target,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        pipeline=pipeline,
        scaling_method=pipeline["scaling_method"],
        scaling_reason=scaling_reason,
    )
    logger.info("Wrote preprocessing outputs to the data and models directories.")

    validate_outputs(x_train, x_test, y_train, y_test, features, target)
    logger.info("Final validation passed.")


if __name__ == "__main__":
    main()
