"""Generate Result 10 from the leakage-free XGBoost model.

The model accepts only the eight independent formulation parameters.  Every
candidate is created in physical units, transformed with the *training* scaler,
and then evaluated by the persisted model.  This prevents the stale-model and
incorrectly-scaled inputs that previously produced constant optimization plots.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "result10"

MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
DATASET_PATH = DATA_DIR / "processed_dataset.csv"

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
BOUNDS = {
    # Result 10 follows the reviewer-approved formulation-design interval.
    "particle_size_nm": (50.0, 200.0),
    "drug_diffusion": (1e-10, 1e-8),
    "nanoparticle_diffusion": (1e-10, 1e-8),
    "release_rate": (0.01, 0.20),
    "uptake_rate": (0.01, 0.10),
    "drug_loading": (0.20, 1.00),
    "tumor_growth_rate": (0.01, 0.05),
    "drug_efficacy": (0.30, 1.00),
}

PARTICLE_SIZE_POINTS = 50
RELEASE_RATE_POINTS = 50
TOP_CANDIDATES = 10
RANDOM_STATE = 42
# The simulation target is capped at 95%; do not select surrogate
# extrapolations above this defined physical maximum.
MAX_PHYSICAL_TUMOR_REDUCTION = 95.0
CANDIDATE_POOL_SIZE = 10_000
NEAR_OPTIMUM_WINDOW = 5.0
MINIMUM_PREDICTION_SEPARATION = 0.25

PARTICLE_FIGURE = RESULTS_DIR / "10.1 Tumor Reduction vs Particle Size.png"
RELEASE_FIGURE = RESULTS_DIR / "10.2 Tumor Reduction vs Drug Release Rate.png"
HEATMAP_FIGURE = RESULTS_DIR / "10.3 Optimization Heatmap.png"
CANDIDATES_PATH = RESULTS_DIR / "10.4 Optimized Parameters.csv"
SUMMARY_PATH = RESULTS_DIR / "result10_summary.txt"


def load_artifacts() -> tuple[object, dict, pd.DataFrame]:
    """Load and validate the current leakage-free training artifacts."""
    for path in (MODEL_PATH, PIPELINE_PATH, DATASET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with PIPELINE_PATH.open("rb") as file:
        pipeline = pickle.load(file)
    model_features = list(getattr(model, "feature_names_in_", []))
    pipeline_features = list(pipeline.get("feature_names", []))
    if model_features != MODEL_FEATURES or pipeline_features != MODEL_FEATURES:
        raise ValueError(
            "Result 10 requires the retrained eight-feature model. Run "
            "ml/split_dataset.py and ml/train_model.py before optimization."
        )
    data = pd.read_csv(DATASET_PATH)
    if data.empty or set(MODEL_FEATURES) - set(data.columns):
        raise ValueError("Processed dataset does not contain all independent features.")
    return model, pipeline, data


def transform(raw: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    """Apply the already-fitted training transformation, without refitting."""
    if list(raw.columns) != MODEL_FEATURES:
        raise ValueError("Prediction features are in the wrong order.")
    if not pipeline.get("scaling_applied", False):
        return raw.copy()
    means = pd.Series(pipeline["means"], dtype=float).reindex(MODEL_FEATURES)
    scales = pd.Series(pipeline["scales"], dtype=float).reindex(MODEL_FEATURES)
    if means.isna().any() or scales.isna().any() or (scales == 0).any():
        raise ValueError("Invalid persisted preprocessing statistics.")
    return (raw - means) / scales


def predict(model: object, pipeline: dict, raw: pd.DataFrame) -> np.ndarray:
    """Predict from physical-unit formulation rows."""
    values = raw.loc[:, MODEL_FEATURES].astype(float)
    return np.asarray(model.predict(transform(values, pipeline)), dtype=float)


def response_range(values: np.ndarray, label: str) -> float:
    """Fail loudly if updated inputs do not change the model prediction."""
    span = float(np.ptp(values))
    if not np.isfinite(values).all() or span <= 1e-8:
        raise RuntimeError(
            f"{label} predictions are constant. The model did not receive varied inputs; "
            "do not use these optimization results."
        )
    return span


def make_baseline(data: pd.DataFrame) -> dict[str, float]:
    """Use the median observed formulation as the fixed reference for one-way sweeps."""
    return {feature: float(data[feature].median()) for feature in MODEL_FEATURES}


def one_way_sweep(
    model: object, pipeline: dict, baseline: dict[str, float], feature: str, points: int
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = BOUNDS[feature]
    values = np.linspace(lower, upper, points)
    raw = pd.DataFrame([baseline] * points, columns=MODEL_FEATURES)
    raw[feature] = values
    fixed_features = [name for name in MODEL_FEATURES if name != feature]
    if any(raw[name].nunique() != 1 for name in fixed_features):
        raise RuntimeError("A non-optimized feature changed during the one-way sweep.")
    predictions = predict(model, pipeline, raw)
    response_range(predictions, feature)
    return values, predictions


def two_way_sweep(
    model: object, pipeline: dict, baseline: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Evaluate all 2,500 physical-unit size/release combinations."""
    sizes = np.linspace(*BOUNDS["particle_size_nm"], PARTICLE_SIZE_POINTS)
    rates = np.linspace(*BOUNDS["release_rate"], RELEASE_RATE_POINTS)
    size_grid, rate_grid = np.meshgrid(sizes, rates, indexing="xy")
    raw = pd.DataFrame([baseline] * size_grid.size, columns=MODEL_FEATURES)
    raw["particle_size_nm"] = size_grid.ravel()
    raw["release_rate"] = rate_grid.ravel()
    if raw["particle_size_nm"].nunique() != PARTICLE_SIZE_POINTS or raw["release_rate"].nunique() != RELEASE_RATE_POINTS:
        raise RuntimeError("The heatmap inputs do not cover both requested optimization ranges.")
    fixed_features = [name for name in MODEL_FEATURES if name not in {"particle_size_nm", "release_rate"}]
    if any(raw[name].nunique() != 1 for name in fixed_features):
        raise RuntimeError("A non-optimized feature changed during the two-way sweep.")
    predictions = predict(model, pipeline, raw).reshape(size_grid.shape)
    response_range(predictions, "particle-size/release-rate grid")
    return sizes, rates, predictions, raw.assign(predicted_tumor_reduction_percent=predictions.ravel())


def save_figures(
    particle_sizes: np.ndarray,
    particle_predictions: np.ndarray,
    release_rates: np.ndarray,
    release_predictions: np.ndarray,
    sizes: np.ndarray,
    rates: np.ndarray,
    grid_predictions: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(particle_sizes, particle_predictions, color="#1f77b4", linewidth=2.2)
    axis.scatter(particle_sizes, particle_predictions, color="#1f77b4", s=14)
    axis.set(title="10.1 Predicted Tumor Reduction vs Particle Size", xlabel="Particle size (nm)", ylabel="Predicted tumor reduction (%)")
    best_index = int(np.argmax(particle_predictions))
    axis.axvline(particle_sizes[best_index], color="#2ca02c", linestyle="--", linewidth=1.5)
    axis.annotate(
        f"Best tested size: {particle_sizes[best_index]:.1f} nm",
        xy=(particle_sizes[best_index], particle_predictions[best_index]),
        xytext=(8, -28), textcoords="offset points", fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(PARTICLE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(release_rates, release_predictions, color="#d62728", linewidth=2.2)
    axis.scatter(release_rates, release_predictions, color="#d62728", s=14)
    axis.set(title="10.2 Predicted Tumor Reduction vs Drug Release Rate", xlabel="Drug release rate (h⁻¹)", ylabel="Predicted tumor reduction (%)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RELEASE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.6))
    contour = axis.contourf(sizes, rates, grid_predictions, levels=30, cmap="viridis")
    colorbar = figure.colorbar(contour, ax=axis)
    colorbar.set_label("Predicted tumor reduction (%)")
    best_rate, best_size = np.unravel_index(np.argmax(grid_predictions), grid_predictions.shape)
    axis.scatter(sizes[best_size], rates[best_rate], marker="*", s=180, color="white", edgecolor="black", label="Best grid point")
    near_optimal = float(grid_predictions.max() - 0.05 * np.ptp(grid_predictions))
    near_contour = axis.contour(sizes, rates, grid_predictions, levels=[near_optimal], colors="white", linewidths=1.4)
    axis.clabel(near_contour, fmt={near_optimal: "Top 5% response"}, fontsize=8)
    axis.set(title="10.3 Joint Optimization: Particle Size and Release Rate", xlabel="Particle size (nm)", ylabel="Drug release rate (h⁻¹)")
    axis.legend(frameon=True, loc="best")
    figure.tight_layout()
    figure.savefig(HEATMAP_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def optimize_all_parameters(model: object, pipeline: dict, baseline: dict[str, float]) -> pd.DataFrame:
    """Return ten diverse, feasible, near-optimal formulations.

    A global optimizer locates the best feasible response.  A reproducible
    random design pool is then used to select high-response candidates that
    are separated in predicted outcome, rather than reporting ten nearly
    identical points from the final optimizer population.
    """
    lower = np.array([BOUNDS[feature][0] for feature in MODEL_FEATURES])
    upper = np.array([BOUNDS[feature][1] for feature in MODEL_FEATURES])

    def objective(unit_values: np.ndarray) -> float:
        physical = lower + np.asarray(unit_values) * (upper - lower)
        row = pd.DataFrame([physical], columns=MODEL_FEATURES)
        prediction = float(predict(model, pipeline, row)[0])
        if prediction > MAX_PHYSICAL_TUMOR_REDUCTION:
            return 1_000.0 + prediction - MAX_PHYSICAL_TUMOR_REDUCTION
        return -prediction

    result = differential_evolution(
        objective, [(0.0, 1.0)] * len(MODEL_FEATURES), seed=RANDOM_STATE,
        maxiter=80, popsize=15, polish=False, workers=1,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    vectors = np.vstack(
        [result.x, result.population, rng.uniform(0.0, 1.0, size=(CANDIDATE_POOL_SIZE, len(MODEL_FEATURES)))]
    )
    physical = lower + vectors * (upper - lower)
    pool = pd.DataFrame(physical, columns=MODEL_FEATURES).drop_duplicates()
    pool["predicted_tumor_reduction_percent"] = predict(model, pipeline, pool)
    best_feasible = float(pool.loc[
        pool["predicted_tumor_reduction_percent"] <= MAX_PHYSICAL_TUMOR_REDUCTION,
        "predicted_tumor_reduction_percent",
    ].max())
    pool = pool.loc[
        (pool["predicted_tumor_reduction_percent"] <= MAX_PHYSICAL_TUMOR_REDUCTION)
        & (pool["predicted_tumor_reduction_percent"] >= best_feasible - NEAR_OPTIMUM_WINDOW)
    ].sort_values("predicted_tumor_reduction_percent", ascending=False)
    selected_rows: list[pd.Series] = []
    for _, candidate in pool.iterrows():
        prediction = float(candidate["predicted_tumor_reduction_percent"])
        if all(
            abs(prediction - float(selected["predicted_tumor_reduction_percent"]))
            >= MINIMUM_PREDICTION_SEPARATION
            for selected in selected_rows
        ):
            selected_rows.append(candidate)
        if len(selected_rows) == TOP_CANDIDATES:
            break
    candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    if len(candidates) != TOP_CANDIDATES:
        raise RuntimeError("Could not find ten diverse, feasible near-optimal candidate formulations.")
    priority = ["rank", "particle_size_nm", "release_rate", "predicted_tumor_reduction_percent"]
    return candidates[priority + [name for name in candidates.columns if name not in priority]]


def write_summary(
    baseline: dict[str, float], particle_span: float, release_span: float, grid_span: float, candidates: pd.DataFrame
) -> None:
    best = candidates.iloc[0]
    lines = [
        "Result 10: Leakage-Free Parameter Optimization",
        "===============================================",
        "",
        "This result uses the retrained XGBoost model with exactly eight independent formulation inputs. final_tumor_volume and every other simulation output are excluded.",
        "Every physical-unit candidate is transformed only with the scaler fitted on the training split before prediction.",
        "",
        f"Particle-size sweep: {PARTICLE_SIZE_POINTS} evenly spaced values from 50 to 200 nm; prediction span = {particle_span:.6f}%.",
        f"Release-rate sweep: {RELEASE_RATE_POINTS} evenly spaced values from 0.01 to 0.20 h^-1; prediction span = {release_span:.6f}%.",
        f"Joint heatmap: {PARTICLE_SIZE_POINTS * RELEASE_RATE_POINTS} combinations; prediction span = {grid_span:.6f}%.",
        "Other inputs were held at their processed-dataset median values for the one- and two-way sweeps:",
        *[f"- {feature}: {value:.6g}" for feature, value in baseline.items()],
        "",
        f"10.4 uses bounded global optimization plus a reproducible {CANDIDATE_POOL_SIZE}-point design pool. It reports ten feasible near-optimal formulations within {NEAR_OPTIMUM_WINDOW:.1f}% of the best prediction, separated by at least {MINIMUM_PREDICTION_SEPARATION:.2f}% in predicted response.",
        f"Predictions above {MAX_PHYSICAL_TUMOR_REDUCTION:.0f}% are excluded because the simulation target is capped at this value.",
        "The particle-size curve may contain tree-model steps because XGBoost is an ensemble of decision trees; no visual smoothing was applied, so every point is a direct model prediction.",
        f"Best candidate predicted tumor reduction: {best['predicted_tumor_reduction_percent']:.6f}%.",
        "These are surrogate-model estimates; simulation validation remains necessary before selecting a final formulation.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, pipeline, data = load_artifacts()
    baseline = make_baseline(data)
    particle_sizes, particle_predictions = one_way_sweep(model, pipeline, baseline, "particle_size_nm", PARTICLE_SIZE_POINTS)
    release_rates, release_predictions = one_way_sweep(model, pipeline, baseline, "release_rate", RELEASE_RATE_POINTS)
    sizes, rates, grid_predictions, _ = two_way_sweep(model, pipeline, baseline)
    candidates = optimize_all_parameters(model, pipeline, baseline)
    save_figures(particle_sizes, particle_predictions, release_rates, release_predictions, sizes, rates, grid_predictions)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    write_summary(baseline, response_range(particle_predictions, "particle-size sweep"), response_range(release_predictions, "release-rate sweep"), response_range(grid_predictions, "joint optimization"), candidates)
    required = [PARTICLE_FIGURE, RELEASE_FIGURE, HEATMAP_FIGURE, CANDIDATES_PATH, SUMMARY_PATH]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("One or more Result 10 artifacts were not created.")
    print(f"Result 10 completed. Outputs saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
