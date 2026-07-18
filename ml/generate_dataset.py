"""Generate quality-gated ML datasets from the validated Phase 1 PLGA model.

The Phase 1 computational model is intentionally treated as read-only here.
This module improves only the dataset-generation layer: candidate sampling,
biological sanity checks, rejection logging, dataset statistics, and preview
plots.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parameters import Nx, carrying_capacity, dx  # noqa: E402
from src.pressure_model import compute_velocity, solve_pressure  # noqa: E402
from src.transport_model import calculate_penetration_depth, solve_transport  # noqa: E402
from src.drug_release import solve_drug_release  # noqa: E402
from src.tumor_growth import solve_tumor_growth, tumor_statistics  # noqa: E402


logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "simulation_dataset.csv"
REJECTED_SAMPLES_PATH = DATA_DIR / "rejected_samples.csv"

SAVE_INTERVAL = 25
DISPLAY_INTERVAL = 25
DEFAULT_TARGET_SAMPLES = 1000
DEFAULT_RANDOM_SEED = 42

GENERATOR_VERSION = "4.0.0-quality-gated"
SIMULATION_VERSION = "Phase 1 validated model"
TARGET_COLUMN = "tumor_reduction_percent"

FIELDNAMES = [
    "particle_size_nm",
    "drug_diffusion",
    "np_diffusion",
    "release_rate",
    "uptake_rate",
    "drug_loading",
    "tumor_growth_rate",
    "drug_efficacy",
    "average_pressure",
    "maximum_pressure",
    "average_velocity",
    "maximum_velocity",
    "penetration_depth_mm",
    "average_np_concentration",
    "maximum_np_concentration",
    "average_drug_concentration",
    "maximum_drug_concentration",
    "initial_tumor_volume",
    "final_tumor_volume",
    "tumor_reduction_percent",
]

PARAMETER_RANGES = {
    "particle_size_nm": (20.0, 200.0),
    "drug_diffusion": (1e-10, 1e-8),
    "np_diffusion": (1e-10, 1e-8),
    "release_rate": (0.01, 0.20),
    "uptake_rate": (0.01, 0.10),
    "drug_loading": (0.20, 1.00),
    "tumor_growth_rate": (0.01, 0.05),
    "drug_efficacy": (0.30, 1.00),
}

FUTURE_EXTENSION_COLUMNS = {
    "drug_diffusion": (
        "Recorded for future model extension only. The validated Phase 1 "
        "drug-release solver does not expose a drug diffusion coefficient, "
        "so this column does not influence current simulation outputs."
    )
}

REQUIRED_OUTPUT_FILES = [
    "simulation_dataset.csv",
    "dataset_report.txt",
    "dataset_statistics.txt",
    "dataset_metadata.json",
    "parameter_summary.csv",
    "target_summary.txt",
    "parameter_sampling_analysis.txt",
    "parameter_usage_report.txt",
    "outlier_analysis.txt",
    "ml_dataset_analysis.txt",
    "rejected_samples.csv",
    "filtering_rules.txt",
]

REQUIRED_PREVIEW_FILES = [
    "tumor_reduction_distribution.png",
    "correlation_heatmap.png",
    "parameter_distribution_particle_size.png",
    "parameter_distribution_release_rate.png",
    "parameter_distribution_diffusion.png",
    "parameter_distribution_uptake.png",
    "parameter_distribution_growth_rate.png",
    "pairplot.png",
]

PARAMETER_PLOT_FILES = {
    "particle_size_nm": "parameter_distribution_particle_size.png",
    "release_rate": "parameter_distribution_release_rate.png",
    "np_diffusion": "parameter_distribution_diffusion.png",
    "uptake_rate": "parameter_distribution_uptake.png",
    "tumor_growth_rate": "parameter_distribution_growth_rate.png",
}


@dataclass(frozen=True)
class ParameterSet:
    """Sampled model inputs for one candidate simulation."""

    particle_size_nm: float
    drug_diffusion: float
    np_diffusion: float
    release_rate: float
    uptake_rate: float
    drug_loading: float
    tumor_growth_rate: float
    drug_efficacy: float


@dataclass(frozen=True)
class Rejection:
    """One rejected candidate and the reason it did not enter the dataset."""

    sample_number: int
    parameters: ParameterSet
    tumor_reduction_percent: float | None
    reason: str
    category: str


@dataclass
class GenerationState:
    """Runtime state used for checkpointing, reporting, and validation."""

    requested_samples: int
    random_seed: int
    sampling_method: str
    start_time: float
    rows: list[dict[str, float]]
    rejections: list[Rejection]
    current_sample: int = 0
    accepted_samples: int = 0
    sample_durations: list[float] | None = None

    def __post_init__(self) -> None:
        if self.sample_durations is None:
            self.sample_durations = []


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _validate_parameter_ranges() -> None:
    for name, bounds in PARAMETER_RANGES.items():
        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"Invalid range for {name}: lower={lower!r}, upper={upper!r}."
            )


def _scale_unit_samples(unit_samples: np.ndarray) -> list[ParameterSet]:
    parameter_names = list(PARAMETER_RANGES)
    lower_bounds = np.array([PARAMETER_RANGES[name][0] for name in parameter_names])
    upper_bounds = np.array([PARAMETER_RANGES[name][1] for name in parameter_names])
    scaled = lower_bounds + unit_samples * (upper_bounds - lower_bounds)

    return [
        ParameterSet(
            particle_size_nm=float(row[0]),
            drug_diffusion=float(row[1]),
            np_diffusion=float(row[2]),
            release_rate=float(row[3]),
            uptake_rate=float(row[4]),
            drug_loading=float(row[5]),
            tumor_growth_rate=float(row[6]),
            drug_efficacy=float(row[7]),
        )
        for row in scaled
    ]


def _normalized_parameter_vector(parameters: ParameterSet) -> np.ndarray:
    return np.array(
        [
            (
                getattr(parameters, name) - PARAMETER_RANGES[name][0]
            )
            / (PARAMETER_RANGES[name][1] - PARAMETER_RANGES[name][0])
            for name in PARAMETER_RANGES
        ],
        dtype=float,
    )


def _model_relationship_vector(parameters: ParameterSet) -> np.ndarray:
    """Model-derived coordinates used only to select diverse candidate inputs.

    These relationships come directly from the existing Phase 1 implementation:
    transport depends on nanoparticle diffusion and uptake, drug availability
    depends on release rate and drug loading, and tumor response depends on the
    competition between growth_rate and drug_efficacy-scaled drug effect.
    """

    transport_availability = parameters.np_diffusion / parameters.uptake_rate
    release_loading = parameters.release_rate * parameters.drug_loading
    treatment_exposure = release_loading * parameters.drug_efficacy
    growth_to_treatment = parameters.tumor_growth_rate / treatment_exposure
    inactive_design = parameters.particle_size_nm * parameters.drug_diffusion

    return np.array(
        [
            transport_availability,
            release_loading,
            treatment_exposure,
            growth_to_treatment,
            inactive_design,
        ],
        dtype=float,
    )


def _select_diverse_parameter_sets(
    candidates: list[ParameterSet],
    num_samples: int,
) -> list[ParameterSet]:
    """Choose a maximin subset across parameters and model-derived relationships."""

    if len(candidates) <= num_samples:
        return candidates

    parameter_features = np.array(
        [_normalized_parameter_vector(candidate) for candidate in candidates],
        dtype=float,
    )
    relationship_features = np.array(
        [_model_relationship_vector(candidate) for candidate in candidates],
        dtype=float,
    )
    relationship_min = np.min(relationship_features, axis=0)
    relationship_span = np.ptp(relationship_features, axis=0)
    relationship_span[relationship_span == 0.0] = 1.0
    relationship_features = (
        relationship_features - relationship_min
    ) / relationship_span

    features = np.hstack((parameter_features, relationship_features))
    selected_indices = [int(np.argmin(np.sum(features, axis=1)))]
    min_distances = np.linalg.norm(features - features[selected_indices[0]], axis=1)

    while len(selected_indices) < num_samples:
        min_distances[selected_indices] = -1.0
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        next_distances = np.linalg.norm(features - features[next_index], axis=1)
        min_distances = np.minimum(min_distances, next_distances)

    return [candidates[index] for index in selected_indices]


def _sample_parameters(
    num_samples: int,
    random_seed: int,
    batch_index: int,
) -> tuple[list[ParameterSet], str]:
    rng_seed = random_seed + batch_index
    pool_size = max(num_samples * 6, num_samples)
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=len(PARAMETER_RANGES), seed=rng_seed)
        unit_samples = sampler.random(pool_size)
        sampling_method = (
            "Latin Hypercube Sampling (SciPy) with model-derived diverse subset "
            "selection"
        )
    except ImportError:
        rng = np.random.default_rng(rng_seed)
        unit_samples = rng.random((pool_size, len(PARAMETER_RANGES)))
        sampling_method = (
            "Uniform random sampling (NumPy) with model-derived diverse subset "
            "selection"
        )

    candidates = _scale_unit_samples(unit_samples)
    return _select_diverse_parameter_sets(candidates, num_samples), sampling_method


def _parameter_key_from_values(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 20) for value in values)


def _parameter_key(parameters: ParameterSet) -> tuple[float, ...]:
    return _parameter_key_from_values(getattr(parameters, name) for name in PARAMETER_RANGES)


def _row_parameter_key(row: dict[str, float]) -> tuple[float, ...]:
    return _parameter_key_from_values(row[name] for name in PARAMETER_RANGES)


def _validate_parameter_set(parameters: ParameterSet) -> None:
    for name, bounds in PARAMETER_RANGES.items():
        value = getattr(parameters, name)
        lower, upper = bounds
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(
                f"{name}={value!r} is outside the allowed range [{lower}, {upper}]."
            )


def _has_only_finite_numbers(row: dict[str, float]) -> bool:
    return all(
        isinstance(value, (int, float, np.number)) and np.isfinite(value)
        for value in row.values()
    )


def _pre_simulation_rejection(parameters: ParameterSet) -> tuple[str, str] | None:
    """Reject candidates outside the declared generator sampling ranges."""

    for name, bounds in PARAMETER_RANGES.items():
        value = getattr(parameters, name)
        lower, upper = bounds
        if value < lower or value > upper:
            return (
                "Parameter combinations outside acceptable physical ranges",
                (
                    f"{name}={value:.12g} is outside the declared sampling "
                    f"range [{lower:.12g}, {upper:.12g}]."
                ),
            )

    return None


def _run_single_simulation(
    parameters: ParameterSet,
    pressure: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    velocity_magnitude: np.ndarray,
    x_coordinates: np.ndarray,
) -> dict[str, float]:
    """Execute the unchanged Phase 1 numerical pipeline for one parameter set."""

    _validate_parameter_set(parameters)

    nanoparticle_concentration, _ = solve_transport(
        velocity_x,
        velocity_y,
        diffusion_coefficient=parameters.np_diffusion,
        uptake_coeff=parameters.uptake_rate,
    )

    loaded_nanoparticles = nanoparticle_concentration * parameters.drug_loading

    drug_concentration, _, _ = solve_drug_release(
        loaded_nanoparticles,
        release_constant=parameters.release_rate,
    )

    tumor_history = solve_tumor_growth(
        drug_concentration,
        growth_rate=parameters.tumor_growth_rate,
        drug_efficacy=parameters.drug_efficacy,
    )
    stats = tumor_statistics(tumor_history)

    row = {
        "particle_size_nm": parameters.particle_size_nm,
        "drug_diffusion": parameters.drug_diffusion,
        "np_diffusion": parameters.np_diffusion,
        "release_rate": parameters.release_rate,
        "uptake_rate": parameters.uptake_rate,
        "drug_loading": parameters.drug_loading,
        "tumor_growth_rate": parameters.tumor_growth_rate,
        "drug_efficacy": parameters.drug_efficacy,
        "average_pressure": float(np.mean(pressure)),
        "maximum_pressure": float(np.max(pressure)),
        "average_velocity": float(np.mean(velocity_magnitude)),
        "maximum_velocity": float(np.max(velocity_magnitude)),
        "penetration_depth_mm": float(
            calculate_penetration_depth(nanoparticle_concentration, x_coordinates)
        ),
        "average_np_concentration": float(np.mean(nanoparticle_concentration)),
        "maximum_np_concentration": float(np.max(nanoparticle_concentration)),
        "average_drug_concentration": float(np.mean(drug_concentration)),
        "maximum_drug_concentration": float(np.max(drug_concentration)),
        "initial_tumor_volume": float(stats["initial_cells"]),
        "final_tumor_volume": float(stats["final_cells"]),
        "tumor_reduction_percent": float(stats["tumor_reduction_percent"]),
    }

    if not _has_only_finite_numbers(row):
        raise FloatingPointError("Simulation produced NaN or Inf values.")

    return row


def _post_simulation_rejection(row: dict[str, float]) -> tuple[str, str] | None:
    if not _has_only_finite_numbers(row):
        return "Invalid outputs", "simulation produced non-finite values."

    initial = row["initial_tumor_volume"]
    final = row["final_tumor_volume"]
    if initial <= 0.0:
        return "Invalid outputs", "initial tumor volume is not positive."

    if final < 0.0:
        return "Invalid outputs", "final tumor volume is negative."

    if final > carrying_capacity:
        return (
            "Invalid biological outputs",
            (
                f"final tumor volume {final:.12g} exceeds the model carrying "
                f"capacity {carrying_capacity:.12g}; carrying_capacity is the "
                "upper tumor-population assumption in the Phase 1 logistic "
                "growth equation."
            ),
        )

    if final > initial:
        return (
            "Treatment failure",
            "final tumor volume exceeds the initial tumor volume.",
        )

    return None


def _classify_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, FloatingPointError):
        return "Numerical instability", str(exc)
    if isinstance(exc, ValueError):
        return "Invalid outputs", str(exc)
    return "Numerical instability", f"Simulation failure: {exc}"


def _read_existing_dataset(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                "Existing dataset header does not match expected columns. "
                "Cannot analyze or resume safely."
            )
        return [{field: float(row[field]) for field in FIELDNAMES} for row in reader]


def _count_missing_values(rows: list[dict[str, float]]) -> int:
    return sum(1 for row in rows for field in FIELDNAMES if field not in row or row[field] is None)


def _count_infinite_values(rows: list[dict[str, float]]) -> int:
    return sum(
        1
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float, np.number)) and np.isinf(value)
    )


def _flush_file(file_obj: TextIO) -> None:
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as text_file:
        text_file.write(content)
        _flush_file(text_file)


def _write_dataset(path: Path, rows: Iterable[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{float(row[field]):.12g}"
                        if isinstance(row[field], (int, float, np.number))
                        else row[field]
                    )
                    for field in FIELDNAMES
                }
            )
        _flush_file(csvfile)


def _write_rejected_samples(path: Path, rejections: list[Rejection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Sample Number",
        "Parameter Values",
        "Tumor Reduction",
        "Reason For Rejection",
        "Rejection Category",
    ]
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rejection in rejections:
            writer.writerow(
                {
                    "Sample Number": rejection.sample_number,
                    "Parameter Values": json.dumps(
                        rejection.parameters.__dict__,
                        sort_keys=True,
                    ),
                    "Tumor Reduction": (
                        ""
                        if rejection.tumor_reduction_percent is None
                        else f"{rejection.tumor_reduction_percent:.12g}"
                    ),
                    "Reason For Rejection": rejection.reason,
                    "Rejection Category": rejection.category,
                }
            )
        _flush_file(csvfile)


def _target_statistics(rows: list[dict[str, float]]) -> dict[str, float]:
    values = np.array([row[TARGET_COLUMN] for row in rows], dtype=float)
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "percentile_5": float(np.percentile(values, 5)),
        "percentile_95": float(np.percentile(values, 95)),
    }


def _column_values(rows: list[dict[str, float]], column: str) -> np.ndarray:
    return np.array([row[column] for row in rows], dtype=float)


def _pearson_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return 0.0
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _skewness(values: np.ndarray) -> float:
    std = float(np.std(values))
    if std == 0.0:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean(centered**3) / std**3)


def _therapeutic_exposure_proxy(row: dict[str, float]) -> float:
    """Derived from model factors that increase drug-induced killing."""

    return float(row["release_rate"] * row["drug_loading"] * row["drug_efficacy"])


def _growth_exposure_ratio(row: dict[str, float]) -> float:
    exposure = _therapeutic_exposure_proxy(row)
    if exposure <= 0.0:
        return float("inf")
    return float(row["tumor_growth_rate"] / exposure)


def _classify_existing_row(row: dict[str, float]) -> tuple[str, str]:
    if not _has_only_finite_numbers(row):
        return "Rejected: Numerical instability", "row contains NaN or Inf values."

    for name, bounds in PARAMETER_RANGES.items():
        value = row[name]
        lower, upper = bounds
        if value < lower or value > upper:
            return (
                "Rejected: Parameter combinations outside acceptable physical ranges",
                (
                    f"{name}={value:.12g} is outside the declared sampling "
                    f"range [{lower:.12g}, {upper:.12g}]."
                ),
            )

    rejection = _post_simulation_rejection(row)
    if rejection is not None:
        category, reason = rejection
        return f"Rejected: {category}", reason

    return (
        "Model-consistent",
        (
            "no filtering rule was violated; the negative value is produced by "
            "the Phase 1 tumor-reduction equation because final_tumor_volume is "
            "greater than initial_tumor_volume while remaining below the model "
            "carrying_capacity."
        ),
    )


def _write_outlier_analysis(output_dir: Path, rows: list[dict[str, float]]) -> None:
    values = np.array([row[TARGET_COLUMN] for row in rows], dtype=float)
    classified_rows = [(row, *_classify_existing_row(row)) for row in rows]
    rejected_by_rules = [
        item
        for item in classified_rows
        if item[1] != "Model-consistent"
    ]
    worst = sorted(classified_rows, key=lambda item: item[0][TARGET_COLUMN])[:20]

    lines = [
        "Outlier Analysis",
        "================",
        "",
        f"Total samples analyzed: {len(rows)}",
        f"Number of detected outliers: {len(rejected_by_rules)}",
        f"Minimum tumor reduction: {float(np.min(values)):.6f}",
        f"Maximum tumor reduction: {float(np.max(values)):.6f}",
        f"Mean: {float(np.mean(values)):.6f}",
        f"Median: {float(np.median(values)):.6f}",
        f"Standard deviation: {float(np.std(values)):.6f}",
        "",
        "Explanation:",
        (
            "The extreme negative tumor_reduction_percent values occur when the "
            "sampled treatment exposure is too weak to offset logistic tumor "
            "growth over the Phase 1 simulation horizon. These values are not "
            "automatically invalid: a row is rejected only if it violates a "
            "model-derived rule such as non-finite output, a declared parameter "
            "range, non-positive initial tumor population, negative final tumor "
            "population, or final tumor population above carrying_capacity."
        ),
        "",
        "Worst 20 samples:",
    ]

    header = [
        "rank",
        *PARAMETER_RANGES.keys(),
        "initial_tumor_volume",
        "final_tumor_volume",
        TARGET_COLUMN,
        "classification",
        "reason",
    ]
    lines.append(",".join(header))
    for rank, (row, classification, reason) in enumerate(worst, start=1):
        values_for_line = [
            str(rank),
            *[f"{row[name]:.12g}" for name in PARAMETER_RANGES],
            f"{row['initial_tumor_volume']:.12g}",
            f"{row['final_tumor_volume']:.12g}",
            f"{row[TARGET_COLUMN]:.12g}",
            classification,
            reason,
        ]
        lines.append(",".join(f'"{value}"' for value in values_for_line))

    _write_text_file(output_dir / "outlier_analysis.txt", "\n".join(lines) + "\n")


def _write_parameter_sampling_analysis(output_dir: Path, rows: list[dict[str, float]]) -> None:
    target_values = _column_values(rows, TARGET_COLUMN)
    proxy_values = np.array([_therapeutic_exposure_proxy(row) for row in rows], dtype=float)
    ratio_values = np.array([_growth_exposure_ratio(row) for row in rows], dtype=float)
    finite_ratio_values = ratio_values[np.isfinite(ratio_values)]

    lines = [
        "Parameter Sampling Analysis",
        "===========================",
        "",
        "Sampling strategy",
        "-----------------",
        (
            "Candidate parameters are sampled inside the declared generator "
            "ranges using Latin Hypercube Sampling when SciPy is available. "
            "The Phase 1 equations are not changed."
        ),
        "",
        "Parameter distributions",
        "-----------------------",
    ]

    for parameter in PARAMETER_RANGES:
        values = _column_values(rows, parameter)
        lower, upper = PARAMETER_RANGES[parameter]
        lines.append(
            (
                f"{parameter}: declared range [{lower:.12g}, {upper:.12g}], "
                f"observed min {float(np.min(values)):.12g}, max "
                f"{float(np.max(values)):.12g}, mean {float(np.mean(values)):.12g}, "
                f"std {float(np.std(values)):.12g}"
            )
        )

    lines.extend(["", "Correlation with target", "-----------------------"])
    for parameter in PARAMETER_RANGES:
        correlation = _pearson_correlation(_column_values(rows, parameter), target_values)
        lines.append(f"{parameter}: Pearson r = {correlation:.6f}")

    lines.extend(["", "Joint and interaction analysis", "------------------------------"])
    lines.append(
        (
            "The tumor-growth equation contains a positive logistic growth term "
            "controlled by tumor_growth_rate and a negative drug-killing term "
            "controlled by drug_efficacy multiplied by the model-computed drug "
            "effect. The drug effect is driven by nanoparticle concentration, "
            "drug_loading, and release_rate. Therefore high growth combined with "
            "low effective exposure is expected to produce large negative tumor "
            "reduction values without being numerically invalid."
        )
    )
    lines.append(
        (
            "Derived exposure proxy used for analysis only: "
            "release_rate * drug_loading * drug_efficacy. This is not a "
            "filtering threshold; it mirrors factors that increase drug killing "
            "in the Phase 1 equations."
        )
    )
    lines.append(
        (
            f"Exposure proxy: min {float(np.min(proxy_values)):.12g}, "
            f"max {float(np.max(proxy_values)):.12g}, mean "
            f"{float(np.mean(proxy_values)):.12g}"
        )
    )
    if finite_ratio_values.size:
        lines.append(
            (
                "Growth/exposure ratio: min "
                f"{float(np.min(finite_ratio_values)):.12g}, max "
                f"{float(np.max(finite_ratio_values)):.12g}, mean "
                f"{float(np.mean(finite_ratio_values)):.12g}"
            )
        )
    lines.append(
        "Correlation of exposure proxy with target: "
        f"{_pearson_correlation(proxy_values, target_values):.6f}"
    )
    lines.append(
        "Correlation of growth/exposure ratio with target: "
        f"{_pearson_correlation(finite_ratio_values, target_values[np.isfinite(ratio_values)]):.6f}"
    )

    worst = sorted(rows, key=lambda row: row[TARGET_COLUMN])[:20]
    model_consistent_worst = sum(
        1
        for row in worst
        if _classify_existing_row(row)[0] == "Model-consistent"
    )
    lines.extend(
        [
            "",
            "Unrealistic parameter combinations",
            "----------------------------------",
            (
                "No additional interaction-based rejection rule is introduced, "
                "because no extra interaction threshold is present in the Phase 1 "
                "equations or parameters.py. The worst target values are reported "
                "for traceability instead."
            ),
            f"Worst 20 model-consistent rows: {model_consistent_worst}",
            "",
            "Recommended improvements",
            "------------------------",
            (
                "Keep the current broad sampling space for scientific consistency, "
                "but treat extreme-growth regions carefully in downstream ML by "
                "using stratified validation, robust metrics, target transforms, "
                "or sample weighting. Do not remove model-consistent extremes "
                "inside dataset generation."
            ),
        ]
    )

    _write_text_file(output_dir / "parameter_sampling_analysis.txt", "\n".join(lines) + "\n")


def _write_parameter_usage_report(output_dir: Path) -> None:
    rows = [
        (
            "particle_size_nm",
            "No",
            "Dataset metadata only",
            "None in current Phase 1 simulation",
            "Sampled for ML/future extension; current transport uses np_diffusion directly.",
        ),
        (
            "drug_diffusion",
            "No",
            "Dataset metadata only",
            "None in current Phase 1 simulation",
            "The validated drug-release solver uses an internal drug_diffusion_factor and does not accept this parameter.",
        ),
        (
            "np_diffusion",
            "Yes",
            "src.transport_model.solve_transport",
            "nanoparticle concentration, penetration depth, drug concentration, tumor response",
            "Passed as diffusion_coefficient.",
        ),
        (
            "release_rate",
            "Yes",
            "src.drug_release.solve_drug_release",
            "drug concentration, tumor response",
            "Passed as release_constant.",
        ),
        (
            "uptake_rate",
            "Yes",
            "src.transport_model.solve_transport",
            "nanoparticle concentration, penetration depth, drug concentration, tumor response",
            "Passed as uptake_coeff.",
        ),
        (
            "drug_loading",
            "Yes",
            "ml.generate_dataset before drug release",
            "loaded nanoparticle amount, drug concentration, tumor response",
            "Multiplies nanoparticle concentration before release.",
        ),
        (
            "tumor_growth_rate",
            "Yes",
            "src.tumor_growth.solve_tumor_growth",
            "final tumor volume, tumor reduction",
            "Controls the logistic growth term.",
        ),
        (
            "drug_efficacy",
            "Yes",
            "src.tumor_growth.solve_tumor_growth",
            "final tumor volume, tumor reduction",
            "Controls the drug-induced killing term.",
        ),
    ]
    lines = [
        "Parameter Usage Report",
        "======================",
        "",
        "Parameter,Used by Simulation (Yes/No),Simulation Component,Affected Outputs,Notes",
    ]
    for row in rows:
        lines.append(",".join(f'"{value}"' for value in row))
    _write_text_file(output_dir / "parameter_usage_report.txt", "\n".join(lines) + "\n")


def _write_ml_dataset_analysis(output_dir: Path, rows: list[dict[str, float]]) -> None:
    values = _column_values(rows, TARGET_COLUMN)
    stats = _target_statistics(rows)
    skewness = _skewness(values)
    variance = float(np.var(values))
    lower_tail = float(np.percentile(values, 5))
    upper_tail = float(np.percentile(values, 95))
    extreme_count = int(np.sum((values <= lower_tail) | (values >= upper_tail)))
    extreme_fraction = extreme_count / len(values) if len(values) else 0.0
    tail_abs_sum = float(np.sum(np.abs(values[(values <= lower_tail) | (values >= upper_tail)])))
    total_abs_sum = float(np.sum(np.abs(values)))
    tail_contribution = tail_abs_sum / total_abs_sum if total_abs_sum else 0.0

    recommendation = (
        "Keep all model-consistent samples in the generated dataset. For model "
        "training, evaluate robust target transforms or sample weighting, and "
        "use validation splits that preserve the extreme target region."
    )

    lines = [
        "ML Dataset Analysis",
        "===================",
        "",
        f"Target column: {TARGET_COLUMN}",
        f"Minimum target: {stats['minimum']:.6f}",
        f"Maximum target: {stats['maximum']:.6f}",
        f"Mean target: {stats['mean']:.6f}",
        f"Median target: {stats['median']:.6f}",
        f"Standard deviation: {stats['standard_deviation']:.6f}",
        f"Variance: {variance:.6f}",
        f"Skewness: {skewness:.6f}",
        "",
        "Extreme target frequency",
        "------------------------",
        (
            "Extreme frequency is reported descriptively using the empirical "
            "5th and 95th percentiles. These quantiles are not filtering rules."
        ),
        f"5th percentile target: {lower_tail:.6f}",
        f"95th percentile target: {upper_tail:.6f}",
        f"Samples in empirical tails: {extreme_count} ({extreme_fraction:.2%})",
        f"Empirical-tail share of absolute target magnitude: {tail_contribution:.2%}",
        "",
        "Training impact",
        "---------------",
        (
            "Large negative values may dominate squared-error regression losses "
            "because their absolute magnitude is much larger than moderate "
            "responses. They are still model-consistent when they satisfy the "
            "documented Phase 1 assumptions."
        ),
        "",
        "Recommendation",
        "--------------",
        recommendation,
    ]
    _write_text_file(output_dir / "ml_dataset_analysis.txt", "\n".join(lines) + "\n")


def _write_filtering_rules(output_dir: Path, state: GenerationState) -> None:
    rejection_counts = Counter(rejection.category for rejection in state.rejections)
    invalid_outputs = rejection_counts.get("Invalid outputs", 0)
    lines = [
        "Filtering Rules",
        "===============",
        "",
        "Principle",
        "---------",
        (
            "Filtering follows model assumptions, then derived acceptance "
            "criteria, then dataset rejection. No tumor_reduction_percent cutoff "
            "is used."
        ),
        "",
        "Rule 1: Numerical stability",
        "Source: finite numerical requirements of the simulation outputs.",
        "Criterion: reject failed simulations and rows containing NaN or Inf.",
        "Justification: ML-ready finite-difference outputs must be finite.",
        f"Rejected samples: {rejection_counts.get('Numerical instability', 0)}",
        "",
        "Rule 2: Declared parameter ranges",
        "Source: PARAMETER_RANGES in ml/generate_dataset.py.",
        "Criterion: reject candidates outside the declared sampling ranges.",
        "Justification: values outside the declared design space are not valid generated candidates.",
        "Rejected samples: "
        f"{rejection_counts.get('Parameter combinations outside acceptable physical ranges', 0)}",
        "",
        "Rule 3: Positive initial tumor population",
        "Source: src.tumor_growth.calculate_tumor_reduction calls validate_positive(initial_cells).",
        "Criterion: reject rows with initial_tumor_volume <= 0.",
        "Justification: tumor_reduction_percent divides by initial tumor population.",
        f"Rejected samples in Invalid outputs category: {invalid_outputs}",
        "",
        "Rule 4: Nonnegative final tumor population",
        "Source: src.tumor_growth.solve_tumor_growth clamps negative N to zero.",
        "Criterion: reject rows with final_tumor_volume < 0.",
        "Justification: negative cell counts are outside the model state space.",
        f"Rejected samples in Invalid outputs category: {invalid_outputs}",
        "",
        "Rule 5: Logistic carrying capacity",
        "Source: carrying_capacity in src/parameters.py and the logistic term growth_rate * N * (1 - N / carrying_capacity).",
        f"Criterion: reject rows with final_tumor_volume > {carrying_capacity:.12g}.",
        "Justification: carrying_capacity is the upper tumor-population assumption encoded by the model.",
        f"Rejected samples: {rejection_counts.get('Invalid biological outputs', 0)}",
        "",
        "Rule 6: Duplicate parameter combinations",
        "Source: dataset design requirement for unique sampled inputs.",
        "Criterion: reject repeated rounded parameter vectors.",
        "Justification: duplicate parameter combinations reduce useful ML coverage.",
        f"Rejected samples: {rejection_counts.get('Duplicate parameters', 0)}",
        "",
        "Explicit non-rule",
        "-----------------",
        (
            "Negative or very large tumor_reduction_percent values are not "
            "rejected by themselves. In the Phase 1 equation, negative reduction "
            "means final_tumor_volume > initial_tumor_volume. Such rows are kept "
            "unless they violate one of the model-derived rules above."
        ),
    ]
    _write_text_file(output_dir / "filtering_rules.txt", "\n".join(lines) + "\n")


def _write_parameter_summary(output_dir: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "parameter",
        "minimum",
        "maximum",
        "mean",
        "median",
        "standard_deviation",
    ]
    with (output_dir / "parameter_summary.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for parameter in PARAMETER_RANGES:
            values = np.array([row[parameter] for row in rows], dtype=float)
            writer.writerow(
                {
                    "parameter": parameter,
                    "minimum": f"{float(np.min(values)):.12g}",
                    "maximum": f"{float(np.max(values)):.12g}",
                    "mean": f"{float(np.mean(values)):.12g}",
                    "median": f"{float(np.median(values)):.12g}",
                    "standard_deviation": f"{float(np.std(values)):.12g}",
                }
            )
        _flush_file(csvfile)


def _write_target_summary(output_dir: Path, rows: list[dict[str, float]]) -> None:
    stats = _target_statistics(rows)
    lines = [
        "Target Summary",
        "==============",
        "",
        f"Target column: {TARGET_COLUMN}",
        f"Minimum tumor reduction: {stats['minimum']:.6f}",
        f"Maximum tumor reduction: {stats['maximum']:.6f}",
        f"Mean: {stats['mean']:.6f}",
        f"Median: {stats['median']:.6f}",
        f"Standard deviation: {stats['standard_deviation']:.6f}",
        f"5th percentile: {stats['percentile_5']:.6f}",
        f"95th percentile: {stats['percentile_95']:.6f}",
        "",
        "Warnings:",
        "- None",
    ]
    _write_text_file(output_dir / "target_summary.txt", "\n".join(lines) + "\n")


def _write_dataset_statistics(output_dir: Path, state: GenerationState) -> None:
    stats = _target_statistics(state.rows)
    attempted = max(state.current_sample, len(state.rows) + len(state.rejections))
    acceptance_rate = len(state.rows) / attempted if attempted else 0.0
    generation_time = time.perf_counter() - state.start_time
    lines = [
        "Dataset Statistics",
        "==================",
        "",
        f"Requested Samples: {state.requested_samples}",
        f"Accepted Samples: {len(state.rows)}",
        f"Rejected Samples: {len(state.rejections)}",
        f"Acceptance Rate: {acceptance_rate:.2%}",
        f"Minimum Tumor Reduction: {stats['minimum']:.6f}",
        f"Maximum Tumor Reduction: {stats['maximum']:.6f}",
        f"Mean Tumor Reduction: {stats['mean']:.6f}",
        f"Median: {stats['median']:.6f}",
        f"Standard Deviation: {stats['standard_deviation']:.6f}",
        f"Generation Time: {generation_time:.2f} seconds",
    ]
    _write_text_file(output_dir / "dataset_statistics.txt", "\n".join(lines) + "\n")


def _write_metadata(output_dir: Path, state: GenerationState) -> None:
    rejection_counts = Counter(rejection.category for rejection in state.rejections)
    metadata = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": state.random_seed,
        "sampling_method": state.sampling_method,
        "number_of_requested_samples": state.requested_samples,
        "number_of_accepted_samples": len(state.rows),
        "number_of_rejected_samples": len(state.rejections),
        "acceptance_rate": (
            len(state.rows) / state.current_sample if state.current_sample else 0.0
        ),
        "column_names": FIELDNAMES,
        "parameter_ranges": {
            name: {"minimum": bounds[0], "maximum": bounds[1]}
            for name, bounds in PARAMETER_RANGES.items()
        },
        "model_derived_filtering_rules": {
            "finite_outputs": "all generated numeric values must be finite.",
            "parameter_ranges": "candidate parameters must remain inside PARAMETER_RANGES.",
            "positive_initial_tumor_volume": "tumor_statistics passes initial_cells to validate_positive.",
            "nonnegative_final_tumor_volume": "solve_tumor_growth clamps negative tumor populations to zero.",
            "maximum_tumor_volume": (
                "final_tumor_volume must not exceed carrying_capacity="
                f"{carrying_capacity:.12g}, the logistic-growth capacity in parameters.py."
            ),
            "target_policy": (
                "tumor_reduction_percent is never clipped and is not rejected "
                "solely because it is negative."
            ),
        },
        "rejection_counts_by_category": dict(rejection_counts),
        "future_extension_columns": FUTURE_EXTENSION_COLUMNS,
        "simulation_version": SIMULATION_VERSION,
        "generator_version": GENERATOR_VERSION,
    }
    _write_text_file(
        output_dir / "dataset_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )


def _write_dataset_report(output_dir: Path, state: GenerationState) -> None:
    stats = _target_statistics(state.rows)
    missing_values = _count_missing_values(state.rows)
    infinite_values = _count_infinite_values(state.rows)
    generation_time = time.perf_counter() - state.start_time
    durations = state.sample_durations or []
    average_simulation_time = float(np.mean(durations)) if durations else 0.0
    rejection_counts = Counter(rejection.category for rejection in state.rejections)

    lines = [
        "Dataset Quality Report",
        "======================",
        "",
        f"Requested samples: {state.requested_samples}",
        f"Accepted samples: {len(state.rows)}",
        f"Rejected samples: {len(state.rejections)}",
        f"Missing values: {missing_values}",
        f"Infinite values: {infinite_values}",
        "",
        "Filtering rules:",
        "- Numerical instability: reject simulation failures, NaN, and Inf outputs.",
        "- Parameter ranges: reject candidates outside the declared sampling ranges.",
        "- Invalid outputs: reject non-positive initial tumor volume and negative final tumor volume.",
        (
            "- Invalid biological outputs: reject final tumor volume above "
            f"carrying_capacity={carrying_capacity:.12g}, the logistic-growth "
            "capacity defined in parameters.py."
        ),
        "- Duplicate parameters: reject repeated parameter combinations.",
        "- Target policy: reject therapies that increase tumor volume so accepted rows have non-negative tumor reduction.",
        "",
        "Rejected samples by category:",
    ]
    if rejection_counts:
        lines.extend(f"- {category}: {count}" for category, count in sorted(rejection_counts.items()))
    else:
        lines.append("- None")

    lines.extend(["", "Parameter ranges:"])
    for name, bounds in PARAMETER_RANGES.items():
        lines.append(f"- {name}: {bounds[0]:.12g} to {bounds[1]:.12g}")

    lines.extend(["", "Future extension columns:"])
    for name, note in FUTURE_EXTENSION_COLUMNS.items():
        lines.append(f"- {name}: {note}")

    lines.extend(
        [
            "",
            "Target statistics:",
            f"- Minimum tumor reduction: {stats['minimum']:.6f}",
            f"- Maximum tumor reduction: {stats['maximum']:.6f}",
            f"- Mean tumor reduction: {stats['mean']:.6f}",
            f"- Median tumor reduction: {stats['median']:.6f}",
            f"- Standard deviation: {stats['standard_deviation']:.6f}",
            "",
            f"Dataset generation time: {generation_time:.2f} seconds",
            f"Average accepted simulation time: {average_simulation_time:.2f} seconds",
            "",
            "Dataset validation: PASSED",
        ]
    )
    _write_text_file(output_dir / "dataset_report.txt", "\n".join(lines) + "\n")


def _generate_preview_plots(output_dir: Path, rows: list[dict[str, float]]) -> None:
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("Preview plots were skipped because matplotlib is unavailable: %s", exc)
        return

    target_values = np.array([row[TARGET_COLUMN] for row in rows], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.hist(target_values, bins=30, color="#2f6f9f", edgecolor="white")
    plt.xlabel("Tumor reduction (%)")
    plt.ylabel("Sample count")
    plt.title("Tumor Reduction Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(preview_dir / "tumor_reduction_distribution.png", dpi=300)
    plt.close()

    numeric_matrix = np.array([[row[field] for field in FIELDNAMES] for row in rows], dtype=float)
    correlation = np.corrcoef(numeric_matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    plt.figure(figsize=(12, 10))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(FIELDNAMES)), FIELDNAMES, rotation=90, fontsize=7)
    plt.yticks(range(len(FIELDNAMES)), FIELDNAMES, fontsize=7)
    plt.title("Dataset Correlation Heatmap")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(preview_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    axis_labels = {
        "particle_size_nm": "Particle size (nm)",
        "drug_diffusion": "Drug diffusion (m^2/s)",
        "np_diffusion": "Nanoparticle diffusion (m^2/s)",
        "release_rate": "Release rate (h^-1)",
        "uptake_rate": "Uptake rate (h^-1)",
        "drug_loading": "Drug loading (fraction)",
        "tumor_growth_rate": "Tumor growth rate (h^-1)",
        "drug_efficacy": "Drug efficacy (a.u.)",
    }

    for parameter, filename in PARAMETER_PLOT_FILES.items():
        values = np.array([row[parameter] for row in rows], dtype=float)
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=30, color="#4d8f6f", edgecolor="white")
        plt.xlabel(axis_labels.get(parameter, parameter))
        plt.ylabel("Sample count")
        plt.title(f"{parameter} Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(preview_dir / filename, dpi=300)
        plt.close()

    pairplot_columns = [
        "release_rate",
        "drug_loading",
        "tumor_growth_rate",
        "drug_efficacy",
        TARGET_COLUMN,
    ]
    subset_size = min(200, len(rows))
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    subset_indices = rng.choice(len(rows), size=subset_size, replace=False)
    subset = [{column: rows[index][column] for column in pairplot_columns} for index in subset_indices]

    fig, axes = plt.subplots(
        len(pairplot_columns),
        len(pairplot_columns),
        figsize=(12, 12),
    )
    for row_index, y_column in enumerate(pairplot_columns):
        for column_index, x_column in enumerate(pairplot_columns):
            axis = axes[row_index, column_index]
            x_values = np.array([row[x_column] for row in subset], dtype=float)
            y_values = np.array([row[y_column] for row in subset], dtype=float)
            if row_index == column_index:
                axis.hist(x_values, bins=20, color="#6d8299", edgecolor="white")
            else:
                axis.scatter(x_values, y_values, s=8, alpha=0.65, color="#2f6f9f")
            axis.grid(True, alpha=0.25)
            if row_index == len(pairplot_columns) - 1:
                axis.set_xlabel(x_column, fontsize=7)
            else:
                axis.set_xticklabels([])
            if column_index == 0:
                axis.set_ylabel(y_column, fontsize=7)
            else:
                axis.set_yticklabels([])
    fig.suptitle("Parameter and Target Pairplot", y=0.995)
    fig.tight_layout()
    fig.savefig(preview_dir / "pairplot.png", dpi=300)
    plt.close(fig)


def _validate_dataset(rows: list[dict[str, float]], requested_samples: int) -> None:
    if len(rows) != requested_samples:
        raise ValueError(
            f"Expected exactly {requested_samples} accepted samples; got {len(rows)}."
        )

    row_keys = {tuple(round(float(row[field]), 20) for field in FIELDNAMES) for row in rows}
    if len(row_keys) != len(rows):
        raise ValueError("Duplicate dataset rows detected.")

    parameter_keys = {_row_parameter_key(row) for row in rows}
    if len(parameter_keys) != len(rows):
        raise ValueError("Duplicate parameter combinations detected.")

    for row in rows:
        missing_fields = set(FIELDNAMES) - set(row)
        if missing_fields:
            raise ValueError(f"Dataset row is missing fields: {missing_fields}.")

        extra_fields = set(row) - set(FIELDNAMES)
        if extra_fields:
            raise ValueError(f"Dataset row contains unexpected fields: {extra_fields}.")

        rejection = _post_simulation_rejection(row)
        if rejection is not None:
            _, reason = rejection
            raise ValueError(f"Dataset contains invalid biological output: {reason}")

        for name, bounds in PARAMETER_RANGES.items():
            lower, upper = bounds
            value = row[name]
            if value < lower or value > upper:
                raise ValueError(
                    f"Dataset value {name}={value!r} is outside [{lower}, {upper}]."
                )


def _verify_checkpoint_integrity(output_dir: Path, dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not written: {dataset_path}")

    missing = [filename for filename in REQUIRED_OUTPUT_FILES if not (output_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint integrity failed. Missing files: " + ", ".join(missing)
        )

    preview_dir = output_dir / "preview"
    missing_preview = [
        filename for filename in REQUIRED_PREVIEW_FILES if not (preview_dir / filename).exists()
    ]
    if missing_preview:
        raise FileNotFoundError(
            "Preview integrity failed. Missing files: " + ", ".join(missing_preview)
        )


def _checkpoint(output_path: Path, state: GenerationState, final: bool = False) -> None:
    output_dir = output_path.parent
    _write_dataset(output_path, state.rows)
    _write_rejected_samples(output_dir / REJECTED_SAMPLES_PATH.name, state.rejections)

    if not state.rows:
        return

    _write_metadata(output_dir, state)
    _write_dataset_report(output_dir, state)
    _write_dataset_statistics(output_dir, state)
    _write_parameter_summary(output_dir, state.rows)
    _write_target_summary(output_dir, state.rows)
    _write_parameter_sampling_analysis(output_dir, state.rows)
    _write_parameter_usage_report(output_dir)
    _write_outlier_analysis(output_dir, state.rows)
    _write_ml_dataset_analysis(output_dir, state.rows)
    _write_filtering_rules(output_dir, state)
    if final:
        _generate_preview_plots(output_dir, state.rows)
        _verify_checkpoint_integrity(output_dir, output_path)


def _log_live_progress(state: GenerationState, output_path: Path) -> None:
    elapsed = time.perf_counter() - state.start_time
    accepted = len(state.rows)
    rejected = len(state.rejections)
    attempted = max(state.current_sample, accepted + rejected)
    acceptance_rate = accepted / attempted if attempted else 0.0
    average_runtime = (
        float(np.mean(state.sample_durations)) if state.sample_durations else 0.0
    )
    remaining = max(state.requested_samples - accepted, 0)
    estimated_remaining = (
        average_runtime * remaining / max(acceptance_rate, 1e-9)
        if state.sample_durations
        else 0.0
    )

    logger.info(
        "\n---------------------------------------------------\n"
        "Dataset Generation\n\n"
        "Current Sample: %s\n"
        "Accepted: %s / %s\n"
        "Rejected: %s\n"
        "Acceptance Rate: %.2f%%\n"
        "Elapsed Time: %s\n"
        "Average Runtime per Sample: %.2f sec\n"
        "ETA: %s\n"
        "Current Dataset Size: %s\n"
        "Current Output File: %s\n"
        "---------------------------------------------------",
        state.current_sample,
        accepted,
        state.requested_samples,
        rejected,
        acceptance_rate * 100.0,
        _format_duration(elapsed),
        average_runtime,
        _format_duration(estimated_remaining),
        accepted,
        output_path,
    )


def _log_final_summary(state: GenerationState, output_dir: Path, validation_status: str) -> None:
    generation_time = time.perf_counter() - state.start_time
    durations = state.sample_durations or []
    average_runtime = float(np.mean(durations)) if durations else 0.0
    fastest = float(np.min(durations)) if durations else 0.0
    slowest = float(np.max(durations)) if durations else 0.0
    acceptance_rate = len(state.rows) / state.current_sample if state.current_sample else 0.0

    logger.info(
        "\n====================================================\n"
        "Dataset Generation Complete\n"
        "====================================================\n\n"
        "Requested Samples\n%s\n\n"
        "Accepted\n%s\n\n"
        "Rejected\n%s\n\n"
        "Acceptance Rate\n%.2f%%\n\n"
        "Generation Time\n%s\n\n"
        "Average Runtime/Accepted Sample\n%.2f sec\n\n"
        "Fastest Accepted Sample\n%.2f sec\n\n"
        "Slowest Accepted Sample\n%.2f sec\n\n"
        "Output Folder\n%s\n\n"
        "Validation\n%s\n\n"
        "====================================================",
        state.requested_samples,
        len(state.rows),
        len(state.rejections),
        acceptance_rate * 100.0,
        _format_duration(generation_time),
        average_runtime,
        fastest,
        slowest,
        output_dir,
        validation_status,
    )


def _append_rejection(
    state: GenerationState,
    parameters: ParameterSet,
    tumor_reduction_percent: float | None,
    category: str,
    reason: str,
) -> None:
    state.rejections.append(
        Rejection(
            sample_number=state.current_sample,
            parameters=parameters,
            tumor_reduction_percent=tumor_reduction_percent,
            reason=reason,
            category=category,
        )
    )


def generate_dataset(
    num_samples: int = DEFAULT_TARGET_SAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    output_path: Path | str = OUTPUT_PATH,
) -> list[dict[str, float]]:
    """Generate exactly ``num_samples`` accepted dataset rows."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples!r}.")

    _configure_logging()
    _validate_parameter_ranges()

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = _read_existing_dataset(output_path)
    if existing_rows:
        _write_outlier_analysis(output_dir, existing_rows)
        logger.info("Existing dataset analyzed in %s", output_dir / "outlier_analysis.txt")

    logger.info(
        "Note: drug_diffusion is recorded for future model extension only "
        "and does not influence the current validated Phase 1 simulation."
    )

    candidate_batch_size = max(num_samples, 250)
    first_batch, sampling_method = _sample_parameters(
        candidate_batch_size,
        random_seed,
        batch_index=0,
    )
    pending_batches: list[tuple[int, list[ParameterSet]]] = [(0, first_batch)]

    state = GenerationState(
        requested_samples=num_samples,
        random_seed=random_seed,
        sampling_method=sampling_method,
        start_time=time.perf_counter(),
        rows=[],
        rejections=[],
    )

    seen_parameter_sets: set[tuple[float, ...]] = set()
    completed_since_checkpoint = 0
    completed_since_display = 0
    batch_index = 1

    x_coordinates = np.arange(Nx, dtype=float) * dx
    pressure = solve_pressure()
    velocity_x, velocity_y = compute_velocity(pressure)
    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

    try:
        while len(state.rows) < num_samples:
            if not pending_batches:
                batch, _ = _sample_parameters(
                    candidate_batch_size,
                    random_seed,
                    batch_index=batch_index,
                )
                pending_batches.append((batch_index, batch))
                batch_index += 1

            _, parameter_sets = pending_batches.pop(0)

            for parameters in parameter_sets:
                if len(state.rows) >= num_samples:
                    break

                state.current_sample += 1
                parameter_key = _parameter_key(parameters)
                if parameter_key in seen_parameter_sets:
                    _append_rejection(
                        state,
                        parameters,
                        None,
                        "Duplicate parameters",
                        "duplicate parameter combination.",
                    )
                    continue
                seen_parameter_sets.add(parameter_key)

                pre_rejection = _pre_simulation_rejection(parameters)
                if pre_rejection is not None:
                    category, reason = pre_rejection
                    _append_rejection(state, parameters, None, category, reason)
                    continue

                try:
                    simulation_start = time.perf_counter()
                    row = _run_single_simulation(
                        parameters,
                        pressure,
                        velocity_x,
                        velocity_y,
                        velocity_magnitude,
                        x_coordinates,
                    )
                    duration = time.perf_counter() - simulation_start
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    category, reason = _classify_failure(exc)
                    _append_rejection(state, parameters, None, category, reason)
                    continue

                post_rejection = _post_simulation_rejection(row)
                if post_rejection is not None:
                    category, reason = post_rejection
                    _append_rejection(
                        state,
                        parameters,
                        row[TARGET_COLUMN],
                        category,
                        reason,
                    )
                    continue

                state.rows.append(row)
                state.accepted_samples = len(state.rows)
                state.sample_durations.append(duration)
                completed_since_checkpoint += 1
                completed_since_display += 1

                if completed_since_display >= DISPLAY_INTERVAL:
                    _log_live_progress(state, output_path)
                    completed_since_display = 0

                if completed_since_checkpoint >= SAVE_INTERVAL:
                    logger.info("Saving checkpoint...")
                    _checkpoint(output_path, state)
                    logger.info("Checkpoint saved.")
                    completed_since_checkpoint = 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving checkpoint...")
        _checkpoint(output_path, state)
        logger.info("Checkpoint saved.")
        return state.rows

    _validate_dataset(state.rows, num_samples)
    _checkpoint(output_path, state, final=True)
    _verify_checkpoint_integrity(output_dir, output_path)

    logger.info("Dataset successfully written")
    logger.info("Metadata successfully written")
    logger.info("Reports successfully written")
    logger.info("Preview plots successfully written")
    logger.info("Checkpoint integrity verified")
    _log_final_summary(state, output_dir, "PASSED")

    return state.rows


if __name__ == "__main__":
    generate_dataset()
