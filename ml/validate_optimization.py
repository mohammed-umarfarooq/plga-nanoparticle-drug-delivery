"""Assess whether surrogate-optimized PLGA formulations can be validated.

The workflow is feasibility-first.  It compares the features required by the
persisted XGBoost model with the independent parameter definitions used by the
validated Phase 1 simulation.  A computational comparison is performed only
when the surrogate prediction for each formulation can be reconstructed from
independent simulation inputs alone.

No simulation is imported or executed when that condition is not met.
"""

from __future__ import annotations

import ast
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OPTIMIZATION_DIR = PROJECT_ROOT / "results" / "optimization"
VALIDATION_DIR = PROJECT_ROOT / "results" / "validation"
MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
OPTIMIZED_FORMULATIONS_PATH = PROJECT_ROOT / "results" / "result10" / "10.4 Optimized Parameters.csv"
GENERATOR_PATH = PROJECT_ROOT / "ml" / "generate_dataset.py"
LIMITATIONS_PATH = VALIDATION_DIR / "validation_limitations.txt"
REPORT_PATH = VALIDATION_DIR / "validation_report.txt"

MODEL_TO_SIMULATION_NAME = {"nanoparticle_diffusion": "np_diffusion"}

logger = logging.getLogger(__name__)


class ValidationFeasibilityError(ValueError):
    """Raised when required validation input artifacts are invalid."""


@dataclass(frozen=True)
class ValidationFeasibility:
    """Immutable outcome of comparing model and simulation input requirements."""

    independent_inputs: list[str]
    model_features: list[str]
    derived_model_inputs: list[str]
    optimized_formulation_count: int

    @property
    def is_feasible(self) -> bool:
        """Return whether model inputs are fully reconstructable from design inputs."""
        return not self.derived_model_inputs


def configure_logging() -> None:
    """Configure concise console logging for the validation workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def load_simulation_inputs(path: Path = GENERATOR_PATH) -> list[str]:
    """Read validated independent inputs without importing any simulation module."""
    if not path.is_file():
        raise FileNotFoundError(f"Simulation parameter-definition file not found: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PARAMETER_RANGES"
            for target in node.targets
        ):
            ranges = ast.literal_eval(node.value)
            if isinstance(ranges, dict) and ranges:
                return [str(name) for name in ranges]
    raise ValidationFeasibilityError(
        "Could not read PARAMETER_RANGES from the validated dataset generator."
    )


def load_model_features(path: Path = MODEL_PATH) -> list[str]:
    """Load feature names from the persisted model without fitting or predicting."""
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    with path.open("rb") as handle:
        model: Any = pickle.load(handle)
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValidationFeasibilityError(
            "The trained model does not retain feature names for feasibility checking."
        )
    return feature_names


def load_optimized_formulations(
    path: Path = OPTIMIZED_FORMULATIONS_PATH,
) -> pd.DataFrame:
    """Load and validate the existing optimization recommendations."""
    if not path.is_file():
        raise FileNotFoundError(f"Optimized formulations file not found: {path}")
    formulations = pd.read_csv(path)
    if formulations.empty:
        raise ValidationFeasibilityError("Optimized formulations file is empty.")
    
    # Map column names if loading from 10.4 Optimized Parameters.csv
    if "predicted_tumor_reduction_percent" in formulations.columns:
        formulations = formulations.rename(columns={
            "predicted_tumor_reduction_percent": "Predicted Tumor Reduction",
            "nanoparticle_diffusion": "np_diffusion",
            "rank": "Rank"
        })

    if "Predicted Tumor Reduction" not in formulations.columns:
        raise ValidationFeasibilityError(
            "Optimized formulations lack the 'Predicted Tumor Reduction' column."
        )
    return formulations


def assess_validation_feasibility(
    model_features: list[str],
    independent_inputs: list[str],
    formulations: pd.DataFrame,
) -> ValidationFeasibility:
    """Compare persisted-model inputs with true simulation design variables.

    Features not present in the independent simulation-input configuration are
    simulation-derived model inputs.  Their formulation-specific values cannot
    be reconstructed without running the simulation, so a comparison against
    predictions made with fixed derived references is not scientifically valid.
    """
    normalized_independent = {
        simulation_name for simulation_name in independent_inputs
    }
    derived = [
        feature
        for feature in model_features
        if MODEL_TO_SIMULATION_NAME.get(feature, feature) not in normalized_independent
    ]
    missing_columns = [name for name in independent_inputs if name not in formulations.columns]
    if missing_columns:
        raise ValidationFeasibilityError(
            "Optimized formulations cannot be reconstructed from independent inputs; "
            f"missing columns: {missing_columns}"
        )
    return ValidationFeasibility(
        independent_inputs=independent_inputs,
        model_features=model_features,
        derived_model_inputs=derived,
        optimized_formulation_count=len(formulations),
    )


def write_limitations(feasibility: ValidationFeasibility) -> None:
    """Write a transparent explanation when full validation is infeasible or feasible."""
    if feasibility.is_feasible:
        lines = [
            "Validation Limitations for Optimized Formulations",
            "===============================================",
            "",
            "Validation status: Full computational validation is feasible.",
            "",
            "Reason",
            "------",
            "The surrogate model is leakage-free and uses only independent parameters. Therefore, the formulations can be directly simulated and validated.",
            "",
            "Interpretation",
            "--------------",
            "The optimized formulations are ready for direct validation against Phase 1 computational simulations.",
            "",
            "Recommendation",
            "--------------",
            "Proceed with Phase 1 validation of the optimized formulations.",
        ]
    else:
        lines = [
            "Validation Limitations for Optimized Formulations",
            "===============================================",
            "",
            "Validation status: Full computational validation is not scientifically feasible for the current surrogate-optimization outputs.",
            "",
            "Reason",
            "------",
            "The optimized formulations contain the independent simulation parameters, but the trained XGBoost model also requires formulation-specific simulation outputs as input features.",
            "Derived model inputs: " + ", ".join(feasibility.derived_model_inputs) + ".",
            "These values are not independent design variables. They are produced only after a Phase 1 simulation has been run for a formulation.",
            "",
            "Why a direct comparison would be invalid",
            "----------------------------------------",
            "The optimization workflow supplied fixed reference values for the derived inputs so that the existing model could be queried without simulation. Running Phase 1 now would produce different, formulation-specific derived values. Comparing those simulation outputs with predictions generated under fixed reference values would compare different surrogate input conditions and would not validate the optimized prediction.",
            "",
            "Interpretation",
            "--------------",
            "The reported formulations must therefore be interpreted as surrogate-model recommendations, not validated computational solutions. Their direct biological and computational interpretability is limited by the derived-output feature dependency.",
            "",
            "Recommendation",
            "--------------",
            "Retraining a surrogate model using only independent simulation inputs would allow each optimized formulation to be reconstructed, simulated, and compared directly with its surrogate prediction. No retraining, dataset generation, or computational simulation was performed by this validation module.",
        ]
    LIMITATIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(feasibility: ValidationFeasibility) -> None:
    """Write the final validation report."""
    if feasibility.is_feasible:
        lines = [
            "Optimized Formulation Validation Report",
            "======================================",
            "",
            "Validation feasibility",
            "----------------------",
            f"Optimized formulations assessed: {feasibility.optimized_formulation_count}",
            "Full computational validation: feasible.",
            "Feasibility conclusion: feasible.",
            "",
            "Methodology",
            "-----------",
            "The feature names retained by models/xgboost_model.pkl were compared with the independent parameter names in PARAMETER_RANGES from ml/generate_dataset.py. The optimized CSV was checked to confirm it contains every independent simulation parameter.",
            "",
            "Validation results and metrics",
            "------------------------------",
            "Validation is feasible because the model uses only independent parameters.",
            "",
            "Model limitations",
            "-----------------",
            "Independent simulation inputs: " + ", ".join(feasibility.independent_inputs) + ".",
            "Derived model inputs: None.",
            "",
            "Overall conclusion",
            "------------------",
            "Validation feasibility check passed successfully. The model is leakage-free.",
        ]
    else:
        lines = [
            "Optimized Formulation Validation Report",
            "======================================",
            "",
            "Validation feasibility",
            "----------------------",
            f"Optimized formulations assessed: {feasibility.optimized_formulation_count}",
            "Full computational validation: not performed.",
            "Feasibility conclusion: not feasible without an invalid comparison.",
            "",
            "Methodology",
            "-----------",
            "The feature names retained by models/xgboost_model.pkl were compared with the independent parameter names in PARAMETER_RANGES from ml/generate_dataset.py. The optimized CSV was checked to confirm it contains every independent simulation parameter.",
            "",
            "Validation results and metrics",
            "------------------------------",
            "No Phase 1 simulations, validation_results.csv, error metrics, or validation figures were generated because full validation is infeasible for the current trained-model feature set.",
            "",
            "Model limitations",
            "-----------------",
            "Independent simulation inputs: " + ", ".join(feasibility.independent_inputs) + ".",
            "Derived model inputs: " + ", ".join(feasibility.derived_model_inputs) + ".",
            "The trained model depends on derived outputs that are unavailable from a formulation alone. The optimization used fixed references for those inputs; a simulation would instead produce formulation-specific values.",
            "",
            "Recommendations for future work",
            "-------------------------------",
            "Train and evaluate a replacement surrogate with only independent simulation inputs, then repeat bounded optimization and compare its predictions with unchanged Phase 1 simulations.",
            "",
            "Overall conclusion",
            "------------------",
            "Validation is transparently limited rather than performed under incompatible model-input conditions. No simulation code was modified or executed, no model was retrained, and no dataset was modified.",
        ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    """Confirm all applicable infeasibility-validation artifacts were created."""
    required = [LIMITATIONS_PATH, REPORT_PATH]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Required validation output(s) were not created: " + ", ".join(missing))


def main() -> None:
    """Assess feasibility and document limitations without invalid simulation use."""
    configure_logging()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    independent_inputs = load_simulation_inputs()
    model_features = load_model_features()
    formulations = load_optimized_formulations()
    feasibility = assess_validation_feasibility(
        model_features, independent_inputs, formulations
    )
    write_limitations(feasibility)
    write_validation_report(feasibility)
    validate_outputs()
    if feasibility.is_feasible:
        logger.info(
            "Full validation is feasible. Reports saved to %s.",
            VALIDATION_DIR,
        )
    else:
        logger.info(
            "Full validation is infeasible because %s derived model inputs are required. "
            "Transparent reports saved to %s.",
            len(feasibility.derived_model_inputs),
            VALIDATION_DIR,
        )


if __name__ == "__main__":
    main()
