"""Generate Result 7 figures from the simulation dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "simulation_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "result7"
PARAMETER_DISTRIBUTION_PATH = RESULTS_DIR / "7.2 Parameter Distribution.png"

PARAMETERS = [
    ("particle_size_nm", "Particle Size (nm)"),
    ("drug_diffusion", "Drug Diffusion Coefficient (m²/s)"),
    ("np_diffusion", "Nanoparticle Diffusion Coefficient (m²/s)"),
    ("release_rate", "Drug Release Rate (1/h)"),
    ("uptake_rate", "Cellular Uptake Coefficient (1/h)"),
    ("drug_loading", "Drug Loading (fraction)"),
    ("tumor_growth_rate", "Tumor Growth Rate (1/h)"),
    ("drug_efficacy", "Drug Efficacy (a.u.)"),
]


def generate_parameter_distribution_figure() -> Path:
    """Create the multi-panel input-parameter distribution figure for Result 7.2."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Simulation dataset not found: {DATASET_PATH}")

    frame = pd.read_csv(DATASET_PATH)
    missing_columns = [column for column, _ in PARAMETERS if column not in frame.columns]
    if missing_columns:
        raise KeyError("Dataset is missing parameter columns: " + ", ".join(missing_columns))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 2, figsize=(14, 16))

    for axis, (column, label) in zip(axes.flat, PARAMETERS):
        values = pd.to_numeric(frame[column], errors="raise")
        axis.hist(values, bins=30, color="#2f6f9f", edgecolor="white", linewidth=0.7)
        axis.set_title(label, fontsize=12, fontweight="bold")
        axis.set_xlabel(label, fontsize=10)
        axis.set_ylabel("Number of simulations", fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    figure.suptitle("Result 7.2: Input Parameter Distributions", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(PARAMETER_DISTRIBUTION_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return PARAMETER_DISTRIBUTION_PATH


if __name__ == "__main__":
    output_path = generate_parameter_distribution_figure()
    print(f"Result 7.2 saved to: {output_path}")
