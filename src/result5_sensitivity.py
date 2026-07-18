import numpy as np

from src.parameters import *
from src.transport_model import solve_transport
from src.drug_release import solve_drug_release
from src.tumor_growth import (
    solve_tumor_growth,
    tumor_statistics
)
from src.visualization import (
    plot_tornado,
    plot_spider,
    plot_tumor_reduction
)
from src.utils import get_logger, write_csv_rows


logger = get_logger(__name__)

PARAMETER_SWEEP = {
    "Particle Size": [0.8, 1.0, 1.2],
    "Diffusion Coefficient": [0.8, 1.0, 1.2],
    "Drug Release Rate": [0.8, 1.0, 1.2],
    "Drug Uptake Rate": [0.8, 1.0, 1.2],
    "Tumor Growth Rate": [0.8, 1.0, 1.2],
    "Drug Efficacy": [0.8, 1.0, 1.2]
}

DISPLAY_LABELS = {
    "Particle Size": "Particle Size",
    "Diffusion Coefficient": "Diffusion Coefficient",
    "Drug Release Rate": "Release Rate",
    "Drug Uptake Rate": "Drug Uptake",
    "Tumor Growth Rate": "Tumor Growth Rate",
    "Drug Efficacy": "Drug Efficacy"
}


def run_sensitivity_study(
    x,
    X,
    Y,
    Vx,
    Vy
):
    """Run Result 5: sensitivity analysis for model parameters.

    Parameters
    ----------
    x : ndarray
        Spatial x-coordinate vector.
    X, Y : ndarray
        Spatial mesh grids retained for API compatibility.
    Vx, Vy : ndarray
        Darcy velocity field components.

    Returns
    -------
    dict
        Sensitivity metrics for each parameter.

    Units
    -----
    Uses model-native units; sensitivity and reduction are percent.
    """

    _ = (x, X, Y)
    logger.info("\nRunning Result 5...")
    logger.info("Sensitivity Analysis")

    results = {}
    baseline = {}

    # Collect baseline values for percentage change
    baseline["Particle Size"] = 100.0
    baseline["Diffusion Coefficient"] = D
    baseline["Drug Release Rate"] = release_rate
    baseline["Drug Uptake Rate"] = uptake_rate
    baseline["Tumor Growth Rate"] = growth_rate
    baseline["Drug Efficacy"] = drug_efficacy

    parameter_scores = []
    parameter_reductions = []
    labels = []

    for param, variations in PARAMETER_SWEEP.items():
        logger.info("\nEvaluating parameter: %s", param)

        final_volumes = []
        variation_results = []
        baseline_volume = None

        for factor in variations:
            if param == "Particle Size":
                effective_diffusion = D * (100.0 / (baseline[param] * factor))
                C_np, _ = solve_transport(
                    Vx,
                    Vy,
                    diffusion_coefficient=effective_diffusion
                )
                Drug, _, _ = solve_drug_release(C_np)

            elif param == "Diffusion Coefficient":
                C_np, _ = solve_transport(
                    Vx,
                    Vy,
                    diffusion_coefficient=baseline[param] * factor
                )
                Drug, _, _ = solve_drug_release(C_np)

            elif param == "Drug Release Rate":
                C_np, _ = solve_transport(Vx, Vy)
                Drug, _, _ = solve_drug_release(
                    C_np,
                    release_constant=baseline[param] * factor
                )

            elif param == "Drug Uptake Rate":
                C_np, _ = solve_transport(
                    Vx,
                    Vy,
                    uptake_coeff=baseline[param] * factor
                )
                Drug, _, _ = solve_drug_release(C_np)

            elif param == "Tumor Growth Rate":
                C_np, _ = solve_transport(Vx, Vy)
                Drug, _, _ = solve_drug_release(C_np)
                tumor_history = solve_tumor_growth(
                    Drug,
                    growth_rate=baseline[param] * factor
                )
                stats = tumor_statistics(tumor_history)
                final_volume = stats["final_cells"]

            elif param == "Drug Efficacy":
                C_np, _ = solve_transport(Vx, Vy)
                Drug, _, _ = solve_drug_release(C_np)
                tumor_history = solve_tumor_growth(
                    Drug,
                    drug_efficacy=baseline[param] * factor
                )
                stats = tumor_statistics(tumor_history)
                final_volume = stats["final_cells"]

            else:
                tumor_history = solve_tumor_growth(Drug)
                stats = tumor_statistics(tumor_history)
                final_volume = stats["final_cells"]

            if param not in ["Tumor Growth Rate", "Drug Efficacy"]:
                tumor_history = solve_tumor_growth(Drug)
                stats = tumor_statistics(tumor_history)
                final_volume = stats["final_cells"]

            if factor == 1.0:
                baseline_volume = final_volume

            final_volumes.append(final_volume)
            variation_results.append({
                "factor": factor,
                "final_volume": final_volume
            })

        if baseline_volume is None:
            baseline_volume = final_volumes[1]

        sensitivity_index = (
            100.0
            * (
                max(final_volumes)
                - min(final_volumes)
            )
            / baseline_volume
        )

        modified_volume = min(final_volumes)
        tumor_reduction = (
            100.0
            * (baseline_volume - modified_volume)
            / baseline_volume
        )

        results[param] = {
            "variations": variation_results,
            "baseline_volume": baseline_volume,
            "modified_volume": modified_volume,
            "sensitivity_index": sensitivity_index,
            "tumor_reduction": tumor_reduction
        }

        parameter_scores.append(sensitivity_index)
        parameter_reductions.append(tumor_reduction)
        labels.append(DISPLAY_LABELS[param])

    max_score = max(parameter_scores) if parameter_scores else 1.0
    if max_score == 0:
        normalized_scores = [0 for _ in parameter_scores]
    else:
        normalized_scores = [score / max_score for score in parameter_scores]

    plot_tornado(
        labels,
        parameter_scores,
        filename="results/result5/tornado.png"
    )

    plot_spider(
        labels,
        np.array(normalized_scores)
    )

    plot_tumor_reduction(
        labels,
        parameter_reductions
    )

    fieldnames = [
        "Parameter",
        "Baseline Tumor Volume",
        "Modified Tumor Volume",
        "Tumor Reduction (%)",
        "Sensitivity Index (%)",
        "80%",
        "100%",
        "120%",
    ]
    csv_rows = [
        {
            "Parameter": param,
            "Baseline Tumor Volume": data["baseline_volume"],
            "Modified Tumor Volume": data["modified_volume"],
            "Tumor Reduction (%)": data["tumor_reduction"],
            "Sensitivity Index (%)": data["sensitivity_index"],
            "80%": data["variations"][0]["final_volume"],
            "100%": data["variations"][1]["final_volume"],
            "120%": data["variations"][2]["final_volume"],
        }
        for param, data in results.items()
    ]
    write_csv_rows("results/result5/summary.csv", fieldnames, csv_rows)

    highest_index = labels[np.argmax(parameter_scores)]
    lowest_index = labels[np.argmin(parameter_scores)]

    with open(
        "results/result5/summary.txt",
        "w"
    ) as txtfile:
        txtfile.write("Sensitivity Analysis\n")
        txtfile.write("====================\n\n")
        txtfile.write("Parameters were varied by 80%, 100%, and 120%.\n")
        txtfile.write("Tumor reduction was calculated directly from simulated tumor volumes.\n\n")

        for param, data in results.items():
            txtfile.write(
                f"{param}: baseline={data['baseline_volume']:.2f}, "
            )
            txtfile.write(
                f"modified={data['modified_volume']:.2f}, "
            )
            txtfile.write(
                f"reduction={data['tumor_reduction']:.2f}%\n"
            )

        txtfile.write("\nInterpretation:\n")
        txtfile.write(
            f"Highest influence parameter: {highest_index}.\n"
        )
        txtfile.write(
            f"Lowest influence parameter: {lowest_index}.\n\n"
        )
        txtfile.write(
            "The tornado chart ranks parameters by the relative change in final tumor volume caused by parameter variation.\n"
        )
        txtfile.write(
            "The spider plot uses fixed normalized sensitivity scaling from 0 to 1.\n"
        )
        txtfile.write(
            "The tumor reduction chart compares the best achievable reduction for each parameter.\n"
        )

    logger.info("\nResult 5 complete")

    return results
