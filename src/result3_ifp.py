"""Result 3: interstitial fluid pressure sensitivity study."""

from __future__ import annotations

import numpy as np

from src.parameters import D
from src.pressure_model import compute_velocity, solve_pressure
from src.transport_model import calculate_penetration_depth, solve_transport
from src.drug_release import solve_drug_release
from src.tumor_growth import solve_tumor_growth, tumor_statistics
from src.visualization import (
    plot_ifp_contour,
    plot_ifp_drug_concentration,
    plot_ifp_penetration,
)
from src.utils import get_logger, validate_positive, write_csv_rows, write_text


MMHG_TO_PA = 133.322
logger = get_logger(__name__)


def convert_mmHg_to_pa(mmHg: float) -> float:
    """Convert interstitial fluid pressure from mmHg to Pa.

    Parameters
    ----------
    mmHg : float
        Pressure in millimeters of mercury.

    Returns
    -------
    float
        Pressure in pascals.

    Units
    -----
    mmHg to Pa.
    """

    validate_positive(mmHg, "mmHg")
    return mmHg * MMHG_TO_PA


def ifp_velocity_scale(mmHg: float) -> float:
    """Calculate the effective transport velocity scale for IFP.

    Parameters
    ----------
    mmHg : float
        Interstitial fluid pressure.

    Returns
    -------
    float
        Dimensionless velocity scale clipped between 0.5 and 1.0.

    Units
    -----
    Pressure is in mmHg.
    """

    validate_positive(mmHg, "mmHg")
    return max(0.5, min(1.0, 1.1 - 0.02 * mmHg))


def _build_summary_text(
    rows: list[dict[str, float]],
    average_drug_concentrations: list[float],
) -> str:
    """Build the Result 3 text summary.

    Parameters
    ----------
    rows : list of dict
        Per-IFP result rows.
    average_drug_concentrations : list of float
        Average drug concentrations.

    Returns
    -------
    str
        Summary file content.

    Units
    -----
    Drug concentration is kg/m^3; penetration is mm.
    """

    lines = [
        "Effect of Interstitial Fluid Pressure (IFP)",
        "========================================",
        "",
        "Simulation parameters: IFP values were swept from 5 to 30 mmHg.",
        "The pressure equation was kept unchanged.",
        "Higher IFP was represented as lower effective transport.",
        "",
        f"Maximum value: {max(average_drug_concentrations):.6f} kg/m^3",
        f"Minimum value: {min(average_drug_concentrations):.6f} kg/m^3",
        f"Average value: {np.mean(average_drug_concentrations):.6f} kg/m^3",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"IFP = {row['IFP (mmHg)']:.0f} mmHg",
                (
                    "  Average Drug Concentration : "
                    f"{row['Average Drug Concentration']:.6f} kg/m^3"
                ),
                (
                    "  Maximum Drug Concentration : "
                    f"{row['Maximum Drug Concentration']:.6f} kg/m^3"
                ),
                f"  Penetration Depth : {row['Penetration Depth (mm)']:.2f} mm",
                f"  Final Tumor Volume : {row['Final Tumor Volume']:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "Scientific observations:",
            "As IFP increases, effective transport decreases.",
            "Short interpretation:",
            "Reduced transport lowers drug delivery and produces shallower penetration.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ifp_study(
    x: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> dict[str, list[float]]:
    """Run Result 3: effect of interstitial fluid pressure.

    Parameters
    ----------
    x, y : ndarray
        One-dimensional spatial coordinates.
    X, Y : ndarray
        Spatial mesh grids.

    Returns
    -------
    dict
        IFP values, drug concentrations, penetration depths, and final tumor
        volumes.

    Units
    -----
    Coordinates are meters. IFP is mmHg. Penetration is mm.
    """

    _ = y
    logger.info("\nRunning Result 3...")
    logger.info("Effect of Interstitial Fluid Pressure")

    ifp_values = [5, 10, 15, 20, 25, 30]
    average_drug_concentrations: list[float] = []
    maximum_drug_concentrations: list[float] = []
    penetration_depths: list[float] = []
    final_tumor_volumes: list[float] = []
    rows: list[dict[str, float]] = []

    base_pressure = solve_pressure()
    base_vx, base_vy = compute_velocity(base_pressure)

    for mmHg in ifp_values:
        pa = convert_mmHg_to_pa(mmHg)
        logger.info("\nSimulating IFP = %s mmHg (%.1f Pa)", mmHg, pa)

        velocity_scale = ifp_velocity_scale(mmHg)
        effective_diffusion = D * velocity_scale

        C_np, _ = solve_transport(
            base_vx,
            base_vy,
            diffusion_coefficient=effective_diffusion,
            velocity_scale=velocity_scale,
        )
        Drug, _, _ = solve_drug_release(C_np)
        tumor_history = solve_tumor_growth(Drug)
        stats = tumor_statistics(tumor_history)

        average_drug = float(np.mean(Drug))
        maximum_drug = float(np.max(Drug))
        penetration = calculate_penetration_depth(C_np, x)

        average_drug_concentrations.append(average_drug)
        maximum_drug_concentrations.append(maximum_drug)
        penetration_depths.append(penetration)
        final_tumor_volumes.append(stats["final_cells"])

        plot_ifp_contour(X, Y, C_np, mmHg)

        rows.append(
            {
                "IFP (mmHg)": mmHg,
                "Velocity Scale": velocity_scale,
                "Effective Diffusion Coefficient": effective_diffusion,
                "Average Drug Concentration": average_drug,
                "Maximum Drug Concentration": maximum_drug,
                "Penetration Depth (mm)": penetration,
                "Final Tumor Volume": stats["final_cells"],
            }
        )

    plot_ifp_drug_concentration(ifp_values, average_drug_concentrations)
    plot_ifp_penetration(ifp_values, penetration_depths)

    fieldnames = [
        "IFP (mmHg)",
        "Velocity Scale",
        "Effective Diffusion Coefficient",
        "Average Drug Concentration",
        "Maximum Drug Concentration",
        "Penetration Depth (mm)",
        "Final Tumor Volume",
    ]
    write_csv_rows("results/result3/summary.csv", fieldnames, rows)
    write_text(
        "results/result3/summary.txt",
        _build_summary_text(rows, average_drug_concentrations),
    )

    logger.info("\nResult 3 complete")

    return {
        "ifp_values": ifp_values,
        "average_drug": average_drug_concentrations,
        "maximum_drug": maximum_drug_concentrations,
        "penetration_depths": penetration_depths,
        "final_tumor_volumes": final_tumor_volumes,
    }
