"""Result 4: diffusion coefficient sensitivity study."""

from __future__ import annotations

import numpy as np

from src.parameters import D
from src.transport_model import (
    calculate_penetration_depth,
    get_centerline_profile,
    solve_transport,
)
from src.visualization import (
    plot_diffusion_average_concentration,
    plot_diffusion_penetration,
    plot_radial_profiles,
)
from src.utils import get_logger, write_csv_rows, write_text


logger = get_logger(__name__)


def _build_summary_text(rows: list[dict[str, float]]) -> str:
    """Build the Result 4 text summary.

    Parameters
    ----------
    rows : list of dict
        Per-diffusion-coefficient result rows.

    Returns
    -------
    str
        Summary file content.

    Units
    -----
    Diffusion coefficient is m^2/s; penetration depth is mm.
    """

    averages = [row["Average Concentration"] for row in rows]
    lines = [
        "Effect of Diffusion Coefficient",
        "==============================",
        "",
        "Simulation parameters: diffusion coefficients were swept from 0.5 to 1.5 times the base D.",
        "",
        f"Maximum value: {max(averages):.6f} kg/m^3",
        f"Minimum value: {min(averages):.6f} kg/m^3",
        f"Average value: {np.mean(averages):.6f} kg/m^3",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                (
                    "Diffusion Coefficient = "
                    f"{row['Diffusion Coefficient']:.3e} m^2/s"
                ),
                f"  Average Concentration : {row['Average Concentration']:.6f} kg/m^3",
                f"  Maximum Concentration : {row['Maximum Concentration']:.6f} kg/m^3",
                f"  Penetration Depth : {row['Penetration Depth (mm)']:.2f} mm",
                "",
            ]
        )

    lines.extend(
        [
            "Scientific observations:",
            "Higher diffusion coefficients lead to deeper penetration and higher average nanoparticle concentrations.",
            "Short interpretation:",
            "Diffusion strengthens spatial spreading without changing the model equations.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diffusion_study(
    x: np.ndarray,
    Vx: np.ndarray,
    Vy: np.ndarray,
) -> dict[str, list[float]]:
    """Run Result 4: effect of diffusion coefficient.

    Parameters
    ----------
    x : ndarray
        One-dimensional x-coordinate vector.
    Vx, Vy : ndarray
        Darcy velocity field components.

    Returns
    -------
    dict
        Diffusion coefficients, penetration depths, and concentration metrics.

    Units
    -----
    Coordinates are meters. Diffusion is m^2/s. Penetration is mm.
    """

    logger.info("\nRunning Result 4...")
    logger.info("Effect of Diffusion Coefficient")

    diffusion_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    diffusion_coefficients = [factor * D for factor in diffusion_factors]
    labels = [f"{factor:.2f} D" for factor in diffusion_factors]

    penetration_depths: list[float] = []
    average_concentrations: list[float] = []
    maximum_concentrations: list[float] = []
    profiles: dict[str, np.ndarray] = {}
    rows: list[dict[str, float]] = []

    for label, D_value in zip(labels, diffusion_coefficients):
        logger.info(
            "\nSimulating diffusion coefficient %s (%.3e m^2/s)",
            label,
            D_value,
        )

        C_np, _ = solve_transport(Vx, Vy, diffusion_coefficient=D_value)

        penetration = calculate_penetration_depth(C_np, x)
        average_conc = float(np.mean(C_np))
        maximum_conc = float(np.max(C_np))

        penetration_depths.append(penetration)
        average_concentrations.append(average_conc)
        maximum_concentrations.append(maximum_conc)
        profiles[label] = get_centerline_profile(C_np)

        rows.append(
            {
                "Diffusion Coefficient": D_value,
                "Average Concentration": average_conc,
                "Maximum Concentration": maximum_conc,
                "Penetration Depth (mm)": penetration,
            }
        )

    plot_diffusion_penetration(diffusion_coefficients, penetration_depths)
    plot_diffusion_average_concentration(
        diffusion_coefficients,
        average_concentrations,
    )
    plot_radial_profiles(x, profiles)

    fieldnames = [
        "Diffusion Coefficient",
        "Average Concentration",
        "Maximum Concentration",
        "Penetration Depth (mm)",
    ]
    write_csv_rows("results/result4/summary.csv", fieldnames, rows)
    write_text("results/result4/summary.txt", _build_summary_text(rows))

    logger.info("\nResult 4 complete")

    return {
        "diffusion_coefficients": diffusion_coefficients,
        "penetration_depths": penetration_depths,
        "average_concentrations": average_concentrations,
        "maximum_concentrations": maximum_concentrations,
    }
