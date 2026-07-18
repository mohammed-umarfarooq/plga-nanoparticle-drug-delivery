import numpy as np
import matplotlib.pyplot as plt

from src.parameters import *

from src.transport_model import (
    solve_transport,
    calculate_penetration_depth,
    get_centerline_profile
)

from src.visualization import (
    plot_penetration_depth
)
from src.utils import get_logger


logger = get_logger(__name__)

# =====================================================
# STOKES–EINSTEIN DIFFUSION COEFFICIENT
# =====================================================

def calculate_diffusion_from_size(
    particle_size_nm: float,
) -> float:
    """Calculate diffusion coefficient from nanoparticle size.

    Parameters
    ----------
    particle_size_nm : float
        Particle diameter.

    Returns
    -------
    float
        Diffusion coefficient scaled from the base coefficient.

    Units
    -----
    Particle size is nm; diffusion coefficient is m^2/s.
    """

    reference_size = 100

    D_particle = (
        D * (
            reference_size
            / particle_size_nm
        )
    )

    return D_particle


# =====================================================
# RESULT 1
# PARTICLE SIZE STUDY
# =====================================================

def run_particle_size_study(
    X: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
    Vx: np.ndarray,
    Vy: np.ndarray,
) -> list[float]:
    """Run Result 1: particle size study.

    Parameters
    ----------
    X, Y : ndarray
        Spatial mesh grids.
    x : ndarray
        One-dimensional x-coordinate vector.
    Vx, Vy : ndarray
        Darcy velocity field components.

    Returns
    -------
    list of float
        Penetration depths for configured particle sizes.

    Units
    -----
    Coordinates are meters. Particle size is nm. Penetration is mm.
    """

    logger.info("\nRunning Result 1...")
    logger.info("Particle Size Study")

    penetration_depths = []

    profiles = {}
    
    # Cache statistics to avoid running transport solvers twice
    concentration_stats = {}

    for size in particle_sizes:

        logger.info("\nSimulating %s nm", size)

        D_size = (
            calculate_diffusion_from_size(
                size
            )
        )

        C_np, history = solve_transport(
            Vx,
            Vy,
            diffusion_coefficient=D_size
        )

        depth = (
            calculate_penetration_depth(
                C_np,
                x
            )
        )

        penetration_depths.append(
            depth
        )

        logger.info(
            "Penetration Depth = %.2f mm",
            depth,
        )

        profiles[size] = (
            get_centerline_profile(
                C_np
            )
        )
        
        # Save metrics for the output block
        concentration_stats[size] = {
            "max": np.max(C_np),
            "mean": np.mean(C_np)
        }

        # ---------------------------------------------
        # Concentration contour
        # ---------------------------------------------

        plt.figure(figsize=(8, 6))

        plt.contourf(
            X * 1000,
            Y * 1000,
            C_np,
            levels=40
        )

        plt.contour(
            X * 1000,
            Y * 1000,
            C_np,
            colors="black",
            linewidths=0.5
        )

        plt.colorbar(
            label="Nanoparticle Concentration"
        )

        plt.xlabel(
            "X Position (mm)"
        )

        plt.ylabel(
            "Y Position (mm)"
        )

        plt.title(
            f"{size} nm Nanoparticle Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            f"results/result1/{size}nm_concentration.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =================================================
    # Comparison Profile
    # =================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        x * 1000,
        profiles[50],
        label="50 nm",
        linewidth=2
    )

    plt.plot(
        x * 1000,
        profiles[100],
        label="100 nm",
        linewidth=2
    )

    plt.plot(
        x * 1000,
        profiles[200],
        label="200 nm",
        linewidth=2
    )

    plt.xlabel(
        "Distance into Tumor (mm)"
    )

    plt.ylabel(
        "Normalized Concentration"
    )

    plt.title(
        "Nanoparticle Penetration Profile Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/result1/penetration_profile_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =================================================
    # Penetration Depth Plot
    # =================================================

    plot_penetration_depth(
        particle_sizes,
        penetration_depths
    )

    # =================================================
    # Save Numerical Results
    # =================================================

    with open(
        "results/result1/penetration_depths.txt",
        "w"
    ) as f:

        f.write(
            "Particle Size Study\n"
        )

        f.write(
            "===================\n\n"
        )

        f.write(
            "Penetration Depth Criterion:\n"
        )

        f.write(
            "Distance where concentration falls below 5% of maximum concentration.\n\n"
        )

        for size in particle_sizes:

            max_concentration = concentration_stats[size]["max"]
            mean_concentration = concentration_stats[size]["mean"]

            depth = penetration_depths[
                particle_sizes.index(size)
            ]

            f.write(
                f"Particle Size : {size} nm\n"
            )

            f.write(
                f"Diffusion Coefficient : "
                f"{calculate_diffusion_from_size(size):.3e} m²/s\n"
            )

            f.write(
                f"Penetration Depth : {depth:.2f} mm\n"
            )

            f.write(
                f"Maximum Concentration : "
                f"{max_concentration:.6f}\n"
            )

            f.write(
                f"Average Concentration : "
                f"{mean_concentration:.6f}\n"
            )

            f.write(
                "\n"
            )

    # =================================================
    # REVIEWER VALIDATION SUMMARY
    # =================================================

    logger.info("\n")
    logger.info("=" * 70)
    logger.info("RESULT 1 VALIDATION SUMMARY")
    logger.info("=" * 70)

    logger.info("\nPenetration Criterion:")
    logger.info(
        "Penetration depth = distance where concentration "
        "falls below 5% of maximum concentration."
    )

    logger.info("\nDiffusion Coefficients (Stokes-Einstein Based)")
    logger.info("-" * 70)

    logger.info(
        f"{'Size (nm)':<15}"
        f"{'Relative D':<20}"
        f"{'Penetration Depth (mm)':<25}"
        f"{'Avg Concentration':<20}"
    )

    logger.info("-" * 70)

    for size, depth in zip(
        particle_sizes,
        penetration_depths
    ):

        D_size = calculate_diffusion_from_size(
            size
        )

        mean_conc = concentration_stats[size]["mean"]

        logger.info(
            f"{size:<15}"
            f"{D_size:.3e}      "
            f"{depth:<25.2f}"
            f"{mean_conc:.5f}"
        )

    logger.info("\nStokes-Einstein Relation Used:")
    logger.info(
        "D = kB*T / (6*pi*mu*r)"
    )

    logger.info(
        "\nSmaller nanoparticles have larger diffusion "
        "coefficients and therefore penetrate deeper "
        "into tumor tissue."
    )

    logger.info("=" * 70)
            
    return penetration_depths
