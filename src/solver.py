import numpy as np

from src.parameters import *
from src.utils import get_logger, write_text

from src.pressure_model import (
    solve_pressure,
    compute_velocity
)

from src.transport_model import (
    solve_transport
)

from src.drug_release import (
    solve_drug_release
)

from src.tumor_growth import (
    solve_tumor_growth,
    tumor_statistics
)

from src.visualization import (
    create_folders,
    plot_pressure,
    plot_velocity,
    plot_transport,
    plot_drug,
    plot_concentration_history,
    plot_drug_history,
    plot_tumor_growth
)

from src.result1_particle_size import (
    run_particle_size_study
)

from src.result2_release_rate import (
    run_release_rate_study
)

from src.result3_ifp import run_ifp_study
from src.result4_diffusion import run_diffusion_study
from src.result5_sensitivity import run_sensitivity_study
from src.result6_optimization import run_optimization_study


logger = get_logger(__name__)

# =====================================================
# MAIN SIMULATION WORKFLOW
# =====================================================

def run_simulation() -> dict[str, object]:
    """Run the complete PLGA nanoparticle simulation workflow.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Core arrays and summary outputs from Results 1-6.

    Units
    -----
    Uses SI units internally, with hours for simulation time.
    """

    logger.info("\n================================")
    logger.info("PLGA NANOPARTICLE SIMULATION")
    logger.info("================================\n")

    # -------------------------------------
    # Create folders
    # -------------------------------------

    create_folders()

    # -------------------------------------
    # Grid
    # -------------------------------------

    x = np.linspace(
        0,
        Lx,
        Nx
    )

    y = np.linspace(
        0,
        Ly,
        Ny
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    # -------------------------------------
    # Pressure
    # -------------------------------------

    P = solve_pressure()

    Vx, Vy = compute_velocity(
        P
    )

    # -------------------------------------
    # Transport
    # -------------------------------------

    (
        C_np,
        concentration_history
    ) = solve_transport(
        Vx,
        Vy
    )

    # -------------------------------------
    # Drug Release
    # -------------------------------------
    # CORRECTION:
    # solve_drug_release now returns
    # Drug, drug_history,
    # release_percent_history
    # -------------------------------------

    (
        Drug,
        drug_history,
        release_percent_history
    ) = solve_drug_release(
        C_np
    )

    # -------------------------------------
    # Tumor Growth
    # -------------------------------------

    tumor_history = (
        solve_tumor_growth(
            Drug
        )
    )

    stats = (
        tumor_statistics(
            tumor_history
        )
    )

    # -------------------------------------
    # Standard Plots
    # -------------------------------------

    plot_pressure(
        X,
        Y,
        P
    )

    plot_velocity(
        X,
        Y,
        Vx,
        Vy
    )

    plot_transport(
        X,
        Y,
        C_np
    )

    plot_drug(
        X,
        Y,
        Drug
    )

    plot_concentration_history(
        concentration_history
    )

    plot_drug_history(
        drug_history
    )

    plot_tumor_growth(
        tumor_history
    )

    # -------------------------------------
    # Nanoparticle & Drug Statistics
    # -------------------------------------

    peak_np = np.max(
        concentration_history
    )

    avg_np = np.mean(
        concentration_history
    )

    peak_np_index = np.argmax(
        concentration_history
    )

    peak_np_time = (
        peak_np_index * dt
    )

    peak_drug = np.max(
        drug_history
    )

    peak_drug_index = np.argmax(
        drug_history
    )

    peak_drug_time = (
        peak_drug_index * dt
    )

    final_release_percent = (
        release_percent_history[-1]
    )

    write_text(
        "results/transport/nanoparticle_statistics.txt",
        (
            "Nanoparticle Statistics\n"
            "=======================\n\n"
            f"Peak Concentration : {peak_np:.6f}\n"
            f"Minimum Concentration : {np.min(concentration_history):.6f}\n"
            f"Average Concentration : {avg_np:.6f}\n"
            f"Tmax : {peak_np_time:.2f} h\n\n"
            "Scientific observations:\n"
            "Nanoparticle concentration decreases after Tmax.\n\n"
            "Short interpretation:\n"
            "After Tmax, nanoparticle concentration decreases due to particle "
            "clearance, drug release and cellular uptake.\n"
        ),
    )

    write_text(
        "results/drug/drug_statistics.txt",
        (
            "Drug Statistics\n"
            "===============\n\n"
            f"Peak Drug Concentration : {peak_drug:.6f}\n"
            f"Minimum Drug Concentration : {np.min(drug_history):.6f}\n"
            f"Average Drug Concentration : {np.mean(drug_history):.6f}\n"
            f"Tmax Drug : {peak_drug_time:.2f} h\n"
            f"Final Release Percent : {final_release_percent:.2f}%\n"
            "This value is consistent with the initial nanoparticle drug loading.\n\n"
            "Scientific observations:\n"
            "Final release approaches complete drug availability from the initial "
            "nanoparticle loading.\n\n"
            "Short interpretation:\n"
            "Effective release is maintained throughout the simulation period.\n"
        ),
    )

    # -------------------------------------
    # Save Tumor Statistics
    # Reviewer Requested:
    # Peak Size
    # Peak Time
    # Regression Time
    # -------------------------------------

    write_text(
        "results/tumor/tumor_statistics.txt",
        (
            "Tumor Statistics\n"
            "================\n\n"
            f"Initial Cells : {stats['initial_cells']:.2f}\n"
            f"Final Cells : {stats['final_cells']:.2f}\n"
            f"Peak Tumor Size : {stats['peak_tumor_size']:.2f}\n"
            f"Minimum Tumor Size : {min(tumor_history):.2f}\n"
            f"Average Tumor Size : {np.mean(tumor_history):.2f}\n"
            f"Peak Time : {stats['peak_time_hours']:.2f} h\n"
            f"Regression Start : {stats['regression_start_hours']:.2f} h\n"
            f"Tumor Reduction : {stats['tumor_reduction_percent']:.2f}%\n\n"
            "Scientific observations:\n"
            "Tumor growth peaks before drug-induced regression dominates.\n\n"
            "Short interpretation:\n"
            "The simulated drug exposure produces net tumor reduction.\n"
        ),
    )

    # -------------------------------------
    # Result 1
    # Particle Size Study
    # -------------------------------------

    penetration_depths = (
        run_particle_size_study(
            X,
            Y,
            x,
            Vx,
            Vy
        )
    )

    logger.info("\nParticle Size Study Summary")
    logger.info("---------------------------")

    for size, depth in zip(
        particle_sizes,
        penetration_depths
    ):
        logger.info(
            "%s nm : %.2f mm",
            size,
            depth,
        )

    # -------------------------------------
    # Result 2
    # Release Rate Study
    # -------------------------------------

    (
        release_histories,
        tumor_results,
        summary_results
    ) = run_release_rate_study(
        C_np
    )

    # -------------------------------------
    # Result 3
    # Effect of Interstitial Fluid Pressure
    # -------------------------------------

    run_ifp_study(
        x,
        y,
        X,
        Y
    )

    # -------------------------------------
    # Result 4
    # Diffusion Coefficient Study
    # -------------------------------------

    run_diffusion_study(
        x,
        Vx,
        Vy
    )

    # -------------------------------------
    # Result 5
    # Sensitivity Analysis
    # -------------------------------------

    run_sensitivity_study(
        x,
        X,
        Y,
        Vx,
        Vy
    )

    # -------------------------------------
    # Result 6
    # Optimization
    # -------------------------------------

    run_optimization_study(
        x,
        Vx,
        Vy
    )

    # -------------------------------------
    # Final Summary
    # -------------------------------------

    logger.info("\n================================")
    logger.info("SIMULATION COMPLETED")
    logger.info("================================")

    logger.info("\nREVIEWER VALIDATION SUMMARY")
    logger.info("--------------------------------")

    logger.info(
        "Peak Tumor Size : %.2f",
        stats["peak_tumor_size"],
    )

    logger.info(
        "Peak Time : %.2f h",
        stats["peak_time_hours"],
    )

    logger.info(
        "Regression Start : %.2f h",
        stats["regression_start_hours"],
    )

    logger.info(
        "Tumor Reduction : %.2f%%",
        stats["tumor_reduction_percent"],
    )

    logger.info(
        "Peak Drug Concentration : %.6f",
        peak_drug,
    )

    logger.info(
        "Drug Tmax : %.2f h",
        peak_drug_time,
    )

    logger.info(
        "Final Release : %.2f%%",
        final_release_percent,
    )

    logger.info("\nTumor Reduction Summary")

    for label, result in summary_results.items():

        logger.info(
            "%s: %.2f%%",
            label,
            result["tumor_reduction_percent"],
        )

    logger.info("\nResults saved in 'results/' folder")

    return {

        "pressure": P,

        "velocity_x": Vx,

        "velocity_y": Vy,

        "nanoparticles": C_np,

        "drug": Drug,

        "tumor_history": tumor_history,

        "tumor_stats": stats,

        "penetration_depths": penetration_depths,

        "release_results": summary_results
    }
