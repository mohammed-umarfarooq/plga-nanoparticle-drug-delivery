import numpy as np

from src.parameters import *

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

# =====================================================
# MAIN SIMULATION WORKFLOW
# =====================================================

def run_simulation():

    print("\n================================")
    print("PLGA NANOPARTICLE SIMULATION")
    print("================================\n")

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

    C_np, concentration_history = (
        solve_transport(
            Vx,
            Vy
        )
    )

    # -------------------------------------
    # Drug Release
    # -------------------------------------

    Drug, drug_history = (
        solve_drug_release(
            C_np
        )
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
    # Save Standard Results
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
    # Save Tumor Statistics
    # -------------------------------------

    with open(
        "results/tumor/tumor_statistics.txt",
        "w"
    ) as f:

        f.write(
            "Tumor Statistics\n"
        )

        f.write(
            "================\n\n"
        )

        f.write(
            f"Initial Cells : "
            f"{stats['initial_cells']:.2f}\n"
        )

        f.write(
            f"Final Cells : "
            f"{stats['final_cells']:.2f}\n"
        )

        f.write(
            f"Tumor Reduction : "
            f"{stats['tumor_reduction_percent']:.2f}%\n"
        )

    # -------------------------------------
    # Result 1
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

    # -------------------------------------
    # Result 2
    # -------------------------------------

    (
        release_histories,
        tumor_results,
        summary_results
    ) = run_release_rate_study(
        C_np
    )

    # -------------------------------------
    # Final Summary
    # -------------------------------------

    print("\n================================")
    print("SIMULATION COMPLETED")
    print("================================")

    print(
        f"\nTumor Reduction = "
        f"{stats['tumor_reduction_percent']:.2f}%"
    )

    print(
        "\nResults saved in "
        "'results/' folder"
    )

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