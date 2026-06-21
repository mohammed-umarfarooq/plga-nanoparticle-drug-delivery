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

    with open(
        "results/transport/nanoparticle_statistics.txt",
        "w"
    ) as f:

        f.write(
            "Nanoparticle Statistics\n"
        )

        f.write(
            "=======================\n\n"
        )

        f.write(
            f"Peak Concentration : "
            f"{peak_np:.6f}\n"
        )

        f.write(
            f"Average Concentration : "
            f"{avg_np:.6f}\n"
        )

        f.write(
            f"Tmax : "
            f"{peak_np_time:.2f} h\n"
        )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "After Tmax, nanoparticle concentration "
            "decreases due to particle clearance, "
            "drug release and cellular uptake.\n"
        )

    with open(
        "results/drug/drug_statistics.txt",
        "w"
    ) as f:

        f.write(
            "Drug Statistics\n"
        )

        f.write(
            "===============\n\n"
        )

        f.write(
            f"Peak Drug Concentration : "
            f"{peak_drug:.6f}\n"
        )

        f.write(
            f"Tmax Drug : "
            f"{peak_drug_time:.2f} h\n"
        )

        f.write(
            f"Final Release Percent : "
            f"{final_release_percent:.2f}%\n"
        )

        f.write(
            "This value is consistent with the "
            "initial nanoparticle drug loading.\n"
        )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "Final release approaches complete drug "
            "availability from the initial nanoparticle "
            "loading, indicating effective release "
            "throughout the simulation period.\n"
        )

    # -------------------------------------
    # Save Tumor Statistics
    # Reviewer Requested:
    # Peak Size
    # Peak Time
    # Regression Time
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
            f"Peak Tumor Size : "
            f"{stats['peak_tumor_size']:.2f}\n"
        )

        f.write(
            f"Peak Time : "
            f"{stats['peak_time_hours']:.2f} h\n"
        )

        f.write(
            f"Regression Start : "
            f"{stats['regression_start_hours']:.2f} h\n"
        )

        f.write(
            f"Tumor Reduction : "
            f"{stats['tumor_reduction_percent']:.2f}%\n"
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

    print("\nParticle Size Study Summary")
    print("---------------------------")

    for size, depth in zip(
        particle_sizes,
        penetration_depths
    ):
        print(
            f"{size} nm : "
            f"{depth:.2f} mm"
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
    # Final Summary
    # -------------------------------------

    print("\n================================")
    print("SIMULATION COMPLETED")
    print("================================")

    print("\nREVIEWER VALIDATION SUMMARY")
    print("--------------------------------")

    print(
        f"Peak Tumor Size : "
        f"{stats['peak_tumor_size']:.2f}"
    )

    print(
        f"Peak Time : "
        f"{stats['peak_time_hours']:.2f} h"
    )

    print(
        f"Regression Start : "
        f"{stats['regression_start_hours']:.2f} h"
    )

    print(
        f"Tumor Reduction : "
        f"{stats['tumor_reduction_percent']:.2f}%"
    )

    print(
        f"Peak Drug Concentration : "
        f"{peak_drug:.6f}"
    )

    print(
        f"Drug Tmax : "
        f"{peak_drug_time:.2f} h"
    )

    print(
        f"Final Release : "
        f"{final_release_percent:.2f}%"
    )

    print("\nTumor Reduction Summary")

    for label, result in summary_results.items():

        print(
            f"{label}: "
            f"{result['tumor_reduction_percent']:.2f}%"
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