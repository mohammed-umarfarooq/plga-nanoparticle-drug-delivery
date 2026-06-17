import numpy as np
import matplotlib.pyplot as plt

from src.parameters import *
from src.transport_model import (
    solve_transport,
    calculate_penetration_depth
)
from src.visualization import (
    plot_penetration_depth
)

# =====================================================
# RESULT 1
# PARTICLE SIZE COMPARISON
# =====================================================

def run_particle_size_study(
    X,
    Y,
    x,
    Vx,
    Vy
):

    print("\nRunning Result 1...")
    print("Particle Size Study")

    penetration_depths = []

    for size in particle_sizes:

        print(f"\nSimulating {size} nm")

        # -----------------------------------
        # Size-dependent diffusion
        # -----------------------------------

        D_size = D * (400 / size)

        # -----------------------------------
        # Transport Simulation
        # -----------------------------------

        C_np, history = solve_transport(
            Vx,
            Vy,
            diffusion_coefficient=D_size
        )

        # -----------------------------------
        # Penetration Depth
        # -----------------------------------

        depth = calculate_penetration_depth(
            C_np,
            x
        )

        penetration_depths.append(
            depth
        )

        print(
            f"Penetration Depth = "
            f"{depth:.2f} mm"
        )

        # -----------------------------------
        # Save Concentration Map
        # -----------------------------------

        plt.figure(figsize=(8, 6))

        plt.contourf(
            X * 1000,
            Y * 1000,
            C_np,
            levels=20
        )

        plt.colorbar(
            label="Concentration"
        )

        plt.xlabel("mm")
        plt.ylabel("mm")

        plt.title(
            f"{size} nm Nanoparticle Distribution"
        )

        plt.savefig(
            f"results/result1/{size}nm_concentration.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ---------------------------------------
    # Penetration Depth Graph
    # ---------------------------------------

    plot_penetration_depth(
        particle_sizes,
        penetration_depths
    )

    # ---------------------------------------
    # Save Numerical Results
    # ---------------------------------------

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

        for size, depth in zip(
            particle_sizes,
            penetration_depths
        ):

            f.write(
                f"{size} nm : "
                f"{depth:.2f} mm\n"
            )

    print(
        "\nResult 1 Complete"
    )

    return penetration_depths