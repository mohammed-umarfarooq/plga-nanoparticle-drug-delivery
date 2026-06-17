import numpy as np

from src.parameters import *

# =====================================================
# NANOPARTICLE TRANSPORT MODEL
# =====================================================

def solve_transport(Vx, Vy, diffusion_coefficient=D):

    print("Solving Nanoparticle Transport...")

    # Nanoparticle concentration

    C_np = np.zeros((Ny, Nx))

    # Injection at left boundary

    C_np[:, 0] = 1.0

    concentration_history = []

    # Time Loop

    for step in range(time_steps):

        C_new = C_np.copy()

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):

                # Diffusion Term

                diffusion = diffusion_coefficient * (
                    (C_np[i + 1, j]
                     - 2 * C_np[i, j]
                     + C_np[i - 1, j]) / dx**2
                    +
                    (C_np[i, j + 1]
                     - 2 * C_np[i, j]
                     + C_np[i, j - 1]) / dy**2
                )

                # Convection Term

                convection = (
                    -Vx[i, j]
                    * (C_np[i, j] - C_np[i, j - 1])
                    / dx
                    -
                    Vy[i, j]
                    * (C_np[i, j] - C_np[i - 1, j])
                    / dy
                )

                # Cell Uptake

                uptake = uptake_rate * C_np[i, j]

                # Update

                C_new[i, j] = (
                    C_np[i, j]
                    + dt * (
                        diffusion
                        + convection
                        - uptake
                    )
                )

                # Prevent negative concentration

                if C_new[i, j] < 0:
                    C_new[i, j] = 0

        C_np = C_new

        concentration_history.append(
            np.mean(C_np)
        )

    print("Nanoparticle Transport Complete")

    return C_np, concentration_history


# =====================================================
# PENETRATION DEPTH
# =====================================================

def calculate_penetration_depth(C_np, x):

    threshold = 0.1 * np.max(C_np)

    penetration_x = 0

    for j in range(Nx):

        if np.any(C_np[:, j] > threshold):
            penetration_x = x[j]

    penetration_depth_mm = penetration_x * 1000

    return penetration_depth_mm