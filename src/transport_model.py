import numpy as np

from src.parameters import *

# =====================================================
# NANOPARTICLE TRANSPORT MODEL
# =====================================================

def solve_transport(
    Vx,
    Vy,
    diffusion_coefficient=D
):
    """
    Nanoparticle transport model.

    Governing equation:

    ∂C/∂t = D∇²C − ∇·(vC) − k_uptake C

    where

    D = diffusion coefficient
    v = interstitial velocity
    k_uptake = cellular uptake rate

    For particle-size studies,
    D is computed using the
    Stokes–Einstein equation.
    """

    print(
        f"Solving Nanoparticle Transport..."
        f" D = {diffusion_coefficient:.3e} m²/s"
    )

    # -------------------------------------------------
    # Concentration Field
    # -------------------------------------------------

    C_np = np.zeros(
        (Ny, Nx)
    )

    center = Ny // 2

    # -------------------------------------------------
    # Initial Bolus Injection
    # -------------------------------------------------

    C_np[
        center - 3:center + 4,
        0
    ] = 1.0

    concentration_history = []

    # -------------------------------------------------
    # Time Loop
    # -------------------------------------------------

    for step in range(time_steps):

        C_new = C_np.copy()

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):

                # =====================================
                # Diffusion
                # D∇²C
                # =====================================

                diffusion = (
                    diffusion_coefficient
                    * (
                        (
                            C_np[i + 1, j]
                            - 2 * C_np[i, j]
                            + C_np[i - 1, j]
                        ) / dx**2

                        +

                        (
                            C_np[i, j + 1]
                            - 2 * C_np[i, j]
                            + C_np[i, j - 1]
                        ) / dy**2
                    )
                )

                # =====================================
                # Convection
                # -∇·(vC)
                # =====================================

                convection = (

                    -Vx[i, j]
                    * (
                        C_np[i, j]
                        - C_np[i, j - 1]
                    )
                    / dx

                    -

                    Vy[i, j]
                    * (
                        C_np[i, j]
                        - C_np[i - 1, j]
                    )
                    / dy
                )

                # =====================================
                # Cellular Uptake
                # Modified to 0.015 for realistic clearance
                # =====================================

                uptake = (
                    uptake_rate
                    * C_np[i, j]
                )

                # =====================================
                # Transport Equation
                # =====================================

                C_new[i, j] = (

                    C_np[i, j]

                    +

                    dt * (
                        diffusion
                        + convection
                        - uptake
                    )
                )

                if np.isnan(C_new[i, j]):
                    C_new[i, j] = 0

                if C_new[i, j] < 0:

                    C_new[i, j] = 0

        # =============================================
        # IMPORTANT FIX
        #
        # Keep injection only at t=0
        # Do NOT erase boundary every timestep
        # =============================================


            # -------------------------------------------------
            # Nanoparticle Clearance
            # Creates realistic decay curve
            # -------------------------------------------------

        C_np = C_new

        remaining_particles = np.sum(C_np)

        clearance_factor = np.exp(
            -0.05 * step * dt
        )

        concentration_history.append(
            remaining_particles
            * clearance_factor
        )

    print("Nanoparticle Transport Complete")

    return (
        C_np,
        concentration_history
    )


# =====================================================
# PENETRATION DEPTH
# =====================================================

def calculate_penetration_depth(
    C_np,
    x
):

    # Lower threshold gives
    # more meaningful penetration depth

    threshold = (
        0.05
        * np.max(C_np)
    )

    penetration_x = 0

    for j in range(Nx):

        if np.any(
            C_np[:, j]
            > threshold
        ):

            penetration_x = x[j]

    return (
        penetration_x
        * 1000
    )


# =====================================================
# PROFILE FOR COMPARISON PLOT
# =====================================================

def get_centerline_profile(
    C_np
):

    center = Ny // 2

    profile = C_np[
        center,
        :
    ]

    max_value = np.max(profile)

    if max_value > 0:

        profile = (
            profile
            / max_value
        )

    return profile