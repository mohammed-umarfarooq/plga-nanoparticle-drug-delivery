import numpy as np

from src.parameters import *

# =====================================================
# PRESSURE MODEL
# =====================================================

def solve_pressure():

    print("Solving Pressure Equation...")

    P = np.zeros((Ny, Nx))

    # -------------------------------------------------
    # Boundary Conditions
    # Left side: high pressure
    # Right side: low pressure
    # -------------------------------------------------

    P[:, 0] = 1000
    P[:, -1] = 0

    # -------------------------------------------------
    # Finite Difference Iteration
    # Solves:
    #
    # ∇²P = 0
    # -------------------------------------------------

    for iteration in range(500):

        P_new = P.copy()

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):

                P_new[i, j] = 0.25 * (

                    P[i + 1, j]
                    + P[i - 1, j]
                    + P[i, j + 1]
                    + P[i, j - 1]

                )

        P = P_new

    print("Pressure Solution Complete")

    return P


# =====================================================
# VELOCITY FIELD
# =====================================================

def compute_velocity(P):
    """
    Computes interstitial fluid velocity
    using Darcy's law:

    v = -K∇P

    where:

    K = hydraulic conductivity
    P = pressure

    The velocity field is used in the
    convection term of the nanoparticle
    transport equation.
    """

    print("Computing Velocity Field...")

    # -------------------------------------------------
    # Darcy Velocity
    #
    # v = -K ∇P
    # -------------------------------------------------

    dPdy, dPdx = np.gradient(
        P,
        dy,
        dx
    )

    Vx = -K * dPdx

    Vy = -K * dPdy

    velocity_mag = np.sqrt(
        Vx**2 + Vy**2
    )

    print(
        f"Maximum Velocity : "
        f"{np.max(velocity_mag):.3e} m/s"
    )

    print(
        f"Average Velocity : "
        f"{np.mean(velocity_mag):.3e} m/s"
    )

    print("Velocity Computation Complete")

    return Vx, Vy