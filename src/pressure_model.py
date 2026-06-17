import numpy as np

from src.parameters import *

# =====================================================
# PRESSURE MODEL
# =====================================================

def solve_pressure():

    print("Solving Pressure Equation...")

    P = np.zeros((Ny, Nx))

    # Boundary Conditions

    P[:, 0] = 1000
    P[:, -1] = 0

    # Finite Difference Iteration

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

    print("Computing Velocity Field...")

    Vy, Vx = np.gradient(-K * P)

    print("Velocity Computation Complete")

    return Vx, Vy