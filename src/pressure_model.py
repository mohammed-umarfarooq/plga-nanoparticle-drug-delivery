import numpy as np

from src.parameters import *
from src.utils import get_logger, validate_array, validate_nonnegative


logger = get_logger(__name__)

# =====================================================
# PRESSURE MODEL
# =====================================================

def solve_pressure(
    inlet_pressure: float = 1000,
    outlet_pressure: float = 0
) -> np.ndarray:

    """
    Solve the steady-state interstitial fluid pressure field.

    Parameters
    ----------
    inlet_pressure : float, optional
        Pressure applied on the left boundary (Pa).
        Default = 1000 Pa (original behaviour).

    outlet_pressure : float, optional
        Pressure applied on the right boundary (Pa).
        Default = 0 Pa (original behaviour).

    Returns
    -------
    P : ndarray
        Pressure field (Pa).
    """

    validate_nonnegative(inlet_pressure, "inlet_pressure")
    validate_nonnegative(outlet_pressure, "outlet_pressure")

    logger.info("Solving Pressure Equation...")

    P = np.zeros((Ny, Nx))

    # -------------------------------------------------
    # Boundary Conditions
    #
    # Left boundary  : inlet pressure
    # Right boundary : outlet pressure
    #
    # Defaults preserve the original simulation.
    # -------------------------------------------------

    P[:, 0] = inlet_pressure
    P[:, -1] = outlet_pressure

    # -------------------------------------------------
    # Finite Difference Iteration
    #
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

    logger.info(
        "Pressure Solution Complete (Inlet = %.2f Pa, Outlet = %.2f Pa)",
        inlet_pressure,
        outlet_pressure,
    )

    return P


# =====================================================
# VELOCITY FIELD
# =====================================================

def compute_velocity(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes interstitial fluid velocity
    using Darcy's law:

        v = -K∇P

    where

        K = hydraulic conductivity
        P = pressure

    Parameters
    ----------
    P : ndarray
        Pressure field (Pa)

    Returns
    -------
    Vx : ndarray
        Velocity in x-direction (m/s)

    Vy : ndarray
        Velocity in y-direction (m/s)
    """

    P = validate_array(P, "P", expected_shape=(Ny, Nx))

    logger.info("Computing Velocity Field...")

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
        Vx ** 2 + Vy ** 2
    )

    logger.info(
        "Maximum Velocity : %.3e m/s",
        np.max(velocity_mag),
    )

    logger.info(
        "Average Velocity : %.3e m/s",
        np.mean(velocity_mag),
    )

    logger.info("Velocity Computation Complete")

    return Vx, Vy
