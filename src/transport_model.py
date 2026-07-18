import numpy as np

from src.parameters import *
from src.utils import (
    get_logger,
    validate_array,
    validate_matching_shapes,
    validate_nonnegative,
    validate_positive,
)


logger = get_logger(__name__)

# =====================================================
# NANOPARTICLE TRANSPORT MODEL
# =====================================================

def solve_transport(
    Vx: np.ndarray,
    Vy: np.ndarray,
    diffusion_coefficient: float = D,
    velocity_scale: float = 1.0,
    uptake_coeff: float = uptake_rate,
) -> tuple[np.ndarray, list[float]]:
    """
    Nanoparticle transport model.

    Governing equation

        ∂C/∂t = D∇²C − ∇·(vC) − k_uptake C

    Parameters
    ----------
    Vx, Vy : ndarray
        Darcy velocity field.

    diffusion_coefficient : float
        Nanoparticle diffusion coefficient (m²/s).

    velocity_scale : float, optional
        Multiplies the velocity field.

        Default = 1.0

        Values below 1.0 reduce convection and are
        used to simulate elevated interstitial
        fluid pressure (IFP).

    uptake_coeff : float, optional
        Cellular uptake coefficient used in the transport model.

    Returns
    -------
    C_np : ndarray
        Final nanoparticle concentration.

    concentration_history : list
        Total nanoparticle concentration history.
    """

    Vx = validate_array(Vx, "Vx", expected_shape=(Ny, Nx))
    Vy = validate_array(Vy, "Vy", expected_shape=(Ny, Nx))
    validate_matching_shapes(("Vx", Vx), ("Vy", Vy))
    validate_positive(diffusion_coefficient, "diffusion_coefficient")
    validate_positive(velocity_scale, "velocity_scale")
    validate_nonnegative(uptake_coeff, "uptake_coeff")

    logger.info(
        "Solving Nanoparticle Transport... D = %.3e m^2/s",
        diffusion_coefficient,
    )

    # -------------------------------------------------
    # Effective Velocity
    #
    # Used for Result 3 (IFP study).
    #
    # Default = 1.0
    # Existing simulation remains unchanged.
    # -------------------------------------------------

    Vx = velocity_scale * Vx
    Vy = velocity_scale * Vy

    # -------------------------------------------------
    # Concentration Field
    # -------------------------------------------------

    C_np = np.zeros((Ny, Nx))

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
    # Time Integration
    # -------------------------------------------------

    for step in range(time_steps):

        C_new = C_np.copy()

        # Compute the same finite-difference update for the whole interior
        # grid at once.  This replaces millions of Python-level cell updates
        # during dataset generation with NumPy operations.
        center_values = C_np[1:-1, 1:-1]
        diffusion = diffusion_coefficient * (
            (C_np[2:, 1:-1] - 2 * center_values + C_np[:-2, 1:-1]) / dx**2
            + (C_np[1:-1, 2:] - 2 * center_values + C_np[1:-1, :-2]) / dy**2
        )
        convection = (
            -Vx[1:-1, 1:-1] * (center_values - C_np[1:-1, :-2]) / dx
            -Vy[1:-1, 1:-1] * (center_values - C_np[:-2, 1:-1]) / dy
        )
        updated_interior = center_values + dt * (
            diffusion + convection - uptake_coeff * center_values
        )
        C_new[1:-1, 1:-1] = np.nan_to_num(
            updated_interior,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        C_new[1:-1, 1:-1] = np.maximum(C_new[1:-1, 1:-1], 0.0)

        # -------------------------------------------------
        # Update Concentration
        # -------------------------------------------------

        C_np = C_new

        # -------------------------------------------------
        # Nanoparticle Clearance
        # -------------------------------------------------

        remaining_particles = np.sum(C_np)

        clearance_factor = np.exp(
            -0.05 * step * dt
        )

        concentration_history.append(
            remaining_particles
            * clearance_factor
        )

    logger.info("Nanoparticle Transport Complete")

    return (
        C_np,
        concentration_history
    )


# =====================================================
# PENETRATION DEPTH
# =====================================================

def calculate_penetration_depth(
    C_np: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Calculate penetration depth based on the
    furthest location where concentration exceeds
    5% of the maximum concentration.
    """

    C_np = validate_array(C_np, "C_np")
    x = validate_array(x, "x")

    if C_np.shape[1] != x.size:
        raise ValueError(
            "x length must match the concentration field x dimension; "
            f"got {x.size} and {C_np.shape[1]}."
        )

    threshold = 0.05 * np.max(C_np)

    penetration_x = 0.0

    for j in range(Nx):

        if np.any(C_np[:, j] > threshold):

            penetration_x = x[j]

    return penetration_x * 1000.0


# =====================================================
# CENTERLINE PROFILE
# =====================================================

def get_centerline_profile(
    C_np: np.ndarray,
) -> np.ndarray:
    """
    Returns the normalized centerline
    concentration profile for comparison plots.
    """

    C_np = validate_array(C_np, "C_np")

    center = C_np.shape[0] // 2

    profile = C_np[
        center,
        :
    ].copy()

    max_value = np.max(profile)

    if max_value > 0:

        profile /= max_value

    return profile
