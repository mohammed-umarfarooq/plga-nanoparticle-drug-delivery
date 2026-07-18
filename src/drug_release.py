import numpy as np

from src.parameters import *
from src.utils import get_logger, validate_array, validate_positive


logger = get_logger(__name__)

# =====================================================
# DRUG RELEASE MODEL
# =====================================================

def solve_drug_release(
    C_np: np.ndarray,
    release_constant: float = release_rate,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Solve PLGA drug release and free-drug diffusion.

    Purpose
    -------
    Drug release model:

    1. PLGA nanoparticles release drug
       according to first-order kinetics.

    2. Released drug diffuses through
       tumor tissue using a discrete
       diffusion operator.

    3. Drug diffusion coefficient is
       intentionally larger than the
       nanoparticle diffusion coefficient
       because free drug molecules are
       substantially smaller than the
       carrier particles.
    Parameters
    ----------
    C_np : ndarray
        Nanoparticle concentration field.
    release_constant : float, optional
        First-order drug release constant (h^-1).

    Returns
    -------
    Drug : ndarray
        Final released drug concentration field.
    drug_history : list of float
        Mean drug concentration over time.
    release_percent_history : list of float
        Cumulative release percentage over time.

    Units
    -----
    Time is in hours. Release constant is h^-1.
    """

    C_np = validate_array(C_np, "C_np", expected_shape=(Ny, Nx))
    validate_positive(release_constant, "release_constant")

    logger.info("Simulating Drug Release...")

    Drug = np.zeros_like(C_np)

    remaining_np = C_np.copy()

    initial_drug_amount = np.sum(C_np)

    if initial_drug_amount <= 0:
        logger.warning("Initial drug amount is zero; using epsilon denominator.")
        initial_drug_amount = 1e-12

    drug_history = []

    release_percent_history = []

    # -------------------------------------------------
    # Time Loop
    # -------------------------------------------------

    for step in range(time_steps):

        released = (
            release_constant
            * remaining_np
            * dt
        )

        # Drug released from nanoparticles

        Drug += released

        # =====================================================
        # Drug Diffusion
        # =====================================================
        #
        # Reviewer requested deeper penetration.
        #
        # Increased diffusion coefficient to represent
        # molecular drug transport after release from PLGA.
        #
        # This produces broader spatial distribution
        # while maintaining numerical stability.
        #
        # =====================================================

        drug_diffusion_factor = 0.10

        laplacian = (
            np.roll(Drug, 1, axis=0)
            + np.roll(Drug, -1, axis=0)
            + np.roll(Drug, 1, axis=1)
            + np.roll(Drug, -1, axis=1)
            - 4 * Drug
        )

        Drug += (
            drug_diffusion_factor
            * laplacian
        )

        Drug = np.maximum(
            Drug,
            0
        )

        remaining_np -= released

        remaining_np = np.maximum(
            remaining_np,
            0
        )

        drug_history.append(
            np.mean(Drug)
        )

        released_fraction = (

            np.sum(Drug)

            / initial_drug_amount

        ) * 100

        release_percent_history.append(
            released_fraction
        )

    logger.info("Drug Release Complete")

    return (

        Drug,

        drug_history,

        release_percent_history

    )


# =====================================================
# T50
# =====================================================

def calculate_t50(
    release_percent_history: list[float],
) -> float | None:
    """Calculate the time required to reach 50 percent release.

    Parameters
    ----------
    release_percent_history : list of float
        Cumulative release percentages.

    Returns
    -------
    float or None
        First time reaching 50 percent release, or None if never reached.

    Units
    -----
    Time is in hours.
    """

    for i, value in enumerate(
        release_percent_history
    ):

        if value >= 50:

            return i * dt

    return None


# =====================================================
# FINAL RELEASE %
# =====================================================

def calculate_final_release_percent(
    release_percent_history: list[float],
) -> float:
    """Return the final cumulative release percentage.

    Parameters
    ----------
    release_percent_history : list of float
        Cumulative release percentages.

    Returns
    -------
    float
        Final release percentage.

    Units
    -----
    Percent.
    """

    return release_percent_history[-1]


# =====================================================
# TIME VECTOR
# =====================================================

def get_time_hours() -> np.ndarray:
    """Return the simulation time vector.

    Parameters
    ----------
    None

    Returns
    -------
    ndarray
        Simulation times.

    Units
    -----
    Hours.
    """

    return (
        np.arange(
            time_steps
        ) * dt
    )


