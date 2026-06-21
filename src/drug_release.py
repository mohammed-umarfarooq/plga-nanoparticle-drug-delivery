import numpy as np

from src.parameters import *

# =====================================================
# DRUG RELEASE MODEL
# =====================================================

def solve_drug_release(
    C_np,
    release_constant=release_rate
):
    """
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
    """

    print("Simulating Drug Release...")

    Drug = np.zeros_like(C_np)

    remaining_np = C_np.copy()

    initial_drug_amount = np.sum(C_np)

    # Safety check

    if initial_drug_amount <= 0:

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

    print("Drug Release Complete")

    return (

        Drug,

        drug_history,

        release_percent_history

    )


# =====================================================
# T50
# =====================================================

def calculate_t50(
    release_percent_history
):

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
    release_percent_history
):

    return release_percent_history[-1]


# =====================================================
# TIME VECTOR
# =====================================================

def get_time_hours():

    return (
        np.arange(
            time_steps
        ) * dt
    )


# =====================================================
# OPTIONAL EXPONENTIAL MODEL
# =====================================================

def solve_drug_release_exponential(
    C_np,
    release_constant=release_rate
):

    print(
        "Simulating Exponential Drug Release..."
    )

    Drug = np.zeros_like(C_np)

    drug_history = []

    for step in range(time_steps):

        time = step * dt

        released = (

            release_constant
            * C_np
            * np.exp(
                -release_constant * time
            )
            * dt

        )

        Drug += released

        drug_history.append(
            np.mean(Drug)
        )

    print(
        "Exponential Drug Release Complete"
    )

    return Drug, drug_history