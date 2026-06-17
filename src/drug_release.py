import numpy as np

from src.parameters import *

# =====================================================
# DRUG RELEASE MODEL
# =====================================================

def solve_drug_release(
    C_np,
    release_constant=release_rate
):

    print("Simulating Drug Release...")

    # Drug concentration

    Drug = np.zeros_like(C_np)

    # Remaining drug inside nanoparticles

    remaining_np = C_np.copy()

    drug_history = []

    # Time Loop

    for step in range(time_steps):

        # Drug released during this step

        released = (
            release_constant
            * remaining_np
            * dt
        )

        # Add released drug

        Drug += released

        # Remove released drug from nanoparticles

        remaining_np -= released

        # Prevent negative values

        remaining_np = np.maximum(
            remaining_np,
            0
        )

        # Store average concentration

        drug_history.append(
            np.mean(Drug)
        )

    print("Drug Release Complete")

    return Drug, drug_history


# =====================================================
# EXPONENTIAL RELEASE MODEL (OPTIONAL)
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