import numpy as np

from src.parameters import *

# =====================================================
# TUMOR GROWTH MODEL
# =====================================================

def solve_tumor_growth(Drug):

    print("Simulating Tumor Growth...")

    # Initial tumor cell population

    N = 1e6

    tumor_history = []

    # Average drug concentration

    avg_drug = np.mean(Drug)

    for step in range(time_steps):

        # PLGA drug effect increases gradually

        drug_effect = avg_drug * (
            1 - np.exp(-0.01 * step)
        )

        # Logistic growth

        growth_term = (
            growth_rate
            * N
            * (
                1 - N / carrying_capacity
            )
        )

        # Drug-induced cell death

        drug_killing = (
            drug_efficacy
            * drug_effect
            * N
        )

        # Net change

        dN = (
            growth_term
            - drug_killing
        )

        N = N + dN * dt

        # Prevent negative cells

        if N < 0:
            N = 0

        tumor_history.append(N)

    print("Tumor Growth Complete")

    return tumor_history


# =====================================================
# TUMOR REDUCTION
# =====================================================

def calculate_tumor_reduction(
    initial_cells,
    final_cells
):

    reduction = (
        (initial_cells - final_cells)
        / initial_cells
    ) * 100

    return reduction


# =====================================================
# SUMMARY STATISTICS
# =====================================================

def tumor_statistics(
    tumor_history
):

    initial_cells = tumor_history[0]

    final_cells = tumor_history[-1]

    reduction = calculate_tumor_reduction(
        initial_cells,
        final_cells
    )

    return {
        "initial_cells": initial_cells,
        "final_cells": final_cells,
        "tumor_reduction_percent": reduction
    }