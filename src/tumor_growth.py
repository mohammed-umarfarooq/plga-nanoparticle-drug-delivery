import numpy as np

from src.parameters import *

# =====================================================
# TUMOR GROWTH MODEL
# =====================================================

def solve_tumor_growth(Drug):

    print("Simulating Tumor Growth...")

    # Initial tumor cells

    N = 1e6

    tumor_history = []

    # -------------------------------------------------
    # Drug availability
    # -------------------------------------------------

    total_drug = np.sum(Drug)

    max_possible = Drug.size

    normalized_drug = total_drug / max_possible

    # -------------------------------------------------
    # Time Loop
    # -------------------------------------------------

    for step in range(time_steps):

        # Drug effect builds gradually

        drug_effect = (

            normalized_drug

            *

            (
                1
                -
                np.exp(
                    -0.03 * step
                )
            )

        )

        # Logistic growth

        growth_term = (

            growth_rate

            * N

            * (
                1
                -
                N / carrying_capacity
            )

        )

        # Drug-induced killing

        drug_killing = (
            drug_efficacy*drug_effect*2.0*N
        )

        # Net population change

        dN = (

            growth_term

            -

            drug_killing

        )

        N = N + dN * dt

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

        (
            initial_cells
            -
            final_cells
        )

        /

        initial_cells

    ) * 100

    return reduction


# =====================================================
# PEAK TUMOR
# =====================================================

def calculate_peak_tumor(
    tumor_history
):

    peak_size = max(
        tumor_history
    )

    peak_index = np.argmax(
        tumor_history
    )

    peak_time = (
        peak_index * dt
    )

    return (
        peak_size,
        peak_time
    )


# =====================================================
# REGRESSION START TIME
# =====================================================

def calculate_regression_time(
    tumor_history
):

    peak_index = np.argmax(
        tumor_history
    )

    regression_time = (
        peak_index * dt
    )

    return regression_time


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
# SUMMARY STATISTICS
# =====================================================

def tumor_statistics(
    tumor_history
):

    initial_cells = (
        tumor_history[0]
    )

    final_cells = (
        tumor_history[-1]
    )

    reduction = (
        calculate_tumor_reduction(
            initial_cells,
            final_cells
        )
    )

    peak_size, peak_time = (
        calculate_peak_tumor(
            tumor_history
        )
    )

    regression_time = (
        calculate_regression_time(
            tumor_history
        )
    )

    return {

        "initial_cells":
        initial_cells,

        "final_cells":
        final_cells,

        "tumor_reduction_percent":
        reduction,

        "peak_tumor_size":
        peak_size,

        "peak_time_hours":
        peak_time,

        "regression_start_hours":
        regression_time
    }


# =====================================================
# MODEL INTERPRETATION
# =====================================================

def get_tumor_model_explanation():

    return (
        "Initial tumor growth occurs because the drug effect "
        "builds gradually over time. During the early phase, "
        "tumor proliferation exceeds drug-induced cell killing. "
        "After sufficient drug accumulation, cell killing "
        "dominates and tumor regression begins."
    )