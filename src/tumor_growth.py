import numpy as np

from src.parameters import *
from src.utils import get_logger, validate_array, validate_positive


logger = get_logger(__name__)

# =====================================================
# TUMOR GROWTH MODEL
# =====================================================

def solve_tumor_growth(
    Drug: np.ndarray,
    growth_rate: float = growth_rate,
    drug_efficacy: float = drug_efficacy
) -> list[float]:
    """Simulate tumor-cell population response to released drug.

    Parameters
    ----------
    Drug : ndarray
        Released drug concentration field.
    growth_rate : float, optional
        Logistic tumor growth rate (h^-1).
    drug_efficacy : float, optional
        Drug killing coefficient.

    Returns
    -------
    list of float
        Tumor-cell population history.

    Units
    -----
    Time is in hours. Cell population is a count.
    """

    Drug = validate_array(Drug, "Drug")
    validate_positive(growth_rate, "growth_rate")
    validate_positive(drug_efficacy, "drug_efficacy")

    logger.info("Simulating Tumor Growth...")

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
            drug_efficacy * drug_effect * 2.0 * N
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

    logger.info("Tumor Growth Complete")

    return tumor_history


# =====================================================
# TUMOR REDUCTION
# =====================================================

def calculate_tumor_reduction(
    initial_cells: float,
    final_cells: float,
) -> float:
    """Calculate tumor reduction from initial and final populations.

    Parameters
    ----------
    initial_cells : float
        Initial tumor-cell count.
    final_cells : float
        Final tumor-cell count.

    Returns
    -------
    float
        Tumor reduction percentage.

    Units
    -----
    Percent.
    """

    validate_positive(initial_cells, "initial_cells")

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
    tumor_history: list[float],
) -> tuple[float, float]:
    """Calculate peak tumor size and time.

    Parameters
    ----------
    tumor_history : list of float
        Tumor-cell population history.

    Returns
    -------
    tuple of float
        Peak tumor-cell count and peak time.

    Units
    -----
    Cell count and hours.
    """

    if not tumor_history:
        raise ValueError("tumor_history must not be empty.")

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
    tumor_history: list[float],
) -> float:
    """Return the regression start time based on the peak index.

    Parameters
    ----------
    tumor_history : list of float
        Tumor-cell population history.

    Returns
    -------
    float
        Regression start time.

    Units
    -----
    Hours.
    """

    if not tumor_history:
        raise ValueError("tumor_history must not be empty.")

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


# =====================================================
# SUMMARY STATISTICS
# =====================================================

def tumor_statistics(
    tumor_history: list[float],
) -> dict[str, float]:
    """Summarize tumor response statistics.

    Parameters
    ----------
    tumor_history : list of float
        Tumor-cell population history.

    Returns
    -------
    dict
        Initial, final, peak, regression, and reduction metrics.

    Units
    -----
    Cell counts, hours, and percent.
    """

    if not tumor_history:
        raise ValueError("tumor_history must not be empty.")

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

def get_tumor_model_explanation() -> str:
    """Return a short interpretation of the tumor-growth model.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Human-readable model interpretation.

    Units
    -----
    Not applicable.
    """

    return (
        "Initial tumor growth occurs because the drug effect "
        "builds gradually over time. During the early phase, "
        "tumor proliferation exceeds drug-induced cell killing. "
        "After sufficient drug accumulation, cell killing "
        "dominates and tumor regression begins."
    )
