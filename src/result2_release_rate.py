from src.parameters import *
from src.drug_release import (
    solve_drug_release
)
from src.tumor_growth import (
    solve_tumor_growth,
    tumor_statistics
)
from src.visualization import (
    plot_release_comparison,
    plot_tumor_comparison
)

# =====================================================
# RESULT 2
# RELEASE RATE COMPARISON
# =====================================================

def run_release_rate_study(
    C_np
):

    print("\nRunning Result 2...")
    print("Drug Release Rate Study")

    release_histories = {}

    tumor_results = {}

    summary_results = {}

    # -------------------------------------
    # Slow / Medium / Fast
    # -------------------------------------

    for label, rate in release_rates.items():

        print(
            f"\nSimulating {label} Release"
        )

        Drug, drug_history = (
            solve_drug_release(
                C_np,
                release_constant=rate
            )
        )

        tumor_history = (
            solve_tumor_growth(
                Drug
            )
        )

        stats = (
            tumor_statistics(
                tumor_history
            )
        )

        release_histories[label] = (
            drug_history
        )

        tumor_results[label] = (
            tumor_history
        )

        summary_results[label] = (
            stats
        )

        print(
            f"Tumor Reduction = "
            f"{stats['tumor_reduction_percent']:.2f}%"
        )

    # -------------------------------------
    # Save Graphs
    # -------------------------------------

    plot_release_comparison(
        release_histories
    )

    plot_tumor_comparison(
        tumor_results
    )

    # -------------------------------------
    # Save Text Results
    # -------------------------------------

    with open(
        "results/result2/release_rate_results.txt",
        "w"
    ) as f:

        f.write(
            "Release Rate Study\n"
        )

        f.write(
            "==================\n\n"
        )

        for label, stats in (
            summary_results.items()
        ):

            f.write(
                f"{label} Release\n"
            )

            f.write(
                f"Initial Cells : "
                f"{stats['initial_cells']:.2f}\n"
            )

            f.write(
                f"Final Cells : "
                f"{stats['final_cells']:.2f}\n"
            )

            f.write(
                f"Tumor Reduction : "
                f"{stats['tumor_reduction_percent']:.2f}%\n"
            )

            f.write(
                "\n"
            )

    print(
        "\nResult 2 Complete"
    )

    return (
        release_histories,
        tumor_results,
        summary_results
    )