from src.parameters import *

from src.drug_release import (
    solve_drug_release,
    calculate_t50,
    calculate_final_release_percent
)

from src.tumor_growth import (
    solve_tumor_growth,
    tumor_statistics,
    get_tumor_model_explanation
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

    # -------------------------------------------------
    # Slow / Medium / Fast
    # -------------------------------------------------

    for label, rate in release_rates.items():

        print(
            f"\nSimulating {label} Release"
        )

        (
            Drug,
            drug_history,
            release_percent_history
        ) = solve_drug_release(
            C_np,
            release_constant=rate
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

        t50 = (
            calculate_t50(
                release_percent_history
            )
        )

        final_release = (
            calculate_final_release_percent(
                release_percent_history
            )
        )

        stats["t50_hours"] = t50

        stats["final_release_percent"] = (
            final_release
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

        print(
            f"t50 = "
            f"{t50:.2f} hours"
        )

        print(
            f"Final Release = "
            f"{final_release:.2f}%"
        )

    # -------------------------------------------------
    # Comparison Graphs
    # -------------------------------------------------

    plot_release_comparison(
        release_histories
    )

    plot_tumor_comparison(
        tumor_results
    )

    # -------------------------------------------------
    # Save Numerical Results
    # -------------------------------------------------

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
                "-----------------\n"
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
                f"Peak Tumor Size : "
                f"{stats['peak_tumor_size']:.2f}\n"
            )

            f.write(
                f"Peak Time : "
                f"{stats['peak_time_hours']:.2f} h\n"
            )

            f.write(
                f"Regression Start : "
                f"{stats['regression_start_hours']:.2f} h\n"
            )

            f.write(
                f"Tumor Reduction : "
                f"{stats['tumor_reduction_percent']:.2f}%\n"
            )

            f.write(
                f"t50 : "
                f"{stats['t50_hours']:.2f} h\n"
            )

            f.write(
                f"Final Release : "
                f"{stats['final_release_percent']:.2f}%\n"
            )

            f.write(
                "\nInterpretation:\n"
            )

            f.write(
                get_tumor_model_explanation()
            )

            f.write(
                "\n\nRelease Rate Interpretation:\n"
            )

            f.write(
                "Faster release rates reduce the time required "
                "to reach therapeutic drug concentrations, "
                "resulting in earlier tumor regression and "
                "greater overall tumor reduction.\n"
            )

            f.write("\n")

            f.write("\n")

        f.write(
            "Summary Table\n"
        )

        f.write(
            "=============\n\n"
        )

        f.write(
            f"{'Case':<10}"
            f"{'t50(h)':<12}"
            f"{'Final Release %':<20}"
            f"{'Tumor Reduction %':<20}\n"
        )

        f.write(
            "-" * 62 + "\n"
        )

        for label, stats in summary_results.items():

            f.write(
                f"{label:<10}"
                f"{stats['t50_hours']:<12.2f}"
                f"{stats['final_release_percent']:<20.2f}"
                f"{stats['tumor_reduction_percent']:<20.2f}\n"
            )

        f.write("\n")

        f.write(
            "Release Constant Justification:\n"
        )

        f.write(
            "Slow  : k = 0.02 h^-1\n"
        )

        f.write(
            "Medium: k = 0.05 h^-1\n"
        )

        f.write(
            "Fast  : k = 0.10 h^-1\n\n"
        )

        f.write(
            "These values represent slow, intermediate and "
            "rapid PLGA degradation/release scenarios used "
            "for sensitivity analysis of controlled drug "
            "delivery systems.\n\n"
        )

        f.write(
            "Higher release constants produce faster drug "
            "availability, smaller t50 values and stronger "
            "tumor regression.\n"
        )

    # =================================================
    # REVIEWER VALIDATION SUMMARY
    # =================================================

    print("\n")
    print("=" * 80)
    print("RESULT 2 VALIDATION SUMMARY")
    print("=" * 80)

    print("\nRelease Constants Used")
    print("----------------------")
    print("Slow   : k = 0.02 h^-1")
    print("Medium : k = 0.05 h^-1")
    print("Fast   : k = 0.10 h^-1")

    print(
        "\nThese values represent slow, intermediate "
        "and rapid PLGA degradation/release scenarios "
        "used for sensitivity analysis of controlled "
        "drug delivery systems."
    )

    print("\n")
    print(
        f"{'Case':<10}"
        f"{'t50(h)':<12}"
        f"{'Final Release %':<20}"
        f"{'Tumor Reduction %':<20}"
    )

    print("-" * 70)

    for label, stats in summary_results.items():

        print(
            f"{label:<10}"
            f"{stats['t50_hours']:<12.2f}"
            f"{stats['final_release_percent']:<20.2f}"
            f"{stats['tumor_reduction_percent']:<20.2f}"
        )

    print("\nInterpretation:")
    print(
        "Higher release constants produce smaller t50 values,"
    )
    print(
        "leading to faster drug availability and stronger"
    )
    print(
        "tumor regression."
    )

    print("=" * 80)

    print(
        "\nResult 2 Complete"
    )

    return (
        release_histories,
        tumor_results,
        summary_results
    )