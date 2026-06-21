import os
import numpy as np
import matplotlib.pyplot as plt

from src.parameters import dt

# =====================================================
# CREATE RESULT FOLDERS
# =====================================================

def create_folders():

    folders = [
        "results/pressure",
        "results/transport",
        "results/drug",
        "results/tumor",
        "results/result1",
        "results/result2"
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


# =====================================================
# PRESSURE CONTOUR
# =====================================================

def plot_pressure(
    X,
    Y,
    P
):

    plt.figure(figsize=(8, 6))

    plt.contourf(
        X * 1000,
        Y * 1000,
        P,
        levels=40
    )

    plt.colorbar(
        label="Pressure (Pa)"
    )

    plt.xlabel(
        "X Position (mm)"
    )

    plt.ylabel(
        "Y Position (mm)"
    )

    plt.title(
        "Interstitial Fluid Pressure (Pa)"
    )

    plt.tight_layout()

    plt.savefig(
        "results/pressure/pressure_contour.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# VELOCITY FIELD
# =====================================================

def plot_velocity(
    X,
    Y,
    Vx,
    Vy
):

    plt.figure(figsize=(8, 6))

    velocity_magnitude = np.sqrt(
        Vx**2 + Vy**2
    )

    max_velocity = np.max(
        velocity_magnitude
    )

    if max_velocity > 0:

        Vx_plot = (
            Vx / max_velocity
        )

        Vy_plot = (
            Vy / max_velocity
        )

    else:

        Vx_plot = Vx
        Vy_plot = Vy

    plt.contourf(
        X * 1000,
        Y * 1000,
        velocity_magnitude,
        levels=30
    )

    plt.colorbar(
        label="Velocity Magnitude (m/s)"
    )

    plt.quiver(
        X[::2, ::2] * 1000,
        Y[::2, ::2] * 1000,
        Vx_plot[::2, ::2],
        Vy_plot[::2, ::2],
        scale=25
    )

    plt.xlabel(
        "X Position (mm)"
    )

    plt.ylabel(
        "Y Position (mm)"
    )

    plt.title(
        "Interstitial Fluid Velocity Field"
    )
    
    plt.tight_layout()

    plt.savefig(
        "results/pressure/velocity_field.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# NANOPARTICLE DISTRIBUTION
# =====================================================

def plot_transport(
    X,
    Y,
    C_np
):

    plt.figure(figsize=(8, 6))

    plt.contourf(
        X * 1000,
        Y * 1000,
        C_np,
        levels=50
    )

    plt.contour(
        X * 1000,
        Y * 1000,
        C_np,
        colors="black",
        linewidths=0.3
    )

    plt.colorbar(
        label="Nanoparticle Concentration"
    )

    plt.xlabel(
        "X Position (mm)"
    )

    plt.ylabel(
        "Y Position (mm)"
    )

    plt.title(
        "Nanoparticle Distribution"
    )
    plt.tight_layout()

    plt.savefig(
        "results/transport/nanoparticle_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# DRUG DISTRIBUTION
# =====================================================

def plot_drug(
    X,
    Y,
    Drug
):

    plt.figure(figsize=(8, 6))

    plt.contourf(
        X * 1000,
        Y * 1000,
        Drug,
        levels=50
    )

    plt.contour(
        X * 1000,
        Y * 1000,
        Drug,
        colors="black",
        linewidths=0.3
    )

    plt.colorbar(
        label="Drug Concentration"
    )

    plt.xlabel(
        "X Position (mm)"
    )

    plt.ylabel(
        "Y Position (mm)"
    )

    plt.title(
        "Drug Distribution"
    )
    plt.tight_layout()

    plt.savefig(
        "results/drug/drug_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# NANOPARTICLE CONCENTRATION HISTORY
# =====================================================

def plot_concentration_history(
    history
):

    time_hours = (
        np.arange(
            len(history)
        ) * dt
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        time_hours,
        history,
        linewidth=2
    )

    peak_index = np.argmax(history)

    plt.scatter(
        time_hours[peak_index],
        history[peak_index]
    )

    plt.annotate(
        f"Tmax = {time_hours[peak_index]:.1f} h",
        (
            time_hours[peak_index],
            history[peak_index]
        )
    )

    plt.xlabel(
        "Time (hours)"
    )

    plt.ylabel(
        "Nanoparticle Concentration"
    )

    plt.title(
        "Nanoparticle Concentration vs Time"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/transport/concentration_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# DRUG RELEASE HISTORY
# =====================================================

def plot_drug_history(
    history
):

    time_hours = (
        np.arange(
            len(history)
        ) * dt
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        time_hours,
        history,
        linewidth=2
    )

    peak_index = np.argmax(history)

    plt.scatter(
        time_hours[peak_index],
        history[peak_index]
    )

    plt.annotate(
        f"Tmax = {time_hours[peak_index]:.1f} h",
        (
            time_hours[peak_index],
            history[peak_index]
        )
    )

    plt.xlabel(
        "Time (hours)"
    )

    plt.ylabel(
        "Drug Concentration"
    )

    plt.title(
        "Drug Release Curve"
    )

    plt.grid(True)
    
    plt.tight_layout()

    plt.savefig(
        "results/drug/drug_release_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# TUMOR GROWTH RESPONSE
# =====================================================

def plot_tumor_growth(
    tumor_history
):

    time_hours = (
        np.arange(
            len(tumor_history)
        ) * dt
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        time_hours,
        tumor_history,
        linewidth=2
    )

    peak_index = np.argmax(
        tumor_history
    )

    plt.scatter(
        time_hours[peak_index],
        tumor_history[peak_index]
    )

    plt.annotate(
        f"Peak = {time_hours[peak_index]:.1f} h",
        (
            time_hours[peak_index],
            tumor_history[peak_index]
        )
    )

    plt.xlabel(
        "Time (hours)"
    )

    plt.ylabel(
        "Tumor Cell Population"
    )

    plt.title(
        "Tumor Growth Response"
    )

    plt.grid(True)
    
    plt.tight_layout()

    plt.savefig(
        "results/tumor/tumor_growth_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# RESULT 1
# PENETRATION DEPTH COMPARISON
# =====================================================

def plot_penetration_depth(
    particle_sizes,
    penetration_depths
):

    plt.figure(figsize=(8, 6))

    plt.plot(
        particle_sizes,
        penetration_depths,
        marker="o",
        linewidth=2
    )

    # Reviewer Annotation: Display values directly on graph points
    for x, y in zip(
        particle_sizes,
        penetration_depths
    ):
        plt.text(
            x,
            y + max(penetration_depths) * 0.02, # Slight vertical offset
            f"{y:.2f} mm",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.xlabel(
        "Particle Size (nm)"
    )

    plt.ylabel(
        "Penetration Depth (mm)"
    )

    plt.title(
        "Penetration Depth vs Particle Size"
    )

    plt.grid(True)

    # Give a tiny bit of vertical headroom for the top labels
    plt.ylim(min(penetration_depths) * 0.8, max(penetration_depths) * 1.15)

    plt.tight_layout()

    plt.savefig(
        "results/result1/penetration_depth_vs_size.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# RESULT 2
# DRUG RELEASE COMPARISON
# =====================================================

def plot_release_comparison(
    release_histories
):

    plt.figure(figsize=(8, 6))

    for label, history in release_histories.items():

        time_hours = (
            np.arange(
                len(history)
            ) * dt
        )

        if label == "Slow":

            legend_label = (
                "Slow (k=0.02)"
            )

        elif label == "Medium":

            legend_label = (
                "Medium (k=0.05)"
            )

        else:

            legend_label = (
                "Fast (k=0.10)"
            )

        plt.plot(
            time_hours,
            history,
            label=legend_label,
            linewidth=2
        )

    plt.xlabel(
        "Time (hours)"
    )

    plt.ylabel(
        "Drug Concentration"
    )

    plt.title(
        "Drug Release Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/result2/drug_release_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# RESULT 2
# TUMOR RESPONSE COMPARISON
# =====================================================

def plot_tumor_comparison(
    tumor_results
):

    plt.figure(figsize=(8, 6))

    for label, history in tumor_results.items():

        time_hours = (
            np.arange(
                len(history)
            ) * dt
        )

        plt.plot(
            time_hours,
            history,
            label=label,
            linewidth=2
        )

    plt.xlabel(
        "Time (hours)"
    )

    plt.ylabel(
        "Tumor Cell Population"
    )

    plt.title(
        "Tumor Response Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/result2/tumor_response_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()