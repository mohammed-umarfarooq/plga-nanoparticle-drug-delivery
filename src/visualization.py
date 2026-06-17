import os
import matplotlib.pyplot as plt

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
        levels=20
    )

    plt.colorbar(
        label="Pressure"
    )

    plt.xlabel("mm")
    plt.ylabel("mm")

    plt.title(
        "Interstitial Fluid Pressure"
    )

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

    plt.quiver(
        X,
        Y,
        Vx,
        Vy
    )

    plt.title(
        "Velocity Field"
    )

    plt.savefig(
        "results/pressure/velocity_field.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# NANOPARTICLE MAP
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
        levels=20
    )

    plt.colorbar(
        label="Concentration"
    )

    plt.xlabel("mm")
    plt.ylabel("mm")

    plt.title(
        "Nanoparticle Distribution"
    )

    plt.savefig(
        "results/transport/nanoparticle_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# DRUG MAP
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
        levels=20
    )

    plt.colorbar(
        label="Drug Concentration"
    )

    plt.xlabel("mm")
    plt.ylabel("mm")

    plt.title(
        "Drug Distribution"
    )

    plt.savefig(
        "results/drug/drug_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# NANOPARTICLE HISTORY
# =====================================================

def plot_concentration_history(
    history
):

    plt.figure(figsize=(8, 6))

    plt.plot(history)

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Average Concentration"
    )

    plt.title(
        "Nanoparticle Concentration vs Time"
    )

    plt.grid(True)

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

    plt.figure(figsize=(8, 6))

    plt.plot(history)

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Drug Concentration"
    )

    plt.title(
        "Drug Release Curve"
    )

    plt.grid(True)

    plt.savefig(
        "results/drug/drug_release_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# TUMOR GROWTH
# =====================================================

def plot_tumor_growth(
    tumor_history
):

    plt.figure(figsize=(8, 6))

    plt.plot(
        tumor_history
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Tumor Cells"
    )

    plt.title(
        "Tumor Growth Response"
    )

    plt.grid(True)

    plt.savefig(
        "results/tumor/tumor_growth_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# RESULT 1
# PENETRATION DEPTH
# =====================================================

def plot_penetration_depth(
    particle_sizes,
    penetration_depths
):

    plt.figure(figsize=(8, 6))

    plt.plot(
        particle_sizes,
        penetration_depths,
        marker="o"
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

        plt.plot(
            history,
            label=label
        )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Drug Concentration"
    )

    plt.title(
        "Drug Release Comparison"
    )

    plt.legend()

    plt.grid(True)

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

        plt.plot(
            history,
            label=label
        )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Tumor Cells"
    )

    plt.title(
        "Tumor Response Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "results/result2/tumor_response_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()