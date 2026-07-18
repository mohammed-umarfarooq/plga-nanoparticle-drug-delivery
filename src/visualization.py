from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.parameters import dt
from src.utils import RESULTS_DIR, ensure_directory, ensure_parent_directory, validate_array

# =====================================================
# GLOBAL PLOT SETTINGS
# =====================================================

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10

# =====================================================
# CREATE RESULT FOLDERS
# =====================================================

def create_folders() -> None:
    """Create the result directory tree.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Units
    -----
    Not applicable.
    """

    folders = [

        RESULTS_DIR,
        RESULTS_DIR / "pressure",
        RESULTS_DIR / "transport",
        RESULTS_DIR / "drug",
        RESULTS_DIR / "tumor",
        RESULTS_DIR / "result1",
        RESULTS_DIR / "result2",
        RESULTS_DIR / "result3",
        RESULTS_DIR / "result4",
        RESULTS_DIR / "result5",
        RESULTS_DIR / "result6",

    ]

    for folder in folders:

        ensure_directory(folder)

# =====================================================
# GENERIC FIGURE INITIALIZER
# =====================================================

def create_figure(
    width: float = 8,
    height: float = 6,
) -> plt.Figure:
    """Create a matplotlib figure with consistent sizing.

    Parameters
    ----------
    width : float
        Figure width.
    height : float
        Figure height.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure.

    Units
    -----
    Inches.
    """

    fig = plt.figure(
        figsize=(width, height)
    )

    return fig


# =====================================================
# GENERIC SAVE FUNCTION
# =====================================================

def save_plot(
    filename: str | Path,
) -> None:
    """Save and close the current plot.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output image path.

    Returns
    -------
    None

    Units
    -----
    Image resolution is 300 dpi.
    """

    plt.tight_layout()

    output_path = ensure_parent_directory(filename)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# GENERIC LINE PLOT
# =====================================================

def plot_line(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str | Path,
    marker: str = "o",
    grid: bool = True,
) -> None:
    """Plot a single line series.

    Parameters
    ----------
    x, y : sequence of float
        X and Y values.
    xlabel, ylabel, title : str
        Plot labels.
    filename : str or pathlib.Path
        Output image path.
    marker : str
        Matplotlib marker style.
    grid : bool
        Whether to show a grid.

    Returns
    -------
    None

    Units
    -----
    Defined by the caller-provided labels.
    """

    create_figure()

    plt.plot(
        x,
        y,
        marker=marker,
        linewidth=2
    )

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    if grid:

        plt.grid(True)

    save_plot(filename)


# =====================================================
# GENERIC HEATMAP
# =====================================================

def plot_heatmap(
    matrix: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str | Path,
    x_ticks: Sequence[float] | None = None,
    y_ticks: Sequence[float] | None = None,
    colorbar_label: str = "Value",
) -> None:
    """Plot a heatmap without modifying the underlying data.

    Parameters
    ----------
    matrix : ndarray
        Heatmap values.
    xlabel, ylabel, title : str
        Plot labels.
    filename : str or pathlib.Path
        Output image path.
    x_ticks, y_ticks : sequence of float, optional
        Tick labels.
    colorbar_label : str
        Colorbar label.

    Returns
    -------
    None

    Units
    -----
    Defined by the caller-provided labels.
    """

    validate_array(matrix, "matrix")

    create_figure()

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto"
    )

    if x_ticks is not None:

        plt.xticks(
            np.arange(len(x_ticks)),
            x_ticks
        )

    if y_ticks is not None:

        plt.yticks(
            np.arange(len(y_ticks)),
            y_ticks
        )

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    cbar = plt.colorbar(im)

    cbar.set_label(colorbar_label)

    save_plot(filename)


# =====================================================
# GENERIC CONTOUR PLOT
# =====================================================

def plot_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    colorbar_label: str,
    filename: str | Path,
    levels: int = 40,
) -> None:
    """Plot a filled contour field.

    Parameters
    ----------
    X, Y, Z : ndarray
        Mesh coordinates and field values.
    xlabel, ylabel, title, colorbar_label : str
        Plot labels.
    filename : str or pathlib.Path
        Output image path.
    levels : int
        Number of contour levels.

    Returns
    -------
    None

    Units
    -----
    Defined by the caller-provided labels.
    """

    validate_array(X, "X")
    validate_array(Y, "Y")
    validate_array(Z, "Z")

    create_figure()

    contour = plt.contourf(
        X,
        Y,
        Z,
        levels=levels
    )

    plt.colorbar(
        contour,
        label=colorbar_label
    )

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(True, alpha=0.2)

    save_plot(filename)


# =====================================================
# TORNADO CHART
# =====================================================

def plot_tornado(
    labels: Sequence[str],
    values: Sequence[float],
    filename: str | Path,
) -> None:
    """Plot a tornado sensitivity chart.

    Parameters
    ----------
    labels : sequence of str
        Parameter labels.
    values : sequence of float
        Sensitivity values.
    filename : str or pathlib.Path
        Output image path.

    Returns
    -------
    None

    Units
    -----
    Sensitivity index in percent.
    """

    order = np.argsort(
        np.abs(values)
    )[::-1]

    labels = np.array(labels)[order]

    values = np.array(values)[order]

    fig = create_figure(width=10, height=6)

    bars = plt.barh(
        labels,
        values,
        color="#1f77b4",
        edgecolor="black"
    )

    plt.axvline(
        0,
        color="black",
        linewidth=1.0
    )

    plt.xlabel(
        "Sensitivity Index (%)"
    )

    plt.title(
        "Sensitivity Analysis (Tornado Chart)"
    )

    plt.grid(axis="x", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + np.sign(width) * 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=9
        )

    fig.subplots_adjust(left=0.28)

    save_plot(filename)

    # =====================================================
# PRESSURE CONTOUR
# =====================================================

def plot_pressure(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
) -> None:
    """Plot interstitial pressure in Pa on a millimeter grid."""

    plot_contour(
        X * 1000,
        Y * 1000,
        P,
        xlabel="X Position (mm)",
        ylabel="Y Position (mm)",
        title="Interstitial Fluid Pressure",
        colorbar_label="Pressure (Pa)",
        filename="results/pressure/pressure_contour.png",
        levels=40
    )


# =====================================================
# VELOCITY FIELD
# =====================================================

def plot_velocity(
    X: np.ndarray,
    Y: np.ndarray,
    Vx: np.ndarray,
    Vy: np.ndarray,
) -> None:
    """Plot interstitial velocity magnitude and direction in m/s."""

    create_figure()

    velocity = np.sqrt(
        Vx**2 + Vy**2
    )

    plt.contourf(
        X * 1000,
        Y * 1000,
        velocity,
        levels=30
    )

    plt.colorbar(
        label="Velocity Magnitude (m/s)"
    )

    vmax = np.max(velocity)

    if vmax > 0:

        plt.quiver(
            X[::2, ::2] * 1000,
            Y[::2, ::2] * 1000,
            Vx[::2, ::2] / vmax,
            Vy[::2, ::2] / vmax,
            scale=25
        )

    plt.xlabel(
        "X Position (mm)"
    )

    plt.ylabel(
        "Y Position (mm)"
    )

    plt.title(
        "Interstitial Fluid Velocity"
    )

    save_plot(
        "results/pressure/velocity_field.png"
    )


# =====================================================
# NANOPARTICLE DISTRIBUTION
# =====================================================

def plot_transport(
    X: np.ndarray,
    Y: np.ndarray,
    C_np: np.ndarray,
) -> None:
    """Plot final nanoparticle concentration over the tissue domain."""

    create_figure()

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

    save_plot(
        "results/transport/nanoparticle_distribution.png"
    )


# =====================================================
# DRUG DISTRIBUTION
# =====================================================

def plot_drug(
    X: np.ndarray,
    Y: np.ndarray,
    Drug: np.ndarray,
) -> None:
    """Plot final released drug concentration over the tissue domain."""

    create_figure()

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

    save_plot(
        "results/drug/drug_distribution.png"
    )


# =====================================================
# NANOPARTICLE CONCENTRATION HISTORY
# =====================================================

def plot_concentration_history(
    history: Sequence[float],
) -> None:
    """Plot total nanoparticle concentration history versus time in hours."""

    time = np.arange(
        len(history)
    ) * dt

    create_figure()

    plt.plot(
        time,
        history,
        linewidth=2
    )

    peak = np.argmax(
        history
    )

    plt.scatter(
        time[peak],
        history[peak],
        s=60
    )

    plt.annotate(
        f"Tmax={time[peak]:.1f} h",
        (
            time[peak],
            history[peak]
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

    save_plot(
        "results/transport/concentration_curve.png"
    )


# =====================================================
# DRUG RELEASE HISTORY
# =====================================================

def plot_drug_history(
    history: Sequence[float],
) -> None:
    """Plot mean drug concentration history versus time in hours."""

    time = np.arange(
        len(history)
    ) * dt

    create_figure()

    plt.plot(
        time,
        history,
        linewidth=2
    )

    peak = np.argmax(
        history
    )

    plt.scatter(
        time[peak],
        history[peak],
        s=60
    )

    plt.annotate(
        f"Tmax={time[peak]:.1f} h",
        (
            time[peak],
            history[peak]
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

    save_plot(
        "results/drug/drug_release_curve.png"
    )


# =====================================================
# TUMOR GROWTH RESPONSE
# =====================================================

def plot_tumor_growth(
    tumor_history: Sequence[float],
) -> None:
    """Plot tumor-cell population history versus time in hours."""

    time = np.arange(
        len(tumor_history)
    ) * dt

    create_figure()

    plt.plot(
        time,
        tumor_history,
        linewidth=2
    )

    peak = np.argmax(
        tumor_history
    )

    plt.scatter(
        time[peak],
        tumor_history[peak],
        s=60
    )

    plt.annotate(
        f"Peak={time[peak]:.1f} h",
        (
            time[peak],
            tumor_history[peak]
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

    save_plot(
        "results/tumor/tumor_growth_curve.png"
    )

    # =====================================================
# RESULT 1
# PENETRATION DEPTH COMPARISON
# =====================================================

def plot_penetration_depth(
    particle_sizes: Sequence[float],
    penetration_depths: Sequence[float],
) -> None:
    """Plot penetration depth in mm against particle size in nm."""

    create_figure()

    plt.plot(
        particle_sizes,
        penetration_depths,
        marker="o",
        linewidth=2,
        markersize=7
    )

    for x, y in zip(
        particle_sizes,
        penetration_depths
    ):

        plt.text(
            x,
            y + 0.02 * max(penetration_depths),
            f"{y:.2f}",
            ha="center",
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
        "Effect of Particle Size on Penetration Depth"
    )

    plt.grid(True)

    save_plot(
        "results/result1/penetration_depth_vs_size.png"
    )


# =====================================================
# RESULT 2
# DRUG RELEASE COMPARISON
# =====================================================

def plot_release_comparison(
    release_histories: Mapping[str, Sequence[float]],
) -> None:
    """Plot drug-release histories for release-rate cases."""

    create_figure()

    labels = {
        "Slow": "Slow (k=0.02)",
        "Medium": "Medium (k=0.05)",
        "Fast": "Fast (k=0.10)"
    }

    for key, history in release_histories.items():

        time = np.arange(
            len(history)
        ) * dt

        plt.plot(
            time,
            history,
            linewidth=2,
            label=labels[key]
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

    plt.grid(True)

    plt.legend()

    save_plot(
        "results/result2/drug_release_comparison.png"
    )


# =====================================================
# RESULT 2
# TUMOR RESPONSE COMPARISON
# =====================================================

def plot_tumor_comparison(
    tumor_results: Mapping[str, Sequence[float]],
) -> None:
    """Plot tumor response histories for release-rate cases."""

    create_figure()

    for label, history in tumor_results.items():

        time = np.arange(
            len(history)
        ) * dt

        plt.plot(
            time,
            history,
            linewidth=2,
            label=label
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

    plt.grid(True)

    plt.legend()

    save_plot(
        "results/result2/tumor_response_comparison.png"
    )


# RESULT 3
# IFP STUDY
# =====================================================

def plot_ifp_drug_concentration(
    ifp_values: Sequence[float],
    drug_values: Sequence[float],
) -> None:
    """Plot average drug concentration against IFP in mmHg."""

    plot_line(
        ifp_values,
        drug_values,
        xlabel="Interstitial Fluid Pressure (mmHg)",
        ylabel="Average Drug Concentration (kg/m³)",
        title="Effect of Interstitial Fluid Pressure on Drug Concentration",
        filename="results/result3/drug_vs_ifp.png"
    )


def plot_ifp_penetration(
    ifp_values: Sequence[float],
    penetration_depths: Sequence[float],
) -> None:
    """Plot penetration depth in mm against IFP in mmHg."""

    plot_line(
        ifp_values,
        penetration_depths,
        xlabel="Interstitial Fluid Pressure (mmHg)",
        ylabel="Penetration Depth (mm)",
        title="Effect of Interstitial Fluid Pressure on Penetration Depth",
        filename="results/result3/penetration_vs_ifp.png"
    )


def plot_ifp_contour(
    X: np.ndarray,
    Y: np.ndarray,
    concentration: np.ndarray,
    pressure_mmHg: float,
) -> None:
    """Plot nanoparticle concentration for one IFP value."""

    create_figure()

    plt.contourf(
        X*1000,
        Y*1000,
        concentration,
        levels=40
    )

    plt.colorbar(
        label="Nanoparticle Concentration (kg/m³)"
    )

    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")

    plt.title(
        f"Nanoparticle Distribution\nIFP = {pressure_mmHg:.0f} mmHg"
    )

    save_plot(
        f"results/result3/ifp_{int(pressure_mmHg)}mmHg.png"
    )


# =====================================================
# RESULT 4
# DIFFUSION STUDY
# =====================================================

def plot_diffusion_penetration(
    diffusion_values: Sequence[float],
    penetration_depths: Sequence[float],
) -> None:
    """Plot penetration depth in mm against diffusion coefficient in m^2/s."""

    plot_line(
        diffusion_values,
        penetration_depths,
        xlabel="Diffusion Coefficient (m²/s)",
        ylabel="Penetration Depth (mm)",
        title="Diffusion Coefficient vs Penetration Depth",
        filename="results/result4/diffusion_vs_penetration.png"
    )


def plot_diffusion_average_concentration(
    diffusion_values: Sequence[float],
    average_concentration: Sequence[float],
) -> None:
    """Plot average nanoparticle concentration against diffusion coefficient."""

    plot_line(
        diffusion_values,
        average_concentration,
        xlabel="Diffusion Coefficient (m²/s)",
        ylabel="Average Nanoparticle Concentration",
        title="Average Concentration vs Diffusion Coefficient",
        filename="results/result4/diffusion_vs_average.png"
    )


def plot_radial_profiles(
    radius: Sequence[float],
    profiles: Mapping[str, Sequence[float]],
) -> None:
    """Plot normalized concentration profiles against radius in mm."""

    create_figure()

    for label, profile in profiles.items():

        plt.plot(
            radius*1000,
            profile,
            linewidth=2,
            label=label
        )

    plt.xlabel(
        "Radius (mm)"
    )

    plt.ylabel(
        "Normalized Concentration"
    )

    plt.title(
        "Radial Concentration Profiles"
    )

    plt.grid(True)

    plt.legend()

    save_plot(
        "results/result4/radial_profiles.png"
    )


# =====================================================
# RESULT 5
# SENSITIVITY ANALYSIS
# =====================================================

def plot_spider(
    labels: Sequence[str],
    values: np.ndarray,
    filename: str | Path = "results/result5/spider.png",
) -> None:
    """Plot normalized sensitivity values on a polar spider chart."""

    angles = np.linspace(
        0,
        2*np.pi,
        len(labels),
        endpoint=False
    )

    values = np.concatenate(
        (values,[values[0]])
    )

    angles = np.concatenate(
        (angles,[angles[0]])
    )

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(
        111,
        polar=True
    )

    ax.plot(
        angles,
        values,
        linewidth=2,
        color="#1f77b4"
    )

    ax.fill(
        angles,
        values,
        alpha=0.25,
        color="#1f77b4"
    )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels,
        fontsize=10
    )

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    ax.grid(True, alpha=0.4)

    plt.title(
        "Normalized Parameter Sensitivity",
        fontsize=14
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_tumor_reduction(
    labels: Sequence[str],
    reductions: Sequence[float],
) -> None:
    """Plot tumor reduction percentage for sensitivity parameters."""

    sorted_indices = np.argsort(reductions)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_reductions = [reductions[i] for i in sorted_indices]

    fig = create_figure(width=10, height=6)

    bars = plt.bar(
        sorted_labels,
        sorted_reductions,
        color="#2ca02c",
        edgecolor="black"
    )

    plt.xlabel("Model Parameter")
    plt.ylabel("Tumor Reduction (%)")
    plt.title("Tumor Reduction Due to Parameter Variations")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    save_plot("results/result5/tumor_reduction.png")


# =====================================================
# RESULT 6
# OPTIMIZATION
# =====================================================

def plot_design_heatmap(
    matrix: np.ndarray,
    particle_sizes: Sequence[float],
    release_rates: Sequence[float],
) -> None:
    """Plot optimization score heatmap by size and release rate."""

    plot_heatmap(
        matrix,
        xlabel="Release Rate (h⁻¹)",
        ylabel="Particle Size (nm)",
        title="Optimization Heatmap",
        filename="results/result6/heatmap.png",
        x_ticks=release_rates,
        y_ticks=particle_sizes,
        colorbar_label="Objective Score"
    )


def plot_design_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> None:
    """Plot optimization score as a 3D surface."""

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
        linewidth=0,
        rcount=100,
        ccount=100
    )

    ax.set_xlabel("Particle Size (nm)")
    ax.set_ylabel("Release Rate (h⁻¹)")
    ax.set_zlabel("Objective Score")
    ax.set_title("Optimization Surface")
    ax.view_init(elev=30, azim=-60)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label("Objective Score")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "results/result6/surface3d.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


