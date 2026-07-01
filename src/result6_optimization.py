import numpy as np

from src.parameters import *
from src.transport_model import solve_transport
from src.drug_release import solve_drug_release
from src.tumor_growth import solve_tumor_growth
from src.visualization import (
    plot_design_heatmap,
    plot_design_surface,
    plot_line
)
from src.utils import get_logger, write_csv_rows


EPSILON = 1e-12
MAX_SIMULATIONS = 60
logger = get_logger(__name__)


def _diffusion_stability_number(diffusion_coefficient: float) -> float:
    """Calculate the explicit diffusion stability number.

    Parameters: diffusion coefficient in m^2/s. Returns: dimensionless
    stability number. Units: m^2/s and dimensionless.
    """

    return (
        diffusion_coefficient
        * dt
        * (
            1.0 / dx**2
            +
            1.0 / dy**2
        )
    )


def _is_stable_diffusion(diffusion_coefficient: float) -> bool:
    """Check whether a diffusion coefficient satisfies the stability limit.

    Parameters: diffusion coefficient in m^2/s. Returns: stability flag.
    Units: m^2/s.
    """

    return _diffusion_stability_number(diffusion_coefficient) <= 0.5


def compute_objective_score(
    average_drug,
    final_tumor,
    max_average_drug,
    max_final_tumor,
    drug_weight=0.7,
    tumor_weight=0.3
) -> float:
    """Compute the optimization objective score.

    Parameters: average drug, final tumor, normalization values, and weights.
    Returns: clipped objective score. Units: model-native normalized score.
    """

    values = [
        average_drug,
        final_tumor,
        max_average_drug,
        max_final_tumor
    ]

    if not np.all(np.isfinite(values)):
        return -1.0

    drug_score = average_drug / (max_average_drug + EPSILON)
    tumor_score = final_tumor / (max_final_tumor + EPSILON)

    objective = (
        drug_weight * drug_score
        -
        tumor_weight * tumor_score
    )

    if not np.isfinite(objective):
        return -1.0

    return float(np.clip(objective, -1.0, 1.0))


def _candidate_key(
    particle_size,
    release_rate_value,
    diffusion_factor
):
    """Build a stable cache key for one design candidate.

    Parameters: particle size in nm, release rate in h^-1, and diffusion
    factor. Returns: hashable cache key. Units: nm, h^-1, dimensionless.
    """

    return (
        int(round(particle_size)),
        round(float(release_rate_value), 5),
        round(float(diffusion_factor), 5)
    )


def _evaluate_candidate(
    particle_size,
    release_rate_value,
    diffusion_factor,
    Vx,
    Vy,
    candidate_cache,
    transport_cache,
    drug_cache
):
    """Evaluate or fetch one optimization candidate.

    Parameters: design variables, velocity arrays, and caches. Returns: result
    dictionary for the candidate. Units: nm, h^-1, m^2/s, and cell count.
    """

    key = _candidate_key(
        particle_size,
        release_rate_value,
        diffusion_factor
    )

    if key in candidate_cache:
        return candidate_cache[key]

    effective_diffusion = (
        D
        * (100.0 / particle_size)
        * diffusion_factor
    )

    if not _is_stable_diffusion(effective_diffusion):
        logger.warning(
            "Skipping unstable design: size=%s nm, release_rate=%.3f h^-1, "
            "diffusion_factor=%.3f, D=%.3e",
            particle_size,
            release_rate_value,
            diffusion_factor,
            effective_diffusion,
        )

        result = {
            "Particle Size": int(round(particle_size)),
            "Release Rate": float(release_rate_value),
            "Diffusion Factor": float(diffusion_factor),
            "Diffusion Coefficient": float(effective_diffusion),
            "Average Drug": 0.0,
            "Final Tumor": float(carrying_capacity),
            "Score": -1.0,
            "Rejected": True
        }

        candidate_cache[key] = result

        return result

    logger.info(
        "Simulating size=%s nm, release_rate=%.3f h^-1, diffusion_factor=%.3f",
        particle_size,
        release_rate_value,
        diffusion_factor,
    )

    diffusion_key = round(
        float(effective_diffusion),
        15
    )

    if diffusion_key not in transport_cache:
        C_np, _ = solve_transport(
            Vx,
            Vy,
            diffusion_coefficient=effective_diffusion
        )
        transport_cache[diffusion_key] = C_np
    else:
        C_np = transport_cache[diffusion_key]

    drug_key = (
        diffusion_key,
        round(float(release_rate_value), 5)
    )

    if drug_key not in drug_cache:
        Drug, _, _ = solve_drug_release(
            C_np,
            release_constant=release_rate_value
        )
        drug_cache[drug_key] = Drug
    else:
        Drug = drug_cache[drug_key]

    tumor_history = solve_tumor_growth(Drug)

    average_drug = float(np.mean(Drug))
    final_tumor = float(tumor_history[-1])

    if not np.isfinite(average_drug):
        average_drug = 0.0

    if not np.isfinite(final_tumor):
        final_tumor = carrying_capacity

    result = {
        "Particle Size": int(round(particle_size)),
        "Release Rate": float(release_rate_value),
        "Diffusion Factor": float(diffusion_factor),
        "Diffusion Coefficient": float(effective_diffusion),
        "Average Drug": average_drug,
        "Final Tumor": final_tumor,
        "Score": -1.0,
        "Rejected": False
    }

    candidate_cache[key] = result

    return result


def _score_candidates(candidates: list[dict[str, float]]) -> list[dict[str, float]]:
    """Score and rank optimization candidates.

    Parameters: candidate dictionaries. Returns: candidates sorted by score.
    Units: model-native normalized score.
    """

    finite_drug = [
        c["Average Drug"]
        for c in candidates
        if np.isfinite(c["Average Drug"])
    ]

    finite_tumor = [
        c["Final Tumor"]
        for c in candidates
        if np.isfinite(c["Final Tumor"])
    ]

    max_average_drug = max(finite_drug) if finite_drug else EPSILON
    max_final_tumor = max(finite_tumor) if finite_tumor else EPSILON

    for candidate in candidates:
        candidate["Score"] = compute_objective_score(
            candidate["Average Drug"],
            candidate["Final Tumor"],
            max_average_drug,
            max_final_tumor
        )

    return sorted(
        candidates,
        key=lambda c: c["Score"],
        reverse=True
    )


def _bounded_values(
    center,
    step,
    lower,
    upper,
    decimals=None
):
    """Generate bounded refinement values around a center.

    Parameters: center, step, lower/upper bounds, and optional decimal places.
    Returns: unique bounded values. Units: same as center.
    """

    values = [
        center - step,
        center,
        center + step
    ]

    bounded = []

    for value in values:
        value = min(
            upper,
            max(
                lower,
                value
            )
        )

        if decimals is None:
            value = int(round(value))
        else:
            value = round(float(value), decimals)

        if value not in bounded:
            bounded.append(value)

    return bounded


def _fill_missing_scores(matrix: np.ndarray) -> np.ndarray:
    """Replace missing heatmap scores with the finite minimum score.

    Parameters: score matrix. Returns: filled score matrix. Units: objective
    score.
    """

    filled = matrix.copy()

    finite_values = filled[np.isfinite(filled)]

    if finite_values.size == 0:
        return np.zeros_like(filled)

    fallback = float(np.min(finite_values))
    filled[~np.isfinite(filled)] = fallback

    return filled


def _build_heatmap(
    candidates: list[dict[str, float]],
) -> tuple[list[int], list[float], np.ndarray]:
    """Build the optimization heatmap axes and score matrix.

    Parameters: ranked candidates. Returns: particle sizes, release rates, and
    heatmap score matrix. Units: nm, h^-1, objective score.
    """

    particle_sizes = sorted(
        {
            c["Particle Size"]
            for c in candidates
        }
    )

    release_rates = sorted(
        {
            c["Release Rate"]
            for c in candidates
        }
    )

    heatmap_scores = np.full(
        (
            len(particle_sizes),
            len(release_rates)
        ),
        -np.inf
    )

    size_index = {
        value: index
        for index, value in enumerate(particle_sizes)
    }

    release_index = {
        value: index
        for index, value in enumerate(release_rates)
    }

    for candidate in candidates:
        i = size_index[candidate["Particle Size"]]
        j = release_index[candidate["Release Rate"]]
        heatmap_scores[i, j] = max(
            heatmap_scores[i, j],
            candidate["Score"]
        )

    return (
        particle_sizes,
        release_rates,
        _fill_missing_scores(heatmap_scores)
    )


def run_optimization_study(
    x,
    Vx,
    Vy
):
    """Run Result 6: hierarchical adaptive nanoparticle design search.

    Parameters
    ----------
    x : ndarray
        Spatial x-coordinate vector retained for API compatibility.
    Vx, Vy : ndarray
        Darcy velocity field components.

    Returns
    -------
    dict
        Heatmap scores, top candidates, and simulation count.

    Units
    -----
    Particle size is nm. Release rate is h^-1. Diffusion is m^2/s.
    """

    _ = x
    logger.info("\nRunning Result 6...")
    logger.info("Optimal Nanoparticle Design")

    cache = {}
    transport_cache = {}
    drug_cache = {}
    candidates = []

    stage1_particle_sizes = [20, 60, 100]
    stage1_release_rates = [0.01, 0.05, 0.09]
    stage1_diffusion_factors = [0.8, 1.0, 1.2]

    search_batches = [
        (
            stage1_particle_sizes,
            stage1_release_rates,
            stage1_diffusion_factors
        )
    ]

    previous_best_score = None
    best_candidate = None
    refinement_iteration = 0

    while search_batches:
        (
            particle_batch,
            release_batch,
            diffusion_batch
        ) = search_batches.pop(0)

        for particle_size in particle_batch:
            for release_rate_value in release_batch:
                for diffusion_factor in diffusion_batch:
                    if len(cache) >= MAX_SIMULATIONS:
                        break

                    candidate = _evaluate_candidate(
                        particle_size,
                        release_rate_value,
                        diffusion_factor,
                        Vx,
                        Vy,
                        cache,
                        transport_cache,
                        drug_cache
                    )

                    if candidate not in candidates:
                        candidates.append(candidate)

                if len(cache) >= MAX_SIMULATIONS:
                    break

            if len(cache) >= MAX_SIMULATIONS:
                break

        ranked_candidates = _score_candidates(candidates)
        best_candidate = ranked_candidates[0]
        best_score = best_candidate["Score"]

        if previous_best_score is not None:
            improvement = (
                best_score - previous_best_score
            ) / (
                abs(previous_best_score) + EPSILON
            )

            if improvement < 0.01:
                break

        previous_best_score = best_score

        if refinement_iteration >= 2:
            break

        if len(cache) >= MAX_SIMULATIONS:
            break

        refinement_iteration += 1

        size_step = max(
            5,
            20 // refinement_iteration
        )

        release_step = max(
            0.005,
            0.02 / refinement_iteration
        )

        diffusion_step = max(
            0.025,
            0.2 / refinement_iteration
        )

        search_batches.append(
            (
                _bounded_values(
                    best_candidate["Particle Size"],
                    size_step,
                    20,
                    120
                ),
                _bounded_values(
                    best_candidate["Release Rate"],
                    release_step,
                    0.01,
                    0.09,
                    decimals=3
                ),
                _bounded_values(
                    best_candidate["Diffusion Factor"],
                    diffusion_step,
                    0.8,
                    1.2,
                    decimals=3
                )
            )
        )

    ranked_candidates = _score_candidates(candidates)
    top10 = ranked_candidates[:10]

    (
        particle_sizes,
        release_rates,
        heatmap_scores
    ) = _build_heatmap(ranked_candidates)

    plot_design_heatmap(
        heatmap_scores,
        particle_sizes,
        release_rates
    )

    X, Y = np.meshgrid(
        particle_sizes,
        release_rates
    )

    plot_design_surface(
        X,
        Y,
        heatmap_scores.T
    )

    objective_vs_size = {
        size: float(np.max(heatmap_scores[i, :]))
        for i, size in enumerate(particle_sizes)
    }

    objective_vs_release = {
        release_rate_value: float(np.max(heatmap_scores[:, j]))
        for j, release_rate_value in enumerate(release_rates)
    }

    plot_line(
        particle_sizes,
        list(objective_vs_size.values()),
        xlabel="Particle Size (nm)",
        ylabel="Objective Score",
        title="Objective Score vs Particle Size",
        filename="results/result6/objective_vs_size.png"
    )

    plot_line(
        release_rates,
        list(objective_vs_release.values()),
        xlabel="Release Rate (h^-1)",
        ylabel="Objective Score",
        title="Objective Score vs Release Rate",
        filename="results/result6/objective_vs_release.png"
    )

    fieldnames = [
        "Rank",
        "Particle Size (nm)",
        "Release Rate",
        "Diffusion Coefficient (m^2/s)",
        "Average Drug Concentration",
        "Final Tumor Volume",
        "Objective Score",
    ]
    write_csv_rows(
        "results/result6/top10.csv",
        fieldnames,
        [
            {
                "Rank": rank,
                "Particle Size (nm)": candidate["Particle Size"],
                "Release Rate": candidate["Release Rate"],
                "Diffusion Coefficient (m^2/s)": candidate["Diffusion Coefficient"],
                "Average Drug Concentration": candidate["Average Drug"],
                "Final Tumor Volume": candidate["Final Tumor"],
                "Objective Score": candidate["Score"],
            }
            for rank, candidate in enumerate(top10, start=1)
        ],
    )

    with open(
        "results/result6/summary.txt",
        "w"
    ) as txtfile:
        txtfile.write("Optimal Nanoparticle Design\n")
        txtfile.write("===========================\n\n")
        txtfile.write("Hierarchical adaptive optimization was used instead of full brute force.\n")
        txtfile.write(f"Total unique simulations: {len(cache)}\n")
        txtfile.write(f"Unique transport solves: {len(transport_cache)}\n")
        txtfile.write(f"Unique drug-release solves: {len(drug_cache)}\n")
        txtfile.write(f"Maximum allowed simulations: {MAX_SIMULATIONS}\n\n")
        txtfile.write("Objective score maximizes normalized average drug concentration\n")
        txtfile.write("and penalizes normalized final tumor volume.\n\n")
        txtfile.write("Equation:\n")
        txtfile.write(
            "Score = 0.7 * (Average Drug / (max(Average Drug) + epsilon)) - "
        )
        txtfile.write(
            "0.3 * (Final Tumor / (max(Final Tumor) + epsilon))\n\n"
        )

        txtfile.write("Top 10 designs:\n")
        for rank, candidate in enumerate(top10, start=1):
            txtfile.write(
                f"{rank}. Size={candidate['Particle Size']} nm, "
            )
            txtfile.write(
                f"Release={candidate['Release Rate']:.3f} h^-1, "
            )
            txtfile.write(
                f"D={candidate['Diffusion Coefficient']:.3e}, "
            )
            txtfile.write(
                f"Average Drug={candidate['Average Drug']:.6f}, "
            )
            txtfile.write(
                f"Final Tumor={candidate['Final Tumor']:.2f}, "
            )
            txtfile.write(
                f"Score={candidate['Score']:.6f}\n"
            )

    with open(
        "results/result6/optimal_design.txt",
        "w"
    ) as txtfile:
        txtfile.write("Optimal Design Summary\n")
        txtfile.write("======================\n\n")

        if top10:
            best = top10[0]
            txtfile.write(
                f"Best design: Size={best['Particle Size']} nm, "
            )
            txtfile.write(
                f"Release Rate={best['Release Rate']:.3f} h^-1, "
            )
            txtfile.write(
                f"Diffusion={best['Diffusion Coefficient']:.3e}.\n"
            )
            txtfile.write(
                f"Objective Score={best['Score']:.6f}\n"
            )
            txtfile.write(
                f"Unique simulations used={len(cache)}\n"
            )

    logger.info("\nResult 6 complete")

    return {
        "heatmap_scores": heatmap_scores,
        "top10": top10,
        "simulation_count": len(cache)
    }
