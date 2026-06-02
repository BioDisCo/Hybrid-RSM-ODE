"""Parameter tuning/fitting for plate data."""

import argparse
import glob
import logging
import pathlib
import sys
from typing import cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import curve_fit

import data_import
from data_import import (
    Experiments,
    set_min,
    shift_by_time,
    until_time,
)

from growth_models import (
    steady_state_haldane,
    growth_rate_haldane_light_only,
    growth_rate_haldane_both,
    growth_rate_haldane_synergistic,
    infer_growth_rate_model,
    evaluate_growth_rate,
    evaluate_steady_state,
)

# Import growth rate analysis functions
sys.path.insert(0, str(pathlib.Path(__file__).parent / "analysis_scripts"))
from growth_rate_analysis import calculate_specific_growth_rate
from data_import import trimmed_mean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]

# Configuration parameters
MAX_TIME_HOURS = 200  # Maximum time to display in plots (hours)
L0_REF = 170.0  # Reference light intensity (µmol m⁻² s⁻¹) - for converting L0_factor to actual L0


def plot_steady_state_analysis(
    experiments: Experiments,
    output_dir: str = "results_plates",
    monod_params: tuple | None = None,
) -> None:
    """Analyze and plot steady state concentrations from plate experiments.

    Creates three figures:
    1. Steady state vs nutrient concentration for each light intensity
    2. Grid showing data with horizontal steady state lines (and optionally fitted model)
    3. Interactive 3D surface of steady state vs C0 and L0

    Args:
        experiments: Dictionary of experimental data
        output_dir: Directory to save output files
        monod_params: Optional tuple of (n_max_ref, k_c, k_l, alpha) for plotting fitted model
    """
    # Group experiments by (C0_factor, L0_factor) and calculate steady states
    steady_states: dict[tuple[float, float], float] = {}
    steady_state_data: dict[tuple[float, float], dict] = {}

    for exp_name, exp_data in experiments.items():
        c0 = exp_data["C0_factor"]
        l0 = exp_data["L0_factor"]
        key = (c0, l0)

        if key not in steady_state_data:
            steady_state_data[key] = exp_data

        # Calculate steady state as mean of last 10% of time points
        concentration = np.array(exp_data["Mean"])
        n_points = len(concentration)
        steady_start_idx = int(0.9 * n_points)

        ss_value = np.nan

        if steady_start_idx < n_points:
            ss_value = np.nanmean(concentration[steady_start_idx:])

        # If mean gives NaN or invalid value, try using all replicates
        if np.isnan(ss_value) and exp_data["replicates"]:
            # Collect last 10% of each replicate
            all_ss_values = []
            for rep in exp_data["replicates"]:
                rep_conc = np.array(rep["Value"])
                rep_n = len(rep_conc)
                rep_ss_idx = int(0.9 * rep_n)
                if rep_ss_idx < rep_n:
                    rep_ss = np.nanmean(rep_conc[rep_ss_idx:])
                    if not np.isnan(rep_ss) and rep_ss > 0:
                        all_ss_values.append(rep_ss)

            if all_ss_values:
                # Use trimmed mean (drop min and max) for fault tolerance
                ss_value = trimmed_mean(all_ss_values)

        if not np.isnan(ss_value) and ss_value > 0:
            steady_states[key] = ss_value
            logger.info(f"C0×{c0:.3f}, L0×{l0:.3f}: SS = {ss_value:.2e}")
        else:
            logger.debug(f"Could not estimate steady state: C0×{c0:.3f}, L0×{l0:.3f}")

    if not steady_states:
        logger.warning("No steady states calculated")
        return

    # Organize data for plotting
    unique_l0 = sorted(set(l0 for c0, l0 in steady_states.keys()))
    unique_c0 = sorted(set(c0 for c0, l0 in steady_states.keys()), reverse=True)

    # ===== FIGURE: Steady State Grid with Horizontal Lines =====
    n_c0 = len(unique_c0)
    n_l0 = len(unique_l0)

    fig, axes = plt.subplots(n_c0, n_l0, figsize=(4 * n_l0, 4 * n_c0))

    # Ensure axes is always 2D
    if n_c0 == 1:
        axes = axes.reshape(1, -1)
    if n_l0 == 1:
        axes = axes.reshape(-1, 1)

    for i, c0 in enumerate(unique_c0):
        for j, l0 in enumerate(unique_l0):
            ax = axes[i, j]

            if (c0, l0) in steady_state_data:
                exp_data = steady_state_data[(c0, l0)]
                time = np.array(exp_data["Time"])
                concentration = np.array(exp_data["Mean"])

                # Plot all replicates (log scale)
                for replicate in exp_data["replicates"]:
                    ax.semilogy(
                        replicate["Time"],
                        replicate["Value"],
                        alpha=0.6,
                        linewidth=1,
                        color="gray",
                    )

                # Plot mean (log scale)
                ax.semilogy(time, concentration, "b-", linewidth=2, label="Mean")

                # Calculate and plot steady state as horizontal line
                if (c0, l0) in steady_states:
                    ss_value = steady_states[(c0, l0)]
                    ax.axhline(
                        ss_value,
                        color="r",
                        linestyle="--",
                        linewidth=2.5,
                        label=f"SS = {ss_value:.2e}",
                    )

                # Plot fitted Monod model prediction if available
                if monod_params is not None and (c0, l0) in steady_states:
                    n_max_ref, k_c, k_l, k_i = monod_params
                    # Convert L0 factor to actual L0 value for Monod model evaluation
                    l0_actual = l0 * L0_REF
                    # Calculate predicted steady state using Monod product with Haldane photoinhibition
                    ss_pred = steady_state_haldane(
                        (c0, l0_actual), n_max_ref, k_c, k_l, k_i
                    )
                    ax.axhline(
                        ss_pred,
                        color="orange",
                        linestyle=":",
                        linewidth=2,
                        label=f"Fit = {ss_pred:.2e}",
                        alpha=0.8,
                    )

                ax.set_title(f"C0×{c0:.3f}, L0×{l0:.3f}")
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(fontsize=8)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j == 0:
                    ax.set_ylabel("Cell Conc. (log)", fontsize=10)
                if i == n_c0 - 1:
                    ax.set_xlabel("Time (h)", fontsize=10)

    fig.text(0.5, 0.02, "Time (h)", ha="center", fontsize=12)
    fig.text(
        0.02, 0.5, "Cell Concentration", va="center", rotation="vertical", fontsize=12
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    output_path_2 = pathlib.Path(output_dir) / "plates_steady_state_lines.png"
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    logger.info(f"Saved steady state grid to {output_path_2}")
    plt.close()

    # Save measured steady states to CSV file (calibration + extrapolation)
    import csv

    intermediate_data = load_intermediate_data()
    steady_states_path = pathlib.Path(output_dir) / "plates_steady_states_measured.csv"
    with open(steady_states_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["C0", "L0", "N_max", "type"])
        for (c0, l0), n_max in sorted(steady_states.items()):
            writer.writerow([f"{c0:.6f}", f"{l0:.6f}", f"{n_max:.6e}", "calibration"])
        if intermediate_data:
            for (c0, l0), (mu, n_max) in sorted(intermediate_data.items()):
                if (c0, l0) not in steady_states:
                    writer.writerow(
                        [f"{c0:.6f}", f"{l0:.6f}", f"{n_max:.6e}", "extrapolation"]
                    )
    logger.info(f"Saved measured steady states to {steady_states_path}")

    return steady_states


def plot_growth_rate_analysis(
    experiments: Experiments, output_dir: str = "results_plates"
) -> None:
    """Analyze and plot initial growth rates from plate experiments.

    Creates two figures:
    1. Growth rate vs nutrient concentration for each light intensity
    2. Grid of exponential fits showing data and model for quality assessment

    Args:
        experiments: Dictionary of experimental data
        output_dir: Directory to save output files
    """
    # Group experiments by (C0_factor, L0_factor)
    growth_data: dict[tuple[float, float], dict] = {}

    for exp_name, exp_data in experiments.items():
        c0 = exp_data["C0_factor"]
        l0 = exp_data["L0_factor"]
        key = (c0, l0)

        if key not in growth_data:
            growth_data[key] = exp_data

    # Calculate growth rates for each condition
    # Use fixed exponential phase window: 0-60 hours
    exp_start_fixed = 0.0
    exp_end_fixed = 60.0

    growth_rates: dict[tuple[float, float], float] = {}

    for (c0, l0), exp_data in growth_data.items():
        time = np.array(exp_data["Time"])
        concentration = np.array(exp_data["Mean"])

        try:
            # Calculate growth rate using fixed time window
            mu = calculate_specific_growth_rate(
                time, concentration, (exp_start_fixed, exp_end_fixed)
            )

            if not np.isnan(mu):
                growth_rates[(c0, l0)] = mu
                logger.info(f"C0×{c0:.3f}, L0×{l0:.3f}: μ = {mu:.4f} h⁻¹")
            else:
                logger.debug(
                    f"Could not calculate reliable growth rate for C0×{c0:.3f}, L0×{l0:.3f}"
                )
        except Exception as e:
            logger.debug(
                f"Error calculating growth rate for C0×{c0:.3f}, L0×{l0:.3f}: {e}"
            )

    if not growth_rates:
        logger.warning("No growth rates calculated")
        return

    # Organize data for plotting
    unique_l0 = sorted(set(l0 for c0, l0 in growth_rates.keys()))
    unique_c0 = sorted(set(c0 for c0, l0 in growth_rates.keys()), reverse=True)

    # ===== FIGURE: Exponential Fits Grid =====
    n_c0 = len(unique_c0)
    n_l0 = len(unique_l0)

    fig, axes = plt.subplots(n_c0, n_l0, figsize=(4 * n_l0, 4 * n_c0))

    # Ensure axes is always 2D
    if n_c0 == 1:
        axes = axes.reshape(1, -1)
    if n_l0 == 1:
        axes = axes.reshape(-1, 1)

    for i, c0 in enumerate(unique_c0):
        for j, l0 in enumerate(unique_l0):
            ax = axes[i, j]

            if (c0, l0) in growth_data:
                exp_data = growth_data[(c0, l0)]
                time = np.array(exp_data["Time"])
                concentration = np.array(exp_data["Mean"])

                # Plot all replicates (log scale)
                for replicate in exp_data["replicates"]:
                    ax.semilogy(
                        replicate["Time"],
                        replicate["Value"],
                        alpha=0.6,
                        linewidth=1,
                        color="gray",
                    )

                # Plot mean (log scale)
                ax.semilogy(time, concentration, "b-", linewidth=2, label="Mean")

                # Use fixed exponential phase window: 0-60 hours
                exp_start = exp_start_fixed
                exp_end = exp_end_fixed
                exp_mask = (time >= exp_start) & (time <= exp_end)

                if np.any(exp_mask):
                    # Highlight exponential phase
                    ax.axvspan(
                        exp_start,
                        exp_end,
                        alpha=0.1,
                        color="green",
                        label="Exponential phase (0-60h)",
                    )

                    # Fit exponential and plot
                    t_exp = time[exp_mask]
                    c_exp = concentration[exp_mask]

                    # Filter out non-positive values for logarithm
                    valid_log_mask = c_exp > 0
                    n_valid = np.sum(valid_log_mask)

                    # If mean has no positive values, try using all replicates
                    if n_valid == 0 and exp_data["replicates"]:
                        # Collect all replicate values in the window
                        all_rep_times = []
                        all_rep_concs = []
                        for rep in exp_data["replicates"]:
                            rep_time = np.array(rep["Time"])
                            rep_conc = np.array(rep["Value"])
                            rep_mask = (rep_time >= exp_start) & (rep_time <= exp_end)
                            if np.any(rep_mask):
                                all_rep_times.extend(rep_time[rep_mask])
                                all_rep_concs.extend(rep_conc[rep_mask])

                        if all_rep_times:
                            t_exp = np.array(all_rep_times)
                            c_exp = np.array(all_rep_concs)
                            valid_log_mask = c_exp > 0
                            n_valid = np.sum(valid_log_mask)

                    if n_valid >= 2:  # Need at least 2 points for a line
                        try:
                            t_exp_valid = t_exp[valid_log_mask]
                            c_exp_valid = c_exp[valid_log_mask]

                            # Linear regression on ln(concentration)
                            ln_c = np.log(c_exp_valid)
                            slope, intercept, r_value, _, _ = stats.linregress(
                                t_exp_valid, ln_c
                            )

                            # Plot exponential fit (log scale shows as straight line)
                            t_fit = np.linspace(exp_start, exp_end, 100)
                            c_fit = np.exp(intercept) * np.exp(slope * t_fit)
                            ax.semilogy(
                                t_fit,
                                c_fit,
                                "r--",
                                linewidth=2.5,
                                label=f"Fit (μ={slope:.3f})",
                            )
                        except Exception as e:
                            pass  # Silently skip if fit fails
                    else:
                        pass  # No fit possible for this condition

                ax.set_title(f"C0×{c0:.3f}, L0×{l0:.3f}")
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(fontsize=8)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j == 0:
                    ax.set_ylabel("Cell Conc. (log)", fontsize=10)
                if i == n_c0 - 1:
                    ax.set_xlabel("Time (h)", fontsize=10)

    fig.text(0.5, 0.02, "Time (h)", ha="center", fontsize=12)
    fig.text(
        0.02, 0.5, "Cell Concentration", va="center", rotation="vertical", fontsize=12
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    output_path_2 = pathlib.Path(output_dir) / "plates_exponential_fits.png"
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    logger.info(f"Saved exponential fits grid to {output_path_2}")
    plt.close()

    # Save measured growth rates to CSV file (calibration + extrapolation)
    import csv

    intermediate_data = load_intermediate_data()
    growth_rates_path = pathlib.Path(output_dir) / "plates_growth_rates_measured.csv"
    with open(growth_rates_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["C0", "L0", "mu_max", "type"])
        for (c0, l0), mu in sorted(growth_rates.items()):
            writer.writerow([f"{c0:.6f}", f"{l0:.6f}", f"{mu:.6e}", "calibration"])
        if intermediate_data:
            for (c0, l0), (mu, n_max) in sorted(intermediate_data.items()):
                if (c0, l0) not in growth_rates:
                    writer.writerow(
                        [f"{c0:.6f}", f"{l0:.6f}", f"{mu:.6e}", "extrapolation"]
                    )
    logger.info(f"Saved measured growth rates to {growth_rates_path}")

    return growth_rates


def calculate_r2_local(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score, handling NaN values.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score, or NaN if insufficient valid data
    """
    # Remove NaN values and keep only positive values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true > 0) & (y_pred > 0)
    if np.sum(mask) < 2:
        return np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - (ss_res / ss_tot)


def plot_simulation_extended(
    experiments: Experiments,
    growth_rates: dict[tuple[float, float], float],
    steady_states: dict[tuple[float, float], float],
    intermediate_data_timeseries: dict[tuple[float, float], dict],
    output_dir: str = "results_plates",
    monod_params: tuple | None = None,
    growth_rate_monod_params: tuple | None = None,
    manual_tlag_adjustments: dict | None = None,
) -> None:
    """Plot extended grid with all C0 values including intermediate dilutions.

    Creates a large grid showing:
    - Rows: L0 values (sorted low to high, bottom to top)
    - Columns: All C0 values including intermediate dilutions (sorted low to high, left to right)
    - Each subplot: gray replicates, blue mean, green "Both fits", and local R²

    The intermediate dilution data serves as VALIDATION data - they are plotted but were NOT
    used to calibrate the Monod models. The "Both fits" curve uses the Monod parameters
    calibrated on the main experimental data, evaluated at the intermediate (C0, L0) values.

    Args:
        experiments: Dictionary of experimental data (calibration set)
        growth_rates: Dict mapping (C0, L0) to initial growth rate μ_max
        steady_states: Dict mapping (C0, L0) to steady state concentration N_max
        intermediate_data_timeseries: Dict mapping (C0, L0) to dict with 'time', 'mean', 'replicates'
        output_dir: Directory to save output files
        monod_params: Tuple of (n_max_ref, k_c, k_l, alpha) for steady state predictions
        growth_rate_monod_params: Tuple for growth rate predictions (4 or 5 parameters)
        manual_tlag_adjustments: Optional dict {(L0_factor, C0_factor): t_lag_hours} for manual lag time adjustments
    """
    if monod_params is None or growth_rate_monod_params is None:
        logger.warning("Both Monod fits required for extended plot. Skipping.")
        return

    # Dictionary to store t_lag values for ALL conditions (keyed by (l0, c0))
    tlag_values_dict = {}

    # Collect all unique C0 and L0 values from both experiments and intermediate data
    all_conditions = {}

    # Add experimental data (calibration set)
    for exp_name, exp_data in experiments.items():
        c0 = exp_data["C0_factor"]
        l0 = exp_data["L0_factor"]
        key = (c0, l0)

        if key not in all_conditions:
            all_conditions[key] = {
                "type": "experiment",
                "data": exp_data,
                "mu_max": growth_rates.get(key),
                "N_max": steady_states.get(key),
            }

    # Add intermediate data (validation set) with complete timeseries
    for (c0, l0), data_dict in intermediate_data_timeseries.items():
        key = (c0, l0)
        if key not in all_conditions:
            all_conditions[key] = {
                "type": "intermediate",
                "time": data_dict["time"],
                "mean": data_dict["mean"],
                "replicates": data_dict["replicates"],
                "mu_max": data_dict["mu_max"],
                "N_max": data_dict["N_max"],
            }

    # Extract unique C0 and L0 values and sort them
    unique_c0 = sorted(set(c0 for c0, l0 in all_conditions.keys()))
    unique_l0 = sorted(set(l0 for c0, l0 in all_conditions.keys()))

    n_c0 = len(unique_c0)
    n_l0 = len(unique_l0)

    logger.info(
        f"Creating extended plot with {n_l0} rows (L0) × {n_c0} columns (C0) = {n_l0 * n_c0} subplots"
    )
    logger.info(f"L0 values ({n_l0}): {[f'{x:.2f}' for x in unique_l0]}")
    logger.info(f"C0 values ({n_c0}): {[f'{x:.2f}' for x in unique_c0]}")

    # Count calibration vs validation points
    n_calib = sum(1 for c in all_conditions.values() if c["type"] == "experiment")
    n_valid = sum(1 for c in all_conditions.values() if c["type"] == "intermediate")
    logger.info(f"Calibration points: {n_calib}, Validation points: {n_valid}")

    # Storage for global R² calculation
    all_data_calib = []
    all_pred_calib = []
    all_data_valid = []
    all_pred_valid = []

    # Storage for R² statistics
    r2_list_calib = []
    r2_list_valid = []

    # Create grid layout - note: we reverse L0 order so lowest is at bottom
    # sharey=False allows each subplot to have its own y-axis scale
    fig, axes = plt.subplots(
        n_l0, n_c0, figsize=(1.2 * n_c0, 2 * n_l0), sharex=True, sharey=True
    )

    # Ensure axes is always 2D
    if n_l0 == 1:
        axes = axes.reshape(1, -1)
    if n_c0 == 1:
        axes = axes.reshape(-1, 1)

    # Unpack Monod parameters
    monod_params_ss = monod_params  # (n_max_ref, k_c, k_l, k_i)

    # Infer growth rate model type
    model_type_gr = infer_growth_rate_model(growth_rate_monod_params)

    # Plot each condition - iterate with L0 reversed (bottom to top)
    for i, l0 in enumerate(reversed(unique_l0)):  # reversed so lowest L0 at bottom
        for j, c0 in enumerate(unique_c0):
            ax = axes[i, j]

            key = (c0, l0)

            if key in all_conditions:
                condition = all_conditions[key]

                # Convert L0 factor to actual L0 value for Monod model evaluation
                l0_actual = l0 * L0_REF

                # Calculate predicted parameters using Both Fits (Haldane models)
                # Use centralized functions from growth_models module
                mu_max_pred = evaluate_growth_rate(
                    c0,
                    l0_actual,
                    growth_rate_monod_params,
                    model_type_gr,
                    use_l0_factors=True,
                )
                N_max_pred = evaluate_steady_state(
                    c0, l0_actual, monod_params_ss, use_l0_factors=True
                )

                # Get time and concentration data based on condition type
                if condition["type"] == "experiment":
                    exp_data = condition["data"]
                    time = np.array(exp_data["Time"])
                    concentration = np.array(exp_data["Mean"])
                    replicates = exp_data["replicates"]
                else:  # intermediate
                    time = condition["time"]
                    concentration = condition["mean"]
                    replicates = condition["replicates"]

                # Plot replicates in gray
                for replicate in replicates:
                    ax.plot(
                        replicate["Time"],
                        replicate["Value"],
                        alpha=0.3,
                        linewidth=0.6,
                        color="gray",
                    )

                # Plot mean
                mean_color = (
                    "mediumslateblue"
                    if condition["type"] == "intermediate"
                    else "black"
                )
                ax.plot(
                    time,
                    concentration,
                    linestyle="--",
                    color=mean_color,
                    linewidth=1.2,
                    label="Data (mean)",
                    alpha=0.9,
                )

                # Get initial condition from first valid point
                t0_idx = None
                N0 = None
                for idx, c in enumerate(concentration):
                    if c > 0 and not np.isnan(c):
                        t0_idx = idx
                        N0 = c
                        break

                if N0 is not None and N0 > 0:
                    # Calculate Both Fits simulation using Monod-predicted parameters
                    t0 = time[t0_idx]

                    # Apply manual t_lag adjustment if specified for this condition
                    if manual_tlag_adjustments is not None:
                        key_tlag = (l0, c0)  # (L0_factor, C0_factor)
                        if key_tlag in manual_tlag_adjustments:
                            t_lag_adjust = manual_tlag_adjustments[key_tlag]
                            t0 += t_lag_adjust  # Positive values delay start, negative values advance it
                            logger.debug(
                                f"Applied t_lag adjustment of {t_lag_adjust:.1f}h for L0={l0}, C0={c0}"
                            )

                    t_sim_shifted = np.array(time) - t0
                    N_sim_both = N_max_pred / (
                        1
                        + ((N_max_pred - N0) / N0)
                        * np.exp(-mu_max_pred * t_sim_shifted)
                    )

                    # MODIF: Both fits → plein + tomato
                    ax.plot(
                        time,
                        N_sim_both,
                        color="tomato",
                        linewidth=1.2,
                        linestyle="-",
                        label="Both fits",
                        alpha=0.9,
                    )

                    # Calculate local R²
                    r2_local = calculate_r2_local(concentration, N_sim_both)

                    # Store data for global R² calculation
                    mask = (
                        ~(np.isnan(concentration) | np.isnan(N_sim_both))
                        & (concentration > 0)
                        & (N_sim_both > 0)
                    )
                    if np.sum(mask) > 0:
                        if condition["type"] == "experiment":
                            all_data_calib.extend(concentration[mask].tolist())
                            all_pred_calib.extend(N_sim_both[mask].tolist())
                            if not np.isnan(r2_local):
                                r2_list_calib.append(r2_local)
                        else:
                            all_data_valid.extend(concentration[mask].tolist())
                            all_pred_valid.extend(N_sim_both[mask].tolist())
                            if not np.isnan(r2_local):
                                r2_list_valid.append(r2_local)

                    # Add R² text in top right corner
                    # Color based on R² value: green if ≥0.845, red if <0.845
                    if not np.isnan(r2_local):
                        if r2_local >= 0.845:
                            bg_color = "lightgreen"
                        else:
                            bg_color = "lightcoral"

                        # Use thicker border for validation points to distinguish from calibration
                        border_width = (
                            1.0 if condition["type"] == "intermediate" else 0.5
                        )

                        ax.text(
                            0.98,
                            0.98,
                            f"R²={r2_local:.2f}",
                            transform=ax.transAxes,
                            fontsize=5.5,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor=bg_color,
                                alpha=0.8,
                                edgecolor="gray",
                                linewidth=border_width,
                            ),
                        )

                    # Add t_lag text in bottom left corner
                    ax.text(
                        0.02,
                        0.98,
                        f"t_lag={t0:.1f}h",
                        transform=ax.transAxes,
                        fontsize=5.5,
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="lightblue",
                            alpha=0.7,
                            edgecolor="gray",
                            linewidth=0.5,
                        ),
                    )

                    # Store t_lag value for saving to file
                    tlag_values_dict[(l0, c0)] = t0

                # Set title with C0 and L0 values
                # Add marker for validation points
                title_suffix = " (val)" if condition["type"] == "intermediate" else ""
                title_color = (
                    "mediumslateblue"
                    if condition["type"] == "intermediate"
                    else "black"
                )
                ax.set_title(
                    f"C0×{c0:.2f}, L0×{l0:.2f}{title_suffix}",
                    fontsize=6,
                    pad=2,
                    color=title_color,
                )

                # MODIF: validation styling
                if condition["type"] == "intermediate":
                    for spine in ax.spines.values():
                        spine.set_color("mediumslateblue")
                        spine.set_linewidth(1.2)
                    ax.tick_params(colors="mediumslateblue")

                ax.grid(True, alpha=0.15, linewidth=0.3)
                ax.set_xlim(0, MAX_TIME_HOURS)

                # Remove tick labels for inner plots
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.tick_params(axis="y", labelsize=5)

                if i < n_l0 - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelsize=5)

                # Legend only for top-left subplot
                if i == 0 and j == 0:
                    ax.legend(fontsize=5, loc="lower right", framealpha=0.8)
            else:
                # Empty subplot - turn off axis
                ax.axis("off")

    # Add common axis labels
    fig.text(0.5, 0.01, "Time (h)", ha="center", fontsize=10, weight="bold")
    fig.text(
        0.005,
        0.5,
        "Cell Concentration",
        va="center",
        rotation="vertical",
        fontsize=10,
        weight="bold",
    )

    plt.tight_layout(rect=[0.015, 0.015, 1, 1])
    output_path = pathlib.Path(output_dir) / "plates_simulation_extended_both_fits.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    # plt.show()
    logger.info(
        f"Saved extended simulation plot with {n_l0}×{n_c0} = {n_l0 * n_c0} panels to {output_path}"
    )
    logger.info(f"  {n_calib} calibration points (white background)")
    logger.info(f"  {n_valid} validation points (yellow background)")

    # Save t_lag values to file - include ALL grid conditions (n_l0 × n_c0)
    tlag_output_path = pathlib.Path(output_dir) / "t_lag_adjustments.txt"
    with open(tlag_output_path, "w") as f:
        f.write("# t_lag values for each (L0_factor, C0_factor) condition\n")
        f.write(
            f"# Grid: {n_l0} L0 values × {n_c0} C0 values = {n_l0 * n_c0} conditions\n"
        )
        f.write("# Format: (L0, C0): t_lag_hours\n\n")

        # Iterate over ALL conditions in the grid (same order as plot: L0 descending, C0 ascending)
        tlag_count = 0
        for l0 in reversed(unique_l0):
            for c0 in unique_c0:
                if (l0, c0) in tlag_values_dict:
                    t0 = tlag_values_dict[(l0, c0)]
                    f.write(f"({l0}, {c0}): {t0:.1f}\n")
                    tlag_count += 1
                else:
                    # Condition exists in grid but no valid data/N0
                    f.write(f"({l0}, {c0}): N/A\n")
                    tlag_count += 1

    logger.info(
        f"Saved {tlag_count} t_lag values to {tlag_output_path} ({len(tlag_values_dict)} with data, {tlag_count - len(tlag_values_dict)} N/A)"
    )

    # ===== CALCULATE AND DISPLAY GLOBAL R² STATISTICS =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("R² STATISTICS")
    logger.info("=" * 80)

    # Calibration statistics
    if r2_list_calib:
        n_good_calib = sum(1 for r2 in r2_list_calib if r2 >= 0.845)
        logger.info(f"CALIBRATION SET:")
        logger.info(f"  Number of conditions: {len(r2_list_calib)}")
        logger.info(
            f"  Conditions with R² ≥ 0.845: {n_good_calib}/{len(r2_list_calib)} ({100 * n_good_calib / len(r2_list_calib):.1f}%)"
        )
        logger.info(
            f"  Mean local R²: {np.mean(r2_list_calib):.4f} ± {np.std(r2_list_calib):.4f}"
        )
        logger.info(f"  Median local R²: {np.median(r2_list_calib):.4f}")
        logger.info(f"  Min local R²: {np.min(r2_list_calib):.4f}")
        logger.info(f"  Max local R²: {np.max(r2_list_calib):.4f}")

        # Calculate GLOBAL R² for calibration
        if all_data_calib and all_pred_calib:
            all_data_calib_arr = np.array(all_data_calib)
            all_pred_calib_arr = np.array(all_pred_calib)
            ss_res_calib = np.sum((all_data_calib_arr - all_pred_calib_arr) ** 2)
            ss_tot_calib = np.sum(
                (all_data_calib_arr - np.mean(all_data_calib_arr)) ** 2
            )
            r2_global_calib = (
                1 - (ss_res_calib / ss_tot_calib) if ss_tot_calib > 0 else np.nan
            )
            logger.info(f"  GLOBAL R² (all data points): {r2_global_calib:.4f}")
            logger.info(f"    (calculated over {len(all_data_calib)} data points)")

    logger.info("")

    # Validation statistics
    if r2_list_valid:
        n_good_valid = sum(1 for r2 in r2_list_valid if r2 >= 0.845)
        logger.info(f"VALIDATION SET (intermediate dilutions):")
        logger.info(f"  Number of conditions: {len(r2_list_valid)}")
        logger.info(
            f"  Conditions with R² ≥ 0.845: {n_good_valid}/{len(r2_list_valid)} ({100 * n_good_valid / len(r2_list_valid):.1f}%)"
        )
        logger.info(
            f"  Mean local R²: {np.mean(r2_list_valid):.4f} ± {np.std(r2_list_valid):.4f}"
        )
        logger.info(f"  Median local R²: {np.median(r2_list_valid):.4f}")
        logger.info(f"  Min local R²: {np.min(r2_list_valid):.4f}")
        logger.info(f"  Max local R²: {np.max(r2_list_valid):.4f}")

        # Calculate GLOBAL R² for validation
        if all_data_valid and all_pred_valid:
            all_data_valid_arr = np.array(all_data_valid)
            all_pred_valid_arr = np.array(all_pred_valid)
            ss_res_valid = np.sum((all_data_valid_arr - all_pred_valid_arr) ** 2)
            ss_tot_valid = np.sum(
                (all_data_valid_arr - np.mean(all_data_valid_arr)) ** 2
            )
            r2_global_valid = (
                1 - (ss_res_valid / ss_tot_valid) if ss_tot_valid > 0 else np.nan
            )
            logger.info(f"  GLOBAL R² (all data points): {r2_global_valid:.4f}")
            logger.info(f"    (calculated over {len(all_data_valid)} data points)")

    logger.info("")

    # Overall statistics (calibration + validation combined)
    if all_data_calib and all_data_valid:
        all_data_combined = np.array(all_data_calib + all_data_valid)
        all_pred_combined = np.array(all_pred_calib + all_pred_valid)
        ss_res_combined = np.sum((all_data_combined - all_pred_combined) ** 2)
        ss_tot_combined = np.sum((all_data_combined - np.mean(all_data_combined)) ** 2)
        r2_global_combined = (
            1 - (ss_res_combined / ss_tot_combined) if ss_tot_combined > 0 else np.nan
        )

        logger.info(f"OVERALL (calibration + validation combined):")
        logger.info(
            f"  Number of conditions: {len(r2_list_calib) + len(r2_list_valid)}"
        )
        n_good_total = sum(1 for r2 in r2_list_calib + r2_list_valid if r2 >= 0.845)
        logger.info(
            f"  Conditions with R² ≥ 0.845: {n_good_total}/{len(r2_list_calib) + len(r2_list_valid)} ({100 * n_good_total / (len(r2_list_calib) + len(r2_list_valid)):.1f}%)"
        )
        logger.info(f"  GLOBAL R² (all data points): {r2_global_combined:.4f}")
        logger.info(f"    (calculated over {len(all_data_combined)} data points)")

    logger.info("=" * 80)
    logger.info("")

    plt.close()


def plot_3d_surface_plate(
    data_dict: dict[tuple[float, float], float],
    model_func: callable,
    model_params: tuple,
    response_var: str,
    output_dir: str,
    intermediate_data: dict = None,
    L0_REF: float = 170.0,
) -> None:
    """
    Create a 3-panel figure with experimental data, RSM surface, and contour map.

    Parameters:
    -----------
    data_dict : dict
        Calibration data mapping (C0, L0_factor) to response value
    model_func : callable
        Model function f(xdata, *params) where xdata is (C0, L0_factor)
    model_params : tuple
        Fitted model parameters
    response_var : str
        Response variable name ('mu_max' or 'Nmax')
    output_dir : str
        Output directory
    intermediate_data : dict, optional
        Validation data mapping (C0, L0_factor) to (mu_max, Nmax) tuple
    L0_REF : float
        Reference L0 value for converting factors to µmol/m²/s
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import MaxNLocator

    # Font sizes
    FONT_TITLE = 16
    FONT_LABEL = 14
    FONT_TICK = 13
    FONT_COLORBAR = 13

    # Extract calibration data
    c0_calib = np.array([c0 for c0, l0 in data_dict.keys()])
    l0_calib = np.array([l0 for c0, l0 in data_dict.keys()])
    response_calib = np.array(list(data_dict.values()))

    # Identify unique L0 values and create color map
    L0_unique = sorted(set(l0_calib))
    colors_L0 = plt.cm.winter(np.linspace(0, 1, len(L0_unique)))
    L0_to_color = {L0: colors_L0[i] for i, L0 in enumerate(L0_unique)}

    # Prepare all data (calibration + validation)
    all_c0 = list(c0_calib)
    all_l0 = list(l0_calib)
    all_response = list(response_calib)
    calib_keys = set(data_dict.keys())

    if intermediate_data:
        for (c0, l0), values in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                all_c0.append(c0)
                all_l0.append(l0)
                # values is (mu_max, Nmax) tuple
                if response_var == "mu_max":
                    all_response.append(values[0])
                else:  # Nmax
                    all_response.append(values[1])

    all_c0 = np.array(all_c0)
    all_l0 = np.array(all_l0)
    all_response = np.array(all_response)

    # Update L0 unique values to include validation data
    L0_unique_all = sorted(set(all_l0))
    colors_L0_all = plt.cm.winter(np.linspace(0, 1, len(L0_unique_all)))
    L0_to_color_all = {L0: colors_L0_all[i] for i, L0 in enumerate(L0_unique_all)}

    # Define grid ranges
    c0_min, c0_max = all_c0.min(), all_c0.max()
    l0_min, l0_max = all_l0.min(), all_l0.max()

    # Create grid
    c0_grid = np.linspace(c0_min, c0_max, 50)
    l0_grid = np.linspace(l0_min, l0_max, 50)
    C0_mesh, L0_mesh = np.meshgrid(c0_grid, l0_grid)

    # Calculate predictions on grid
    xdata_grid = np.array([C0_mesh.flatten(), L0_mesh.flatten()])
    response_mesh = model_func(xdata_grid, *model_params).reshape(C0_mesh.shape)

    # Calculate R² values
    xdata_calib = np.array([c0_calib, l0_calib])
    response_pred_calib = model_func(xdata_calib, *model_params)
    r2_calib = 1 - (
        np.sum((response_calib - response_pred_calib) ** 2)
        / np.sum((response_calib - np.mean(response_calib)) ** 2)
    )

    xdata_all = np.array([all_c0, all_l0])
    response_pred_all = model_func(xdata_all, *model_params)
    r2_all = 1 - (
        np.sum((all_response - response_pred_all) ** 2)
        / np.sum((all_response - np.mean(all_response)) ** 2)
    )

    # Choose colormap based on response variable
    cmap_choice = "viridis" if response_var == "mu_max" else "plasma"

    # Labels
    if response_var == "mu_max":
        z_label = r"$\mu_{\mathrm{max}}$ (h$^{-1}$)"
    else:
        z_label = r"$N_{\mathrm{max}}$ (cells mL$^{-1}$)"

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[3, 0.4],
        hspace=0.3,
        wspace=0.3,
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )

    # === Subplot 1: Experimental data with error bars (ALL data, color by L0) ===
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")

    for L0_val in L0_unique_all:
        mask = all_l0 == L0_val
        if np.any(mask):
            # L0 displayed in µmol/m²/s
            ax1.scatter(
                all_l0[mask] * L0_REF,
                all_c0[mask],
                all_response[mask],
                c=[L0_to_color_all[L0_val]],
                s=50,
                linewidth=2,
                alpha=0.8,
            )

    ax1.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL, labelpad=10
    )
    ax1.set_ylabel("$C_0$", fontsize=FONT_LABEL, labelpad=10)
    ax1.set_zlabel(z_label, fontsize=FONT_LABEL, labelpad=10)
    ax1.set_title("Experimental data", fontsize=FONT_TITLE, pad=20)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.xaxis.set_major_locator(MaxNLocator(6))
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.zaxis.set_major_locator(MaxNLocator(6))

    # === Subplot 2: RSM Surface with calibration/extrapolation distinction ===
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    # Plot surface (L0 in µmol/m²/s for display)
    surf = ax2.plot_surface(
        L0_mesh * L0_REF,
        C0_mesh,
        response_mesh,
        cmap=cmap_choice,
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Calibration points (black)
    ax2.scatter(
        l0_calib * L0_REF,
        c0_calib,
        response_calib,
        color="black",
        s=50,
        marker="o",
        alpha=0.9,
        label="Calibration",
    )

    # Extrapolation points (mediumslateblue)
    if intermediate_data:
        ext_c0, ext_l0, ext_response = [], [], []
        for (c0, l0), values in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                ext_c0.append(c0)
                ext_l0.append(l0)
                if response_var == "mu_max":
                    ext_response.append(values[0])
                else:
                    ext_response.append(values[1])

        if ext_c0:
            ax2.scatter(
                np.array(ext_l0) * L0_REF,
                ext_c0,
                ext_response,
                color="mediumslateblue",
                s=100,
                marker="o",
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
                label="Extrapolation",
            )
            ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)

    ax2.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL, labelpad=10
    )
    ax2.set_ylabel("$C_0$", fontsize=FONT_LABEL, labelpad=10)
    ax2.set_zlabel(z_label, fontsize=FONT_LABEL, labelpad=10)
    ax2.set_title(
        f"RSM Surface\nR² calib = {r2_calib:.3f}, R² global = {r2_all:.3f}",
        fontsize=FONT_TITLE,
        pad=20,
    )
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.xaxis.set_major_locator(MaxNLocator(6))
    ax2.yaxis.set_major_locator(MaxNLocator(6))
    ax2.zaxis.set_major_locator(MaxNLocator(6))

    # === Subplot 3: Contour map ===
    ax3 = fig.add_subplot(gs[0, 2])

    # L0 in µmol/m²/s for display
    contour = ax3.contourf(
        L0_mesh * L0_REF, C0_mesh, response_mesh, levels=15, cmap=cmap_choice, alpha=0.8
    )
    contour_lines = ax3.contour(
        L0_mesh * L0_REF,
        C0_mesh,
        response_mesh,
        levels=15,
        colors="black",
        alpha=0.3,
        linewidths=0.5,
    )
    ax3.clabel(
        contour_lines,
        inline=True,
        fontsize=8,
        fmt="%.2f" if response_var == "mu_max" else "%.1e",
    )

    # Calibration points (black)
    ax3.scatter(
        l0_calib * L0_REF,
        c0_calib,
        s=50,
        c="black",
        marker="o",
        linewidths=2,
        alpha=0.9,
        zorder=5,
        label="Calibration",
    )

    # Extrapolation points
    if intermediate_data and ext_c0:
        ax3.scatter(
            np.array(ext_l0) * L0_REF,
            ext_c0,
            s=100,
            c="mediumslateblue",
            marker="o",
            edgecolor="white",
            linewidths=2,
            alpha=0.9,
            zorder=5,
            label="Extrapolation",
        )
        ax3.legend(loc="best", fontsize=10, framealpha=0.95)

    ax3.set_xlabel(r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL)
    ax3.set_ylabel("$C_0$", fontsize=FONT_LABEL)
    ax3.set_title("Contour map", fontsize=FONT_TITLE)
    ax3.tick_params(labelsize=FONT_TICK)

    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label(z_label, rotation=270, labelpad=25, fontsize=FONT_COLORBAR)

    # === Equation at bottom ===
    if response_var == "mu_max":
        mu_max_ref, k_c, k_l, k_i = model_params
        equation_text = (
            r"$\mu_{\mathrm{max}}(C_0, L_0) = "
            f"{mu_max_ref:.4f}" + r" \times "
            r"\frac{C_0}{" + f"{k_c:.4f}" + r" + C_0} \times "
            r"\frac{L_0}{" + f"{k_l:.4f}" + r" + L_0 + L_0^2/" + f"{k_i:.4f}" + r"}$"
        )
    else:
        n_max_ref, k_c, k_l, k_i = model_params
        equation_text = (
            r"$N_{\mathrm{max}}(C_0, L_0) = "
            f"{n_max_ref:.2e}" + r" \times "
            r"\frac{C_0}{" + f"{k_c:.4f}" + r" + C_0} \times "
            r"\frac{L_0}{" + f"{k_l:.4f}" + r" + L_0 + L_0^2/" + f"{k_i:.4f}" + r"}$"
        )

    ax_eq = fig.add_subplot(gs[1, :])
    ax_eq.axis("off")
    ax_eq.text(
        0.5,
        0.5,
        equation_text,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, pad=0.8),
    )

    # Save figure
    output_filename = f"RSM_3D_{response_var}.png"
    output_path = pathlib.Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 3D surface plot: {output_path}")
    logger.info(f"  R² calibration = {r2_calib:.4f}")
    logger.info(f"  R² global = {r2_all:.4f}")


def fit_steady_state_monod(
    steady_states: dict[tuple[float, float], float], output_dir: str = "results_plates"
) -> None:
    """Fit a Monod product with Haldane photoinhibition to the measured steady state data.

    Model: N_max(C,L) = N_max_ref × [C / (K_C + C)] × [L / (K_L + L + L²/K_I)]

    The L²/K_I term in the denominator causes steady state to decrease at very high light
    intensities, modeling photoinhibition stress (Haldane/Andrews model).

    Creates:
    1. A comparison plot of measured vs predicted steady states
    2. A 3D surface plot of the fitted model
    3. Prints fitted parameters

    Args:
        steady_states: Dict mapping (C0, L0) to measured steady state concentration
        output_dir: Directory to save output files
    """
    # Extract data
    c0_vals = []
    l0_vals = []
    n_max_vals = []

    for (c0, l0), n_max in steady_states.items():
        c0_vals.append(c0)
        l0_vals.append(l0)
        n_max_vals.append(n_max)

    c0_vals = np.array(c0_vals)
    l0_vals = np.array(l0_vals)
    n_max_vals = np.array(n_max_vals)

    # Use L0 factors directly (0.07, 0.15, 0.3, 0.6, 1.0) instead of converting to µmol/m²/s
    # This keeps C0 and L0 in comparable ranges [0, 1]

    # Prepare data for curve_fit
    # curve_fit expects: curve_fit(func, xdata, ydata)
    # xdata should be (2, N) for our 2D function
    xdata = np.array([c0_vals, l0_vals])  # Use L0 factors, not actual values

    # Initial guess for parameters
    # N_max_ref ~ max of measurements
    # K_C, K_L ~ moderate values relative to factor range [0, 1]
    # K_I ~ moderate photoinhibition (higher = less inhibition)
    p0 = [np.max(n_max_vals) * 1.5, 0.3, 0.2, 5.0]

    # Bounds: Allow more flexibility while avoiding pathological extremes
    # The original problem was K_I too small (0.001) causing L²/K_I to explode
    # K_I >= 0.5 prevents extreme photoinhibition while allowing fit flexibility
    bounds = ([0, 0, 0.001, 0.5], [np.inf, 10.0, 10.0, 100.0])

    try:
        # Fit the model with photoinhibition
        popt, pcov = curve_fit(
            steady_state_haldane, xdata, n_max_vals, p0=p0, bounds=bounds, maxfev=10000
        )
        n_max_ref, k_c, k_l, k_i = popt

        logger.info(f"Monod product fit parameters (Haldane photoinhibition):")
        logger.info(f"  N_max_ref = {n_max_ref:.3e}")
        logger.info(f"  K_C = {k_c:.6f} (C0 factor scale)")
        logger.info(f"  K_L = {k_l:.6f} (L0 factor scale)")
        logger.info(f"  K_I (photoinhibition) = {k_i:.6f} (L0 factor scale)")
        logger.info(f"  Note: K parameters can be >> 1 even with factors in [0,1]")

        # Calculate predictions and residuals
        n_max_pred = steady_state_haldane(xdata, n_max_ref, k_c, k_l, k_i)
        residuals = n_max_vals - n_max_pred
        r_squared = 1 - (
            np.sum(residuals**2) / np.sum((n_max_vals - np.mean(n_max_vals)) ** 2)
        )
        rmse = np.sqrt(np.mean(residuals**2))

        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  RMSE = {rmse:.3e}")

        # ===== FIGURE 1: Measured vs Predicted =====
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            n_max_vals, n_max_pred, alpha=0.6, s=100, edgecolors="black", linewidth=1.5
        )
        # Add diagonal line (perfect fit)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.75, zorder=0, linewidth=2)

        ax.set_xlabel("Measured N_max (cells/mL)", fontsize=12)
        ax.set_ylabel("Predicted N_max (cells/mL)", fontsize=12)
        ax.set_title(
            f"Steady State Fit: Monod Product Model (R² = {r_squared:.4f})", fontsize=14
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = pathlib.Path(output_dir) / "plates_steady_state_monod_fit.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved measured vs predicted plot to {output_path}")
        plt.close()

        # ===== FIGURE 3D with 3 pannels  =====
        try:
            intermediate_data = load_intermediate_data()
            plot_3d_surface_plate(
                data_dict=steady_states,
                model_func=steady_state_haldane,
                model_params=(n_max_ref, k_c, k_l, k_i),
                response_var="Nmax",
                output_dir=output_dir,
                intermediate_data=intermediate_data,
                L0_REF=170.0,
            )
        except Exception as e:
            logger.error(f"Failed to create 3D matplotlib plot for steady state: {e}")

        # Save parameters to YAML file
        params_dict = {
            "n_max_ref": float(n_max_ref),
            "k_l": float(k_l),
            "k_c": float(k_c),
            "k_i": float(k_i),
        }
        params_path = pathlib.Path(output_dir) / "parameters_steady_state.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params_dict, f)
        logger.info(f"Saved steady state parameters to {params_path}")

        return n_max_ref, k_c, k_l, k_i

    except Exception as e:
        logger.error(f"Failed to fit Monod product model: {e}")
        return None


def plot_growth_rate_with_fit(
    growth_rates: dict[tuple[float, float], float],
    monod_params_gr: tuple,
    output_dir: str = "results_plates",
) -> None:
    """Plot growth rate data with fitted Monod model curves as dashed lines.

    Args:
        growth_rates: Dict mapping (C0, L0) to μ_max values
        monod_params_gr: Tuple of (μ_max_ref, k_c, k_l, alpha) or (μ_max_ref, k_c, k_l, alpha, beta)
        output_dir: Directory to save output files
    """
    unique_c0 = sorted(set(c0 for c0, _ in growth_rates.keys()), reverse=True)
    unique_l0 = sorted(set(l0 for _, l0 in growth_rates.keys()))

    # ===== FIGURE 1: Growth Rate vs Nutrients with Fit =====
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_l0)))

    for color, l0 in zip(colors, unique_l0):
        c0_values = []
        mu_values = []

        for c0 in sorted(unique_c0, reverse=False):
            if (c0, l0) in growth_rates:
                c0_values.append(c0)
                mu_values.append(growth_rates[(c0, l0)])

        if c0_values:
            # Plot measured data
            ax.plot(
                c0_values,
                mu_values,
                "o-",
                label=f"L0×{l0:.3f} (measured)",
                color=color,
                linewidth=2,
                markersize=8,
            )

            # Plot fitted model predictions as dashed line
            c0_model = np.linspace(min(c0_values), max(c0_values), 100)
            # Convert L0 factor to actual L0 value for Monod model evaluation
            l0_actual = l0 * L0_REF
            # Use evaluate_growth_rate which handles all three model types
            model_type_gr = infer_growth_rate_model(monod_params_gr)
            mu_model = evaluate_growth_rate(
                c0_model, l0_actual, monod_params_gr, model_type_gr, use_l0_factors=True
            )

            ax.plot(c0_model, mu_model, "--", color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel("Nutrient Concentration (C0 factor)", fontsize=12)
    ax.set_ylabel("Initial Growth Rate μ (h⁻¹)", fontsize=12)
    ax.set_title("Growth Rate vs Nutrient Concentration (with Monod Fit)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, title="Light Intensity", loc="best")

    plt.tight_layout()
    output_path = (
        pathlib.Path(output_dir) / "plates_growth_rate_vs_nutrients_with_fit.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved growth rate vs nutrients with fit to {output_path}")
    plt.close()

    # ===== FIGURE 2: Growth Rate vs Light with Fit =====
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_c0)))

    for color, c0 in zip(colors, sorted(unique_c0, reverse=False)):
        l0_values = []
        mu_values = []

        for l0 in sorted(unique_l0):
            if (c0, l0) in growth_rates:
                l0_values.append(l0)
                mu_values.append(growth_rates[(c0, l0)])

        if l0_values:
            # Plot measured data
            ax.plot(
                l0_values,
                mu_values,
                "o-",
                label=f"C0×{c0:.3f} (measured)",
                color=color,
                linewidth=2,
                markersize=8,
            )

            # Plot fitted model predictions as dashed line
            l0_model = np.linspace(min(l0_values), max(l0_values), 100)
            # Convert L0 factor to actual L0 value for Monod model evaluation
            l0_model_actual = l0_model * L0_REF
            # Use evaluate_growth_rate which handles all three model types
            model_type_gr = infer_growth_rate_model(monod_params_gr)
            mu_model = evaluate_growth_rate(
                c0, l0_model_actual, monod_params_gr, model_type_gr, use_l0_factors=True
            )

            ax.plot(l0_model, mu_model, "--", color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel("Light Intensity (L0 factor)", fontsize=12)
    ax.set_ylabel("Initial Growth Rate μ (h⁻¹)", fontsize=12)
    ax.set_title("Growth Rate vs Light Intensity (with Monod Fit)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, title="Nutrient Concentration", loc="best")

    plt.tight_layout()
    output_path = pathlib.Path(output_dir) / "plates_growth_rate_vs_light_with_fit.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved growth rate vs light with fit to {output_path}")
    plt.close()


def plot_steady_state_with_fit(
    steady_states: dict[tuple[float, float], float],
    monod_params_ss: tuple,
    output_dir: str = "results_plates",
) -> None:
    """Plot steady state data with fitted Monod model curves as dashed lines.

    Args:
        steady_states: Dict mapping (C0, L0) to N_max values
        monod_params_ss: Tuple of (n_max_ref, k_c, k_l, k_i) from Haldane photoinhibition model
        output_dir: Directory to save output files
    """
    unique_c0 = sorted(set(c0 for c0, _ in steady_states.keys()), reverse=True)
    unique_l0 = sorted(set(l0 for _, l0 in steady_states.keys()))

    n_max_ref, k_c, k_l, k_i = monod_params_ss

    # ===== FIGURE: Steady State vs Nutrients with Fit =====
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_l0)))

    for color, l0 in zip(colors, unique_l0):
        c0_values = []
        n_max_values = []

        for c0 in sorted(unique_c0, reverse=False):
            if (c0, l0) in steady_states:
                c0_values.append(c0)
                n_max_values.append(steady_states[(c0, l0)])

        if c0_values:
            # Plot measured data
            ax.plot(
                c0_values,
                n_max_values,
                "o-",
                label=f"L0×{l0:.3f} (measured)",
                color=color,
                linewidth=2,
                markersize=8,
            )

            # Plot fitted model predictions as dashed line (Haldane model)
            c0_model = np.linspace(min(c0_values), max(c0_values), 100)
            # Convert L0 factor to actual L0 value for Monod model evaluation
            l0_actual = l0 * L0_REF
            n_max_model = steady_state_haldane(
                (c0_model, l0_actual), n_max_ref, k_c, k_l, k_i
            )

            ax.plot(c0_model, n_max_model, "--", color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel("Nutrient Concentration (C0 factor)", fontsize=12)
    ax.set_ylabel("Steady State Concentration (cells/mL)", fontsize=12)
    ax.set_title(
        "Steady State vs Nutrient Concentration (with Monod + Photoinhibition Fit)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, title="Light Intensity", loc="best")

    plt.tight_layout()
    output_path = (
        pathlib.Path(output_dir) / "plates_steady_state_vs_nutrients_with_fit.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved steady state vs nutrients with fit to {output_path}")
    plt.close()


def plot_intermediate_exponential_fits(
    conv_OD_plate_to_OD_erlen: float = 6.01,
    conv_OD_to_cell: float = 4.77e6,
    output_dir: str = "results_plates",
) -> None:
    """Plot intermediate dilution data in grid format with exponential fits.

    Similar to plates_exponential_fits.png but for intermediate data.
    Shows replicates (gray), mean (blue), exponential fit (red dashed),
    and extracted μ_max value.

    Args:
        conv_OD_plate_to_OD_erlen: Conversion factor
        conv_OD_to_cell: Conversion factor
        output_dir: Directory to save output files
    """
    # Map dates to light intensities
    date_to_l0 = {
        "01_07_2025": 0.6,
        "04_11_2024": 0.15,
        "21_10_2024": 0.15,  # CORRECTED: 25.5 µmol/m²/s (was 0.07)
        "07_07_2025": 1.0,
        "17_02_2025": 0.3,
    }

    intermediate_files = sorted(
        glob.glob(
            "all_data/final_corrected_data/replicate_OD_intermediate_dilution_*.csv"
        )
    )

    # Collect all intermediate data
    intermediate_conditions = {}  # (C0, L0) → {time, replicates, mean, mu}

    for fp in intermediate_files:
        filename = pathlib.Path(fp).name
        date = None
        for d, l0 in date_to_l0.items():
            if d in filename:
                date = d
                l0_value = l0
                break

        if date is None:
            continue

        try:
            df = pd.read_csv(fp, sep=";")
            time_col = df.columns[0]
            time = df[time_col].values

            # Extract C0 values from column names
            c0_values = set()
            for col in df.columns[1:]:
                if "C0 = " in col:
                    c0_str = col.split("C0 = ")[1].split(" ")[0]
                    try:
                        c0_values.add(float(c0_str))
                    except ValueError:
                        pass

            # For each C0 value, collect data
            for c0 in sorted(c0_values):
                replicate_cols = [col for col in df.columns if f"C0 = {c0:.2f}" in col]

                if len(replicate_cols) == 0:
                    continue

                # Convert OD to cell concentration
                replicates_data = []
                for col in replicate_cols:
                    od_values = df[col].values
                    cell_conc = od_values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                    replicates_data.append({"Time": time, "Value": cell_conc})

                # Calculate mean with trimmed mean
                mean_conc = []
                for t_idx in range(len(time)):
                    values_at_t = [
                        rep["Value"][t_idx]
                        for rep in replicates_data
                        if rep["Value"][t_idx] > 0
                    ]
                    if len(values_at_t) > 0:
                        mean_conc.append(trimmed_mean(values_at_t))
                    else:
                        mean_conc.append(np.nan)

                mean_conc = np.array(mean_conc)

                # Calculate μ_max from 0-60h window
                exp_start = 0.0
                exp_end = 60.0
                idx_start = np.argmin(np.abs(time - exp_start))
                idx_end = np.argmin(np.abs(time - exp_end))

                mu_max = np.nan
                if idx_start < idx_end:
                    time_window = time[idx_start : idx_end + 1]
                    conc_window = mean_conc[idx_start : idx_end + 1]
                    valid_mask = conc_window > 0

                    if np.sum(valid_mask) > 2:
                        time_valid = time_window[valid_mask]
                        conc_valid = conc_window[valid_mask]
                        ln_conc = np.log(conc_valid)
                        coeffs = np.polyfit(time_valid, ln_conc, 1)
                        mu_max = coeffs[0]

                key = (c0, l0_value)
                intermediate_conditions[key] = {
                    "time": time,
                    "mean": mean_conc,
                    "replicates": replicates_data,
                    "mu_max": mu_max,
                    "exp_start": exp_start,
                    "exp_end": exp_end,
                }

        except Exception as e:
            logger.warning(f"Failed to load intermediate data from {fp}: {e}")

    if not intermediate_conditions:
        logger.warning("No intermediate data to plot")
        return

    # Create grid plot
    unique_c0 = sorted(
        set(c0 for c0, _ in intermediate_conditions.keys()), reverse=True
    )
    unique_l0 = sorted(set(l0 for _, l0 in intermediate_conditions.keys()))

    n_c0 = len(unique_c0)
    n_l0 = len(unique_l0)

    fig, axes = plt.subplots(n_c0, n_l0, figsize=(4 * n_l0, 4 * n_c0))

    if n_c0 == 1:
        axes = axes.reshape(1, -1)
    if n_l0 == 1:
        axes = axes.reshape(-1, 1)

    for i, c0 in enumerate(unique_c0):
        for j, l0 in enumerate(unique_l0):
            ax = axes[i, j]

            key = (c0, l0)
            if key in intermediate_conditions:
                data = intermediate_conditions[key]
                time = data["time"]
                mean_conc = data["mean"]
                replicates = data["replicates"]
                mu_max = data["mu_max"]
                exp_start = data["exp_start"]
                exp_end = data["exp_end"]

                # Plot replicates
                for replicate in replicates:
                    ax.semilogy(
                        replicate["Time"],
                        replicate["Value"],
                        alpha=0.6,
                        linewidth=1,
                        color="gray",
                    )

                # Plot mean
                ax.semilogy(time, mean_conc, "b-", linewidth=2, label="Mean")

                # Highlight exponential phase
                ax.axvspan(
                    exp_start,
                    exp_end,
                    alpha=0.1,
                    color="green",
                    label="Exponential phase (0-60h)",
                )

                # Fit and overlay exponential
                exp_mask = (time >= exp_start) & (time <= exp_end)
                if np.any(exp_mask) and not np.isnan(mu_max):
                    t_exp = time[exp_mask]
                    c_exp = mean_conc[exp_mask]

                    # Plot exponential fit
                    c0_fit = c_exp[0] if len(c_exp) > 0 else 1
                    t_exp_all = np.linspace(exp_start, exp_end, 100)
                    c_exp_fit = c0_fit * np.exp(mu_max * (t_exp_all - t_exp[0]))
                    ax.semilogy(
                        t_exp_all,
                        c_exp_fit,
                        "r--",
                        linewidth=2.5,
                        label=f"Exp fit (μ={mu_max:.4f})",
                    )

                ax.set_title(f"C0×{c0:.3f}, L0×{l0:.3f}")
                ax.set_xlim(0, MAX_TIME_HOURS)
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(fontsize=8)

                if j == 0:
                    ax.set_ylabel("Cell Conc. (log)", fontsize=10)
                if i == n_c0 - 1:
                    ax.set_xlabel("Time (h)", fontsize=10)

    fig.text(0.5, 0.02, "Time (h)", ha="center", fontsize=12)
    fig.text(
        0.02,
        0.5,
        "Cell Concentration (log)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    output_path = pathlib.Path(output_dir) / "plates_exponential_fits_intermediate.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved intermediate exponential fits grid to {output_path}")
    plt.close()


def load_intermediate_data(
    conv_OD_plate_to_OD_erlen: float = 6.01, conv_OD_to_cell: float = 4.77e6
) -> dict[tuple[float, float], tuple[float, float]]:
    """Load intermediate dilution data and calculate μ_max and N_max for visualization.

    Returns:
        Dict mapping (C0, L0) to (μ_max, N_max) tuples
    """
    intermediate_data = {}

    # Map dates to light intensities
    date_to_l0 = {
        "01_07_2025": 0.6,
        "04_11_2024": 0.15,
        "21_10_2024": 0.15,  # CORRECTED: 25.5 µmol/m²/s (was 0.07)
        "07_07_2025": 1.0,
        "17_02_2025": 0.3,
    }

    intermediate_files = sorted(
        glob.glob(
            "all_data/final_corrected_data/replicate_OD_intermediate_dilution_*.csv"
        )
    )

    # Process intermediate_dilution files
    for fp in intermediate_files:
        # Extract date from filename
        filename = pathlib.Path(fp).name
        date = None
        for d, l0 in date_to_l0.items():
            if d in filename:
                date = d
                l0_value = l0
                break

        if date is None:
            logger.warning(f"Could not extract date from {filename}")
            continue

        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(fp, sep=";")
            time_col = df.columns[0]
            time = df[time_col].values

            # Extract C0 values from column names (format: "C0 = X.XX RN")
            c0_values = set()
            for col in df.columns[1:]:
                if "C0 = " in col:
                    c0_str = col.split("C0 = ")[1].split(" ")[0]
                    try:
                        c0_values.add(float(c0_str))
                    except ValueError:
                        pass

            # For each C0 value, calculate μ_max and N_max
            for c0 in sorted(c0_values):
                # Get all replicates for this C0
                # Use flexible matching to handle different decimal formats (0.2 vs 0.20)
                replicate_cols = []
                for col in df.columns:
                    if "C0 = " in col:
                        try:
                            col_c0_str = col.split("C0 = ")[1].split(" ")[0]
                            col_c0 = float(col_c0_str)
                            # Match if C0 values are close (within 0.001)
                            if abs(col_c0 - c0) < 0.001:
                                replicate_cols.append(col)
                        except (ValueError, IndexError):
                            pass

                if len(replicate_cols) == 0:
                    continue

                # Convert OD to cell concentration
                replicates_data = []
                for col in replicate_cols:
                    od_values = df[col].values
                    # Convert: OD_plate → OD_erlen → cells
                    cell_conc = od_values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                    replicates_data.append(cell_conc)

                # Calculate mean concentration with trimmed mean
                mean_conc = []
                for t_idx in range(len(time)):
                    values_at_t = [
                        rep[t_idx] for rep in replicates_data if rep[t_idx] > 0
                    ]
                    if len(values_at_t) > 0:
                        mean_conc.append(trimmed_mean(values_at_t))
                    else:
                        mean_conc.append(np.nan)

                mean_conc = np.array(mean_conc)

                # Calculate μ_max from 0-60h exponential phase
                exp_start = 0.0
                exp_end = 60.0

                idx_start = np.argmin(np.abs(time - exp_start))
                idx_end = np.argmin(np.abs(time - exp_end))

                if idx_start < idx_end:
                    time_window = time[idx_start : idx_end + 1]
                    conc_window = mean_conc[idx_start : idx_end + 1]

                    # Filter positive values
                    valid_mask = conc_window > 0
                    if np.sum(valid_mask) > 2:
                        time_valid = time_window[valid_mask]
                        conc_valid = conc_window[valid_mask]

                        # Linear regression on ln(concentration) vs time
                        ln_conc = np.log(conc_valid)
                        coeffs = np.polyfit(time_valid, ln_conc, 1)
                        mu_max = coeffs[0]
                    else:
                        mu_max = np.nan
                else:
                    mu_max = np.nan

                # Calculate N_max from last 10% of time points
                n_last = max(1, len(mean_conc) // 10)
                ss_values = mean_conc[-n_last:]
                ss_values = ss_values[ss_values > 0]

                if len(ss_values) > 0:
                    n_max = trimmed_mean(ss_values)
                else:
                    n_max = np.nan

                if not np.isnan(mu_max) and not np.isnan(n_max) and mu_max > 0:
                    intermediate_data[(c0, l0_value)] = (mu_max, n_max)
                    logger.debug(
                        f"Intermediate: C0={c0:.3f}, L0={l0_value:.3f}: μ={mu_max:.4f}, N={n_max:.2e}"
                    )

        except Exception as e:
            logger.warning(f"Failed to load intermediate data from {fp}: {e}")

    # Load 16-09-24 data (validation points at L0=0.07)
    hand_cleaned_16_09 = sorted(
        glob.glob("all_data/hand_cleaned/replicates_OD_16_09_2024*.csv")
    )

    for fp in hand_cleaned_16_09:
        try:
            # Use the dedicated function to read hand_cleaned data
            experiments_16_09 = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell,
            )

            for exp_name, exp_data in experiments_16_09.items():
                c0 = exp_data["C0_factor"]
                l0 = exp_data["L0_factor"]
                time = np.array(exp_data["Time"])
                concentration = np.array(exp_data["Mean"])

                # Calculate μ_max from 0-60h exponential phase
                exp_start = 0.0
                exp_end = 60.0
                idx_start = np.argmin(np.abs(time - exp_start))
                idx_end = np.argmin(np.abs(time - exp_end))

                mu_max = np.nan
                if idx_start < idx_end:
                    time_window = time[idx_start : idx_end + 1]
                    conc_window = concentration[idx_start : idx_end + 1]

                    # Filter positive values
                    valid_mask = conc_window > 0
                    if np.sum(valid_mask) > 2:
                        time_valid = time_window[valid_mask]
                        conc_valid = conc_window[valid_mask]

                        # Linear regression on ln(concentration) vs time
                        ln_conc = np.log(conc_valid)
                        coeffs = np.polyfit(time_valid, ln_conc, 1)
                        mu_max = coeffs[0]

                # Calculate N_max from last 10% of time points
                n_last = max(1, len(concentration) // 10)
                ss_values = concentration[-n_last:]
                ss_values = ss_values[ss_values > 0]

                n_max = np.nan
                if len(ss_values) > 0:
                    n_max = trimmed_mean(ss_values)

                if not np.isnan(mu_max) and not np.isnan(n_max) and mu_max > 0:
                    intermediate_data[(c0, l0)] = (mu_max, n_max)
                    logger.debug(
                        f"16-09-24: C0={c0:.3f}, L0={l0:.3f}: μ={mu_max:.4f}, N={n_max:.2e}"
                    )

        except Exception as e:
            logger.warning(f"Failed to load 16-09-24 data from {fp}: {e}")

    logger.info(f"Loaded {len(intermediate_data)} intermediate data points")
    return intermediate_data


def load_intermediate_data_with_timeseries(
    conv_OD_plate_to_OD_erlen: float = 6.01, conv_OD_to_cell: float = 4.77e6
) -> dict[tuple[float, float], dict]:
    """Load intermediate dilution data with complete timeseries for validation plotting.

    Returns:
        Dict mapping (C0, L0) to dict with keys:
            - 'time': time array
            - 'mean': mean concentration array
            - 'replicates': list of dicts with 'Time' and 'Value' keys
            - 'mu_max': calculated growth rate
            - 'N_max': calculated steady state
    """
    intermediate_data = {}

    # Map dates to light intensities
    date_to_l0 = {
        "01_07_2025": 0.6,
        "04_11_2024": 0.15,
        "21_10_2024": 0.15,  # CORRECTED: 25.5 µmol/m²/s (was 0.07)
        "07_07_2025": 1.0,
        "17_02_2025": 0.3,
    }

    intermediate_files = sorted(
        glob.glob(
            "all_data/final_corrected_data/replicate_OD_intermediate_dilution_*.csv"
        )
    )

    # Process intermediate_dilution files
    for fp in intermediate_files:
        # Extract date from filename
        filename = pathlib.Path(fp).name
        date = None
        for d, l0 in date_to_l0.items():
            if d in filename:
                date = d
                l0_value = l0
                break

        if date is None:
            logger.warning(f"Could not extract date from {filename}")
            continue

        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(fp, sep=";")
            time_col = df.columns[0]
            time = df[time_col].values

            # Extract C0 values from column names (format: "C0 = X.XX RN")
            c0_values = set()
            for col in df.columns[1:]:
                if "C0 = " in col:
                    c0_str = col.split("C0 = ")[1].split(" ")[0]
                    try:
                        c0_values.add(float(c0_str))
                    except ValueError:
                        pass

            # For each C0 value, load complete timeseries
            for c0 in sorted(c0_values):
                # Get all replicates for this C0
                # Use flexible matching to handle different decimal formats (0.2 vs 0.20)
                replicate_cols = []
                for col in df.columns:
                    if "C0 = " in col:
                        try:
                            col_c0_str = col.split("C0 = ")[1].split(" ")[0]
                            col_c0 = float(col_c0_str)
                            # Match if C0 values are close (within 0.001)
                            if abs(col_c0 - c0) < 0.001:
                                replicate_cols.append(col)
                        except (ValueError, IndexError):
                            pass

                if len(replicate_cols) == 0:
                    continue

                # Convert OD to cell concentration and store as replicate dicts
                replicates_list = []
                replicates_data = []
                for col in replicate_cols:
                    od_values = df[col].values
                    # Convert: OD_plate → OD_erlen → cells
                    cell_conc = od_values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                    replicates_data.append(cell_conc)
                    replicates_list.append(
                        {"Time": time.copy(), "Value": cell_conc.copy()}
                    )

                # Calculate mean concentration with trimmed mean
                mean_conc = []
                for t_idx in range(len(time)):
                    values_at_t = [
                        rep[t_idx] for rep in replicates_data if rep[t_idx] > 0
                    ]
                    if len(values_at_t) > 0:
                        mean_conc.append(trimmed_mean(values_at_t))
                    else:
                        mean_conc.append(np.nan)

                mean_conc = np.array(mean_conc)

                # Calculate μ_max from 0-60h exponential phase
                exp_start = 0.0
                exp_end = 60.0

                idx_start = np.argmin(np.abs(time - exp_start))
                idx_end = np.argmin(np.abs(time - exp_end))

                mu_max = np.nan
                if idx_start < idx_end:
                    time_window = time[idx_start : idx_end + 1]
                    conc_window = mean_conc[idx_start : idx_end + 1]

                    # Filter positive values
                    valid_mask = conc_window > 0
                    if np.sum(valid_mask) > 2:
                        time_valid = time_window[valid_mask]
                        conc_valid = conc_window[valid_mask]

                        # Linear regression on ln(concentration) vs time
                        ln_conc = np.log(conc_valid)
                        coeffs = np.polyfit(time_valid, ln_conc, 1)
                        mu_max = coeffs[0]

                # Calculate N_max from last 10% of time points
                n_last = max(1, len(mean_conc) // 10)
                ss_values = mean_conc[-n_last:]
                ss_values = ss_values[ss_values > 0]

                n_max = np.nan
                if len(ss_values) > 0:
                    n_max = trimmed_mean(ss_values)

                # Store complete data regardless of validity (for plotting)
                key = (c0, l0_value)
                intermediate_data[key] = {
                    "time": time.copy(),
                    "mean": mean_conc.copy(),
                    "replicates": replicates_list,
                    "mu_max": mu_max,
                    "N_max": n_max,
                }

                if not np.isnan(mu_max) and not np.isnan(n_max) and mu_max > 0:
                    logger.debug(
                        f"Intermediate: C0={c0:.3f}, L0={l0_value:.3f}: μ={mu_max:.4f}, N={n_max:.2e}"
                    )
                else:
                    logger.debug(
                        f"Intermediate: C0={c0:.3f}, L0={l0_value:.3f}: incomplete data"
                    )

        except Exception as e:
            logger.warning(f"Failed to load intermediate data from {fp}: {e}")

    # Load 16-09-24 data (validation points at L0=0.07)
    hand_cleaned_16_09 = sorted(
        glob.glob("all_data/hand_cleaned/replicates_OD_16_09_2024*.csv")
    )

    for fp in hand_cleaned_16_09:
        try:
            # Use the dedicated function to read hand_cleaned data
            experiments_16_09 = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell,
            )

            for exp_name, exp_data in experiments_16_09.items():
                c0 = exp_data["C0_factor"]
                l0 = exp_data["L0_factor"]
                time = np.array(exp_data["Time"])
                concentration = np.array(exp_data["Mean"])
                replicates = exp_data["replicates"]

                # Calculate μ_max from 0-60h exponential phase
                exp_start = 0.0
                exp_end = 60.0
                idx_start = np.argmin(np.abs(time - exp_start))
                idx_end = np.argmin(np.abs(time - exp_end))

                mu_max = np.nan
                if idx_start < idx_end:
                    time_window = time[idx_start : idx_end + 1]
                    conc_window = concentration[idx_start : idx_end + 1]

                    # Filter positive values
                    valid_mask = conc_window > 0
                    if np.sum(valid_mask) > 2:
                        time_valid = time_window[valid_mask]
                        conc_valid = conc_window[valid_mask]

                        # Linear regression on ln(concentration) vs time
                        ln_conc = np.log(conc_valid)
                        coeffs = np.polyfit(time_valid, ln_conc, 1)
                        mu_max = coeffs[0]

                # Calculate N_max from last 10% of time points
                n_last = max(1, len(concentration) // 10)
                ss_values = concentration[-n_last:]
                ss_values = ss_values[ss_values > 0]

                n_max = np.nan
                if len(ss_values) > 0:
                    n_max = trimmed_mean(ss_values)

                # Store complete data
                key = (c0, l0)
                intermediate_data[key] = {
                    "time": time.copy(),
                    "mean": concentration.copy(),
                    "replicates": replicates,
                    "mu_max": mu_max,
                    "N_max": n_max,
                }

                if not np.isnan(mu_max) and not np.isnan(n_max) and mu_max > 0:
                    logger.debug(
                        f"16-09-24: C0={c0:.3f}, L0={l0:.3f}: μ={mu_max:.4f}, N={n_max:.2e}"
                    )
                else:
                    logger.debug(f"16-09-24: C0={c0:.3f}, L0={l0:.3f}: incomplete data")

        except Exception as e:
            logger.warning(f"Failed to load 16-09-24 data from {fp}: {e}")

    logger.info(
        f"Loaded {len(intermediate_data)} intermediate data points with timeseries"
    )
    return intermediate_data


def fit_growth_rate_monod(
    growth_rates: dict[tuple[float, float], float], output_dir: str = "results_plates"
) -> tuple | None:
    """Fit a Monod product with light and nutrient inhibition to the measured growth rate data.

    Model: μ_max(C,L) = μ_max_ref × [C / (K_C + C)] × [L / (K_L + L)] / (1 + α×L² + β×C²)

    The denominator includes:
    - α×L²: Light inhibition at high light intensities
    - β×C²: Nutrient toxicity/osmotic stress at high nutrient concentrations

    This captures the non-separable interaction where high nutrients amplify growth limitations.

    Excludes outliers:
    - L0=0.6, C0=0.25 point (dip in otherwise monotonic trend)

    Creates:
    1. A comparison plot of measured vs predicted growth rates
    2. A 3D surface plot of the fitted model
    3. Prints fitted parameters

    Args:
        growth_rates: Dict mapping (C0, L0) to measured growth rate μ (h⁻¹)
        output_dir: Directory to save output files

    Returns:
        Tuple of (μ_max_ref, k_c, k_l, alpha, beta) or None if fit fails
    """
    # Extract data and filter outliers
    c0_vals = []
    l0_vals = []
    mu_vals = []

    for (c0, l0), mu in growth_rates.items():
        # Skip L0=0.6, C0=0.25 outlier
        if np.isclose(l0, 0.6, atol=1e-6) and np.isclose(c0, 0.25, atol=1e-6):
            logger.debug(f"Skipping outlier: C0={c0:.3f}, L0={l0:.3f}, μ={mu:.4f}")
            continue

        c0_vals.append(c0)
        l0_vals.append(l0)
        mu_vals.append(mu)

    if len(mu_vals) < 4:
        logger.warning(
            f"Not enough data points after filtering ({len(mu_vals)} points). Cannot fit."
        )
        return None

    c0_vals = np.array(c0_vals)
    l0_vals = np.array(l0_vals)
    mu_vals = np.array(mu_vals)

    # Use L0 factors directly (0.07, 0.15, 0.3, 0.6, 1.0) instead of converting to µmol/m²/s
    # This keeps C0 and L0 in comparable ranges [0, 1]

    logger.info(
        f"Fitting growth rate model using {len(mu_vals)} data points (excluded {len(growth_rates) - len(mu_vals)} outliers)"
    )
    logger.info(f"L0 range (factors): {np.min(l0_vals):.2f} - {np.max(l0_vals):.2f}")

    # Prepare data for curve_fit
    xdata = np.array([c0_vals, l0_vals])  # Use L0 factors, not actual values

    try:
        # ===== Haldane Light-Only Model (photoinhibition on light, no synergy) =====
        # μ(C,L) = μ_max_ref × [C/(K_C+C)] × [L/(K_L+L+L²/K_I)]

        p0 = [
            np.max(mu_vals),
            np.mean(c0_vals),
            np.mean(l0_vals),
            np.mean(l0_vals) * 10,
        ]
        # Bounds: Allow K_L and K_I to be much larger than the factor range [0,1]
        # This is normal in RSM - coded factors vs model parameters are different scales
        bounds = ([0.01, 0.001, 0.001, 0.001], [1.0, 10.0, 1000.0, 10000.0])

        popt, pcov = curve_fit(
            growth_rate_haldane_light_only,
            xdata,
            mu_vals,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        mu_max_ref, k_c, k_l, k_i = popt

        logger.info(f"Haldane Light-Only Model (no synergistic term):")
        logger.info(f"  μ = μ_max_ref × [C/(K_C+C)] × [L/(K_L+L+L²/K_I)]")
        logger.info(f"  μ_max_ref = {mu_max_ref:.6f} h⁻¹")
        logger.info(f"  K_C = {k_c:.6f} (C0 factor scale)")
        logger.info(f"  K_L = {k_l:.6f} (L0 factor scale)")
        logger.info(f"  K_I = {k_i:.6f} (L0 factor scale)")
        logger.info(f"  Note: K parameters can be >> 1 even with factors in [0,1]")

        # Calculate predictions and residuals
        mu_pred = growth_rate_haldane_light_only(xdata, *popt)
        residuals = mu_vals - mu_pred
        r_squared = 1 - (
            np.sum(residuals**2) / np.sum((mu_vals - np.mean(mu_vals)) ** 2)
        )
        rmse = np.sqrt(np.mean(residuals**2))

        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  RMSE = {rmse:.6f} h⁻¹")
        logger.info("")
        logger.info("")

        # Save parameters to YAML
        params_dict = {
            "mu_max_ref": float(mu_max_ref),
            "k_l": float(k_l),
            "k_c": float(k_c),
            "k_i": float(k_i),
        }
        params_path = pathlib.Path(output_dir) / "parameters_growth_rate.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params_dict, f)
        logger.info(f"Saved growth rate parameters to {params_path}")

        # ===== FIGURE 1: Measured vs Predicted =====
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            mu_vals, mu_pred, alpha=0.6, s=100, edgecolors="black", linewidth=1.5
        )
        # Add diagonal line (perfect fit)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.75, zorder=0, linewidth=2)

        ax.set_xlabel("Measured μ_max (h⁻¹)", fontsize=12)
        ax.set_ylabel("Predicted μ_max (h⁻¹)", fontsize=12)
        ax.set_title(
            f"Growth Rate Fit: Haldane Light-Only (R² = {r_squared:.4f})", fontsize=14
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = pathlib.Path(output_dir) / "plates_growth_rate_monod_fit.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved measured vs predicted growth rate plot to {output_path}")
        plt.close()

        # ===== FIGURE 3D with 3 pannels  =====
        try:
            intermediate_data = load_intermediate_data()
            plot_3d_surface_plate(
                data_dict=growth_rates,
                model_func=growth_rate_haldane_light_only,
                model_params=popt,
                response_var="mu_max",
                output_dir=output_dir,
                intermediate_data=intermediate_data,
                L0_REF=170.0,
            )
        except Exception as e:
            logger.error(f"Failed to create 3D matplotlib plot: {e}")

        # Return model parameters
        return tuple(popt)  # 4 parameters: mu_max_ref, k_c, k_l, k_i

    except Exception as e:
        logger.error(f"Failed to fit Haldane light-only model for growth rate: {e}")
        return None


def create_full_rsm_panel_plate(
    growth_rates: dict[tuple[float, float], float],
    steady_states: dict[tuple[float, float], float],
    growth_rate_monod_params: tuple,
    steady_state_monod_params: tuple,
    output_dir: str = "results_plates",
) -> None:
    """Create full RSM panel for plate data.

    Creates a 3-row panel with:
    - Row 1: 3D surfaces for growth rate and steady state
    - Row 2: 2D cuts vs L0 for growth rate and steady state
    - Row 3: Sensitivity analysis (4 plots)

    Args:
        growth_rates: Dict (C0, L0) -> μ_max for calibration conditions
        steady_states: Dict (C0, L0) -> N_max for calibration conditions
        growth_rate_monod_params: Fitted Haldane light-only parameters (mu_max_ref, k_c, k_l, k_i)
        steady_state_monod_params: Fitted Haldane parameters (n_max_ref, k_c, k_l, k_i)
        output_dir: Directory to save output
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm, Normalize

    # Font sizes
    FONT_TITLE_3D = 16  # 3D plot titles
    FONT_TITLE_2D = 16  # 2D plot titles
    FONT_TITLE_SENS = 18  # Sensitivity plot titles
    FONT_LABEL_3D = 14  # 3D axis labels
    FONT_LABEL_2D = 14  # 2D axis labels
    FONT_LABEL_SENS = 14  # Sensitivity axis labels
    FONT_TICK_3D = 13  # 3D tick marks
    FONT_TICK_2D = 13  # 2D tick marks
    FONT_TICK_SENS = 13  # Sensitivity tick marks
    FONT_LEGEND = 13  # Legend
    FONT_CONTOUR = 13  # Contour values
    FONT_COLORBAR = 13  # Colorbar labels

    # Load intermediate data for validation points
    intermediate_data = load_intermediate_data()

    # Combine calibration and intermediate data
    all_keys = set(growth_rates.keys())
    if intermediate_data:
        all_keys.update(intermediate_data.keys())

    calib_keys = set(growth_rates.keys())

    # Determine domain from all data (in L0 factor units)
    all_c0 = [k[0] for k in all_keys]
    all_l0 = [k[1] for k in all_keys]
    C0_min, C0_max = min(all_c0), max(all_c0)
    L0_min, L0_max = min(all_l0), max(all_l0)

    # Create grids (in L0 factor units for model evaluation)
    L0_range = np.linspace(L0_min, L0_max, 50)
    C0_range = np.linspace(C0_min, C0_max, 50)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # L0 in µmol/m²/s for display
    L0_range_display = L0_range * L0_REF
    L0_grid_display = L0_grid * L0_REF

    # Extract parameters
    mu_max_ref_gr, k_c_gr, k_l_gr, k_i_gr = growth_rate_monod_params
    n_max_ref_ss, k_c_ss, k_l_ss, k_i_ss = steady_state_monod_params

    # Predictions using Haldane models
    mu_grid = growth_rate_haldane_light_only(
        (C0_grid, L0_grid), *growth_rate_monod_params
    )
    N_grid = steady_state_haldane((C0_grid, L0_grid), *steady_state_monod_params)

    # Calculate R² for calibration data
    mu_vals_calib = np.array([growth_rates[k] for k in calib_keys])
    c0_vals_calib = np.array([k[0] for k in calib_keys])
    l0_vals_calib = np.array([k[1] for k in calib_keys])
    mu_pred_calib = growth_rate_haldane_light_only(
        (c0_vals_calib, l0_vals_calib), *growth_rate_monod_params
    )
    r2_mu = 1 - (
        np.sum((mu_vals_calib - mu_pred_calib) ** 2)
        / np.sum((mu_vals_calib - np.mean(mu_vals_calib)) ** 2)
    )

    n_vals_calib = np.array([steady_states[k] for k in calib_keys])
    n_pred_calib = steady_state_haldane(
        (c0_vals_calib, l0_vals_calib), *steady_state_monod_params
    )
    r2_N = 1 - (
        np.sum((n_vals_calib - n_pred_calib) ** 2)
        / np.sum((n_vals_calib - np.mean(n_vals_calib)) ** 2)
    )

    # Calculate global R² including intermediate data
    if intermediate_data:
        # Combine all growth rate data
        all_mu_vals = list(mu_vals_calib)
        all_c0_vals = list(c0_vals_calib)
        all_l0_vals = list(l0_vals_calib)
        for (c0, l0), (mu, n_max) in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                all_mu_vals.append(mu)
                all_c0_vals.append(c0)
                all_l0_vals.append(l0)

        all_mu_vals = np.array(all_mu_vals)
        all_c0_vals = np.array(all_c0_vals)
        all_l0_vals = np.array(all_l0_vals)
        all_mu_pred = growth_rate_haldane_light_only(
            (all_c0_vals, all_l0_vals), *growth_rate_monod_params
        )
        r2_mu_global = 1 - (
            np.sum((all_mu_vals - all_mu_pred) ** 2)
            / np.sum((all_mu_vals - np.mean(all_mu_vals)) ** 2)
        )

        # Combine all steady state data
        all_n_vals = list(n_vals_calib)
        all_c0_vals_n = list(c0_vals_calib)
        all_l0_vals_n = list(l0_vals_calib)
        for (c0, l0), (mu, n_max) in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                all_n_vals.append(n_max)
                all_c0_vals_n.append(c0)
                all_l0_vals_n.append(l0)

        all_n_vals = np.array(all_n_vals)
        all_c0_vals_n = np.array(all_c0_vals_n)
        all_l0_vals_n = np.array(all_l0_vals_n)
        all_n_pred = steady_state_haldane(
            (all_c0_vals_n, all_l0_vals_n), *steady_state_monod_params
        )
        r2_N_global = 1 - (
            np.sum((all_n_vals - all_n_pred) ** 2)
            / np.sum((all_n_vals - np.mean(all_n_vals)) ** 2)
        )
    else:
        r2_mu_global = r2_mu
        r2_N_global = r2_N

    # C0 colors using truncated 'RdPu_r' colormap (violet to light pink, avoiding white)
    # Truncate colormap from 0.15 to 0.85 to avoid extreme colors (white at the end)
    from matplotlib.colors import LinearSegmentedColormap

    C0_unique = sorted(set(all_c0))
    cmap_rdpu_r = plt.cm.RdPu_r
    colors_C0 = cmap_rdpu_r(np.linspace(0.15, 0.85, len(C0_unique)))
    colors_C0_dict = {c0: colors_C0[i] for i, c0 in enumerate(C0_unique)}

    # Create truncated colormap for colorbar
    cmap_rdpu_r_truncated = LinearSegmentedColormap.from_list(
        "RdPu_r_truncated", cmap_rdpu_r(np.linspace(0.15, 0.85, 256))
    )

    # Create figure
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(
        3,
        4,
        height_ratios=[1.3, 1.0, 1.0],
        hspace=0.35,
        wspace=0.30,
        left=0.06,
        right=0.94,
        top=0.97,
        bottom=0.08,
    )

    # ========== ROW 1: 3D SURFACES ==========
    # Growth rate surface
    ax_3d_mu = fig.add_subplot(gs[0, 0:2], projection="3d")
    surf_mu = ax_3d_mu.plot_surface(
        L0_grid_display,
        C0_grid,
        mu_grid,
        cmap="viridis",
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Plot calibration points (black) and validation points (purple)
    for (c0, l0), mu in growth_rates.items():
        ax_3d_mu.scatter(
            l0 * L0_REF, c0, mu, color="black", s=50, marker="o", alpha=0.9
        )

    if intermediate_data:
        for (c0, l0), (mu, _) in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                ax_3d_mu.scatter(
                    l0 * L0_REF,
                    c0,
                    mu,
                    color="mediumslateblue",
                    s=100,
                    marker="o",
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.9,
                )

    ax_3d_mu.set_xlabel(
        r"Light intensity (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_mu.set_ylabel(r"$C_0$", fontsize=FONT_LABEL_3D, labelpad=10)
    ax_3d_mu.set_zlabel(
        r"$\mu_{\mathrm{max}}$ (h$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_mu.set_title(
        f"R² = {r2_mu:.3f} (calibration), R² = {r2_mu_global:.3f} (global)",
        fontsize=FONT_TITLE_3D,
        pad=10,
    )
    ax_3d_mu.tick_params(labelsize=FONT_TICK_3D)
    ax_3d_mu.xaxis.set_major_locator(MaxNLocator(6))
    ax_3d_mu.yaxis.set_major_locator(MaxNLocator(6))
    ax_3d_mu.zaxis.set_major_locator(MaxNLocator(6))

    cbar_mu = plt.colorbar(surf_mu, ax=ax_3d_mu, shrink=0.5, aspect=10, pad=0.1)
    cbar_mu.set_label(
        r"$\mu_{\mathrm{max}}$ (h$^{-1}$)",
        fontsize=FONT_COLORBAR,
        rotation=270,
        labelpad=20,
    )
    cbar_mu.ax.tick_params(labelsize=FONT_TICK_3D)

    # Steady state surface (scaled by 1e7 for display)
    SCALE_N_3D = 1e7
    ax_3d_N = fig.add_subplot(gs[0, 2:4], projection="3d")
    surf_N = ax_3d_N.plot_surface(
        L0_grid_display,
        C0_grid,
        N_grid / SCALE_N_3D,
        cmap="plasma",
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    for (c0, l0), n_max in steady_states.items():
        ax_3d_N.scatter(
            l0 * L0_REF,
            c0,
            n_max / SCALE_N_3D,
            color="black",
            s=50,
            marker="o",
            alpha=0.9,
        )

    if intermediate_data:
        for (c0, l0), (_, n_max) in intermediate_data.items():
            if (c0, l0) not in calib_keys:
                ax_3d_N.scatter(
                    l0 * L0_REF,
                    c0,
                    n_max / SCALE_N_3D,
                    color="mediumslateblue",
                    s=100,
                    marker="o",
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.9,
                )

    ax_3d_N.set_xlabel(
        r"Light intensity (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_N.set_ylabel(r"$C_0$", fontsize=FONT_LABEL_3D, labelpad=10)
    ax_3d_N.set_zlabel(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ $\times 10^7$)",
        fontsize=FONT_LABEL_3D,
        labelpad=10,
    )
    ax_3d_N.set_title(
        f"R² = {r2_N:.3f} (calibration), R² = {r2_N_global:.3f} (global)",
        fontsize=FONT_TITLE_3D,
        pad=10,
    )
    ax_3d_N.tick_params(labelsize=FONT_TICK_3D)
    ax_3d_N.xaxis.set_major_locator(MaxNLocator(6))
    ax_3d_N.yaxis.set_major_locator(MaxNLocator(6))
    ax_3d_N.zaxis.set_major_locator(MaxNLocator(6))

    cbar_N = plt.colorbar(surf_N, ax=ax_3d_N, shrink=0.5, aspect=10, pad=0.1)
    cbar_N.set_label(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ $\times 10^7$)",
        fontsize=FONT_COLORBAR,
        rotation=270,
        labelpad=20,
    )
    cbar_N.ax.tick_params(labelsize=FONT_TICK_3D)

    # ========== ROW 2: 2D CUTS vs L0 ==========
    # Growth rate vs L0
    ax_2d_mu = fig.add_subplot(gs[1, 0:2])

    for C0_val in C0_unique:
        color = colors_C0_dict[C0_val]

        # Plot calibration points for this C0 (convert L0 to µmol/m²/s)
        for (c0, l0), mu in growth_rates.items():
            if abs(c0 - C0_val) < 0.01:
                ax_2d_mu.plot(
                    l0 * L0_REF, mu, "o", color=color, markersize=7, alpha=0.8
                )

        # Plot validation points
        if intermediate_data:
            for (c0, l0), (mu, _) in intermediate_data.items():
                if abs(c0 - C0_val) < 0.01 and (c0, l0) not in calib_keys:
                    ax_2d_mu.plot(
                        l0 * L0_REF, mu, "^", color=color, markersize=9, alpha=0.8
                    )

        # Plot model curve
        L0_curve = np.linspace(L0_min, L0_max, 100)
        mu_curve = growth_rate_haldane_light_only(
            (C0_val * np.ones_like(L0_curve), L0_curve), *growth_rate_monod_params
        )
        ax_2d_mu.plot(
            L0_curve * L0_REF, mu_curve, "-", color=color, linewidth=2, alpha=0.9
        )

    ax_2d_mu.set_xlabel(
        r"Light intensity (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_2D
    )
    ax_2d_mu.set_ylabel(r"$\mu_{\mathrm{max}}$ (h$^{-1}$)", fontsize=FONT_LABEL_2D)
    # ax_2d_mu.grid(True, alpha=0.3)
    ax_2d_mu.grid(False)
    ax_2d_mu.tick_params(labelsize=FONT_TICK_2D)

    # Add legend for marker types (Calibration / Extrapolation)
    ax_2d_mu.plot([], [], "o", color="gray", markersize=7, label="Calibration")
    ax_2d_mu.plot([], [], "^", color="gray", markersize=9, label="Extrapolation")
    ax_2d_mu.legend(
        fontsize=FONT_LEGEND, framealpha=0.95, loc="lower right", frameon=False
    )

    # Steady state vs L0 (scaled by 1e7 for display)
    ax_2d_N = fig.add_subplot(gs[1, 2:4])
    SCALE_N = 1e7

    for C0_val in C0_unique:
        color = colors_C0_dict[C0_val]

        # Plot calibration points for this C0 (convert L0 to µmol/m²/s, scale N by 1e7)
        for (c0, l0), n_max in steady_states.items():
            if abs(c0 - C0_val) < 0.01:
                ax_2d_N.plot(
                    l0 * L0_REF,
                    n_max / SCALE_N,
                    "o",
                    color=color,
                    markersize=7,
                    alpha=0.8,
                )

        # Plot validation points
        if intermediate_data:
            for (c0, l0), (_, n_max) in intermediate_data.items():
                if abs(c0 - C0_val) < 0.01 and (c0, l0) not in calib_keys:
                    ax_2d_N.plot(
                        l0 * L0_REF,
                        n_max / SCALE_N,
                        "^",
                        color=color,
                        markersize=9,
                        alpha=0.8,
                    )

        # Plot model curve
        L0_curve = np.linspace(L0_min, L0_max, 100)
        N_curve = steady_state_haldane(
            (C0_val * np.ones_like(L0_curve), L0_curve), *steady_state_monod_params
        )
        ax_2d_N.plot(
            L0_curve * L0_REF,
            N_curve / SCALE_N,
            "-",
            color=color,
            linewidth=2,
            alpha=0.9,
        )

    ax_2d_N.set_xlabel(
        r"Light intensity (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_2D
    )
    ax_2d_N.set_ylabel(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ $\times 10^7$)", fontsize=FONT_LABEL_2D
    )
    # ax_2d_N.grid(True, alpha=0.3)
    ax_2d_N.grid(False)
    ax_2d_N.tick_params(labelsize=FONT_TICK_2D)

    # Add horizontal colorbar for C0 above the two 2D plots
    cbar_ax_c0 = fig.add_axes([0.30, 0.635, 0.40, 0.015])
    norm_c0 = Normalize(vmin=C0_min, vmax=C0_max)
    sm_c0 = plt.cm.ScalarMappable(norm=norm_c0, cmap=cmap_rdpu_r_truncated)
    sm_c0.set_array([])
    cbar_c0 = plt.colorbar(sm_c0, cax=cbar_ax_c0, orientation="horizontal")
    cbar_c0.set_label(r"$C_0$", fontsize=FONT_COLORBAR)
    cbar_c0.ax.tick_params(labelsize=FONT_TICK_2D)

    # ========== ROW 3: SENSITIVITY ANALYSIS ==========
    # Create mesh for sensitivity calculation (in L0 factor units)
    L0_sens = np.linspace(L0_min, L0_max, 100)
    C0_sens = np.linspace(C0_min, C0_max, 100)
    L0_mesh, C0_mesh = np.meshgrid(L0_sens, C0_sens)

    # L0 in µmol/m²/s for display
    L0_sens_display = L0_sens * L0_REF

    # Calculate analytical derivatives for growth rate (Haldane light-only model)
    # μ = μ_max_ref · [C/(K_C+C)] · [L/(K_L+L+L²/K_I)]
    monod_c = C0_mesh / (k_c_gr + C0_mesh)
    d_monod_c_dC = k_c_gr / (k_c_gr + C0_mesh) ** 2

    haldane_denom = k_l_gr + L0_mesh + L0_mesh**2 / k_i_gr
    haldane_l = L0_mesh / haldane_denom
    d_haldane_l_dL = (k_l_gr - L0_mesh**2 / k_i_gr) / haldane_denom**2

    dmu_dL0 = mu_max_ref_gr * monod_c * d_haldane_l_dL
    dmu_dC0 = mu_max_ref_gr * haldane_l * d_monod_c_dC

    # Calculate analytical derivatives for steady state (Haldane model)
    # N = N_ref · [C/(K_C+C)] · [L/(K_L+L+L²/K_I)]
    monod_c_ss = C0_mesh / (k_c_ss + C0_mesh)
    d_monod_c_ss_dC = k_c_ss / (k_c_ss + C0_mesh) ** 2

    haldane_denom_ss = k_l_ss + L0_mesh + L0_mesh**2 / k_i_ss
    haldane_l_ss = L0_mesh / haldane_denom_ss
    d_haldane_l_ss_dL = (k_l_ss - L0_mesh**2 / k_i_ss) / haldane_denom_ss**2

    dN_dL0 = n_max_ref_ss * monod_c_ss * d_haldane_l_ss_dL
    dN_dC0 = n_max_ref_ss * haldane_l_ss * d_monod_c_ss_dC

    # Use absolute values for LogNorm (derivatives can be negative at high L with inhibition)
    dmu_dL0_abs = np.abs(dmu_dL0)
    dmu_dC0_abs = np.abs(dmu_dC0)
    dN_dL0_abs = np.abs(dN_dL0)
    dN_dC0_abs = np.abs(dN_dC0)

    # Common scales for each variable
    vmin_mu = min(dmu_dL0_abs.min(), dmu_dC0_abs.min())
    vmax_mu = max(dmu_dL0_abs.max(), dmu_dC0_abs.max())
    vmin_N = min(dN_dL0_abs.min(), dN_dC0_abs.min())
    vmax_N = max(dN_dL0_abs.max(), dN_dC0_abs.max())

    # Avoid zero for LogNorm
    vmin_mu = max(vmin_mu, 1e-10)
    vmin_N = max(vmin_N, 1e-10)

    sensitivity_data = [
        (
            dmu_dL0_abs,
            r"$\partial\mu_{\mathrm{max}}/\partial L_0$",
            vmin_mu,
            vmax_mu,
            "YlOrBr",
            "black",
        ),
        (
            dmu_dC0_abs,
            r"$\partial\mu_{\mathrm{max}}/\partial C_0$",
            vmin_mu,
            vmax_mu,
            "YlOrBr",
            "black",
        ),
        (
            dN_dL0_abs,
            r"$\partial N_{\mathrm{max}}/\partial L_0$",
            vmin_N,
            vmax_N,
            "YlGnBu",
            "black",
        ),
        (
            dN_dC0_abs,
            r"$\partial N_{\mathrm{max}}/\partial C_0$",
            vmin_N,
            vmax_N,
            "YlGnBu",
            "white",
        ),
    ]

    for idx, (field, title, vmin, vmax, cmap, contour_color) in enumerate(
        sensitivity_data
    ):
        ax = fig.add_subplot(gs[2, idx])

        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Filled contours (use L0 in µmol/m²/s for display)
        cs = ax.contourf(
            L0_sens_display, C0_sens, field, levels=40, cmap=cmap, norm=norm
        )

        # Contour lines with specific levels for each plot
        field_min = max(field.min(), vmin)
        field_max = field.max()

        if idx == 3:  # ∂N_max/∂C_0 - specific levels, fewer to avoid overlap
            cont_levels = [4e7, 5e7, 7e7]  # Only 3 well-spaced levels
        else:
            cont_levels = np.logspace(np.log10(field_min), np.log10(field_max), 5)[
                1:-1
            ]  # 3 levels

        cont = ax.contour(
            L0_sens_display,
            C0_sens,
            field,
            levels=cont_levels,
            colors=contour_color,
            linewidths=1.5,
            alpha=0.7,
        )

        # Automatic label placement with appropriate spacing
        labels = ax.clabel(
            cont,
            inline=True,
            fontsize=FONT_CONTOUR,
            fmt="%.0e",
            inline_spacing=30,
            colors=contour_color,
            rightside_up=True,
        )

        # Background for labels and prevent clipping at edges
        if labels:
            for label in labels:
                label.set_clip_on(False)  # Prevent labels from being clipped at edges
                if contour_color == "white":
                    label.set_bbox(
                        dict(
                            boxstyle="round,pad=0.3",
                            facecolor="black",
                            edgecolor="none",
                            alpha=0.5,
                        )
                    )
                else:
                    label.set_bbox(
                        dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.7,
                        )
                    )

            # For ∂N_max/∂C_0 (idx==3): adjust label positions manually
            # to avoid overlap between 4e7/5e7 and keep 7e7 within frame
            if idx == 3:
                ax_xlim = ax.get_xlim()
                x_range = ax_xlim[1] - ax_xlim[0]
                for label in labels:
                    txt = label.get_text().strip()
                    x, y = label.get_position()
                    if txt == "7e+07":
                        # Move label lower and to the right, within frame boundary
                        new_x = min(x, ax_xlim[1] - x_range * 0.10)
                        label.set_position((new_x, y * 0.78))
                    elif txt == "5e+07":
                        # Move slightly lower to separate from 4e+07
                        label.set_position((x, y * 0.86))

        ax.set_title(title, fontsize=FONT_TITLE_SENS, fontweight="bold", pad=8)
        ax.set_xlim([L0_sens_display.min(), L0_sens_display.max()])
        ax.set_ylim([C0_sens.min(), C0_sens.max()])

        # Set log scale for C0 axis with custom ticks
        ax.set_yscale("log")
        c0_tick_values = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
        c0_tick_labels = ["1", "1/2", "1/4", "1/8", "1/16"]
        ax.set_yticks(c0_tick_values)
        ax.set_yticklabels(c0_tick_labels)

        ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5, color="gray")

        ax.set_xlabel(
            r"Light intensity (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_SENS
        )
        if idx == 0:
            ax.set_ylabel(r"$C_0$", fontsize=FONT_LABEL_SENS)

        ax.tick_params(labelsize=FONT_TICK_SENS)

    # Add horizontal colorbars at the bottom for sensitivity plots
    # Colorbar for mu_max (panels 0 and 1) - left side
    cbar_ax1 = fig.add_axes([0.08, 0.02, 0.38, 0.015])
    norm_mu_sens = LogNorm(vmin=vmin_mu, vmax=vmax_mu)
    sm_mu = plt.cm.ScalarMappable(norm=norm_mu_sens, cmap="YlOrBr")
    sm_mu.set_array([])
    cbar1 = plt.colorbar(sm_mu, cax=cbar_ax1, orientation="horizontal", format="%.0e")
    cbar1.set_label(
        r"$\mu_{\mathrm{max}}$ sensitivity (h$^{-1}$)", fontsize=FONT_COLORBAR
    )
    cbar1.ax.tick_params(labelsize=FONT_TICK_SENS)

    # Colorbar for N_max (panels 2 and 3) - right side
    cbar_ax2 = fig.add_axes([0.54, 0.02, 0.38, 0.015])
    norm_N_sens = LogNorm(vmin=vmin_N, vmax=vmax_N)
    sm_N = plt.cm.ScalarMappable(norm=norm_N_sens, cmap="YlGnBu")
    sm_N.set_array([])
    cbar2 = plt.colorbar(sm_N, cax=cbar_ax2, orientation="horizontal", format="%.0e")
    cbar2.set_label(
        r"$N_{\mathrm{max}}$ sensitivity (cells mL$^{-1}$)", fontsize=FONT_COLORBAR
    )
    cbar2.ax.tick_params(labelsize=FONT_TICK_SENS)

    # Save
    plt.tight_layout()
    output_path = pathlib.Path(output_dir) / "RSM_full_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved RSM full panel to {output_path}")
    plt.close()


def evaluate_rsm_extrapolation_by_L0_plate(
    growth_rate_monod_params: tuple,
    steady_state_monod_params: tuple,
    manual_tlag_adjustments: dict,
    conv_OD_plate_to_OD_erlen: float = 6.01,
    conv_OD_to_cell: float = 4.77e6,
    output_dir: str = "results_plates",
) -> None:
    """
    Evaluates the extrapolation capacity of the RSM model for plates.
    Creates a row of 3 plots, one for each L0 value (170, 102, 51 µmol/m²/s).
    Each plot contains 6 curves (C0 = 1, 0.95, 0.75, 0.55, 0.30, 0.0625).
    Replicates are differentiated by markers.

    Args:
        growth_rate_monod_params: Haldane parameters for µ_max (mu_max_ref, k_c, k_l, k_i)
        steady_state_monod_params: Haldane parameters for N_max (n_max_ref, k_c, k_l, k_i)
        manual_tlag_adjustments: dict {(L0_factor, C0_factor): t_lag_hours}
        conv_OD_plate_to_OD_erlen: OD plate → OD erlen conversion factor
        conv_OD_to_cell: OD → cells/mL conversion factor
        output_dir: output directory
    """
    from matplotlib.lines import Line2D

    # =========================================================================
    # GLOBAL FONT SIZE VARIABLES
    # =========================================================================
    FONT_TITLE = 23
    FONT_LABEL = 23
    FONT_TICK = 20
    FONT_LEGEND = 17

    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION EVALUATION BY L0 (PLATES)")
    logger.info(f"{'=' * 80}\n")

    # L0 configurations and corresponding files
    # L0 values in µmol/m²/s and their corresponding L0_factor and dates
    L0_configs = [
        {"L0_display": 170, "L0_factor": 1.0, "date": "07_07_2025"},
        {"L0_display": 102, "L0_factor": 0.6, "date": "01_07_2025"},
        {"L0_display": 51, "L0_factor": 0.3, "date": "17_02_2025"},
    ]

    # C0 values to plot
    C0_values_to_plot = [1.0, 0.95, 0.75, 0.55, 0.30, 0.0625]

    # Calibration C0 values (others are extrapolation)
    calibration_C0 = {1.0, 0.5, 0.25, 0.125, 0.0625}

    # Define colors per C0
    colors_C0 = {
        1.0000: "mediumseagreen",
        0.9500: "dodgerblue",
        0.7500: "orange",
        0.5500: "tomato",
        0.3000: "purple",
        0.0625: "teal",
    }

    # Define markers per replicate (R1-R5)
    markers_rep = {
        "R1": "o",  # circle
        "R2": "^",  # triangle up
        "R3": "s",  # square
        "R4": "D",  # diamond
        "R5": "v",  # triangle down
    }

    # =========================================================================
    # LOAD INTERMEDIATE DATA
    # =========================================================================
    all_data = {}  # {(L0_factor, C0): {'time': [], 'replicates': [{name, values}], 'mean': []}}

    for config in L0_configs:
        date = config["date"]
        l0_factor = config["L0_factor"]

        # Find the corresponding file
        file_pattern = f"all_data/final_corrected_data/replicate_OD_intermediate_dilution_{date}.csv"
        matching_files = glob.glob(file_pattern)

        if not matching_files:
            logger.warning(
                f"No file found for date {date} (L0={config['L0_display']} µmol/m²/s)"
            )
            continue

        fp = matching_files[0]
        logger.info(f"Loading data from {fp}")

        try:
            df = pd.read_csv(fp, sep=";")
            time_col = df.columns[0]
            time = df[time_col].values

            # For each desired C0, load the data
            for c0_target in C0_values_to_plot:
                # Find matching columns
                replicate_cols = []
                for col in df.columns:
                    if "C0 = " in col:
                        try:
                            col_c0_str = col.split("C0 = ")[1].split(" ")[0]
                            col_c0 = float(col_c0_str)
                            if abs(col_c0 - c0_target) < 0.01:
                                replicate_cols.append(col)
                        except (ValueError, IndexError):
                            pass

                if not replicate_cols:
                    logger.debug(f"No data for C0={c0_target} at L0_factor={l0_factor}")
                    continue

                # Load replicates
                replicates_list = []
                replicates_data = []
                for col in replicate_cols:
                    od_values = df[col].values
                    cell_conc = od_values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                    # Extract replicate name (R1, R2, etc.)
                    rep_name = col.split()[-1] if " " in col else "R1"
                    replicates_list.append(
                        {
                            "name": rep_name,
                            "time": time.copy(),
                            "values": cell_conc.copy(),
                        }
                    )
                    replicates_data.append(cell_conc)

                # Calculate the mean
                mean_conc = []
                for t_idx in range(len(time)):
                    values_at_t = [
                        rep[t_idx] for rep in replicates_data if rep[t_idx] > 0
                    ]
                    if len(values_at_t) > 0:
                        mean_conc.append(trimmed_mean(values_at_t))
                    else:
                        mean_conc.append(np.nan)

                all_data[(l0_factor, c0_target)] = {
                    "time": time.copy(),
                    "replicates": replicates_list,
                    "mean": np.array(mean_conc),
                }

        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")

    # =========================================================================
    # LOAD HAND_CLEANED DATA FOR C0 = 0.0625 (1/16)
    # =========================================================================
    # Mapping C0 values to their fractional representation in hand_cleaned files
    c0_fraction_map = {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2", 1.0: "1"}

    # Mapping dates to hand_cleaned files
    date_to_hand_cleaned = {
        "07_07_2025": [
            "all_data/hand_cleaned/replicates_OD_07_07_2025_plate_1.csv",
            "all_data/hand_cleaned/replicates_OD_07_07_2025_plate_2.csv",
        ],
        "01_07_2025": [
            "all_data/hand_cleaned/replicates_OD_01_07_2025_plate_1.csv",
            "all_data/hand_cleaned/replicates_OD_01_07_2025_plate_2.csv",
        ],
        "17_02_2025": [
            "all_data/hand_cleaned/replicates_OD_17_02_2025_plate_1.csv",
            "all_data/hand_cleaned/replicates_OD_17_02_2025_plate_2.csv",
        ],
    }

    for config in L0_configs:
        date = config["date"]
        l0_factor = config["L0_factor"]

        if date not in date_to_hand_cleaned:
            continue

        # C0 values that need to be loaded from hand_cleaned (not in intermediate files)
        c0_from_hand_cleaned = [0.0625]  # Only 0.0625 is missing from intermediate

        for c0_target in c0_from_hand_cleaned:
            if c0_target not in C0_values_to_plot:
                continue

            key = (l0_factor, c0_target)
            if key in all_data:
                continue  # Already loaded

            c0_fraction = c0_fraction_map.get(c0_target)
            if c0_fraction is None:
                continue

            replicates_list = []
            replicates_data = []
            time_ref = None

            for hc_file in date_to_hand_cleaned[date]:
                if not pathlib.Path(hc_file).exists():
                    continue

                try:
                    df = pd.read_csv(hc_file, sep=";")
                    time_col = df.columns[0]
                    time = df[time_col].values

                    if time_ref is None:
                        time_ref = time.copy()

                    # Find columns matching C0 = 1/16 (or other fraction)
                    for col in df.columns:
                        if f"C0 = {c0_fraction} " in col:
                            od_values = df[col].values
                            cell_conc = (
                                od_values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                            )
                            rep_name = (
                                col.split()[-1]
                                if " " in col
                                else f"R{len(replicates_list) + 1}"
                            )
                            replicates_list.append(
                                {
                                    "name": rep_name,
                                    "time": time.copy(),
                                    "values": cell_conc.copy(),
                                }
                            )
                            replicates_data.append(cell_conc)

                except Exception as e:
                    logger.warning(
                        f"Failed to load hand_cleaned data from {hc_file}: {e}"
                    )

            if replicates_list and time_ref is not None:
                # Calculate mean
                mean_conc = []
                for t_idx in range(len(time_ref)):
                    values_at_t = [
                        rep[t_idx]
                        for rep in replicates_data
                        if t_idx < len(rep) and rep[t_idx] > 0
                    ]
                    if len(values_at_t) > 0:
                        mean_conc.append(trimmed_mean(values_at_t))
                    else:
                        mean_conc.append(np.nan)

                all_data[key] = {
                    "time": time_ref,
                    "replicates": replicates_list,
                    "mean": np.array(mean_conc),
                }
                logger.info(
                    f"Loaded C0={c0_target} from hand_cleaned for L0_factor={l0_factor}"
                )

    if not all_data:
        logger.error("No data loaded for extrapolation evaluation")
        return

    # =========================================================================
    # COMPUTE COMMON Y-AXIS LIMITS
    # =========================================================================
    y_min_global = float("inf")
    y_max_global = float("-inf")

    for (l0, c0), data in all_data.items():
        for rep in data["replicates"]:
            values = rep["values"]
            valid_values = values[values > 0]
            if len(valid_values) > 0:
                y_min_global = min(y_min_global, np.nanmin(valid_values))
                y_max_global = max(y_max_global, np.nanmax(valid_values))

    # Add a 5% margin
    y_range = y_max_global - y_min_global
    y_min_plot = max(0, y_min_global - 0.05 * y_range)
    y_max_plot = y_max_global + 0.05 * y_range

    # Convert to 10^7 for display
    y_min_plot_scaled = y_min_plot / 1e7
    y_max_plot_scaled = y_max_plot / 1e7

    logger.info(f"Global Y-axis range: {y_min_plot:.2e} to {y_max_plot:.2e} cells/mL")

    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # For each L0 value, create a plot
    for idx_L0, config in enumerate(L0_configs):
        ax = axes[idx_L0]
        L0_display = config["L0_display"]
        l0_factor = config["L0_factor"]

        # For each C0 in this L0
        for c0 in C0_values_to_plot:
            key = (l0_factor, c0)

            if key not in all_data:
                continue

            data = all_data[key]
            time = data["time"]

            # Color for this C0 value
            color = colors_C0.get(c0, "black")

            # Determine whether this is a calibration or extrapolation condition
            is_calibration = c0 in calibration_C0
            linestyle = "-" if is_calibration else "--"

            # Plot experimental points for each replicate
            for rep in data["replicates"]:
                rep_name = rep["name"]
                rep_time = rep["time"]
                rep_values = rep["values"]

                marker = markers_rep.get(rep_name, "o")

                # Plot points (convert to 10^7)
                ax.plot(
                    rep_time,
                    rep_values / 1e7,
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    markersize=6,
                    linestyle="none",
                )

            # Compute N0 from the first valid point of the mean
            mean_conc = data["mean"]
            valid_mask = mean_conc > 0
            if np.sum(valid_mask) == 0:
                continue
            N0 = mean_conc[valid_mask][0]

            # Compute t_max for this condition
            t_max = np.nanmax(time)

            # Get the adjusted t_lag
            tlag_key = (l0_factor, c0)
            t_lag = 0.0
            if manual_tlag_adjustments and tlag_key in manual_tlag_adjustments:
                t_lag = manual_tlag_adjustments[tlag_key]

            # Compute µ_max and N_max from RSM (Haldane) models
            # L0 must be in factor units for the models
            mu_max_pred = growth_rate_haldane_light_only(
                (c0, l0_factor), *growth_rate_monod_params
            )
            N_max_pred = steady_state_haldane(
                (c0, l0_factor), *steady_state_monod_params
            )

            # Compute the logistic curve
            time_model = np.linspace(0, t_max, 500)
            t_shifted = time_model - t_lag
            # Logistic model: N(t) = Nmax / (1 + ((Nmax - N0) / N0) * exp(-mu_max * (t - t_lag)))
            N_model = N_max_pred / (
                1 + ((N_max_pred - N0) / N0) * np.exp(-mu_max_pred * t_shifted)
            )

            # Plot model curve
            ax.plot(
                time_model,
                N_model / 1e7,
                color=color,
                linewidth=2.5,
                linestyle=linestyle,
                alpha=0.9,
            )

        # Plot configuration
        ax.set_title(
            rf"$L_{{0}}$ = {L0_display} µmol$_{{h\nu}}$ m$^{{-2}}$ s$^{{-1}}$",
            fontsize=FONT_TITLE,
            pad=20,
            fontweight="bold",
        )
        ax.set_xlabel("Time (h)", fontsize=FONT_LABEL)

        # Y label only on the first plot
        if idx_L0 == 0:
            ax.set_ylabel(
                r"Biomass (cells mL$^{-1}$ × $10^{7}$)",
                fontsize=FONT_LABEL,
                labelpad=20,
            )

        # Apply common Y-axis limits
        ax.set_ylim([y_min_plot_scaled, y_max_plot_scaled])

        # Hide Y-axis tick labels on all plots except the first
        if idx_L0 != 0:
            ax.set_yticklabels([])

        ax.grid(False)
        ax.tick_params(labelsize=FONT_TICK)

    # =========================================================================
    # CREATE LEGEND
    # =========================================================================
    legend_elements = []

    # Add C0 colors
    for C0 in C0_values_to_plot:
        color = colors_C0.get(C0, "black")
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=f"$C_{{0}}$={C0:.4f}")
        )

    # Add a separator
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Add replicate markers
    for rep, marker in markers_rep.items():
        # Extract replicate number (R1 -> 1, R2 -> 2, etc.)
        rep_num = rep[1:] if rep.startswith("R") else rep
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                linestyle="none",
                markersize=8,
                label=f"Replicate {rep_num}",
            )
        )

    # Add a separator
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Add RSM curve types
    legend_elements.append(
        Line2D(
            [0], [0], color="gray", linewidth=2.5, linestyle="-", label="Calibration"
        )
    )
    legend_elements.append(
        Line2D(
            [0], [0], color="gray", linewidth=2.5, linestyle="--", label="Extrapolation"
        )
    )

    # Place legend to the right of all plots
    fig.legend(
        handles=legend_elements,
        loc="center right",
        fontsize=FONT_LEGEND,
        framealpha=0.9,
        bbox_to_anchor=(0.985, 0.5),
    )

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])

    output_path = pathlib.Path(output_dir) / "rsm_extrapolation_by_L0.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"RSM extrapolation by L0 figure saved: {output_path}")
    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION BY L0 (PLATES) COMPLETED")
    logger.info(f"{'=' * 80}\n")


def main(args=None):
    # Parse command line arguments
    if args is None:
        parser = argparse.ArgumentParser(
            description="Run parameter tuning for microalgae plate ODE models"
        )
        parser.add_argument(
            "--over",
            action="store_true",
            help="Create overlay plot (over.png) that combines existing and new combined plots",
        )
        args = parser.parse_args()

    # Create results directory if it doesn't exist
    results_dir = pathlib.Path("results_plates")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir.absolute()}")

    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Manual t_lag adjustments for specific (L0_factor, C0_factor) conditions
    # Positive values shift the start of the logistic fit later (increase t_lag)
    # Negative values shift the start earlier (decrease t_lag)
    # Format: {(L0_factor, C0_factor): t_lag_adjustment_hours}
    manual_tlag_adjustments = {
        # L0 = 1.0 (170 µmol/m²/s)
        (1, 1.0): 9,  # 8.5,
        (1, 0.5): 0,
        (1, 0.250): 5,
        (1, 0.125): 8,
        (1, 0.0625): 15,
        (1, 0.95): 10,  # -2
        (1, 0.90): -2,
        (1, 0.85): -2,
        (1, 0.80): -2,
        (1, 0.75): -2,
        (1, 0.70): -2,
        (1, 0.65): -2,
        (1, 0.60): -2,
        (1, 0.55): -4,  # 1, #-4,
        (1, 0.45): -14,
        (1, 0.40): -8,  # ok
        (1, 0.35): -8,  # -9
        (1, 0.3): 0,  # 0
        (1, 0.20): -10,
        (1, 0.15): -12,
        (1, 0.125): -8,
        (1, 0.0625): -10,  # -15
        # L0 = 0.6 (102 µmol/m²/s)
        (0.6, 1.0): 5,
        (0.6, 0.95): 7,
        (0.6, 0.9): 12,
        (0.6, 0.85): 4,
        (0.6, 0.8): 7,
        (0.6, 0.5): 5,
        (0.6, 0.45): -9,  # -5
        (0.6, 0.4): -5,  # -3,
        (0.6, 0.35): -5,  # 0
        (0.6, 0.30): -6,  # -5,
        (0.6, 0.25): 4,  # -3,
        (0.6, 0.20): -5,
        (0.6, 0.15): -20,
        (0.6, 0.125): -15,
        (0.6, 0.0625): 5,
        # L0 = 0.3 (51 µmol/m²/s)
        (0.3, 1.0): -1,  # 0
        (0.3, 0.95): -5,
        (0.3, 0.9): -5,
        (0.3, 0.85): -5,
        (0.3, 0.8): -4,  # -2
        (0.3, 0.75): -7,
        (0.3, 0.70): -7,
        (0.3, 0.65): -5,
        (0.3, 0.60): -9,
        (0.3, 0.55): -16,  # -13
        (0.3, 0.5): -5,  # -3
        (0.3, 0.45): -15,  # -10
        (0.3, 0.40): -10,  # -5
        (0.3, 0.35): -10,  # -5
        (0.3, 0.30): -12,
        (0.3, 0.25): -5,
        (0.3, 0.20): -5,
        (0.3, 0.15): -34,  # -34
        (0.3, 0.125): -5,  # 0
        (0.3, 0.0625): -5,
        # L0 = 0.15 (25.5 µmol/m²/s) - MOVED from 0.07 after mapping fix
        # These adjustments were originally for (0.07, C0) but those were the 21/10 and 04/11 data
        # which are actually at L0=25.5 µmol/m²/s = 0.15×L0_REF
        (0.15, 1): -4,  # -1
        (0.15, 0.95): -11,
        (0.15, 0.9): 1,
        (0.15, 0.85): 1,
        (0.15, 0.8): 1,
        (0.15, 0.75): -1,
        (0.15, 0.70): 5,
        (0.15, 0.65): -20,
        (0.15, 0.60): -1,
        (0.15, 0.55): -2,
        (0.15, 0.50): -11,  # -9,
        (0.15, 0.45): -1,
        (0.15, 0.40): -17,  # -4,
        (0.15, 0.35): -4,
        (0.15, 0.30): 1,
        (0.15, 0.25): -18,  # -17,
        (0.15, 0.20): -10,  # 0
        (0.15, 0.15): 0,
        (0.15, 0.125): -25,  # -20,
        (0.15, 0.0625): -5,
        (0.07, 1): -25,  # -9,
        (0.07, 0.5): -9,
        (0.07, 0.25): -9,
        (0.07, 0.125): -9,
        (0.07, 0.0625): -9,
    }

    # get all the hand_cleaned plate data files
    # Exclude 16_09_2024 (L=0.07) - single plate experiment with anomalous growth depression
    all_plate_files = sorted(glob.glob("all_data/hand_cleaned/replicates_OD_*.csv"))
    plate_files = [f for f in all_plate_files if "16_09_2024" not in f]

    if not plate_files:
        logger.warning("No plate data files found in all_data/hand_cleaned/")
        return

    # Read plate data from hand_cleaned directory
    experiments_plate: Experiments = {}
    for fp in plate_files:
        try:
            new_exp = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell,
            )
            experiments_plate.update(new_exp)
            logger.info(f"Loaded cleaned plate data from {fp}")
        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")

    if experiments_plate:
        # Plot intermediate exponential fits for quality control
        plot_intermediate_exponential_fits(conv_OD_plate_to_OD_erlen, conv_OD_to_cell)

        # Analyze and plot growth rates
        growth_rates = plot_growth_rate_analysis(experiments_plate)

        # Fit Monod model to growth rates (with outlier exclusion)
        growth_rate_monod_params = None
        if growth_rates:
            growth_rate_monod_params = fit_growth_rate_monod(growth_rates)

        # Analyze and plot steady states (first pass without fit)
        steady_states = plot_steady_state_analysis(experiments_plate)

        # Fit Monod model and get parameters to plot on steady state figure
        monod_params = None
        if steady_states:
            monod_params = fit_steady_state_monod(steady_states)
            # Re-plot steady states with fitted model overlaid
            if monod_params is not None:
                steady_states = plot_steady_state_analysis(
                    experiments_plate, monod_params=monod_params
                )

        # Plot growth rates and steady states with fitted model overlaid
        if growth_rates and growth_rate_monod_params:
            plot_growth_rate_with_fit(growth_rates, growth_rate_monod_params)

        if steady_states and monod_params:
            plot_steady_state_with_fit(steady_states, monod_params)

        # Create full RSM panel with 3D surfaces, 2D cuts, and sensitivity analysis
        if growth_rates and steady_states and monod_params and growth_rate_monod_params:
            create_full_rsm_panel_plate(
                growth_rates, steady_states, growth_rate_monod_params, monod_params
            )

        # Generate extended plot with all C0 values including intermediate dilutions
        if growth_rates and steady_states and monod_params and growth_rate_monod_params:
            # Load intermediate data with complete timeseries for validation
            intermediate_data_ts = load_intermediate_data_with_timeseries(
                conv_OD_plate_to_OD_erlen, conv_OD_to_cell
            )
            if intermediate_data_ts:
                logger.info(
                    f"Creating extended plot with {len(intermediate_data_ts)} intermediate data points"
                )
                plot_simulation_extended(
                    experiments_plate,
                    growth_rates,
                    steady_states,
                    intermediate_data_ts,
                    monod_params=monod_params,
                    growth_rate_monod_params=growth_rate_monod_params,
                    manual_tlag_adjustments=manual_tlag_adjustments,
                )

                # Generate RSM extrapolation evaluation by L0 figure
                evaluate_rsm_extrapolation_by_L0_plate(
                    growth_rate_monod_params=growth_rate_monod_params,
                    steady_state_monod_params=monod_params,
                    manual_tlag_adjustments=manual_tlag_adjustments,
                    conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                    conv_OD_to_cell=conv_OD_to_cell,
                )
            else:
                logger.warning("No intermediate data available for extended plot")
    else:
        logger.warning("No plate data was successfully loaded")


if __name__ == "__main__":
    main()
