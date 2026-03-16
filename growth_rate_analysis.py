"""
Script to calculate specific growth rates from microalgae growth data
and plot growth rate vs normalized nutrient concentration.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple
import warnings
import os

warnings.filterwarnings("ignore")


def load_and_process_data(filepath: str) -> pd.DataFrame:
    """Load and process the CSV data from the growth experiment."""
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

    # Clean column names and remove special characters
    df.columns = [col.replace("﻿", "").strip() for col in df.columns]

    return df


def extract_growth_data(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract time and OD data for each dilution condition."""
    growth_data = {}

    # The dilution factors correspond to nutrient concentrations
    # dilution 0 = C0*1.000, dilution 1 = C0*0.500, etc.
    dilution_to_c0 = {0: 1.000, 1: 0.500, 2: 0.250, 3: 0.125, 4: 0.062}

    # Get time data (convert from seconds to hours)
    time_col = "Time (s)"
    if time_col in df.columns:
        time_seconds = df[time_col].dropna()
        time_hours = time_seconds / 3600  # convert to hours

        # Extract mean OD values for each dilution
        for dilution, c0_factor in dilution_to_c0.items():
            mean_col = f"mean {dilution}"
            if mean_col in df.columns:
                od_values = df[mean_col].dropna()

                # Ensure same length as time
                min_length = min(len(time_hours), len(od_values))
                if min_length > 0:
                    growth_data[f"C0*{c0_factor}"] = (
                        time_hours.iloc[:min_length].values,
                        od_values.iloc[:min_length].values,
                    )

    return growth_data


def calculate_specific_growth_rate(
    time: np.ndarray, concentration: np.ndarray, time_window: Tuple[float, float] = None
) -> float:
    """
    Calculate specific growth rate (μ) from concentration data (cells/mL or OD).
    μ = (ln(C2) - ln(C1)) / (t2 - t1)

    If time_window is provided, use only data within that window.
    Otherwise, use exponential phase (detect automatically).
    """
    # Filter out zero or negative values
    valid_mask = concentration > 0
    time_valid = time[valid_mask]
    concentration_valid = concentration[valid_mask]

    if len(time_valid) < 3:
        return np.nan

    # If time window specified, filter data
    if time_window is not None:
        t_start, t_end = time_window
        window_mask = (time_valid >= t_start) & (time_valid <= t_end)
        time_valid = time_valid[window_mask]
        concentration_valid = concentration_valid[window_mask]

        if len(time_valid) < 3:
            return np.nan

    # Calculate log(concentration) and fit linear regression
    ln_concentration = np.log(concentration_valid)

    # Use linear regression to find growth rate
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_valid, ln_concentration
        )

        # Only accept if R² > 0.8 for good fit
        if r_value**2 > 0.8:
            return slope  # This is μ (specific growth rate)
        else:
            return np.nan
    except:
        return np.nan


def find_exponential_phase(
    time: np.ndarray, concentration: np.ndarray
) -> Tuple[float, float]:
    """
    Automatically detect the exponential growth phase.
    Returns time window (start, end) for exponential phase.
    Prioritizes early exponential phase detection.
    """
    # Filter out zero or negative values
    valid_mask = concentration > 0
    time_valid = time[valid_mask]
    concentration_valid = concentration[valid_mask]

    if len(time_valid) < 5:
        return time_valid[0], time_valid[-1]

    ln_concentration = np.log(concentration_valid)

    # Strategy: Look for exponential phase in early to mid time points
    # Most exponential growth happens in the first 50-70% of the experiment
    max_end_idx = int(0.7 * len(time_valid))  # Don't go beyond 70% of data

    # Use a sliding window approach to find the most linear region
    window_size = max(
        5, len(time_valid) // 6
    )  # Smaller windows for more precise detection
    best_r2 = 0
    best_start_idx = 0
    best_end_idx = len(time_valid) - 1

    # Prioritize earlier start times by adding a bias
    for start_idx in range(
        min(len(time_valid) // 3, len(time_valid) - window_size)
    ):  # Start in first third
        for window_len in range(
            window_size,
            min(max_end_idx - start_idx + 1, len(time_valid) - start_idx + 1),
        ):
            end_idx = start_idx + window_len - 1

            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_valid[start_idx : end_idx + 1],
                    ln_concentration[start_idx : end_idx + 1],
                )

                if slope > 0:  # Only positive growth rates
                    # Add bias for earlier start times and good R²
                    early_bias = 1.2 - (
                        start_idx / len(time_valid)
                    )  # Earlier = higher bias
                    adjusted_r2 = r_value**2 * early_bias

                    if adjusted_r2 > best_r2 and r_value**2 > 0.7:  # Require decent fit
                        best_r2 = adjusted_r2
                        best_start_idx = start_idx
                        best_end_idx = end_idx
            except:
                continue

    return time_valid[best_start_idx], time_valid[best_end_idx]


def calculate_theoretical_growth_rates(
    c0_factors: list, base_c0: float, mu: float, gamma: float
) -> list:
    """
    Calculate theoretical growth rates using Monod kinetics: μ * C / (Gamma + C)

    Args:
        c0_factors: List of C0 multiplication factors (e.g., [1.0, 0.5, 0.25, ...])
        base_c0: Base C0 concentration in cells/mL
        mu: Maximum specific growth rate
        gamma: Half-saturation constant

    Returns:
        List of theoretical growth rates
    """
    theoretical_rates = []
    for factor in c0_factors:
        C = base_c0 * factor  # Actual nutrient concentration
        theoretical_mu = mu * C / (gamma + C)  # Monod kinetics
        theoretical_rates.append(theoretical_mu)

    return theoretical_rates


def calculate_logistic_growth_rates(
    c0_factors: list, base_c0: float, mu: float
) -> list:
    """
    Calculate theoretical growth rates using logistic model: μ * C / C₀
    This gives a linear relationship with concentration.

    Args:
        c0_factors: List of C0 multiplication factors (e.g., [1.0, 0.5, 0.25, ...])
        base_c0: Base C0 concentration in cells/mL
        mu: Maximum specific growth rate

    Returns:
        List of theoretical growth rates
    """
    theoretical_rates = []
    for factor in c0_factors:
        C = base_c0 * factor  # Actual nutrient concentration
        theoretical_mu = mu * C / base_c0  # Logistic: μ * C/C₀ = μ * factor
        theoretical_rates.append(theoretical_mu)

    return theoretical_rates


def monod_model(C, mu_max, gamma):
    """
    Monod kinetics model function for curve fitting.
    μ = μ_max * C / (γ + C)

    Args:
        C: Nutrient concentration (cells/mL)
        mu_max: Maximum specific growth rate (h^-1)
        gamma: Half-saturation constant (cells/mL)

    Returns:
        Specific growth rate (h^-1)
    """
    return mu_max * C / (gamma + C)


def fit_monod_parameters(c0_concentrations: list, experimental_mu: list) -> tuple:
    """
    Fit Monod model parameters to experimental data using least squares optimization.

    Args:
        c0_concentrations: List of nutrient concentrations (cells/mL)
        experimental_mu: List of experimental growth rates (h^-1)

    Returns:
        Tuple of (fitted_mu_max, fitted_gamma, r_squared, fitted_params_std_errors)
    """
    try:
        # Initial guess for parameters
        mu_max_guess = max(experimental_mu) * 1.2  # Slightly higher than max observed
        gamma_guess = np.median(
            c0_concentrations
        )  # Median concentration as initial guess

        # Fit the curve
        popt, pcov = curve_fit(
            monod_model,
            c0_concentrations,
            experimental_mu,
            p0=[mu_max_guess, gamma_guess],
            bounds=([0, 0], [1.0, 1e10]),  # Reasonable bounds for biological parameters
        )

        fitted_mu_max, fitted_gamma = popt

        # Calculate R²
        predicted_mu = monod_model(
            np.array(c0_concentrations), fitted_mu_max, fitted_gamma
        )
        ss_res = np.sum((experimental_mu - predicted_mu) ** 2)
        ss_tot = np.sum((experimental_mu - np.mean(experimental_mu)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate standard errors
        param_errors = np.sqrt(np.diag(pcov))

        return fitted_mu_max, fitted_gamma, r_squared, param_errors

    except Exception as e:
        print(f"Error in curve fitting: {e}")
        return None, None, None, None


def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters from ode_tuning.py
    conv_OD_to_cell = 4.77e6  # OD to cells/mL conversion factor

    # Theoretical model parameters (from erlen_bounds in tuning_ODE.py)
    mu_max = 0.17  # h^-1, maximum specific growth rate
    Gamma = 1e5  # cells/mL, half-saturation constant
    base_C0 = 5.5e7  # cells/mL, base nutrient concentration, from perfect light condition (L = 300) in Erlen flask data

    # Load and process data
    # The Erlen data:
    # C0 is encoded as the key in column name
    # 16-09-24: L0 = 22 µmol/m2/s
    # 21-10-24: L0 = 45 µmol/m2/s
    # 04-11-24: L0 = 45 µmol/m2/s
    # 17-02-25: L0 = 90 µmol/m2/s
    # 01-07-25: L0 = 180 µmol/m2/s
    # 07-07-25: L0 = 300 µmol/m2/s
    #
    # These are in the files:
    # "all_data/data_exp_Chlamy_07-07-25.csv"
    # "all_data/data_exp_Chlamy_01-07-25.csv",
    # "all_data/data_exp_Chlamy_17-02-25.csv",
    # dont fit for now:
    # "all_data/data_exp_Chlamy_21-10-24.csv",
    # "all_data/data_exp_Chlamy_04-11-24.csv",
    # "all_data/data_exp_Chlamy_16-09-24.csv"

    # 07-07-25: L0 = 300 µmol/m2/s
    # filepath = "all_data/data_exp_Chlamy_07-07-25.csv"

    # 01-07-25: L0 = 180 µmol/m2/s
    filepath = "all_data/data_exp_Chlamy_07-07-25.csv"

    # 17-02-25: L0 = 90 µmol/m2/s
    # filepath = "all_data/data_exp_Chlamy_17-02-25.csv",

    # load it
    filepath = str(Path("..") / filepath)
    df = load_and_process_data(filepath)

    # Extract growth data for each condition
    growth_data = extract_growth_data(df)

    if not growth_data:
        print("No growth data found. Please check the CSV file structure.")
        return

    print("Found growth data for conditions:", list(growth_data.keys()))
    print(
        "\nNote: Using original (uncorrected) data due to issues with linearity correction:"
    )
    print("- Correction function produces negative OD values (physically impossible)")
    print("- Function: corrected_OD = -1.40 * OD / (OD - 2.68) has singularities")
    print("- Results in unrealistic growth rates and data artifacts")

    # Calculate specific growth rates
    growth_rates = {}
    c0_concentrations = []

    for condition, (time, od) in growth_data.items():
        print(f"\nAnalyzing condition: {condition}")
        print(f"Time range: {time[0]:.1f} - {time[-1]:.1f} hours")
        print(f"OD range: {od.min():.4f} - {od.max():.4f}")

        # Convert OD to cell concentration for growth rate calculation
        cells_per_ml = od * conv_OD_to_cell
        print(
            f"Cell concentration range: {cells_per_ml.min():.2e} - {cells_per_ml.max():.2e} cells/mL"
        )

        # Find exponential phase automatically (using cell concentration)
        exp_start, exp_end = find_exponential_phase(time, cells_per_ml)
        print(f"Exponential phase detected: {exp_start:.1f} - {exp_end:.1f} hours")

        # Calculate growth rate in exponential phase (using cell concentration)
        mu = calculate_specific_growth_rate(time, cells_per_ml, (exp_start, exp_end))

        if not np.isnan(mu):
            growth_rates[condition] = mu
            # Extract C0 concentration from condition name
            c0_factor = float(condition.split("*")[1])
            c0_concentrations.append(c0_factor)
            print(f"Specific growth rate (μ): {mu:.4f} h⁻¹ (calculated from cells/mL)")
        else:
            print("Could not calculate reliable growth rate")

    # Print results summary
    print("\n" + "=" * 50)
    print("GROWTH RATE SUMMARY")
    print("=" * 50)

    # Sort by C0 concentration
    sorted_conditions = sorted(
        growth_rates.keys(), key=lambda x: float(x.split("*")[1]), reverse=True
    )

    for condition in sorted_conditions:
        mu = growth_rates[condition]
        c0_factor = float(condition.split("*")[1])
        print(f"{condition:<12}: μ = {mu:.4f} h⁻¹")

    # Create plot
    if len(growth_rates) > 1:
        # Prepare data for plotting
        c0_values = []
        mu_values = []

        for condition in sorted_conditions:
            c0_factor = float(condition.split("*")[1])
            mu = growth_rates[condition]
            c0_values.append(c0_factor)
            mu_values.append(mu)

        # Convert C0 factors to actual cell concentrations
        c0_cell_concentrations = [base_C0 * factor for factor in c0_values]

        # Fit Monod parameters to experimental data
        print("\n" + "=" * 60)
        print("FITTING MONOD MODEL TO EXPERIMENTAL DATA")
        print("=" * 60)

        fitted_mu_max, fitted_gamma, r_squared_fit, param_errors = fit_monod_parameters(
            c0_cell_concentrations, mu_values
        )

        if fitted_mu_max is not None:
            print(f"Fitted Parameters:")
            print(f"  μ_max = {fitted_mu_max:.4f} ± {param_errors[0]:.4f} h⁻¹")
            print(f"  Γ = {fitted_gamma:.2e} ± {param_errors[1]:.2e} cells/mL")
            print(f"  R² = {r_squared_fit:.4f}")

            # Calculate fitted theoretical growth rates
            fitted_theoretical_mu = monod_model(
                np.array(c0_cell_concentrations), fitted_mu_max, fitted_gamma
            )
        else:
            print("Failed to fit Monod model parameters")
            fitted_mu_max, fitted_gamma = mu_max, Gamma
            fitted_theoretical_mu = calculate_theoretical_growth_rates(
                c0_values, base_C0, mu_max, Gamma
            )

        # Calculate original theoretical growth rates for comparison
        original_theoretical_mu = calculate_theoretical_growth_rates(
            c0_values, base_C0, mu_max, Gamma
        )

        # Calculate logistic growth rates for comparison (using fitted mu_max if available)
        logistic_mu_param = fitted_mu_max if fitted_mu_max is not None else mu_max
        logistic_theoretical_mu = calculate_logistic_growth_rates(
            c0_values, base_C0, logistic_mu_param
        )

        # Create smooth curve for plotting - limit to experimental data range
        C_min = min(c0_cell_concentrations) * 0.5  # Slightly below minimum
        C_max = max(c0_cell_concentrations) * 1.1  # Slightly above maximum
        C_smooth = np.linspace(
            C_min, C_max, 100
        )  # Linear spacing within experimental range
        C_factors_smooth = C_smooth / base_C0  # Convert to factors for logistic
        if fitted_mu_max is not None:
            mu_smooth_fitted = monod_model(C_smooth, fitted_mu_max, fitted_gamma)
        mu_smooth_original = monod_model(C_smooth, mu_max, Gamma)
        mu_smooth_logistic = (
            logistic_mu_param * C_smooth / base_C0
        )  # Linear relationship

        # Create two plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Normalized concentrations
        ax1.plot(
            c0_values,
            mu_values,
            "bo-",
            linewidth=2,
            markersize=8,
            label="Experimental data",
        )
        ax1.plot(
            c0_values,
            original_theoretical_mu,
            "r--",
            linewidth=2,
            markersize=6,
            label=f"Original Monod: μ_max={mu_max:.2f}, Γ={Gamma:.0e}",
        )
        if fitted_mu_max is not None:
            ax1.plot(
                c0_values,
                fitted_theoretical_mu,
                "g-",
                linewidth=2,
                markersize=6,
                label=f"Fitted Monod: μ_max={fitted_mu_max:.3f}, Γ={fitted_gamma:.1e}",
            )
        ax1.plot(
            c0_values,
            logistic_theoretical_mu,
            "m:",
            linewidth=2,
            label=f"Logistic: μ×C/C₀ (μ={logistic_mu_param:.3f})",
        )
        ax1.set_xlabel("Normalized nutrient concentration (C₀)", fontsize=12)
        ax1.set_ylabel("Specific growth rate μ (h⁻¹)", fontsize=12)
        ax1.set_title("Growth Rate vs Normalized Nutrient Concentration", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # Add data point labels for experimental data only
        for i, (c0, mu) in enumerate(zip(c0_values, mu_values)):
            ax1.annotate(
                f"{mu:.3f}",
                (c0, mu),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="blue",
            )

        # Plot 2: Cell concentrations with smooth curves
        ax2.plot(
            c0_cell_concentrations,
            mu_values,
            "bo-",
            linewidth=2,
            markersize=8,
            label="Experimental data",
        )
        ax2.plot(
            C_smooth,
            mu_smooth_original,
            "r--",
            linewidth=2,
            label=f"Original Monod: μ_max={mu_max:.2f}, Γ={Gamma:.0e}",
        )
        if fitted_mu_max is not None:
            ax2.plot(
                C_smooth,
                mu_smooth_fitted,
                "g-",
                linewidth=2,
                label=f"Fitted Monod: μ_max={fitted_mu_max:.3f}, Γ={fitted_gamma:.1e}",
            )
        ax2.plot(
            C_smooth,
            mu_smooth_logistic,
            "m:",
            linewidth=2,
            label=f"Logistic: μ×C/C₀ (μ={logistic_mu_param:.3f})",
        )
        ax2.set_xlabel("Nutrient concentration (cells/mL)", fontsize=12)
        ax2.set_ylabel("Specific growth rate μ (h⁻¹)", fontsize=12)
        ax2.set_title(
            "Growth Rate vs Nutrient Concentration (Cell Counts)", fontsize=14
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        # Using linear scale for better visualization of concentration differences

        # Add data point labels for experimental data only
        for i, (c0_cells, mu) in enumerate(zip(c0_cell_concentrations, mu_values)):
            ax2.annotate(
                f"{mu:.3f}",
                (c0_cells, mu),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="blue",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_dir, "growth_rate_vs_nutrient_concentration_with_theory.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"\nPlot with theory saved as '{os.path.join(results_dir, 'growth_rate_vs_nutrient_concentration_with_theory.png')}'"
        )

        # Print comparison
        print("\n" + "=" * 130)
        print(
            "COMPARISON: EXPERIMENTAL vs ORIGINAL vs FITTED vs LOGISTIC THEORETICAL GROWTH RATES"
        )
        print("=" * 130)
        print(
            f"{'Condition':<12} {'C₀ (cells/mL)':<15} {'Experimental μ':<13} {'Original μ':<13} {'Fitted μ':<13} {'Logistic μ':<13} {'Fitted Diff':<12}"
        )
        print("-" * 130)
        for i, condition in enumerate(sorted_conditions):
            c0_factor = c0_values[i]
            c0_cells = c0_cell_concentrations[i]
            exp_mu = mu_values[i]
            orig_mu = original_theoretical_mu[i]
            logistic_mu = logistic_theoretical_mu[i]
            if fitted_mu_max is not None:
                fitted_mu = fitted_theoretical_mu[i]
                fitted_diff = exp_mu - fitted_mu
            else:
                fitted_mu = orig_mu
                fitted_diff = exp_mu - orig_mu
            print(
                f"C0*{c0_factor:<7} {c0_cells:<15.2e} {exp_mu:<13.4f} {orig_mu:<13.4f} {fitted_mu:<13.4f} {logistic_mu:<13.4f} {fitted_diff:<12.4f}"
            )

        # Calculate R² for original and logistic models
        ss_res_orig = sum(
            (exp - theo) ** 2 for exp, theo in zip(mu_values, original_theoretical_mu)
        )
        ss_res_logistic = sum(
            (exp - theo) ** 2 for exp, theo in zip(mu_values, logistic_theoretical_mu)
        )
        ss_tot = sum((exp - np.mean(mu_values)) ** 2 for exp in mu_values)
        r_squared_orig = 1 - (ss_res_orig / ss_tot)
        r_squared_logistic = 1 - (ss_res_logistic / ss_tot)

        print(f"\nModel Performance:")
        print(f"  Original Monod R² = {r_squared_orig:.4f}")
        print(f"  Logistic Model R² = {r_squared_logistic:.4f}")
        if fitted_mu_max is not None:
            print(f"  Fitted Monod R²   = {r_squared_fit:.4f}")
            print(f"  Improvement in R² = {r_squared_fit - r_squared_orig:.4f}")
            print(f"  Fitted vs Logistic = {r_squared_fit - r_squared_logistic:.4f}")

        print(f"\nOriginal Parameters (from erlen_bounds):")
        print(f"  μ_max = {mu_max} h⁻¹")
        print(f"  Γ = {Gamma:.0e} cells/mL")

        if fitted_mu_max is not None:
            print(f"\nOptimally Fitted Parameters:")
            print(f"  μ_max = {fitted_mu_max:.4f} ± {param_errors[0]:.4f} h⁻¹")
            print(f"  Γ = {fitted_gamma:.2e} ± {param_errors[1]:.2e} cells/mL")
            print(
                f"  Base C₀ = {base_C0:.0e} cells/mL (reference nutrient concentration)"
            )

            # Physical interpretation
            if fitted_gamma > base_C0:
                print(f"\nPhysical Interpretation:")
                print(
                    f"  Γ/C₀_max = {fitted_gamma / base_C0:.2f} - Nutrient limitation is significant"
                )
                print(
                    f"  Growth becomes nutrient-limited when C < {fitted_gamma:.1e} cells/mL"
                )
            else:
                print(f"\nPhysical Interpretation:")
                print(
                    f"  Γ/C₀_max = {fitted_gamma / base_C0:.2f} - Nutrient limitation is moderate"
                )
                print(
                    f"  Growth becomes nutrient-limited when C < {fitted_gamma:.1e} cells/mL"
                )

        # Also plot individual growth curves for verification (converted to cell concentration)
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(growth_data)))

        for i, (condition, (time, od)) in enumerate(
            sorted(
                growth_data.items(),
                key=lambda x: float(x[0].split("*")[1]),
                reverse=True,
            )
        ):
            plt.subplot(2, 3, i + 1)

            # Convert OD to cell concentration
            cells_per_ml = od * conv_OD_to_cell

            plt.semilogy(time, cells_per_ml, "o-", color=colors[i], label=condition)

            # Highlight exponential phase (using cell concentration for detection)
            exp_start, exp_end = find_exponential_phase(time, cells_per_ml)
            exp_mask = (time >= exp_start) & (time <= exp_end)
            if np.any(exp_mask):
                cells_exp = cells_per_ml[exp_mask]
                plt.semilogy(
                    time[exp_mask],
                    cells_exp,
                    "s-",
                    color="red",
                    alpha=0.7,
                    markersize=4,
                    label="Exponential phase",
                )

            plt.xlabel("Time (hours)")
            plt.ylabel("Cell concentration (cells/mL)")
            plt.title(condition)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, "individual_growth_curves_cells.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"Individual growth curves (in cells/mL) saved as '{os.path.join(results_dir, 'individual_growth_curves_cells.png')}'"
        )
        print(f"Conversion factor used: 1 OD = {conv_OD_to_cell:.2e} cells/mL")

        # Plot steady states vs dilution factor
        plt.figure(figsize=(10, 6))

        # Extract steady state data (final time points)
        dilution_factors = []
        steady_state_concentrations = []
        condition_labels = []

        for condition, (time, od) in sorted(
            growth_data.items(), key=lambda x: float(x[0].split("*")[1]), reverse=True
        ):
            c0_factor = float(condition.split("*")[1])
            cells_per_ml = od * conv_OD_to_cell

            # Take steady state as average of last 10% of time points
            n_points = len(cells_per_ml)
            steady_start_idx = int(0.9 * n_points)
            steady_state = np.mean(cells_per_ml[steady_start_idx:])

            dilution_factors.append(c0_factor)
            steady_state_concentrations.append(steady_state)
            condition_labels.append(condition)

        # Plot steady states (linear x-axis)
        plt.plot(
            dilution_factors,
            steady_state_concentrations,
            "ro-",
            linewidth=2,
            markersize=8,
            label="Experimental steady states",
        )

        # Add line with slope 1: starts at full normalized concentration 1 and has slope 1 (y = x)
        x_line = np.linspace(1.0, 0.0, 100)  # From 1 to 0 normalized concentration
        y_line = x_line * max(steady_state_concentrations)  # Scale to match data range
        plt.plot(
            x_line,
            y_line,
            "k-",
            linewidth=2,
            alpha=0.7,
            label="y = x (slope=1, normalized)",
        )

        # Add data point labels
        for i, (df, ss, label) in enumerate(
            zip(dilution_factors, steady_state_concentrations, condition_labels)
        ):
            plt.annotate(
                f"{label}\n({ss:.1e} cells/mL)",
                (df, ss),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.7),
            )

        plt.xlabel("Normalized nutrient concentration (C₀ factor)", fontsize=12)
        plt.ylabel("Steady-state cell concentration (cells/mL)", fontsize=12)
        plt.title(
            "Steady-State Cell Concentrations vs Dilution Factor\n(Chlamydomonas reinhardtii)",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set y-axis to scientific notation
        plt.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, "steady_states_vs_dilution_factor.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"\nSteady-state analysis saved as '{os.path.join(results_dir, 'steady_states_vs_dilution_factor.png')}'"
        )

        # Print steady state summary with ratios
        print("\n" + "=" * 90)
        print("STEADY STATE CONCENTRATIONS WITH DILUTION RATIOS")
        print("=" * 90)
        print(
            f"{'Condition':<12} {'Dilution Factor':<16} {'Steady State (cells/mL)':<20} {'SS Ratio':<10} {'Expected':<10}"
        )
        print("-" * 90)

        for i, (condition, df, ss) in enumerate(
            zip(condition_labels, dilution_factors, steady_state_concentrations)
        ):
            if i == 0:
                # First condition (highest concentration) - no ratio to calculate
                print(
                    f"{condition:<12} {df:<16.3f} {ss:<20.2e} {'---':<10} {'---':<10}"
                )
            else:
                # Calculate ratio of current steady state to previous steady state
                ss_ratio = ss / steady_state_concentrations[i - 1]
                # Expected ratio is the dilution factor ratio
                expected_ratio = df / dilution_factors[i - 1]
                print(
                    f"{condition:<12} {df:<16.3f} {ss:<20.2e} {ss_ratio:<10.3f} {expected_ratio:<10.3f}"
                )

        print("\nNotes:")
        print("- SS Ratio: Actual steady-state ratio (current/previous)")
        print("- Expected: Dilution factor ratio (current/previous)")
        print("- If SS Ratio = Expected, steady states scale linearly with dilution")
        print("- If SS Ratio < Expected, there's nutrient limitation effect")


if __name__ == "__main__":
    main()
