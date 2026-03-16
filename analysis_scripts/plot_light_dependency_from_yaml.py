"""
Script to plot light dependency of Monod parameters from YAML analysis results.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import pandas as pd

# Add parent directory to path to import ode module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ode import AlgalSysParameters, Model, dxdt_algae_model


def monod_light_model(light_intensity, mu_max_light, K_L):
    """Monod model for light dependency: μ = μ_max_light * L₀ / (K_L + L₀)"""
    return mu_max_light * light_intensity / (K_L + light_intensity)


def offset_monod_light_model(light_intensity, mu_offset, mu_max_light, K_L):
    """Offset Monod model for light dependency: μ = μ_offset + μ_max_light * L₀ / (K_L + L₀)"""
    return mu_offset + mu_max_light * light_intensity / (K_L + light_intensity)


def fit_monod_light_parameters(light_intensities, growth_rates):
    """Fit Monod parameters for light dependency."""
    try:
        # Initial guesses: mu_max_light = max growth rate, K_L = median light intensity
        p0 = [max(growth_rates) * 1.2, np.median(light_intensities)]

        # Bounds: mu_max_light > 0, K_L > 0
        bounds = ([0, 0], [np.inf, np.inf])

        popt, pcov = curve_fit(
            monod_light_model,
            light_intensities,
            growth_rates,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        # Calculate parameter errors and R²
        param_errors = np.sqrt(np.diag(pcov))
        y_pred = monod_light_model(light_intensities, *popt)
        ss_res = np.sum((growth_rates - y_pred) ** 2)
        ss_tot = np.sum((growth_rates - np.mean(growth_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt[0], popt[1], r_squared, param_errors

    except Exception as e:
        print(f"Failed to fit Monod light model: {e}")
        return None, None, 0, [0, 0]


def fit_offset_monod_light_parameters(light_intensities, growth_rates):
    """Fit offset Monod parameters for light dependency."""
    try:
        # Initial guesses: mu_offset = min growth rate, mu_max_light = max - min, K_L = median light intensity
        min_growth = min(growth_rates)
        max_growth = max(growth_rates)
        p0 = [min_growth, max_growth - min_growth, np.median(light_intensities)]

        # Bounds: mu_offset can be negative, mu_max_light > 0, K_L > 0
        bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])

        popt, pcov = curve_fit(
            offset_monod_light_model,
            light_intensities,
            growth_rates,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        # Calculate parameter errors and R²
        param_errors = np.sqrt(np.diag(pcov))
        y_pred = offset_monod_light_model(light_intensities, *popt)
        ss_res = np.sum((growth_rates - y_pred) ** 2)
        ss_tot = np.sum((growth_rates - np.mean(growth_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt[0], popt[1], popt[2], r_squared, param_errors

    except Exception as e:
        print(f"Failed to fit offset Monod light model: {e}")
        return None, None, None, 0, [0, 0, 0]


def calculate_steady_state(
    params, light_intensity, c0_factor=1.0, model=Model.MORTALITYv2, time_max=1000
):
    """Calculate steady state cell density for given parameters and light intensity."""
    try:
        # Adjust parameters for this light intensity and C0 factor
        adjusted_params = params._replace(L0=light_intensity, C0=params.C0 * c0_factor)

        # Run ODE to steady state
        sol = solve_ivp(
            fun=lambda t, N: dxdt_algae_model(t, N, adjusted_params, model=model),
            t_span=(0, time_max),
            y0=[adjusted_params.N0],
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        if sol.success and len(sol.y[0]) > 0:
            # Return final value (steady state)
            return sol.y[0][-1]
        else:
            print(f"ODE integration failed for L0={light_intensity}")
            return None

    except Exception as e:
        print(f"Error calculating steady state for L0={light_intensity}: {e}")
        return None


def gamma_monod_model(light_intensity, gamma_max, K_gamma):
    """Monod model for light-dependent carrying capacity: Γ_eff = Γ_max * L₀ / (K_γ + L₀)"""
    return gamma_max * light_intensity / (K_gamma + light_intensity)


def steele_light_model(light_intensity, mu_max_steele, K_steele, beta):
    """Steele model for light dependency with photoinhibition: μ = μ_max * (I / (K + I)) * exp(-β * I)"""
    return (
        mu_max_steele
        * (light_intensity / (K_steele + light_intensity))
        * np.exp(-beta * light_intensity)
    )


def steele_gamma_model(light_intensity, gamma_max_steele, K_gamma_steele, beta_gamma):
    """Steele model for light-dependent carrying capacity with photoinhibition: Γ_eff = Γ_max * (I / (K + I)) * exp(-β * I)"""
    return (
        gamma_max_steele
        * (light_intensity / (K_gamma_steele + light_intensity))
        * np.exp(-beta_gamma * light_intensity)
    )


def fit_gamma_monod_parameters(light_intensities, steady_states):
    """Fit Monod parameters for light-dependent carrying capacity."""
    try:
        # Initial guesses: gamma_max = max steady state * 1.2, K_gamma = median light intensity
        p0 = [max(steady_states) * 1.2, np.median(light_intensities)]

        # Bounds: gamma_max > 0, K_gamma > 0
        bounds = ([0, 0], [np.inf, np.inf])

        popt, pcov = curve_fit(
            gamma_monod_model,
            light_intensities,
            steady_states,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        # Calculate parameter errors and R²
        param_errors = np.sqrt(np.diag(pcov))
        y_pred = gamma_monod_model(light_intensities, *popt)
        ss_res = np.sum((steady_states - y_pred) ** 2)
        ss_tot = np.sum((steady_states - np.mean(steady_states)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt[0], popt[1], r_squared, param_errors

    except Exception as e:
        print(f"Failed to fit Γ_eff Monod model: {e}")
        return None, None, 0, [0, 0]


def fit_steele_light_parameters(light_intensities, growth_rates):
    """Fit Steele parameters for light dependency with photoinhibition."""
    try:
        # Initial guesses: mu_max = max growth rate * 1.2, K = median light, beta = small positive
        p0 = [max(growth_rates) * 1.2, np.median(light_intensities), 0.001]

        # Bounds: all parameters > 0, beta should be small
        bounds = ([0, 0, 0], [np.inf, np.inf, 0.1])

        popt, pcov = curve_fit(
            steele_light_model,
            light_intensities,
            growth_rates,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )

        # Calculate parameter errors and R²
        param_errors = np.sqrt(np.diag(pcov))
        y_pred = steele_light_model(light_intensities, *popt)
        ss_res = np.sum((growth_rates - y_pred) ** 2)
        ss_tot = np.sum((growth_rates - np.mean(growth_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt[0], popt[1], popt[2], r_squared, param_errors

    except Exception as e:
        print(f"Failed to fit Steele light model: {e}")
        return None, None, None, 0, [0, 0, 0]


def fit_steele_gamma_parameters(light_intensities, steady_states):
    """Fit Steele parameters for light-dependent carrying capacity with photoinhibition."""
    try:
        # Initial guesses: gamma_max = max steady state * 1.2, K = median light, beta = small positive
        p0 = [max(steady_states) * 1.2, np.median(light_intensities), 0.001]

        # Bounds: all parameters > 0, beta should be small
        bounds = ([0, 0, 0], [np.inf, np.inf, 0.1])

        popt, pcov = curve_fit(
            steele_gamma_model,
            light_intensities,
            steady_states,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )

        # Calculate parameter errors and R²
        param_errors = np.sqrt(np.diag(pcov))
        y_pred = steele_gamma_model(light_intensities, *popt)
        ss_res = np.sum((steady_states - y_pred) ** 2)
        ss_tot = np.sum((steady_states - np.mean(steady_states)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt[0], popt[1], popt[2], r_squared, param_errors

    except Exception as e:
        print(f"Failed to fit Steele gamma model: {e}")
        return None, None, None, 0, [0, 0, 0]


def load_analysis_results(yaml_filepath):
    """Load analysis results from YAML file."""
    with open(yaml_filepath, "r") as f:
        data = yaml.safe_load(f)
    return data


def extract_parameters_for_plotting(data):
    """Extract relevant parameters from YAML data for plotting."""
    datasets = []

    for dataset in data["datasets"]:
        fitted = dataset["fitted_parameters"]

        # Only include datasets with successful fits
        if fitted.get("mu_max") is not None and fitted.get("r_squared", 0) > 0.0:
            dataset_info = {
                "date": dataset["date"],
                "light_intensity": dataset["light_intensity"],
                "mu_max": fitted["mu_max"],
                "mu_max_error": fitted.get("mu_max_error", 0),
                "gamma": fitted["gamma"],
                "gamma_error": fitted.get("gamma_error", 0),
                "r_squared": fitted["r_squared"],
                "conditions": dataset["conditions"],
            }

            # Get growth rates for all conditions
            dataset_info["growth_rates_by_condition"] = {}
            for condition in dataset["conditions"]:
                c0_factor = condition["c0_factor"]
                dataset_info["growth_rates_by_condition"][c0_factor] = condition[
                    "growth_rate"
                ]

            # Keep the C0*1.0 for backward compatibility
            dataset_info["mu_at_c0_1"] = dataset_info["growth_rates_by_condition"].get(
                1.0, dataset_info["mu_max"]
            )

            # Use steady states directly from YAML data (already calculated)
            steady_states = dataset.get("steady_states", {})
            dataset_info["steady_states_by_condition"] = {}

            # Map from condition names to C0 factors
            condition_mapping = {
                "C0*1.0": 1.0,
                "C0*0.5": 0.5,
                "C0*0.25": 0.25,
                "C0*0.125": 0.125,
                "C0*0.062": 0.062,
            }

            for condition_name, c0_factor in condition_mapping.items():
                if condition_name in steady_states:
                    dataset_info["steady_states_by_condition"][c0_factor] = (
                        steady_states[condition_name]
                    )

            datasets.append(dataset_info)

    return datasets


def plot_light_dependency(datasets, output_dir="results"):
    """Create comprehensive light dependency plots."""
    # Extract data for plotting
    light_intensities = [d["light_intensity"] for d in datasets]
    mu_max_values = [d["mu_max"] for d in datasets]
    mu_max_errors = [d["mu_max_error"] for d in datasets]
    gamma_values = [d["gamma"] for d in datasets]
    gamma_errors = [d["gamma_error"] for d in datasets]
    r_squared_values = [d["r_squared"] for d in datasets]
    mu_at_c0_1_values = [d["mu_at_c0_1"] for d in datasets]
    dates = [d["date"] for d in datasets]

    # Convert to numpy arrays
    light_intensities = np.array(light_intensities)
    mu_max_values = np.array(mu_max_values)
    mu_max_errors = np.array(mu_max_errors)
    gamma_values = np.array(gamma_values)
    gamma_errors = np.array(gamma_errors)
    r_squared_values = np.array(r_squared_values)
    mu_at_c0_1_values = np.array(mu_at_c0_1_values)

    # Identify good quality fits (R² > 0.8)
    good_fits = r_squared_values > 0.8
    anomalous = r_squared_values <= 0.8

    # Create the plot - original 2x2 layout for parameter plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Get normalization factor from linear fit at L0=300
    normalization_factor = 1.0
    if np.sum(good_fits) > 1:
        # Linear fit to get value at 300
        z = np.polyfit(light_intensities[good_fits], mu_max_values[good_fits], 1)
        linear_value_at_300 = z[0] * 300 + z[1]
        normalization_factor = linear_value_at_300

    # Plot 1: μ_max vs Light Intensity (normalized)
    if np.any(good_fits):
        ax1.errorbar(
            light_intensities[good_fits],
            mu_max_values[good_fits] / normalization_factor,
            yerr=mu_max_errors[good_fits] / normalization_factor,
            fmt="o",
            color="blue",
            capsize=3,
            capthick=2,
            markersize=8,
            alpha=0.7,
            label="Good fits (R² > 0.8)",
        )
    if np.any(anomalous):
        ax1.errorbar(
            light_intensities[anomalous],
            mu_max_values[anomalous] / normalization_factor,
            yerr=mu_max_errors[anomalous] / normalization_factor,
            fmt="x",
            color="red",
            capsize=3,
            capthick=2,
            markersize=10,
            alpha=0.7,
            label="Poor fits (R² ≤ 0.8)",
        )

    # Add dataset labels
    for i, (x, y, date, r2) in enumerate(
        zip(
            light_intensities,
            mu_max_values / normalization_factor,
            dates,
            r_squared_values,
        )
    ):
        color = "blue" if r2 > 0.8 else "red"
        ax1.annotate(
            date,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
            color=color,
        )

    # Fit both linear and Monod models for good data points
    light_range = np.linspace(0, max(light_intensities) * 1.1, 100)

    if np.sum(good_fits) > 1:
        # Linear fit
        z = np.polyfit(light_intensities[good_fits], mu_max_values[good_fits], 1)
        p = np.poly1d(z)
        ax1.plot(
            light_range,
            p(light_range) / normalization_factor,
            "b--",
            alpha=0.7,
            label=f"Linear: μ_max = {z[0]:.2e}×L₀ + {z[1]:.3f} (norm. at 300)",
        )

        # Monod light fit
        mu_max_light, K_L, r2_monod, param_errors = fit_monod_light_parameters(
            light_intensities[good_fits], mu_max_values[good_fits]
        )

        # Offset Monod light fit
        (
            mu_offset,
            mu_max_light_offset,
            K_L_offset,
            r2_offset_monod,
            offset_param_errors,
        ) = fit_offset_monod_light_parameters(
            light_intensities[good_fits], mu_max_values[good_fits]
        )

        # Steele light fit
        mu_max_steele, K_steele, beta_steele, r2_steele, steele_param_errors = (
            fit_steele_light_parameters(
                light_intensities[good_fits], mu_max_values[good_fits]
            )
        )

        if mu_max_light is not None:
            monod_curve = monod_light_model(light_range, mu_max_light, K_L)
            ax1.plot(
                light_range,
                monod_curve / normalization_factor,
                "g-",
                alpha=0.7,
                linewidth=2,
                label=f"Monod: normalized",
            )

        if mu_offset is not None:
            offset_monod_curve = offset_monod_light_model(
                light_range, mu_offset, mu_max_light_offset, K_L_offset
            )
            ax1.plot(
                light_range,
                offset_monod_curve / normalization_factor,
                "m-",
                alpha=0.7,
                linewidth=2,
                label=f"Offset Monod: normalized",
            )

        if mu_max_steele is not None:
            steele_curve = steele_light_model(
                light_range, mu_max_steele, K_steele, beta_steele
            )
            ax1.plot(
                light_range,
                steele_curve / normalization_factor,
                "r-",
                alpha=0.8,
                linewidth=2,
                label=f"Steele (photoinhibition): normalized",
            )

        # Display fit quality
        corr = np.corrcoef(light_intensities[good_fits], mu_max_values[good_fits])[0, 1]
        text_info = f"Linear r = {corr:.3f}"
        if mu_max_light is not None:
            text_info += f"\nMonod R² = {r2_monod:.3f}"
        if mu_offset is not None:
            text_info += f"\nOffset Monod R² = {r2_offset_monod:.3f}"
        if mu_max_steele is not None:
            text_info += f"\nSteele R² = {r2_steele:.3f}"

        ax1.text(
            0.05,
            0.95,
            text_info,
            transform=ax1.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        if mu_max_light is None and mu_offset is None:
            ax1.text(
                0.05,
                0.95,
                f"Linear r = {corr:.4f}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            )

    ax1.set_xlabel("Light Intensity (μmol/m²/s)")
    ax1.set_ylabel("μ_max (normalized to linear fit at L₀=300)")
    ax1.set_title(
        "Maximum Growth Rate vs Light Intensity (Normalized)\n(Fitted parameter from Monod model)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Plot 2: Γ vs Light Intensity with Monod fitting
    if np.any(good_fits):
        ax2.errorbar(
            light_intensities[good_fits],
            gamma_values[good_fits] / 1e6,
            yerr=gamma_errors[good_fits] / 1e6,
            fmt="o",
            color="blue",
            capsize=3,
            capthick=2,
            markersize=8,
            alpha=0.7,
            label="Good fits",
        )
    if np.any(anomalous):
        ax2.errorbar(
            light_intensities[anomalous],
            gamma_values[anomalous] / 1e6,
            yerr=gamma_errors[anomalous] / 1e6,
            fmt="x",
            color="red",
            capsize=3,
            capthick=2,
            markersize=10,
            alpha=0.7,
            label="Poor fits",
        )

    # Fit Γ Monod and Steele models for good data points
    if np.sum(good_fits) > 1:
        # Monod fit
        gamma_max_fitted, K_gamma_fitted, r2_gamma_fitted, _ = (
            fit_gamma_monod_parameters(
                light_intensities[good_fits], gamma_values[good_fits]
            )
        )

        # Steele fit
        (
            gamma_max_steele_fitted,
            K_gamma_steele_fitted,
            beta_gamma_fitted,
            r2_gamma_steele_fitted,
            _,
        ) = fit_steele_gamma_parameters(
            light_intensities[good_fits], gamma_values[good_fits]
        )

        if gamma_max_fitted is not None:
            gamma_curve_fitted = gamma_monod_model(
                light_range, gamma_max_fitted, K_gamma_fitted
            )
            ax2.plot(
                light_range,
                gamma_curve_fitted / 1e6,
                "g-",
                alpha=0.8,
                linewidth=2,
                label=f"Γ Monod (R² = {r2_gamma_fitted:.3f})",
            )

        if gamma_max_steele_fitted is not None:
            gamma_steele_curve_fitted = steele_gamma_model(
                light_range,
                gamma_max_steele_fitted,
                K_gamma_steele_fitted,
                beta_gamma_fitted,
            )
            ax2.plot(
                light_range,
                gamma_steele_curve_fitted / 1e6,
                "r-",
                alpha=0.8,
                linewidth=2,
                label=f"Γ Steele (R² = {r2_gamma_steele_fitted:.3f})",
            )

        # Display fit quality and parameters (prefer Steele if better)
        if (
            gamma_max_steele_fitted is not None
            and r2_gamma_steele_fitted > r2_gamma_fitted
        ):
            ax2.text(
                0.05,
                0.95,
                f"Best: Steele Model\nR² = {r2_gamma_steele_fitted:.3f}\nβ = {beta_gamma_fitted:.4f}",
                transform=ax2.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )
        elif gamma_max_fitted is not None:
            ax2.text(
                0.05,
                0.95,
                f"Best: Monod Model\nR² = {r2_gamma_fitted:.3f}",
                transform=ax2.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

    # Add dataset labels
    for i, (x, y, date, r2) in enumerate(
        zip(light_intensities, gamma_values / 1e6, dates, r_squared_values)
    ):
        color = "blue" if r2 > 0.8 else "red"
        ax2.annotate(
            date,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
            color=color,
        )

    ax2.set_xlabel("Light Intensity (μmol/m²/s)")
    ax2.set_ylabel("Γ (×10⁶ cells/mL)")
    ax2.set_title(
        "Half-Saturation Constant vs Light Intensity\n(Fitted parameter Γ with potential Monod dependency)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Plot 3: R² vs Light Intensity (fit quality)
    colors = ["red" if r < 0.8 else "blue" for r in r_squared_values]
    scatter = ax3.scatter(
        light_intensities, r_squared_values, c=colors, s=80, alpha=0.7
    )

    # Add dataset labels
    for i, (x, y, date) in enumerate(zip(light_intensities, r_squared_values, dates)):
        ax3.annotate(
            date,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax3.axhline(
        y=0.8, color="gray", linestyle="--", alpha=0.7, label="R² = 0.8 threshold"
    )
    ax3.set_xlabel("Light Intensity (μmol/m²/s)")
    ax3.set_ylabel("R² (Model Fit Quality)")
    ax3.set_title(
        "Monod Model Fit Quality vs Light Intensity\n(How well μ vs C₀ fits Monod model)"
    )
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # Plot 4: Growth rate for all C0 conditions vs Light Intensity
    c0_conditions = [1.0, 0.5, 0.25, 0.125, 0.062]
    colors_c0 = plt.cm.viridis(np.linspace(0, 1, len(c0_conditions)))

    for i, c0_factor in enumerate(c0_conditions):
        # Extract growth rates for this C0 condition across all datasets
        c0_growth_rates = []
        c0_light_intensities = []
        c0_good_fits = []

        for j, dataset in enumerate(datasets):
            if c0_factor in dataset["growth_rates_by_condition"]:
                c0_growth_rates.append(dataset["growth_rates_by_condition"][c0_factor])
                c0_light_intensities.append(dataset["light_intensity"])
                c0_good_fits.append(dataset["r_squared"] > 0.8)

        c0_growth_rates = np.array(c0_growth_rates)
        c0_light_intensities = np.array(c0_light_intensities)
        c0_good_fits = np.array(c0_good_fits)

        if len(c0_growth_rates) > 0:
            # Plot good and bad fits with different markers
            if np.any(c0_good_fits):
                ax4.scatter(
                    c0_light_intensities[c0_good_fits],
                    c0_growth_rates[c0_good_fits],
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    label=f"C0*{c0_factor}",
                )
            if np.any(~c0_good_fits):
                ax4.scatter(
                    c0_light_intensities[~c0_good_fits],
                    c0_growth_rates[~c0_good_fits],
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    marker="x",
                )

            # Fit both linear and Monod models for good data points
            if np.sum(c0_good_fits) > 1:
                # Linear fit
                z_c0 = np.polyfit(
                    c0_light_intensities[c0_good_fits], c0_growth_rates[c0_good_fits], 1
                )
                p_c0 = np.poly1d(z_c0)
                ax4.plot(
                    light_range,
                    p_c0(light_range),
                    "--",
                    alpha=0.5,
                    color=colors_c0[i],
                    linewidth=1.5,
                    label=f"C0*{c0_factor} linear" if i < 2 else "",
                )

                # Monod light fit
                mu_max_light_c0, K_L_c0, r2_monod_c0, _ = fit_monod_light_parameters(
                    c0_light_intensities[c0_good_fits], c0_growth_rates[c0_good_fits]
                )

                # Offset Monod light fit
                (
                    mu_offset_c0,
                    mu_max_light_offset_c0,
                    K_L_offset_c0,
                    r2_offset_monod_c0,
                    _,
                ) = fit_offset_monod_light_parameters(
                    c0_light_intensities[c0_good_fits], c0_growth_rates[c0_good_fits]
                )

                if (
                    mu_max_light_c0 is not None and r2_monod_c0 > 0.5
                ):  # Only show if reasonable fit
                    monod_curve_c0 = monod_light_model(
                        light_range, mu_max_light_c0, K_L_c0
                    )
                    ax4.plot(
                        light_range,
                        monod_curve_c0,
                        "-",
                        alpha=0.7,
                        color=colors_c0[i],
                        linewidth=2,
                        label=f"C0*{c0_factor} Monod" if i < 2 else "",
                    )

                if (
                    mu_offset_c0 is not None and r2_offset_monod_c0 > 0.5
                ):  # Only show if reasonable fit
                    offset_monod_curve_c0 = offset_monod_light_model(
                        light_range, mu_offset_c0, mu_max_light_offset_c0, K_L_offset_c0
                    )
                    ax4.plot(
                        light_range,
                        offset_monod_curve_c0,
                        ":",
                        alpha=0.8,
                        color=colors_c0[i],
                        linewidth=2.5,
                        label=f"C0*{c0_factor} Offset Monod" if i < 2 else "",
                    )

    ax4.set_xlabel("Light Intensity (μmol/m²/s)")
    ax4.set_ylabel("Growth Rate (h⁻¹)")
    ax4.set_title(
        "Growth Rates for All C0 Conditions vs Light Intensity\n(Experimental μ values for each dilution)"
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "light_dependency_parameters_from_yaml.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()  # Show the plot like in light_dependency_analysis.py

    return output_path


def load_experimental_data(filepath):
    """Load experimental growth curve data from CSV file."""
    try:
        # Convert relative path to absolute path
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), filepath)

        df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

        # Clean column names
        df.columns = [col.replace("﻿", "").strip() for col in df.columns]

        return df
    except Exception as e:
        print(f"Error loading experimental data from {filepath}: {e}")
        return None


def extract_growth_curves(df):
    """Extract time and OD data for each dilution condition."""
    if df is None:
        return {}

    growth_curves = {}

    # Dilution mapping
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
                    # Convert OD to cells/mL (using conversion factor from analysis)
                    conv_OD_to_cell = 4.46e6
                    cells = od_values.iloc[:min_length].values * conv_OD_to_cell

                    growth_curves[f"C0*{c0_factor}"] = {
                        "time": time_hours.iloc[:min_length].values,
                        "od": od_values.iloc[:min_length].values,
                        "cells": cells,
                    }

    return growth_curves


def load_fitted_parameters_from_tuning(
    model_dir="Model.MORTALITYv2", param_file="erlen.yaml"
):
    """Load fitted parameters from tuning_ODE.py output."""
    param_path = os.path.join(model_dir, param_file)
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            param_data = yaml.safe_load(f)
        return AlgalSysParameters(**{k: np.float64(v) for k, v in param_data.items()})
    else:
        print(f"Warning: Parameter file {param_path} not found")
        return None


def calculate_simulated_growth_rate(
    params, light_intensity, c0_factor, N_initial=None, model=Model.MORTALITYv2
):
    """Calculate simulated specific growth rate at given conditions."""
    try:
        # Adjust parameters for this light intensity and C0 factor
        adjusted_params = params._replace(L0=light_intensity, C0=params.C0 * c0_factor)

        # Use a typical initial cell density if not provided
        if N_initial is None:
            N_initial = adjusted_params.N0

        # For MORTALITYv2 model, extract the specific growth rate formula
        if model == Model.MORTALITYv2:
            # From the ODE: light and resource factors
            I = adjusted_params.L0 * np.exp(-1e-8 * N_initial)
            light_factor = I / (I + 30) * np.exp(-I / 30)

            C = adjusted_params.C0 - (N_initial - adjusted_params.N0)
            C = max(C, 0.0)
            resource_factor = C / (adjusted_params.Gamma + C)

            # The specific growth rate from the model
            mu_specific = adjusted_params.mu * light_factor * resource_factor
            return mu_specific
        else:
            print(f"Growth rate calculation not implemented for model {model}")
            return None

    except Exception as e:
        print(
            f"Error calculating growth rate for L0={light_intensity}, C0*{c0_factor}: {e}"
        )
        return None


def generate_simulated_growth_rates(
    fitted_params, light_intensities, c0_factors, model=Model.MORTALITYv2
):
    """Generate simulated growth rate data using fitted parameters."""
    simulated_data = {}

    for c0_factor in c0_factors:
        growth_rates = []
        valid_lights = []

        for light in light_intensities:
            # Calculate growth rate at initial conditions
            growth_rate = calculate_simulated_growth_rate(
                fitted_params, light, c0_factor, model=model
            )
            if growth_rate is not None:
                growth_rates.append(growth_rate)
                valid_lights.append(light)
            else:
                print(f"Failed to calculate growth rate for L0={light}, C0*{c0_factor}")

        if len(growth_rates) > 0:
            simulated_data[c0_factor] = {
                "light_intensities": np.array(valid_lights),
                "growth_rates": np.array(growth_rates),
            }

    return simulated_data


def generate_simulated_steady_states(
    fitted_params, light_intensities, c0_factors, model=Model.MORTALITYv2
):
    """Generate simulated steady state data using fitted parameters."""
    simulated_data = {}

    for c0_factor in c0_factors:
        steady_states = []
        valid_lights = []

        for light in light_intensities:
            steady_state = calculate_steady_state(
                fitted_params, light, c0_factor, model=model
            )
            if steady_state is not None:
                steady_states.append(steady_state)
                valid_lights.append(light)
            else:
                print(
                    f"Failed to calculate steady state for L0={light}, C0*{c0_factor}"
                )

        if len(steady_states) > 0:
            simulated_data[c0_factor] = {
                "light_intensities": np.array(valid_lights),
                "steady_states": np.array(steady_states),
            }

    return simulated_data


def plot_steady_states(datasets, output_dir="results"):
    """Create steady state analysis plots."""
    # Load fitted parameters from tuning_ODE.py
    fitted_params = load_fitted_parameters_from_tuning()

    # Create the plot for steady state analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data for plotting
    light_intensities = [d["light_intensity"] for d in datasets]
    r_squared_values = [d["r_squared"] for d in datasets]
    dates = [d["date"] for d in datasets]

    # Convert to numpy arrays
    light_intensities = np.array(light_intensities)
    r_squared_values = np.array(r_squared_values)

    # Identify good quality fits (R² > 0.8)
    good_fits = r_squared_values > 0.8

    # Light range for fitting curves
    light_range = np.linspace(0, max(light_intensities) * 1.1, 100)

    # Generate simulated data if fitted parameters are available
    c0_conditions = [1.0, 0.5, 0.25, 0.125, 0.062]
    simulated_data = None
    if fitted_params is not None:
        print("Generating simulated steady state data...")
        simulated_data = generate_simulated_steady_states(
            fitted_params, light_range, c0_conditions
        )
        print(f"Generated simulated data for {len(simulated_data)} C0 conditions")

    # Plot 1: Steady States for all C0 conditions vs Light Intensity
    colors_c0 = plt.cm.viridis(np.linspace(0, 1, len(c0_conditions)))

    for i, c0_factor in enumerate(c0_conditions):
        # Extract steady states for this C0 condition across all datasets
        c0_steady_states = []
        c0_light_intensities = []
        c0_good_fits = []

        for j, dataset in enumerate(datasets):
            if c0_factor in dataset.get("steady_states_by_condition", {}):
                c0_steady_states.append(
                    dataset["steady_states_by_condition"][c0_factor]
                )
                c0_light_intensities.append(dataset["light_intensity"])
                c0_good_fits.append(dataset["r_squared"] > 0.8)

        c0_steady_states = np.array(c0_steady_states)
        c0_light_intensities = np.array(c0_light_intensities)
        c0_good_fits = np.array(c0_good_fits)

        if len(c0_steady_states) > 0:
            # Plot good and bad fits with different markers
            if np.any(c0_good_fits):
                ax1.scatter(
                    c0_light_intensities[c0_good_fits],
                    c0_steady_states[c0_good_fits] / 1e6,
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    label=f"C0*{c0_factor}",
                )
            if np.any(~c0_good_fits):
                ax1.scatter(
                    c0_light_intensities[~c0_good_fits],
                    c0_steady_states[~c0_good_fits] / 1e6,
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    marker="x",
                )

            # Fit Monod and Steele models for good data points (for Γ_eff)
            if np.sum(c0_good_fits) > 1:
                # Monod fit
                gamma_max_c0, K_gamma_c0, r2_gamma_c0, _ = fit_gamma_monod_parameters(
                    c0_light_intensities[c0_good_fits], c0_steady_states[c0_good_fits]
                )

                # Steele fit
                (
                    gamma_max_steele_c0,
                    K_gamma_steele_c0,
                    beta_gamma_c0,
                    r2_gamma_steele_c0,
                    _,
                ) = fit_steele_gamma_parameters(
                    c0_light_intensities[c0_good_fits], c0_steady_states[c0_good_fits]
                )

                # Choose best fit and plot it
                best_is_steele = (
                    gamma_max_steele_c0 is not None
                    and r2_gamma_steele_c0 > r2_gamma_c0
                    and r2_gamma_steele_c0 > 0.3
                )

                if best_is_steele:
                    gamma_steele_curve_c0 = steele_gamma_model(
                        light_range,
                        gamma_max_steele_c0,
                        K_gamma_steele_c0,
                        beta_gamma_c0,
                    )
                    ax1.plot(
                        light_range,
                        gamma_steele_curve_c0 / 1e6,
                        "-",
                        alpha=0.7,
                        color=colors_c0[i],
                        linewidth=2,
                        label=f"C0*{c0_factor} Steele" if c0_factor == 1.0 else "",
                    )
                    best_r2 = r2_gamma_steele_c0
                    model_type = "Steele"
                elif gamma_max_c0 is not None and r2_gamma_c0 > 0.3:
                    gamma_curve_c0 = gamma_monod_model(
                        light_range, gamma_max_c0, K_gamma_c0
                    )
                    ax1.plot(
                        light_range,
                        gamma_curve_c0 / 1e6,
                        "-",
                        alpha=0.7,
                        color=colors_c0[i],
                        linewidth=2,
                        label=f"C0*{c0_factor} Monod" if c0_factor == 1.0 else "",
                    )
                    best_r2 = r2_gamma_c0
                    model_type = "Monod"
                else:
                    best_r2 = 0
                    model_type = "None"

                # Add R² annotation for C0=1.0 (the purple curve)
                if c0_factor == 1.0 and best_r2 > 0.3:
                    ax1.text(
                        0.6,
                        0.95,
                        f"C0*{c0_factor}: {model_type}\nR² = {best_r2:.3f}",
                        transform=ax1.transAxes,
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor=colors_c0[i], alpha=0.3
                        ),
                    )

        # Add simulated data for this C0 condition if available
        if simulated_data is not None and c0_factor in simulated_data:
            sim_data = simulated_data[c0_factor]
            ax1.plot(
                sim_data["light_intensities"],
                sim_data["steady_states"] / 1e6,
                "--",
                alpha=0.8,
                color=colors_c0[i],
                linewidth=2,
                label=f"Simulated C0*{c0_factor}" if c0_factor == 1.0 else "",
            )

    ax1.set_xlabel("Light Intensity (μmol/m²/s)")
    ax1.set_ylabel("Steady State Cell Density (×10⁶ cells/mL)")
    ax1.set_title(
        "Experimental vs Simulated Steady States vs Light Intensity\n(All C0 conditions - dashed lines: tuning_ODE.py simulations)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Plot 2: C0*1.0 steady state with Γ_eff Monod fit
    # Extract C0*1.0 steady states
    steady_states_c0_1 = []
    light_intensities_ss = []
    good_fits_ss = []

    for dataset in datasets:
        if 1.0 in dataset.get("steady_states_by_condition", {}):
            steady_states_c0_1.append(dataset["steady_states_by_condition"][1.0])
            light_intensities_ss.append(dataset["light_intensity"])
            good_fits_ss.append(dataset["r_squared"] > 0.8)

    steady_states_c0_1 = np.array(steady_states_c0_1)
    light_intensities_ss = np.array(light_intensities_ss)
    good_fits_ss = np.array(good_fits_ss)

    if len(steady_states_c0_1) > 0:
        # Plot good and bad fits
        if np.any(good_fits_ss):
            ax2.scatter(
                light_intensities_ss[good_fits_ss],
                steady_states_c0_1[good_fits_ss] / 1e6,
                color="blue",
                s=80,
                alpha=0.7,
                label="Good fits (R² > 0.8)",
            )
        if np.any(~good_fits_ss):
            ax2.scatter(
                light_intensities_ss[~good_fits_ss],
                steady_states_c0_1[~good_fits_ss] / 1e6,
                color="red",
                s=80,
                alpha=0.7,
                marker="x",
                label="Poor fits (R² ≤ 0.8)",
            )

        # Fit both Γ_eff Monod and Steele models for good fits
        if np.sum(good_fits_ss) > 1:
            # Monod fit
            gamma_max, K_gamma, r2_gamma, param_errors_gamma = (
                fit_gamma_monod_parameters(
                    light_intensities_ss[good_fits_ss], steady_states_c0_1[good_fits_ss]
                )
            )

            # Steele fit
            (
                gamma_max_steele,
                K_gamma_steele,
                beta_gamma_steele,
                r2_gamma_steele,
                param_errors_steele,
            ) = fit_steele_gamma_parameters(
                light_intensities_ss[good_fits_ss], steady_states_c0_1[good_fits_ss]
            )

            if gamma_max is not None:
                gamma_curve = gamma_monod_model(light_range, gamma_max, K_gamma)
                ax2.plot(
                    light_range,
                    gamma_curve / 1e6,
                    "g-",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Monod: R² = {r2_gamma:.3f}",
                )

            if gamma_max_steele is not None:
                gamma_steele_curve = steele_gamma_model(
                    light_range, gamma_max_steele, K_gamma_steele, beta_gamma_steele
                )
                ax2.plot(
                    light_range,
                    gamma_steele_curve / 1e6,
                    "r-",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Steele: R² = {r2_gamma_steele:.3f}",
                )

            # Choose and display the best model
            best_is_steele = gamma_max_steele is not None and r2_gamma_steele > r2_gamma

            if best_is_steele:
                ax2.text(
                    0.05,
                    0.95,
                    f"BEST: Steele Model\nR² = {r2_gamma_steele:.3f}\nβ = {beta_gamma_steele:.5f}",
                    transform=ax2.transAxes,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7
                    ),
                )

                # Print the fitted parameters for potential model implementation
                print(
                    f"\n=== BEST: ΓEFF STEELE PARAMETERS FOR MODEL IMPLEMENTATION ==="
                )
                print(
                    f"Gamma_max = {gamma_max_steele:.0f} cells/mL  ({gamma_max_steele / 1e6:.1f}×10⁶)"
                )
                print(f"K_gamma = {K_gamma_steele:.0f} μmol/m²/s")
                print(f"Beta = {beta_gamma_steele:.6f} (photoinhibition parameter)")
                print(f"R² = {r2_gamma_steele:.3f}")
                print(
                    f"Equation: Gamma_eff = {gamma_max_steele:.0f} * (L0 / ({K_gamma_steele:.0f} + L0)) * exp(-{beta_gamma_steele:.6f} * L0)"
                )
            elif gamma_max is not None:
                ax2.text(
                    0.05,
                    0.95,
                    f"BEST: Monod Model\nR² = {r2_gamma:.3f}\nΓ_max = {gamma_max / 1e6:.1f}×10⁶\nK_γ = {K_gamma:.0f}",
                    transform=ax2.transAxes,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

                # Print the fitted parameters for potential model implementation
                print(f"\n=== BEST: ΓEFF MONOD PARAMETERS FOR MODEL IMPLEMENTATION ===")
                print(
                    f"Gamma_max = {gamma_max:.0f} cells/mL  ({gamma_max / 1e6:.1f}×10⁶)"
                )
                print(f"K_gamma = {K_gamma:.0f} μmol/m²/s")
                print(f"R² = {r2_gamma:.3f}")
                print(
                    f"Equation: Gamma_eff = {gamma_max:.0f} * L0 / ({K_gamma:.0f} + L0)"
                )

        # Add simulated data for C0*1.0 if available
        if simulated_data is not None and 1.0 in simulated_data:
            sim_data = simulated_data[1.0]
            ax2.plot(
                sim_data["light_intensities"],
                sim_data["steady_states"] / 1e6,
                "k--",
                alpha=0.8,
                linewidth=3,
                label="Simulated (tuning_ODE.py)",
            )

        # Add dataset labels
        for i, (x, y, date, r2) in enumerate(
            zip(
                light_intensities_ss,
                steady_states_c0_1 / 1e6,
                [
                    d["date"]
                    for d in datasets
                    if 1.0 in d.get("steady_states_by_condition", {})
                ],
                good_fits_ss,
            )
        ):
            color = "blue" if r2 else "red"
            ax2.annotate(
                date,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                color=color,
            )

    ax2.set_xlabel("Light Intensity (μmol/m²/s)")
    ax2.set_ylabel("Steady State Cell Density (×10⁶ cells/mL)")
    ax2.set_title(
        "C0*1.0: Experimental vs Simulated Steady States\n(Black dashed: tuning_ODE.py simulation)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "steady_state_light_dependency.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return output_path


def plot_growth_rates(datasets, output_dir="results"):
    """Create growth rate analysis plots."""
    # Load fitted parameters from tuning_ODE.py
    fitted_params = load_fitted_parameters_from_tuning()

    # Create the plot for growth rate analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data for plotting
    light_intensities = [d["light_intensity"] for d in datasets]
    r_squared_values = [d["r_squared"] for d in datasets]
    dates = [d["date"] for d in datasets]

    # Convert to numpy arrays
    light_intensities = np.array(light_intensities)
    r_squared_values = np.array(r_squared_values)

    # Identify good quality fits (R² > 0.8)
    good_fits = r_squared_values > 0.8

    # Light range for fitting curves
    light_range = np.linspace(0, max(light_intensities) * 1.1, 100)

    # Generate simulated growth rate data if fitted parameters are available
    c0_conditions = [1.0, 0.5, 0.25, 0.125, 0.062]
    simulated_growth_data = None
    if fitted_params is not None:
        print("Generating simulated growth rate data...")
        simulated_growth_data = generate_simulated_growth_rates(
            fitted_params, light_range, c0_conditions
        )
        print(
            f"Generated simulated growth rate data for {len(simulated_growth_data)} C0 conditions"
        )

    # Plot 1: Growth Rates for all C0 conditions vs Light Intensity
    colors_c0 = plt.cm.viridis(np.linspace(0, 1, len(c0_conditions)))

    for i, c0_factor in enumerate(c0_conditions):
        # Extract growth rates for this C0 condition across all datasets
        c0_growth_rates = []
        c0_light_intensities = []
        c0_good_fits = []

        for j, dataset in enumerate(datasets):
            if c0_factor in dataset.get("growth_rates_by_condition", {}):
                c0_growth_rates.append(dataset["growth_rates_by_condition"][c0_factor])
                c0_light_intensities.append(dataset["light_intensity"])
                c0_good_fits.append(dataset["r_squared"] > 0.8)

        c0_growth_rates = np.array(c0_growth_rates)
        c0_light_intensities = np.array(c0_light_intensities)
        c0_good_fits = np.array(c0_good_fits)

        if len(c0_growth_rates) > 0:
            # Plot good and bad fits with different markers
            if np.any(c0_good_fits):
                ax1.scatter(
                    c0_light_intensities[c0_good_fits],
                    c0_growth_rates[c0_good_fits],
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    label=f"C0*{c0_factor}",
                )
            if np.any(~c0_good_fits):
                ax1.scatter(
                    c0_light_intensities[~c0_good_fits],
                    c0_growth_rates[~c0_good_fits],
                    c=[colors_c0[i]],
                    s=60,
                    alpha=0.8,
                    marker="x",
                )

            # Fit Steele models for good data points
            if np.sum(c0_good_fits) > 1:
                # Steele fit for growth rates
                mu_max_steele_c0, K_steele_c0, beta_steele_c0, r2_steele_c0, _ = (
                    fit_steele_light_parameters(
                        c0_light_intensities[c0_good_fits],
                        c0_growth_rates[c0_good_fits],
                    )
                )

                if mu_max_steele_c0 is not None and r2_steele_c0 > 0.3:
                    steele_curve_c0 = steele_light_model(
                        light_range, mu_max_steele_c0, K_steele_c0, beta_steele_c0
                    )
                    ax1.plot(
                        light_range,
                        steele_curve_c0,
                        "-",
                        alpha=0.7,
                        color=colors_c0[i],
                        linewidth=2,
                        label=f"C0*{c0_factor} Steele" if c0_factor == 1.0 else "",
                    )

        # Add simulated data for this C0 condition if available
        if simulated_growth_data is not None and c0_factor in simulated_growth_data:
            sim_data = simulated_growth_data[c0_factor]
            ax1.plot(
                sim_data["light_intensities"],
                sim_data["growth_rates"],
                "--",
                alpha=0.8,
                color=colors_c0[i],
                linewidth=2,
                label=f"Simulated C0*{c0_factor}" if c0_factor == 1.0 else "",
            )

    ax1.set_xlabel("Light Intensity (μmol/m²/s)")
    ax1.set_ylabel("Specific Growth Rate (h⁻¹)")
    ax1.set_title(
        "Experimental vs Simulated Growth Rates vs Light Intensity\n(All C0 conditions - dashed lines: tuning_ODE.py simulations)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Plot 2: C0*1.0 growth rate with fitting
    # Extract C0*1.0 growth rates
    growth_rates_c0_1 = []
    light_intensities_gr = []
    good_fits_gr = []

    for dataset in datasets:
        if 1.0 in dataset.get("growth_rates_by_condition", {}):
            growth_rates_c0_1.append(dataset["growth_rates_by_condition"][1.0])
            light_intensities_gr.append(dataset["light_intensity"])
            good_fits_gr.append(dataset["r_squared"] > 0.8)

    growth_rates_c0_1 = np.array(growth_rates_c0_1)
    light_intensities_gr = np.array(light_intensities_gr)
    good_fits_gr = np.array(good_fits_gr)

    if len(growth_rates_c0_1) > 0:
        # Plot good and bad fits
        if np.any(good_fits_gr):
            ax2.scatter(
                light_intensities_gr[good_fits_gr],
                growth_rates_c0_1[good_fits_gr],
                color="blue",
                s=80,
                alpha=0.7,
                label="Good fits (R² > 0.8)",
            )
        if np.any(~good_fits_gr):
            ax2.scatter(
                light_intensities_gr[~good_fits_gr],
                growth_rates_c0_1[~good_fits_gr],
                color="red",
                s=80,
                alpha=0.7,
                marker="x",
                label="Poor fits (R² ≤ 0.8)",
            )

        # Fit Steele model for good fits
        if np.sum(good_fits_gr) > 1:
            # Steele fit
            mu_max_steele, K_steele, beta_steele, r2_steele, param_errors_steele = (
                fit_steele_light_parameters(
                    light_intensities_gr[good_fits_gr], growth_rates_c0_1[good_fits_gr]
                )
            )

            # Monod fit for comparison
            mu_max_monod, K_monod, r2_monod, param_errors_monod = (
                fit_monod_light_parameters(
                    light_intensities_gr[good_fits_gr], growth_rates_c0_1[good_fits_gr]
                )
            )

            if mu_max_steele is not None:
                steele_curve = steele_light_model(
                    light_range, mu_max_steele, K_steele, beta_steele
                )
                ax2.plot(
                    light_range,
                    steele_curve,
                    "r-",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Steele: R² = {r2_steele:.3f}",
                )

            if mu_max_monod is not None:
                monod_curve = monod_light_model(light_range, mu_max_monod, K_monod)
                ax2.plot(
                    light_range,
                    monod_curve,
                    "g-",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Monod: R² = {r2_monod:.3f}",
                )

            # Choose and display the best model
            best_is_steele = mu_max_steele is not None and r2_steele > r2_monod

            if best_is_steele:
                ax2.text(
                    0.05,
                    0.95,
                    f"BEST: Steele Model\nR² = {r2_steele:.3f}\nβ = {beta_steele:.5f}",
                    transform=ax2.transAxes,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7
                    ),
                )
            elif mu_max_monod is not None:
                ax2.text(
                    0.05,
                    0.95,
                    f"BEST: Monod Model\nR² = {r2_monod:.3f}\nμ_max = {mu_max_monod:.4f}\nK_L = {K_monod:.0f}",
                    transform=ax2.transAxes,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

        # Add simulated data for C0*1.0 if available
        if simulated_growth_data is not None and 1.0 in simulated_growth_data:
            sim_data = simulated_growth_data[1.0]
            ax2.plot(
                sim_data["light_intensities"],
                sim_data["growth_rates"],
                "k--",
                alpha=0.8,
                linewidth=3,
                label="Simulated (tuning_ODE.py)",
            )

        # Add dataset labels
        for i, (x, y, date, r2) in enumerate(
            zip(
                light_intensities_gr,
                growth_rates_c0_1,
                [
                    d["date"]
                    for d in datasets
                    if 1.0 in d.get("growth_rates_by_condition", {})
                ],
                good_fits_gr,
            )
        ):
            color = "blue" if r2 else "red"
            ax2.annotate(
                date,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                color=color,
            )

    ax2.set_xlabel("Light Intensity (μmol/m²/s)")
    ax2.set_ylabel("Specific Growth Rate (h⁻¹)")
    ax2.set_title(
        "C0*1.0: Experimental vs Simulated Growth Rates\n(Black dashed: tuning_ODE.py simulation)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "growth_rate_light_dependency.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return output_path


def plot_experimental_data_with_growth_windows(datasets, output_dir="results"):
    """Create plots showing experimental data with growth rate calculation windows marked."""
    # Create a figure with subplots for different datasets (light intensities)
    n_datasets = len(datasets)

    # Create a grid of subplots - 2 columns, adjust rows as needed
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))

    # Make sure axes is always a 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Colors for different C0 conditions
    c0_conditions = [1.0, 0.5, 0.25, 0.125, 0.062]
    colors_c0 = plt.cm.viridis(np.linspace(0, 1, len(c0_conditions)))
    condition_colors = {f"C0*{c0}": colors_c0[i] for i, c0 in enumerate(c0_conditions)}

    for i, dataset in enumerate(datasets):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        date = dataset["date"]
        light_intensity = dataset["light_intensity"]
        filepath = dataset.get("filepath", f"../all_data/data_exp_Chlamy_{date}.csv")

        # Load experimental data
        df = load_experimental_data(filepath)
        if df is None:
            ax.text(
                0.5,
                0.5,
                f"Data not available\nfor {date}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{date} (L0={light_intensity} μmol/m²/s)")
            continue

        growth_curves = extract_growth_curves(df)

        if not growth_curves:
            ax.text(
                0.5,
                0.5,
                f"No growth curves\nfound for {date}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{date} (L0={light_intensity} μmol/m²/s)")
            continue

        # Plot each condition
        for condition in dataset["conditions"]:
            c0_factor = condition["c0_factor"]
            condition_name = f"C0*{c0_factor}"

            if condition_name in growth_curves:
                curve_data = growth_curves[condition_name]
                time = curve_data["time"]
                cells = curve_data["cells"]

                # Plot the full growth curve
                ax.semilogy(
                    time,
                    cells,
                    "-",
                    color=condition_colors[condition_name],
                    alpha=0.7,
                    linewidth=1,
                    label=condition_name,
                )

                # Mark the exponential phase window used for growth rate calculation
                exp_start = condition["exp_start"]
                exp_end = condition["exp_end"]

                # Find indices for the exponential phase
                exp_mask = (time >= exp_start) & (time <= exp_end)
                if np.any(exp_mask):
                    ax.semilogy(
                        time[exp_mask],
                        cells[exp_mask],
                        "o",
                        color=condition_colors[condition_name],
                        markersize=4,
                        alpha=0.8,
                        markerfacecolor="white",
                        markeredgewidth=1.5,
                    )

                    # Add a line showing the fitted exponential growth
                    growth_rate = condition["growth_rate"]
                    if not np.isnan(growth_rate) and growth_rate > 0:
                        # Calculate the fitted line: N(t) = N0 * exp(μ * t)
                        # Use the first point in the exponential window as N0
                        t_exp = time[exp_mask]
                        cells_exp = cells[exp_mask]
                        if len(t_exp) > 0:
                            # Fit: ln(N) = ln(N0) + μ*t
                            ln_cells = np.log(cells_exp)
                            t_rel = (
                                t_exp - t_exp[0]
                            )  # relative time from start of exp phase

                            # Calculate fitted line
                            N0_fit = cells_exp[0]
                            fitted_cells = N0_fit * np.exp(growth_rate * t_rel)

                            ax.semilogy(
                                t_exp,
                                fitted_cells,
                                "--",
                                color=condition_colors[condition_name],
                                linewidth=2,
                                alpha=0.9,
                            )

                            # Add growth rate annotation for C0*1.0 only to avoid clutter
                            if c0_factor == 1.0:
                                mid_idx = len(t_exp) // 2
                                ax.annotate(
                                    f"μ = {growth_rate:.3f} h⁻¹",
                                    xy=(t_exp[mid_idx], fitted_cells[mid_idx]),
                                    xytext=(10, 10),
                                    textcoords="offset points",
                                    fontsize=8,
                                    ha="left",
                                    bbox=dict(
                                        boxstyle="round,pad=0.3",
                                        facecolor="white",
                                        alpha=0.8,
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="->",
                                        color=condition_colors[condition_name],
                                    ),
                                )

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Cell Density (cells/mL)")
        ax.set_title(f"{date} (L0={light_intensity} μmol/m²/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Add text box explaining the markers
        ax.text(
            0.02,
            0.98,
            "Circles: exponential phase\nDashed: fitted growth rate",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        )

    # Hide empty subplots
    for i in range(n_datasets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "experimental_data_with_growth_windows.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return output_path


def print_summary_table(datasets):
    """Print a summary table of all results."""
    print("=" * 100)
    print("LIGHT DEPENDENCY ANALYSIS SUMMARY (from YAML)")
    print("=" * 100)
    print(
        f"{'Dataset':<12} {'L0':<5} {'μ_max':<10} {'Γ(×10⁶)':<12} {'R²':<8} {'μ@C0*1.0':<10} {'Conditions':<10}"
    )
    print("-" * 100)

    for dataset in sorted(datasets, key=lambda x: x["light_intensity"]):
        print(
            f"{dataset['date']:<12} {dataset['light_intensity']:<5.0f} "
            f"{dataset['mu_max']:<10.4f} {dataset['gamma'] / 1e6:<12.2f} "
            f"{dataset['r_squared']:<8.4f} {dataset['mu_at_c0_1']:<10.4f} "
            f"{len(dataset['conditions']):<10}"
        )

    # Calculate linear relationships for good quality data (R² > 0.8)
    good_datasets = [d for d in datasets if d["r_squared"] > 0.8]

    if len(good_datasets) > 1:
        light_vals = np.array([d["light_intensity"] for d in good_datasets])
        mu_max_vals = np.array([d["mu_max"] for d in good_datasets])
        mu_c0_vals = np.array([d["mu_at_c0_1"] for d in good_datasets])

        # Fit linear relationships
        z_mu_max = np.polyfit(light_vals, mu_max_vals, 1)
        z_mu_c0 = np.polyfit(light_vals, mu_c0_vals, 1)

        # Calculate correlation coefficients
        corr_mu_max = np.corrcoef(light_vals, mu_max_vals)[0, 1]
        corr_mu_c0 = np.corrcoef(light_vals, mu_c0_vals)[0, 1]

        print("\nLinear relationships for good quality data (R² > 0.8):")
        print(
            f"μ_max vs L₀:     μ_max = {z_mu_max[0]:.2e} × L₀ + {z_mu_max[1]:.4f} (r = {corr_mu_max:.4f})"
        )
        print(
            f"μ(C0*1.0) vs L₀: μ = {z_mu_c0[0]:.2e} × L₀ + {z_mu_c0[1]:.4f} (r = {corr_mu_c0:.4f})"
        )

        # Monod light fits
        print("\nMonod light relationships for good quality data (R² > 0.8):")
        mu_max_light_fit, K_L_fit, r2_monod_fit, _ = fit_monod_light_parameters(
            light_vals, mu_max_vals
        )
        if mu_max_light_fit is not None:
            print(
                f"μ_max vs L₀:     μ_max = {mu_max_light_fit:.4f} × L₀ / ({K_L_fit:.0f} + L₀) (R² = {r2_monod_fit:.4f})"
            )

        mu_max_light_c0_fit, K_L_c0_fit, r2_monod_c0_fit, _ = (
            fit_monod_light_parameters(light_vals, mu_c0_vals)
        )
        if mu_max_light_c0_fit is not None:
            print(
                f"μ(C0*1.0) vs L₀: μ = {mu_max_light_c0_fit:.4f} × L₀ / ({K_L_c0_fit:.0f} + L₀) (R² = {r2_monod_c0_fit:.4f})"
            )

        # Offset Monod light fits
        print("\nOffset Monod light relationships for good quality data (R² > 0.8):")
        (
            mu_offset_fit,
            mu_max_light_offset_fit,
            K_L_offset_fit,
            r2_offset_monod_fit,
            _,
        ) = fit_offset_monod_light_parameters(light_vals, mu_max_vals)
        if mu_offset_fit is not None:
            print(
                f"μ_max vs L₀:     μ = {mu_offset_fit:.4f} + {mu_max_light_offset_fit:.4f} × L₀ / ({K_L_offset_fit:.0f} + L₀) (R² = {r2_offset_monod_fit:.4f})"
            )

        (
            mu_offset_c0_fit,
            mu_max_light_offset_c0_fit,
            K_L_offset_c0_fit,
            r2_offset_monod_c0_fit,
            _,
        ) = fit_offset_monod_light_parameters(light_vals, mu_c0_vals)
        if mu_offset_c0_fit is not None:
            print(
                f"μ(C0*1.0) vs L₀: μ = {mu_offset_c0_fit:.4f} + {mu_max_light_offset_c0_fit:.4f} × L₀ / ({K_L_offset_c0_fit:.0f} + L₀) (R² = {r2_offset_monod_c0_fit:.4f})"
            )

        # Show linear relationships for all C0 conditions
        print("\nLinear relationships for individual C0 conditions (good fits only):")
        c0_conditions = [1.0, 0.5, 0.25, 0.125, 0.062]
        for c0_factor in c0_conditions:
            # Extract growth rates for this C0 condition across all datasets
            c0_growth_rates = []
            c0_light_intensities = []

            for dataset in good_datasets:
                if c0_factor in dataset["growth_rates_by_condition"]:
                    c0_growth_rates.append(
                        dataset["growth_rates_by_condition"][c0_factor]
                    )
                    c0_light_intensities.append(dataset["light_intensity"])

            if len(c0_growth_rates) > 1:
                c0_light_vals = np.array(c0_light_intensities)
                c0_mu_vals = np.array(c0_growth_rates)

                # Fit linear relationship
                z_c0 = np.polyfit(c0_light_vals, c0_mu_vals, 1)
                corr_c0 = np.corrcoef(c0_light_vals, c0_mu_vals)[0, 1]

                print(
                    f"μ(C0*{c0_factor:<4}) vs L₀: μ = {z_c0[0]:.2e} × L₀ + {z_c0[1]:.4f} (r = {corr_c0:.4f})"
                )

        # Show Monod light relationships for individual C0 conditions
        print(
            "\nMonod light relationships for individual C0 conditions (good fits only):"
        )
        for c0_factor in c0_conditions:
            # Extract growth rates for this C0 condition across all datasets
            c0_growth_rates = []
            c0_light_intensities = []

            for dataset in good_datasets:
                if c0_factor in dataset["growth_rates_by_condition"]:
                    c0_growth_rates.append(
                        dataset["growth_rates_by_condition"][c0_factor]
                    )
                    c0_light_intensities.append(dataset["light_intensity"])

            if len(c0_growth_rates) > 1:
                c0_light_vals = np.array(c0_light_intensities)
                c0_mu_vals = np.array(c0_growth_rates)

                # Fit Monod light relationship
                mu_max_light_c0, K_L_c0, r2_monod_c0, _ = fit_monod_light_parameters(
                    c0_light_vals, c0_mu_vals
                )

                if mu_max_light_c0 is not None:
                    print(
                        f"μ(C0*{c0_factor:<4}) vs L₀: μ = {mu_max_light_c0:.4f} × L₀ / ({K_L_c0:.0f} + L₀) (R² = {r2_monod_c0:.4f})"
                    )

        # Show Offset Monod light relationships for individual C0 conditions
        print(
            "\nOffset Monod light relationships for individual C0 conditions (good fits only):"
        )
        for c0_factor in c0_conditions:
            # Extract growth rates for this C0 condition across all datasets
            c0_growth_rates = []
            c0_light_intensities = []

            for dataset in good_datasets:
                if c0_factor in dataset["growth_rates_by_condition"]:
                    c0_growth_rates.append(
                        dataset["growth_rates_by_condition"][c0_factor]
                    )
                    c0_light_intensities.append(dataset["light_intensity"])

            if len(c0_growth_rates) > 1:
                c0_light_vals = np.array(c0_light_intensities)
                c0_mu_vals = np.array(c0_growth_rates)

                # Fit Offset Monod light relationship
                (
                    mu_offset_c0,
                    mu_max_light_offset_c0,
                    K_L_offset_c0,
                    r2_offset_monod_c0,
                    _,
                ) = fit_offset_monod_light_parameters(c0_light_vals, c0_mu_vals)

                if mu_offset_c0 is not None:
                    print(
                        f"μ(C0*{c0_factor:<4}) vs L₀: μ = {mu_offset_c0:.4f} + {mu_max_light_offset_c0:.4f} × L₀ / ({K_L_offset_c0:.0f} + L₀) (R² = {r2_offset_monod_c0:.4f})"
                    )

    # Identify anomalous datasets
    anomalous_datasets = [d for d in datasets if d["r_squared"] <= 0.8]
    if anomalous_datasets:
        print(f"\nAnomalous datasets (R² ≤ 0.8):")
        for d in anomalous_datasets:
            print(
                f"  {d['date']} (L₀ = {d['light_intensity']} μmol/m²/s): R² = {d['r_squared']:.4f}"
            )


def main():
    """Main function to load YAML and create plots."""
    yaml_filepath = "analysis_scripts/results/microalgae_growth_analysis.yaml"

    if not os.path.exists(yaml_filepath):
        print(f"Error: YAML file not found at {yaml_filepath}")
        print("Please run multi_dataset_analysis.py first to generate the YAML file.")
        return

    # Load results
    print(f"Loading analysis results from {yaml_filepath}...")
    data = load_analysis_results(yaml_filepath)

    # Extract parameters for plotting
    print("Extracting parameters...")
    datasets = extract_parameters_for_plotting(data)

    if not datasets:
        print("No valid datasets found in YAML file.")
        return

    print(f"Found {len(datasets)} valid datasets for analysis.")

    # Print summary of steady states found
    for ds in datasets:
        ss_count = len(ds["steady_states_by_condition"])
        print(f"  {ds['date']} (L0={ds['light_intensity']}): {ss_count} steady states")

    # Create parameter dependency plots
    print("Creating parameter dependency plots...")
    output_path = plot_light_dependency(datasets)
    print(f"\nLight dependency plot saved to: {output_path}")

    # Create separate steady state plots
    print("Creating steady state analysis plots...")
    steady_state_path = plot_steady_states(datasets)
    print(f"Steady state analysis plot saved to: {steady_state_path}")

    # Create separate growth rate plots
    print("Creating growth rate analysis plots...")
    growth_rate_path = plot_growth_rates(datasets)
    print(f"Growth rate analysis plot saved to: {growth_rate_path}")

    # Create experimental data plots with growth windows
    print("Creating experimental data plots with growth rate calculation windows...")
    experimental_data_path = plot_experimental_data_with_growth_windows(datasets)
    print(
        f"Experimental data with growth windows plot saved to: {experimental_data_path}"
    )

    # Print summary table
    print_summary_table(datasets)


if __name__ == "__main__":
    main()
