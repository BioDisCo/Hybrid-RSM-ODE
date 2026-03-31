"""
Response Surface Methodology (RSM) Analysis for Erlen Growth Curves of C. reinhardtii
With integrated RSM-based logistic model predictions
Analyzes growth parameters (mu_max, Nmax) as a function of L0 (light) and C0 (nutrients) with error bars on technical replicates.
Also generates predictions based on RSM surfaces integrated into the logistic model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import pathlib
import logging
from matplotlib.lines import Line2D
import matplotlib as mpl
import yaml
from scipy.interpolate import interp1d
from data_import import read_csv_data_erlen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]

# ================================================================================
# RSM MODEL CONFIGURATION - DOUBLE MONOD ONLY
# ================================================================================


class ModelConfig:
    """
    RSM model configuration: Double Monod
    f(L0, C0) = β₀ × [L0/(L0 + K_L0)] × [C0/(C0 + K_C0)]
    """

    # =========================================================================
    # DOUBLE MONOD MODEL FOR MU_MAX AND NMAX
    # =========================================================================

    @staticmethod
    def monod_model(L0, C0, params):
        """
        Double Monod model:
        params[0] × [L0/(L0 + params[1])] × [C0/(C0 + params[2])]

        Args:
            L0: array, light intensity
            C0: array, nutrient concentration
            params: [β₀ (max), K_L0, K_C0]
        """
        return params[0] * (L0 / (L0 + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def monod_initial_params(L0, C0, y):
        """
        Initial parameters for the fit.

        Returns:
            [β₀, K_L0, K_C0]
        """
        return [
            np.max(y) * 1.2,  # β₀: maximum value
            np.median(L0),  # K_L0: half-saturation constant for L0
            np.median(C0),  # K_C0: half-saturation constant for C0
        ]

    @staticmethod
    def monod_bounds():
        """
        Bounds for the parameters.

        Returns:
            (lower_bounds, upper_bounds)
        """
        return (
            [0, 10, 0.001],  # lower bounds [β₀, K_L0, K_C0]
            [np.inf, 1000, 10],  # upper bounds
        )

    @staticmethod
    def monod_latex(params):
        """LaTeX formula for display."""
        return (
            f"$= {params[0]:.3e} \\cdot "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3e}}} \\cdot "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3e}}}$"
        )

    @staticmethod
    def monod_param_names():
        """Parameter names."""
        return ["β₀ (max)", "K_L0", "K_C0"]

    # -------------------------------------------------------------------------
    # INTERFACES FOR MU_MAX
    # -------------------------------------------------------------------------

    @classmethod
    def mu_max_model(cls, L0, C0, params):
        return cls.monod_model(L0, C0, params)

    @classmethod
    def mu_max_initial_params(cls, L0, C0, y):
        return cls.monod_initial_params(L0, C0, y)

    @classmethod
    def mu_max_bounds(cls):
        return cls.monod_bounds()

    @classmethod
    def mu_max_latex_equation(cls, params):
        prefix = "$\\mu_{\\mathrm{max}}$ "
        return prefix + cls.monod_latex(params)

    @classmethod
    def mu_max_param_names(cls):
        return cls.monod_param_names()

    # -------------------------------------------------------------------------
    # INTERFACES FOR NMAX
    # -------------------------------------------------------------------------

    @classmethod
    def Nmax_model(cls, L0, C0, params):
        return cls.monod_model(L0, C0, params)

    @classmethod
    def Nmax_initial_params(cls, L0, C0, y):
        return cls.monod_initial_params(L0, C0, y)

    @classmethod
    def Nmax_bounds(cls):
        return cls.monod_bounds()

    @classmethod
    def Nmax_latex_equation(cls, params):
        prefix = "$N_{\\mathrm{max}}$ "
        return prefix + cls.monod_latex(params)

    @classmethod
    def Nmax_param_names(cls):
        return cls.monod_param_names()

    # ================================================================================
    # END OF CONFIGURATION
    # ================================================================================
    """
    Centralized model configuration for mu_max and Nmax.
    ONLY MODIFY THIS CLASS to change the equations!
    """

    # -------------------------------------------------------------------------
    # MU_MAX MODEL
    # -------------------------------------------------------------------------

    @staticmethod
    def mu_max_model(L0, C0, params):
        """
        Defines the equation for mu_max.
        Double Monod model (without quadratic term).

        Args:
            L0: light intensity
            C0: nutrient dilution factor
            params: parameter array [beta_0, beta_1, beta_2]

        Returns:
            Predicted mu_max value
        """
        # Double Monod model: beta_0 * [L0/(L0 + beta_1)] * [C0/(C0 + beta_2)]
        return params[0] * (L0 / (L0 + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def mu_max_initial_params(L0, C0, y):
        """
        Initial values for mu_max parameters.
        Returns: [beta_0_init, beta_1_init, beta_2_init]
        """
        return [
            np.max(y) * 1.2,  # beta_0: maximum value
            np.median(L0),  # beta_1: K_L0 (half-saturation constant for L0)
            np.median(C0),  # beta_2: K_C0 (half-saturation constant for C0)
        ]

    @staticmethod
    def mu_max_bounds():
        """
        Bounds for mu_max parameters.
        Returns: (lower_bounds, upper_bounds)
        """
        return (
            [0, 10, 0.001],  # lower bounds
            [np.inf, 1000, 10],  # upper bounds
        )

    @staticmethod
    def mu_max_latex_equation(params):
        """
        Format mu_max equation in LaTeX for display.
        """
        return (
            f"$\\mu_{{\\mathrm{{max}}}} = {params[0]:.3e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3e}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3e}}}$"
        )

    @staticmethod
    def mu_max_param_names():
        """Parameter names for log display."""
        return ["β₀ (maximum)", "β₁ (K_L0)", "β₂ (K_C0)"]

    # -------------------------------------------------------------------------
    # NMAX MODEL
    # -------------------------------------------------------------------------

    @staticmethod
    def Nmax_model(L0, C0, params):
        """
        Defines the equation for Nmax.
        Double Monod model (without quadratic term).
        """
        return params[0] * (L0 / (L0 + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def Nmax_initial_params(L0, C0, y):
        """Initial values for Nmax parameters."""
        return [
            np.max(y) * 1.2,  # beta_0: maximum value
            np.median(L0),  # beta_1: K_L0 (half-saturation constant for L0)
            np.median(C0),  # beta_2: K_C0 (half-saturation constant for C0)
        ]

    @staticmethod
    def Nmax_bounds():
        """Bounds for Nmax parameters."""
        return ([0, 10, 0.001], [np.inf, 1000, 10])

    @staticmethod
    def Nmax_latex_equation(params):
        """Format Nmax equation in LaTeX."""
        return (
            f"$N_{{\\mathrm{{max}}}} = {params[0]:.3e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3e}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3e}}}$"
        )

    @staticmethod
    def Nmax_param_names():
        """Parameter names for display."""
        return ["β₀ (maximum)", "β₁ (K_L0)", "β₂ (K_C0)"]


# ================================================================================
# END OF CONFIGURATION - DO NOT MODIFY BELOW EXCEPT FOR DEBUG
# ================================================================================


def calculate_r2(y_true, y_pred, x_true=None, x_pred=None):
    """
    Robust R² computation, backward-compatible with older call signatures.
    - If x_true and x_pred provided: interpolates y_pred onto x_true (no extrapolation).
    - If x_true/x_pred not provided:
        * if len(y_true) == len(y_pred) -> direct comparison
        * otherwise -> creates normalized abscissas (linspace) to align
          and restricts to the intersection of domains (NO extrapolation).
    Returns a float R² (can be negative if the model is worse than the mean).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # --- simple case: same lengths and no time arrays provided ---
    if x_true is None and x_pred is None and len(y_true) == len(y_pred):
        y_t = y_true
        y_p = y_pred
    else:
        # build abscissas if not provided
        if x_true is None:
            x_true = np.linspace(0.0, 1.0, len(y_true))
        else:
            x_true = np.asarray(x_true, dtype=float)

        if x_pred is None:
            x_pred = np.linspace(0.0, 1.0, len(y_pred))
        else:
            x_pred = np.asarray(x_pred, dtype=float)

        # Determine common domain (no extrapolation allowed)
        xmin = max(np.min(x_true), np.min(x_pred))
        xmax = min(np.max(x_true), np.max(x_pred))

        # restrict x_true to points covered by x_pred
        mask = (x_true >= xmin) & (x_true <= xmax)
        if np.sum(mask) < 2:
            # not enough common points: cannot compute correctly
            return 0.0

        x_t_common = x_true[mask]
        y_t_common = y_true[mask]

        # linear interpolation (NO extrapolation) of y_pred onto x_t_common
        # interpolator created only on x_pred; interp1d will raise if x_t_common
        # contains values outside [min(x_pred), max(x_pred)]
        try:
            f = interp1d(x_pred, y_pred, kind="linear", bounds_error=True)
            y_p_common = f(x_t_common)
        except ValueError:
            # if x_pred doesn't cover the interval (should be avoided by xmin/xmax),
            # return 0.0 to indicate comparison is not possible.
            return 0.0

        y_t = y_t_common
        y_p = y_p_common

    # --- NaN / inf cleanup ---
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if np.sum(valid) < 2:
        return 0.0

    y_t = y_t[valid]
    y_p = y_p[valid]

    # --- standard sum of squares computation ---
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot < 1e-12:
        # zero variance in y_true -> no information
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)

    return float(r2)


def print_r2_statistics_summary(df_r2, df_params_all, coeffs_mu_max, coeffs_Nmax):
    """
    Print R² statistics in a formatted summary for calibration, validation, and overall.

    Args:
        df_r2: DataFrame with R2_rsm_mean, used_for_fit columns
        df_params_all: DataFrame with all conditions and replicate_fits
        coeffs_mu_max: RSM coefficients for mu_max
        coeffs_Nmax: RSM coefficients for Nmax
    """

    def get_params_subset_by_L0_C0(df_r2_subset, df_params_all):
        """Get df_params rows matching L0, C0 pairs from df_r2_subset."""
        result_rows = []
        for _, r2_row in df_r2_subset.iterrows():
            L0, C0 = r2_row["L0"], r2_row["C0"]
            matching = df_params_all[
                (df_params_all["L0"] == L0) & (df_params_all["C0"] == C0)
            ]
            if len(matching) > 0:
                result_rows.append(matching.iloc[0])
        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame()

    def compute_stats_for_subset(df_r2_subset, df_params_subset, label):
        """Compute and print stats for a subset of conditions."""
        r2_values = df_r2_subset["R2_rsm_mean"].values
        valid_r2 = r2_values[~np.isnan(r2_values)]

        if len(valid_r2) == 0:
            logger.info(f"{label}:")
            logger.info("  No valid R² values found!")
            return None, 0

        n_conditions = len(valid_r2)
        n_good = np.sum(valid_r2 >= 0.85)
        pct_good = 100.0 * n_good / n_conditions if n_conditions > 0 else 0.0

        mean_r2 = np.mean(valid_r2)
        std_r2 = np.std(valid_r2)
        median_r2 = np.median(valid_r2)
        min_r2 = np.min(valid_r2)
        max_r2 = np.max(valid_r2)

        # Collect all data points for global R²
        all_biomass_exp = []
        all_biomass_pred = []

        for _, row in df_params_subset.iterrows():
            L0, C0 = row["L0"], row["C0"]
            replicate_fits = row["replicate_fits"]

            N0_values = [
                fit["fit"]["N0"]
                for fit in replicate_fits
                if "fit" in fit and fit["fit"] is not None
            ]
            if not N0_values:
                continue
            N0_mean = np.mean(N0_values)

            for fit_data in replicate_fits:
                time_exp, biomass_exp = fit_data["time"], fit_data["biomass"]
                valid_mask = ~np.isnan(biomass_exp) & (biomass_exp > 0)
                time_clean, biomass_clean = (
                    time_exp[valid_mask],
                    biomass_exp[valid_mask],
                )
                if len(time_clean) < 4:
                    continue

                biomass_rsm = logistic_growth_rsm(
                    time_clean,
                    N0_mean,
                    L0,
                    C0,
                    coeffs_mu_max,
                    coeffs_Nmax,
                    df_params_all,
                )

                all_biomass_exp.extend(biomass_clean)
                all_biomass_pred.extend(biomass_rsm)

        n_data_points = len(all_biomass_exp)
        if n_data_points > 0:
            global_r2 = calculate_r2(
                np.array(all_biomass_exp), np.array(all_biomass_pred)
            )
        else:
            global_r2 = np.nan

        # Print formatted output
        logger.info(f"{label}:")
        logger.info(f"  Number of conditions: {n_conditions}")
        logger.info(
            f"  Conditions with R² ≥ 0.85: {n_good}/{n_conditions} ({pct_good:.1f}%)"
        )
        logger.info(f"  Mean local R²: {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"  Median local R²: {median_r2:.4f}")
        logger.info(f"  Min local R²: {min_r2:.4f}")
        logger.info(f"  Max local R²: {max_r2:.4f}")
        if not np.isnan(global_r2):
            logger.info(f"  GLOBAL R² (all data points): {global_r2:.4f}")
            logger.info(f"    (calculated over {n_data_points} data points)")
        else:
            logger.info(f"  GLOBAL R² (all data points): N/A")

        return (all_biomass_exp, all_biomass_pred), n_conditions

    logger.info("=" * 80)
    logger.info("R² STATISTICS")
    logger.info("=" * 80)

    # CALIBRATION SET (used_for_fit == True)
    df_r2_calibration = df_r2[df_r2["used_for_fit"] == True]
    df_params_calibration = get_params_subset_by_L0_C0(df_r2_calibration, df_params_all)
    calib_data, n_calib = compute_stats_for_subset(
        df_r2_calibration, df_params_calibration, "CALIBRATION SET"
    )

    logger.info("")

    # VALIDATION SET (used_for_fit == False) - "intermediate dilutions"
    df_r2_validation = df_r2[df_r2["used_for_fit"] == False]
    df_params_validation = get_params_subset_by_L0_C0(df_r2_validation, df_params_all)
    valid_data, n_valid = compute_stats_for_subset(
        df_r2_validation,
        df_params_validation,
        "VALIDATION SET (intermediate dilutions)",
    )

    logger.info("")

    # OVERALL (calibration + validation combined)
    if (
        n_calib > 0
        and n_valid > 0
        and calib_data is not None
        and valid_data is not None
    ):
        # Combine all data
        all_r2_values = df_r2["R2_rsm_mean"].values
        valid_r2_overall = all_r2_values[~np.isnan(all_r2_values)]

        n_overall = len(valid_r2_overall)
        n_good_overall = np.sum(valid_r2_overall >= 0.85)
        pct_good_overall = 100.0 * n_good_overall / n_overall if n_overall > 0 else 0.0

        # Combine biomass data
        all_exp_combined = list(calib_data[0]) + list(valid_data[0])
        all_pred_combined = list(calib_data[1]) + list(valid_data[1])
        n_data_points_overall = len(all_exp_combined)

        if n_data_points_overall > 0:
            global_r2_overall = calculate_r2(
                np.array(all_exp_combined), np.array(all_pred_combined)
            )
        else:
            global_r2_overall = np.nan

        logger.info("OVERALL (calibration + validation combined):")
        logger.info(f"  Number of conditions: {n_overall}")
        logger.info(
            f"  Conditions with R² ≥ 0.85: {n_good_overall}/{n_overall} ({pct_good_overall:.1f}%)"
        )
        if not np.isnan(global_r2_overall):
            logger.info(f"  GLOBAL R² (all data points): {global_r2_overall:.4f}")
            logger.info(f"    (calculated over {n_data_points_overall} data points)")
        else:
            logger.info(f"  GLOBAL R² (all data points): N/A")
    elif n_calib > 0 or n_valid > 0:
        # Only one set has data
        all_r2_values = df_r2["R2_rsm_mean"].values
        valid_r2_overall = all_r2_values[~np.isnan(all_r2_values)]

        n_overall = len(valid_r2_overall)
        n_good_overall = np.sum(valid_r2_overall >= 0.85)
        pct_good_overall = 100.0 * n_good_overall / n_overall if n_overall > 0 else 0.0

        logger.info("OVERALL (calibration + validation combined):")
        logger.info(f"  Number of conditions: {n_overall}")
        logger.info(
            f"  Conditions with R² ≥ 0.85: {n_good_overall}/{n_overall} ({pct_good_overall:.1f}%)"
        )

    logger.info("=" * 80)


def logistic_growth(t, N0, Nmax, mu_max, t_lag):
    """
    Logistic growth model with lag phase.

    N(t) = Nmax / (1 + ((Nmax - N0) / N0) * exp(-mu_max * (t - t_lag)))
    """
    return Nmax / (1 + ((Nmax - N0) / N0) * np.exp(-mu_max * (t - t_lag)))


def logistic_growth_rsm(t, N0, L0, C0, coeffs_mu_max, coeffs_Nmax, df_params_tlag):
    """
    Logistic growth model with parameters predicted by RSM surfaces.
    For t_lag, uses the fixed experimental mean value of the condition (L0, C0).

    Args:
        t: time
        N0: initial population
        L0: light intensity
        C0: dilution factor
        coeffs_mu_max: RSM coefficients for mu_max
        coeffs_Nmax: RSM coefficients for Nmax
        df_params_tlag: DataFrame with columns ['L0', 'C0', 't_lag_mean']
    """
    # Use RSM models for mu_max and Nmax
    mu_max = ModelConfig.mu_max_model(L0, C0, coeffs_mu_max)
    Nmax = ModelConfig.Nmax_model(L0, C0, coeffs_Nmax)

    # Find the t_lag corresponding to this condition (L0, C0)
    # Tolerance for floating-point comparison
    mask = (np.abs(df_params_tlag["L0"] - L0) < 1.0) & (
        np.abs(df_params_tlag["C0"] - C0) < 0.001
    )
    matching_rows = df_params_tlag[mask]

    if len(matching_rows) > 0:
        t_lag = matching_rows.iloc[0]["t_lag_mean"]
    else:
        # If no exact match, use the closest value
        distances = np.sqrt(
            (df_params_tlag["L0"] - L0) ** 2 + (df_params_tlag["C0"] - C0) ** 2
        )
        closest_idx = distances.idxmin()
        t_lag = df_params_tlag.loc[closest_idx, "t_lag_mean"]
        logger.warning(
            f"No exact match for L0={L0}, C0={C0}, using closest: t_lag={t_lag:.2f}"
        )

    # Compute the logistic curve with these parameters
    return logistic_growth(t, N0, Nmax, mu_max, t_lag)


def estimate_growth_parameters(time, biomass):
    """
    Estimates growth parameters from data.

    Returns:
        dict: {mu_max, Nmax, t_lag, r_squared, N0} or None if failed
    """
    # Convert to NumPy arrays
    time = np.array(time)
    biomass = np.array(biomass)

    # Remove NaN values
    valid_mask = ~np.isnan(biomass) & (biomass > 0)
    time_clean = time[valid_mask]
    biomass_clean = biomass[valid_mask]

    if len(time_clean) < 4:
        return None

    # Initial estimates
    N0_guess = biomass_clean[0]
    Nmax_guess = np.max(biomass_clean)

    # Estimate mu_max from exponential phase
    if len(biomass_clean) > 5:
        growth_rates = []
        for i in range(1, len(biomass_clean) - 1):
            if (
                biomass_clean[i] > N0_guess * 1.5
                and biomass_clean[i] < Nmax_guess * 0.7
            ):
                dt = time_clean[i + 1] - time_clean[i - 1]
                dN = np.log(biomass_clean[i + 1]) - np.log(biomass_clean[i - 1])
                if dt > 0:
                    growth_rates.append(dN / dt)

        if growth_rates:
            mu_max_guess = np.max(growth_rates)
        else:
            mu_max_guess = 0.1
    else:
        mu_max_guess = 0.1

    t_lag_guess = time_clean[0]

    # Bounds for fitting
    bounds_lower = [N0_guess * 0.5, Nmax_guess * 0.7, 0.01, 0]
    bounds_upper = [N0_guess * 1.5, Nmax_guess * 2.0, 1.0, time_clean[-1] * 0.3]

    try:
        popt, pcov = curve_fit(
            logistic_growth,
            time_clean,
            biomass_clean,
            p0=[N0_guess, Nmax_guess, mu_max_guess, t_lag_guess],
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000,
        )

        N0_fit, Nmax_fit, mu_max_fit, t_lag_fit = popt

        # Compute R² on experimental data points
        biomass_pred = logistic_growth(time_clean, *popt)
        r_squared = calculate_r2(biomass_clean, biomass_pred)

        # Generate a smooth curve for display
        t_max = time_clean[-1]
        time_smooth = np.linspace(0, t_max, 500)
        biomass_smooth = logistic_growth(time_smooth, *popt)

        return {
            "N0": N0_fit,
            "Nmax": Nmax_fit,
            "mu_max": mu_max_fit,
            "t_lag": t_lag_fit,
            "r_squared": r_squared,
            "time_fit": time_smooth,
            "biomass_fit": biomass_smooth,
        }

    except Exception as e:
        logger.warning(f"Fitting failed: {e}")
        return None


def apply_manual_tlag_adjustments(df_params, manual_adjustments):
    """
    Applies manual t_lag adjustments to a DataFrame.

    Args:
        df_params: DataFrame with columns ['L0', 'C0', 't_lag_mean', 't_lag_std', ...]
        manual_adjustments: dict {(L0, C0): new_t_lag_value}

    Returns:
        Modified DataFrame with adjustments applied
    """
    if not manual_adjustments:
        logger.info("No manual t_lag adjustments to apply.")
        return df_params

    logger.info(f"\n{'=' * 80}")
    logger.info("APPLYING MANUAL T_LAG ADJUSTMENTS")
    logger.info(f"{'=' * 80}\n")

    df_modified = df_params.copy()

    for (L0_target, C0_target), new_tlag in manual_adjustments.items():
        # Find the matching row
        mask = (np.abs(df_modified["L0"] - L0_target) < 1.0) & (
            np.abs(df_modified["C0"] - C0_target) < 0.001
        )

        matching_indices = df_modified[mask].index

        if len(matching_indices) > 0:
            idx = matching_indices[0]
            old_tlag = df_modified.loc[idx, "t_lag_mean"]
            df_modified.loc[idx, "t_lag_mean"] = new_tlag

            logger.info(f"  L0={L0_target:.1f}, C0={C0_target:.4f}:")
            logger.info(f"    Old t_lag = {old_tlag:.2f} h")
            logger.info(f"    New t_lag = {new_tlag:.2f} h")
            logger.info(f"    Δt_lag    = {new_tlag - old_tlag:+.2f} h")
        else:
            logger.warning(
                f"  ⚠ Condition L0={L0_target:.1f}, C0={C0_target:.4f} not found in DataFrame!"
            )

    logger.info(f"\n{'=' * 80}\n")

    return df_modified


def filter_conditions(df_params, selected_conditions):
    """
    Filters the DataFrame to keep only the selected conditions.

    Args:
        df_params: DataFrame with all parameters
        selected_conditions: list of tuples (L0, C0) to keep
                            If None, keeps all conditions

    Returns:
        Filtered DataFrame
    """
    if selected_conditions is None:
        logger.info("No filtering applied - using all conditions")
        return df_params

    logger.info(f"\n{'=' * 80}")
    logger.info("FILTERING CONDITIONS")
    logger.info(f"{'=' * 80}")
    logger.info(f"Selected conditions: {len(selected_conditions)}")

    # Create a filter mask
    mask = pd.Series([False] * len(df_params))

    for L0_target, C0_target in selected_conditions:
        # Tolerance for floating-point comparison
        tolerance_L0 = 1.0  # ±1 µmol/m²/s
        tolerance_C0 = 0.001  # ±0.001 for C0

        condition_mask = (np.abs(df_params["L0"] - L0_target) < tolerance_L0) & (
            np.abs(df_params["C0"] - C0_target) < tolerance_C0
        )

        mask |= condition_mask

        n_matches = condition_mask.sum()
        if n_matches > 0:
            logger.info(f"  ✓ Found condition: L0={L0_target}, C0={C0_target}")
        else:
            logger.warning(f"  ✗ Condition not found: L0={L0_target}, C0={C0_target}")

    df_filtered = df_params[mask].copy()

    logger.info(
        f"\nFiltered: {len(df_filtered)} / {len(df_params)} conditions retained"
    )
    logger.info(f"{'=' * 80}\n")

    if len(df_filtered) == 0:
        logger.error("ERROR: No conditions match the selection criteria!")
        logger.error("Available conditions in dataset:")
        for _, row in df_params.iterrows():
            logger.error(f"  L0={row['L0']:.1f}, C0={row['C0']:.4f}")
        raise ValueError("No conditions match the selection criteria")

    return df_filtered



def extract_parameters_with_replicates(files_dict, conv_OD_to_cell=4.77e6, selected_conditions=None):
    """
    Extracts growth parameters for each file and each condition,
    computing parameters for each replicate A, B, C separately.

    Args:
        files_dict: dict of the form {L0_value: filepath}
        conv_OD_to_cell: OD to cells/mL conversion factor
        selected_conditions: list of tuples (L0, C0) to analyze (None = all)

    Returns:
        pd.DataFrame avec colonnes [L0, C0, mu_max_mean, mu_max_std, mu_max_A, mu_max_B, mu_max_C,
                                     Nmax_mean, Nmax_std, Nmax_A, Nmax_B, Nmax_C,
                                     t_lag_mean, t_lag_std, t_lag_A, t_lag_B, t_lag_C, n_replicates]
    """
    results = []

    # Convert selected_conditions to set for fast lookup
    if selected_conditions is not None:
        selected_set = set(selected_conditions)
        logger.info(
            f"Filtering enabled: will process only {len(selected_set)} selected conditions"
        )
    else:
        selected_set = None
        logger.info("No filtering: will process all conditions")

    for L0_factor, filepath in files_dict.items():
        logger.info(f"\nProcessing file: {filepath} (L0={L0_factor})")

        # Read data using data_import.read_csv_data_erlen
        experiments = read_csv_data_erlen(filepath, conv_OD_to_cell)

        for exp_name, exp_data in experiments.items():
            # Recover exact C0_factor from the truncated name value
            # The name format is "Erlen_C0x{:.3f}_L0x{:.3f}" with C0 = 0.5^n
            # We round to the nearest power of 0.5 to avoid :.3f truncation (e.g. 0.062 vs 0.0625)
            c0_approx = float(exp_name.split("_C0x")[1].split("_L0x")[0])
            n = round(np.log(c0_approx) / np.log(0.5)) if c0_approx > 0 else 0
            C0_factor = 0.5**n
            L0_absolute = L0_factor * 170  # L0 max = 170

            # Check if this condition is selected
            if selected_set is not None:
                condition_selected = False
                for L0_sel, C0_sel in selected_set:
                    if (
                        np.abs(L0_absolute - L0_sel) < 1.0
                        and np.abs(C0_factor - C0_sel) < 0.001
                    ):
                        condition_selected = True
                        break

                if not condition_selected:
                    logger.debug(
                        f"  Skipping L0={L0_absolute:.1f}, C0={C0_factor:.3f} (not selected)"
                    )
                    continue

            replicate_params = {"mu_max": [], "Nmax": [], "t_lag": [], "r_squared": []}
            rep_names = ["A", "B", "C"]
            replicate_values = {r: {} for r in rep_names}
            replicate_fits = []

            # Fit each replicate A, B, C
            for i, rep_name in enumerate(rep_names):
                if i >= len(exp_data["replicates"]):
                    break
                time = np.array(exp_data["replicates"][i]["Time"])
                biomass = np.array(exp_data["replicates"][i]["Value"])

                params = estimate_growth_parameters(time, biomass)

                if params and params["r_squared"] > 0.7:
                    replicate_params["mu_max"].append(params["mu_max"])
                    replicate_params["Nmax"].append(params["Nmax"])
                    replicate_params["t_lag"].append(params["t_lag"])
                    replicate_params["r_squared"].append(params["r_squared"])

                    replicate_values[rep_name]["mu_max"] = params["mu_max"]
                    replicate_values[rep_name]["Nmax"] = params["Nmax"]
                    replicate_values[rep_name]["t_lag"] = params["t_lag"]

                    replicate_fits.append(
                        {
                            "replicate": rep_name,
                            "time": time,
                            "biomass": biomass,
                            "fit": params,
                        }
                    )
                else:
                    replicate_values[rep_name]["mu_max"] = np.nan
                    replicate_values[rep_name]["Nmax"] = np.nan
                    replicate_values[rep_name]["t_lag"] = np.nan
                    logger.warning(
                        f"  L0={L0_absolute}, C0={C0_factor}, Rep {rep_name}: Fitting failed"
                    )

            # Compute mean and std if at least 2 replicates succeeded
            if len(replicate_params["mu_max"]) >= 2:
                results.append(
                    {
                        "L0": L0_absolute,
                        "C0": C0_factor,
                        "mu_max_mean": np.mean(replicate_params["mu_max"]),
                        "mu_max_std": np.std(replicate_params["mu_max"], ddof=1),
                        "mu_max_A": replicate_values["A"].get("mu_max", np.nan),
                        "mu_max_B": replicate_values["B"].get("mu_max", np.nan),
                        "mu_max_C": replicate_values["C"].get("mu_max", np.nan),
                        "Nmax_mean": np.mean(replicate_params["Nmax"]),
                        "Nmax_std": np.std(replicate_params["Nmax"], ddof=1),
                        "Nmax_A": replicate_values["A"].get("Nmax", np.nan),
                        "Nmax_B": replicate_values["B"].get("Nmax", np.nan),
                        "Nmax_C": replicate_values["C"].get("Nmax", np.nan),
                        "t_lag_mean": np.mean(replicate_params["t_lag"]),
                        "t_lag_std": np.std(replicate_params["t_lag"], ddof=1),
                        "t_lag_A": replicate_values["A"].get("t_lag", np.nan),
                        "t_lag_B": replicate_values["B"].get("t_lag", np.nan),
                        "t_lag_C": replicate_values["C"].get("t_lag", np.nan),
                        "r_squared_mean": np.mean(replicate_params["r_squared"]),
                        "n_replicates": len(replicate_params["mu_max"]),
                        "replicate_fits": replicate_fits,
                    }
                )

                logger.info(
                    f"  ✓ L0={L0_absolute:.1f}, C0={C0_factor:.3f}: "
                    f"mu_max={np.mean(replicate_params['mu_max']):.3f}±{np.std(replicate_params['mu_max'], ddof=1):.3f} h⁻¹, "
                    f"Nmax={np.mean(replicate_params['Nmax']):.2e}±{np.std(replicate_params['Nmax'], ddof=1):.2e}, "
                    f"t_lag={np.mean(replicate_params['t_lag']):.1f}±{np.std(replicate_params['t_lag'], ddof=1):.1f}h "
                    f"(n={len(replicate_params['mu_max'])})"
                )
            else:
                logger.warning(
                    f"  L0={L0_absolute}, C0={C0_factor}: Not enough successful fits"
                )

    return pd.DataFrame(results)


def fit_response_surface(df, response_var_mean, degree=2):
    """
    Fits a response surface on the means.
    Automatically uses the model defined in ModelConfig.
    """
    L0 = df["L0"].values
    C0 = df["C0"].values
    y = df[response_var_mean].values

    # Determine which model to use
    if "mu_max" in response_var_mean:
        model_func = ModelConfig.mu_max_model
        initial_params_func = ModelConfig.mu_max_initial_params
        bounds_func = ModelConfig.mu_max_bounds
    elif "Nmax" in response_var_mean:
        model_func = ModelConfig.Nmax_model
        initial_params_func = ModelConfig.Nmax_initial_params
        bounds_func = ModelConfig.Nmax_bounds
    else:
        raise ValueError(f"Unknown response variable: {response_var_mean}")

    # Get initial parameters
    p0 = initial_params_func(L0, C0, y)
    bounds = bounds_func()

    # Use curve_fit
    from scipy.optimize import curve_fit

    def model_to_fit(X, *params):
        L0_vals, C0_vals = X
        return model_func(L0_vals, C0_vals, np.array(params))

    try:
        if bounds is not None:
            popt, pcov = curve_fit(
                model_to_fit, (L0, C0), y, p0=p0, bounds=bounds, maxfev=10000
            )
        else:
            popt, pcov = curve_fit(model_to_fit, (L0, C0), y, p0=p0, maxfev=10000)

        coeffs = popt
        y_pred = model_func(L0, C0, coeffs)
        r2 = calculate_r2(y, y_pred)

        def predict(L0_new, C0_new):
            return model_func(L0_new, C0_new, coeffs)

        logger.info(f"  Fit successful: R² = {r2:.4f}, params = {coeffs}")

    except Exception as e:
        logger.warning(
            f"Fit failed for {response_var_mean}: {e}. Using initial values."
        )
        coeffs = np.array(p0)
        r2 = 0.0

        def predict(L0_new, C0_new):
            return model_func(L0_new, C0_new, coeffs)

    return predict, coeffs, r2


def format_coefficient_latex(coeff, is_first=False):
    """
    Formats a coefficient in LaTeX scientific notation with 10^n.
    """
    if coeff == 0:
        return ""

    exp = int(np.floor(np.log10(abs(coeff))))
    mantissa = coeff / (10**exp)

    if is_first:
        sign = "" if coeff >= 0 else "-"
        mantissa = abs(mantissa)
    else:
        sign = " + " if coeff >= 0 else " - "
        mantissa = abs(mantissa)

    if exp == 0:
        return f"{sign}{mantissa:.3f}"

    return f"{sign}{mantissa:.3f} \\times 10^{{{exp}}}"


def format_polynomial(coeffs, response_var):
    """
    Formats the equation in LaTeX using ModelConfig.
    """
    if response_var == "mu_max":
        return ModelConfig.mu_max_latex_equation(coeffs)
    elif response_var == "Nmax":
        return ModelConfig.Nmax_latex_equation(coeffs)
    else:
        raise ValueError(f"Unknown response variable: {response_var}")


def get_latex_label(response_var):
    """Returns the appropriate LaTeX label for each variable."""
    latex_labels = {
        "mu_max": r"$\mu_{\mathrm{max}}$ (h$^{-1}$)",
        "Nmax": r"$N_{\mathrm{max}}$ (cells/mL)",
        "t_lag": r"$t_{\mathrm{lag}}$ (h)",
    }
    return latex_labels.get(response_var, response_var)


def plot_3d_surface(df, response_var, output_dir, df_all_conditions=None):
    """
    Creates a 3D plot with experimental data points (with error bars) and RSM surface.

    Parameters:
    -----------
    df : DataFrame
        Conditions used to calibrate the RSM surface (selected_conditions)
    response_var : str
        Response variable ('mu_max', 'Nmax', etc.)
    output_dir : Path
        Output directory
    df_all_conditions : DataFrame, optional
        All experimental conditions (25 conditions)
        If provided, points not present in df will be displayed differently
        to show extrapolation capability
    """
    # L0 normalization for display
    L0_max_norm = 170.0

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

    # Colors for each L0
    L0_unique = sorted(df["L0"].unique())
    colors_L0 = plt.cm.winter(np.linspace(0, 1, len(L0_unique)))
    L0_to_color = {L0: colors_L0[i] for i, L0 in enumerate(L0_unique)}

    latex_label = get_latex_label(response_var)
    response_mean = f"{response_var}_mean"
    response_std = f"{response_var}_std"

    # Choose the colormap based on response_var
    if response_var == "mu_max":
        cmap_choice = "viridis"
    elif response_var == "Nmax":
        cmap_choice = "plasma"
    else:
        cmap_choice = "viridis"  # Default

    # --- Subplot 1: Experimental data with error bars (ALL data)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")

    # Use df_all_conditions if available, otherwise df
    df_to_plot = df_all_conditions if df_all_conditions is not None else df

    # Update colors for all L0 values
    L0_all_unique = sorted(df_to_plot["L0"].unique())
    colors_L0_all = plt.cm.winter(np.linspace(0, 1, len(L0_all_unique)))
    L0_to_color_all = {L0: colors_L0_all[i] for i, L0 in enumerate(L0_all_unique)}

    for L0_val in L0_all_unique:
        df_L0 = df_to_plot[df_to_plot["L0"] == L0_val]

        # Points with error bars (L0 in µmol/m²/s)
        for _, row in df_L0.iterrows():
            ax1.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                c=[L0_to_color_all[L0_val]],
                s=50,
                linewidth=2,
                alpha=0.8,
            )

            # Vertical error bars
            ax1.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row[response_mean] - row[response_std],
                    row[response_mean] + row[response_std],
                ],
                color=L0_to_color_all[L0_val],
                linewidth=2,
                alpha=0.6,
            )

    ax1.set_xlabel(r"$L_0$ (µmol m$^{-2}$ s$^{-1}$)", fontsize=11, labelpad=10)
    ax1.set_ylabel("$C_0$", fontsize=11, labelpad=10)
    ax1.set_zlabel(latex_label, fontsize=11, labelpad=10)
    ax1.set_title("Experimental data", fontsize=12, pad=20)

    # --- Subplot 2: RSM Surface
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    predict_func, coeffs, r2 = fit_response_surface(df, response_mean)

    # Extended grid over the FULL experimental domain
    if df_all_conditions is not None:
        # Use bounds from all experimental conditions
        L0_min, L0_max = df_all_conditions["L0"].min(), df_all_conditions["L0"].max()
        C0_min, C0_max = df_all_conditions["C0"].min(), df_all_conditions["C0"].max()
    else:
        # Otherwise, use bounds from calibration data
        L0_min, L0_max = df["L0"].min(), df["L0"].max()
        C0_min, C0_max = df["C0"].min(), df["C0"].max()

    L0_range = np.linspace(L0_min, L0_max, 50)
    C0_range = np.linspace(C0_min, C0_max, 50)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # Prediction
    response_grid = np.zeros_like(L0_grid)
    for i in range(L0_grid.shape[0]):
        for j in range(L0_grid.shape[1]):
            response_grid[i, j] = predict_func(L0_grid[i, j], C0_grid[i, j])

    # Surface with chosen colormap (L0 in µmol/m²/s)
    surf = ax2.plot_surface(
        L0_grid,
        C0_grid,
        response_grid,
        cmap=cmap_choice,
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Identify calibration and extrapolation points
    if df_all_conditions is not None:
        # Create a unique key to identify each condition
        df_calib_keys = set(zip(df["L0"], df["C0"]))

        # Separate calibration and extrapolation points
        df_calibration = df_all_conditions[
            df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()
        df_extrapolation = df_all_conditions[
            ~df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()

        # CALIBRATION points (black, circles) - L0 in µmol/m²/s
        for idx, row in df_calibration.iterrows():
            ax2.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                color="black",
                s=50,
                marker="o",
                linewidth=1.5,
                alpha=0.9,
                label="Calibration" if idx == df_calibration.index[0] else "",
            )
            ax2.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row[response_mean] - row[response_std],
                    row[response_mean] + row[response_std],
                ],
                color="black",
                linewidth=2,
                alpha=0.7,
            )

        # EXTRAPOLATION points (mediumslateblue, circles) - L0 in µmol/m²/s
        for idx, row in df_extrapolation.iterrows():
            ax2.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                color="mediumslateblue",
                s=100,
                marker="o",
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
                label="Extrapolation" if idx == df_extrapolation.index[0] else "",
            )
            ax2.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row[response_mean] - row[response_std],
                    row[response_mean] + row[response_std],
                ],
                color="mediumslateblue",
                linewidth=2,
                alpha=0.7,
            )

        # Add a legend
        if len(df_extrapolation) > 0:
            ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
    else:
        # Default behavior (if df_all_conditions is not provided)
        for _, row in df.iterrows():
            ax2.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                color="black",
                s=50,
                linewidth=1,
            )
            ax2.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row[response_mean] - row[response_std],
                    row[response_mean] + row[response_std],
                ],
                color="black",
                linewidth=1.5,
                alpha=0.6,
            )

    ax2.set_xlabel(r"$L_0$ (µmol m$^{-2}$ s$^{-1}$)", fontsize=11, labelpad=10)
    ax2.set_ylabel("$C_0$", fontsize=11, labelpad=10)
    ax2.set_zlabel(latex_label, fontsize=11, labelpad=10)
    ax2.set_title(f"RSM Surface\nR² = {r2:.3f}", fontsize=12, pad=20)

    # --- Subplot 3: Contour map (L0 in µmol/m²/s)
    ax3 = fig.add_subplot(gs[0, 2])
    contour = ax3.contourf(
        L0_grid, C0_grid, response_grid, levels=15, cmap=cmap_choice, alpha=0.8
    )
    contour_lines = ax3.contour(
        L0_grid,
        C0_grid,
        response_grid,
        levels=15,
        colors="black",
        alpha=0.3,
        linewidths=0.5,
    )
    ax3.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    # Points with calibration/extrapolation distinction (L0 in µmol/m²/s)
    if df_all_conditions is not None:
        # Create a unique key to identify each condition
        df_calib_keys = set(zip(df["L0"], df["C0"]))

        # Separate points
        df_calibration = df_all_conditions[
            df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()
        df_extrapolation = df_all_conditions[
            ~df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()

        # Calibration points (black, circles)
        ax3.scatter(
            df_calibration["L0"],
            df_calibration["C0"],
            s=50,
            c="black",
            marker="o",
            linewidths=2,
            alpha=0.9,
            zorder=5,
            label="Calibration",
        )

        # Extrapolation points (mediumslateblue, circles)
        if len(df_extrapolation) > 0:
            ax3.scatter(
                df_extrapolation["L0"],
                df_extrapolation["C0"],
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
    else:
        # Default behavior
        ax3.errorbar(
            df["L0"],
            df["C0"],
            xerr=0,
            yerr=0,
            fmt="o",
            color="black",
            markersize=8,
            ecolor="gray",
            elinewidth=2,
            capsize=3,
            capthick=2,
        )

    ax3.set_xlabel(r"$L_0$ (µmol m$^{-2}$ s$^{-1}$)", fontsize=11)
    ax3.set_ylabel("$C_0$", fontsize=11)
    ax3.set_title("Contour map", fontsize=12)

    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label(latex_label, rotation=270, labelpad=25, fontsize=11)

    # --- Equation (with K_L0 normalized by L0_max)
    coeffs_display = coeffs.copy()
    coeffs_display[1] = coeffs[1] / L0_max_norm  # K_L0 / L0_max
    poly_text = format_polynomial(coeffs_display, response_var)

    ax_eq = fig.add_subplot(gs[1, :])
    ax_eq.axis("off")
    ax_eq.text(
        0.5,
        0.5,
        poly_text,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, pad=0.8),
    )

    output_path = output_dir / f"RSM_3D_{response_var}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    logger.info(f"Saved 3D surface plot: {output_path}")
    logger.info(f"{response_var} : R² = {r2:.4f}")

    # =========================================================================
    # NEW 2D FIGURE: Cross-sections at constant C0 (ONLY FOR mu_max)
    # =========================================================================

    if response_var == "mu_max":
        fig2d = plt.figure(figsize=(14, 7))
        gs2d = fig2d.add_gridspec(
            1, 2, wspace=0.25, left=0.08, right=0.98, top=0.92, bottom=0.12
        )

        # Identify unique C0 values
        if df_all_conditions is not None:
            C0_unique = sorted(df_all_conditions["C0"].unique())
        else:
            C0_unique = sorted(df["C0"].unique())

        # SPECIFIC color palette for the different C0 values
        colors_C0_specific = {
            1.0000: "mediumseagreen",
            0.5000: "grey",
            0.2500: "tomato",
            0.1250: "teal",
            0.0625: "orange",
        }

        # Identify calibration and extrapolation points
        df_calib_keys = set(zip(df["L0"], df["C0"]))

        # --- Plot 1: mu_max vs L0 ---
        ax_left = fig2d.add_subplot(gs2d[0, 0])

        if df_all_conditions is not None:
            # For each C0 value
            for C0_val in C0_unique:
                color = colors_C0_specific.get(
                    C0_val, "black"
                )  # Specific color for this C0

                # Filter data for this C0
                df_C0 = df_all_conditions[df_all_conditions["C0"] == C0_val].copy()

                # Separate calibration and extrapolation
                df_C0_calib = df_C0[
                    df_C0.apply(
                        lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                    )
                ]
                df_C0_extrap = df_C0[
                    ~df_C0.apply(
                        lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                    )
                ]

                # Calibration points (CIRCLES, no edge)
                if len(df_C0_calib) > 0:
                    ax_left.errorbar(
                        df_C0_calib["L0"],
                        df_C0_calib[response_mean],
                        yerr=df_C0_calib[response_std],
                        fmt="o",
                        color=color,
                        markersize=7,
                        ecolor=color,
                        elinewidth=2,
                        capsize=4,
                        capthick=2,
                        alpha=0.8,
                        zorder=5,
                    )

                # Extrapolation points (TRIANGLES, no edge)
                if len(df_C0_extrap) > 0:
                    ax_left.errorbar(
                        df_C0_extrap["L0"],
                        df_C0_extrap[response_mean],
                        yerr=df_C0_extrap[response_std],
                        fmt="^",
                        color=color,
                        markersize=7,
                        ecolor=color,
                        elinewidth=2,
                        capsize=4,
                        capthick=2,
                        alpha=0.8,
                        zorder=5,
                    )

                # RSM prediction curve for this C0 (with specific color)
                L0_curve = np.linspace(L0_min, L0_max, 100)
                response_curve = np.array([predict_func(L0, C0_val) for L0 in L0_curve])

                ax_left.plot(
                    L0_curve,
                    response_curve,
                    "-",
                    color=color,
                    linewidth=2,
                    label=f"$C_0$ = {C0_val:.4f}",
                    alpha=0.9,
                )

        ax_left.set_xlabel(r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=14)
        ax_left.set_ylabel(latex_label, fontsize=14)

        # Create the legend with C0 curves + calibration/extrapolation indicators
        from matplotlib.lines import Line2D

        # Retrieve existing handles and labels (C0 curves)
        handles, labels = ax_left.get_legend_handles_labels()

        # Add an empty separator
        handles.append(Line2D([0], [0], color="none", label=""))
        labels.append("")

        # Add calibration/extrapolation indicators (no edge)
        calib_marker = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markersize=8,
            linestyle="none",
            label="Calibration",
        )
        extrap_marker = Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="grey",
            markersize=10,
            linestyle="none",
            label="Extrapolation",
        )

        handles.extend([calib_marker, extrap_marker])
        labels.extend(["Calibration", "Extrapolation"])

        ax_left.legend(handles, labels, fontsize=11, framealpha=0.95, loc="best")
        ax_left.grid(True, alpha=0.3)
        ax_left.tick_params(labelsize=12)

        # --- Plot 2: Nmax vs L0 ---
        ax_right = fig2d.add_subplot(gs2d[0, 1])

        other_var = "Nmax"
        other_mean = f"{other_var}_mean"
        other_std = f"{other_var}_std"
        # New label for Nmax
        other_latex = r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ × $10^7$)"

        # Fit an RSM surface for Nmax
        predict_func_other, coeffs_other, r2_other = fit_response_surface(
            df, other_mean
        )

        if df_all_conditions is not None:
            for C0_val in C0_unique:
                color = colors_C0_specific.get(
                    C0_val, "black"
                )  # Specific color for this C0

                df_C0 = df_all_conditions[df_all_conditions["C0"] == C0_val].copy()
                df_C0_calib = df_C0[
                    df_C0.apply(
                        lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                    )
                ]
                df_C0_extrap = df_C0[
                    ~df_C0.apply(
                        lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                    )
                ]

                # Calibration points (CIRCLES, no edge)
                if len(df_C0_calib) > 0:
                    ax_right.errorbar(
                        df_C0_calib["L0"],
                        df_C0_calib[other_mean],
                        yerr=df_C0_calib[other_std],
                        fmt="o",
                        color=color,
                        markersize=7,
                        ecolor=color,
                        elinewidth=2,
                        capsize=4,
                        capthick=2,
                        alpha=0.8,
                        zorder=5,
                    )

                # Extrapolation points (TRIANGLES, no edge)
                if len(df_C0_extrap) > 0:
                    ax_right.errorbar(
                        df_C0_extrap["L0"],
                        df_C0_extrap[other_mean],
                        yerr=df_C0_extrap[other_std],
                        fmt="^",
                        color=color,
                        markersize=7,
                        ecolor=color,
                        elinewidth=2,
                        capsize=4,
                        capthick=2,
                        alpha=0.8,
                        zorder=5,
                    )

                # RSM curve (with specific color) - NO LABEL to avoid legend duplication
                L0_curve = np.linspace(L0_min, L0_max, 100)
                response_curve_other = np.array(
                    [predict_func_other(L0, C0_val) for L0 in L0_curve]
                )

                ax_right.plot(
                    L0_curve,
                    response_curve_other,
                    "-",
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                )

        ax_right.set_xlabel(r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=14)
        ax_right.set_ylabel(other_latex, fontsize=14)
        # NO LEGEND on the right plot
        ax_right.grid(True, alpha=0.3)
        ax_right.tick_params(labelsize=12)
        ax_right.yaxis.get_offset_text().set_visible(False)

        # Save the 2D figure
        output_path_2d = output_dir / f"RSM_2D_cuts_{response_var}.png"
        plt.savefig(output_path_2d, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Saved 2D cuts plot: {output_path_2d}")

    return coeffs


def plot_fits_overview(df, output_dir):
    """
    Displays an overview of logistic fits for all replicates.
    Includes mu_max, Nmax and t_lag with their standard deviations in the title.
    """
    # Count the total number of conditions
    n_conditions = len(df)
    ncols = 5
    nrows = int(np.ceil(n_conditions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    colors_rep = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71"}

    for idx, row in df.iterrows():
        ax = axes[idx]

        L0 = row["L0"]
        C0 = row["C0"]

        # Plot each replicate
        for fit_data in row["replicate_fits"]:
            rep_name = fit_data["replicate"]
            time = fit_data["time"]
            biomass = fit_data["biomass"]
            fit = fit_data["fit"]

            # Experimental data
            ax.plot(
                time, biomass, "o", color=colors_rep[rep_name], markersize=5, alpha=0.6
            )

            # Fit
            ax.plot(
                fit["time_fit"],
                fit["biomass_fit"],
                "-",
                color=colors_rep[rep_name],
                linewidth=1.5,
                label=f"{rep_name}: R²={fit['r_squared']:.2f}",
            )

        # Title with ALL parameters and their standard deviations
        title_text = (
            f"L0={L0:.0f} µmol/m²/s, C0={C0:.3f}\n"
            f"$\\mu_{{max}}$={row['mu_max_mean']:.3f}±{row['mu_max_std']:.3f} h⁻¹\n"
            f"$N_{{max}}$={row['Nmax_mean']:.2e}±{row['Nmax_std']:.2e} cells/mL\n"
            f"$t_{{lag}}$={row['t_lag_mean']:.1f}±{row['t_lag_std']:.1f} h"
        )

        ax.set_title(title_text, fontsize=8)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_ylabel("N (cells/mL)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide empty axes
    for idx in range(n_conditions, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "logistic_fits_calibration.png"
    plt.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved fits overview: {output_path}")


def plot_fits_overview_rsm(df, coeffs_mu_max, coeffs_Nmax, df_params_tlag, output_dir):
    """
    Displays an overview of logistic fits with integrated RSM model predictions.
    Replicates A, B, C are plotted in red, blue, green.
    The RSM-logistic model is plotted in black (one per condition).
    The RSM fit R² is displayed in the legend.

    Args:
        df: DataFrame with parameters and replicate_fits
        coeffs_mu_max: RSM coefficients for mu_max
        coeffs_Nmax: RSM coefficients for Nmax
        df_params_tlag: DataFrame with columns ['L0', 'C0', 't_lag_mean'] for t_lag lookup
        output_dir: output directory
    """
    # Count the total number of conditions
    n_conditions = len(df)
    ncols = 5
    nrows = int(np.ceil(n_conditions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    colors_rep = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71"}

    for idx, row in df.iterrows():
        ax = axes[idx]

        L0 = row["L0"]
        C0 = row["C0"]

        # Compute parameters predicted by RSM using ModelConfig
        mu_max_rsm = ModelConfig.mu_max_model(L0, C0, coeffs_mu_max)
        Nmax_rsm = ModelConfig.Nmax_model(L0, C0, coeffs_Nmax)

        # For t_lag, use the experimental mean value of this condition
        t_lag_rsm = row["t_lag_mean"]  # Directly from the DataFrame

        # Variables to store N0 and time range
        N0_estimate = None
        t_max = 0

        # Collect all experimental data to compute the RSM R²
        all_time_exp = []
        all_biomass_exp = []

        # Plot the replicates
        for fit_data in row["replicate_fits"]:
            rep_name = fit_data["replicate"]
            time = fit_data["time"]
            biomass = fit_data["biomass"]
            fit = fit_data["fit"]

            # Extract N0 from first replicate
            if N0_estimate is None:
                N0_estimate = fit["N0"]

            # Update t_max
            t_max = max(t_max, np.nanmax(time))

            # Collect experimental data (removing NaN values)
            valid_mask = ~np.isnan(biomass) & (biomass > 0)
            all_time_exp.extend(time[valid_mask])
            all_biomass_exp.extend(biomass[valid_mask])

            # Experimental data
            ax.plot(
                time, biomass, "o", color=colors_rep[rep_name], markersize=5, alpha=0.6
            )

            # Individual fit
            ax.plot(
                fit["time_fit"],
                fit["biomass_fit"],
                "-",
                color=colors_rep[rep_name],
                linewidth=1.5,
                alpha=0.7,
                label=f"{rep_name}: R²={fit['r_squared']:.2f}",
            )

        # Generate and plot the RSM-logistic prediction
        r2_rsm = np.nan  # Default value

        if N0_estimate is not None and t_max > 0:
            # Create a smooth time vector
            time_smooth = np.linspace(0, t_max, 500)

            # Check that RSM parameters are valid
            if mu_max_rsm > 0 and Nmax_rsm > 0:
                # Compute the logistic curve with RSM parameters
                try:
                    biomass_rsm = logistic_growth(
                        time_smooth, N0_estimate, Nmax_rsm, mu_max_rsm, t_lag_rsm
                    )

                    # Compute the R² of the RSM model on experimental data
                    if len(all_time_exp) > 0 and len(all_biomass_exp) > 0:
                        all_time_exp = np.array(all_time_exp)
                        all_biomass_exp = np.array(all_biomass_exp)

                        # RSM predictions at experimental data points
                        biomass_rsm_exp = logistic_growth(
                            all_time_exp, N0_estimate, Nmax_rsm, mu_max_rsm, t_lag_rsm
                        )

                        # R² computation
                        r2_rsm = calculate_r2(all_biomass_exp, biomass_rsm_exp)

                    # Plot the RSM prediction in black with R² in the legend
                    ax.plot(
                        time_smooth,
                        biomass_rsm,
                        "-",
                        color="black",
                        linewidth=2.5,
                        alpha=0.9,
                        label=f"RSM-Logistic: R²={r2_rsm:.2f}",
                        zorder=10,
                    )

                    logger.info(
                        f"RSM curve plotted for L0={L0:.1f}, C0={C0:.3f}: "
                        f"mu_max={mu_max_rsm:.3f}, Nmax={Nmax_rsm:.2e}, t_lag={t_lag_rsm:.2f}, R²={r2_rsm:.3f}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to plot RSM curve for L0={L0:.1f}, C0={C0:.3f}: {e}"
                    )
            else:
                logger.warning(
                    f"Invalid RSM parameters for L0={L0:.1f}, C0={C0:.3f}: "
                    f"mu_max={mu_max_rsm:.3f}, Nmax={Nmax_rsm:.2e}"
                )
        else:
            logger.warning(
                f"No valid data for RSM plot: L0={L0:.1f}, C0={C0:.3f}, "
                f"N0={N0_estimate}, t_max={t_max}"
            )

        # Title with experimental parameters AND RSM predictions
        title_text = (
            f"L0={L0:.0f} µmol/m²/s, C0={C0:.3f}\n"
            f"Exp: $\\mu_{{max}}$={row['mu_max_mean']:.3f}±{row['mu_max_std']:.3f}, "
            f"$N_{{max}}$={row['Nmax_mean']:.2e}±{row['Nmax_std']:.2e}\n"
            f"RSM: $\\mu_{{max}}$={mu_max_rsm:.3f}, "
            f"$N_{{max}}$={Nmax_rsm:.2e}, "
            f"$t_{{lag}}$={t_lag_rsm:.1f}h"
        )

        ax.set_title(title_text, fontsize=7)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_ylabel("N (cells/mL)", fontsize=8)
        ax.legend(fontsize=5, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide empty axes
    for idx in range(n_conditions, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "RSM_ODE_calibration.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved RSM-logistic fits overview: {output_path}")


def evaluate_rsm_extrapolation(
    df_params_selected,
    coeffs_mu_max,
    coeffs_Nmax,
    df_params_tlag,
    files_dict,
    conv_OD_to_cell,
    output_dir,
):
    """
    Evaluates the extrapolation capability of the RSM model on all 25 conditions.
    Displays RSM predictions on experimental data and computes R² for each condition.

    Subplots corresponding to certain conditions (L0, C0) are highlighted
    with a colored background (region where the model performs well).

    Args:
        df_params_selected: DataFrame with conditions used for RSM fit
        coeffs_mu_max: RSM coefficients for mu_max
        coeffs_Nmax: RSM coefficients for Nmax
        df_params_tlag: DataFrame with columns ['L0', 'C0', 't_lag_mean'] for t_lag lookup
        files_dict: dictionary of data files
        conv_OD_to_cell: conversion factor
        output_dir: output directory
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION EVALUATION ON ALL 25 CONDITIONS")
    logger.info(f"{'=' * 80}\n")

    # *** FIX: Use df_params_tlag which contains manual adjustments ***
    # Instead of re-extracting data, we use the DataFrame passed as parameter
    df_params_all = df_params_tlag.copy()
    logger.info(
        f"Using {len(df_params_all)} conditions (with manual t_lag adjustments applied)"
    )

    # Identify conditions used for the fit
    selected_pairs = set(zip(df_params_selected["L0"], df_params_selected["C0"]))
    df_params_all["used_for_fit"] = df_params_all.apply(
        lambda row: (row["L0"], row["C0"]) in selected_pairs, axis=1
    )

    # Define the region where the model "performs well"
    highlight_pairs = {
        (170.0, 1.0000),
        (170.0, 0.5000),
        (170.0, 0.2500),
        (170.0, 0.1250),
        (170.0, 0.0625),
        (102.0, 1.0000),
        (102.0, 0.5000),
        (102.0, 0.2500),
        (51.0, 1.0000),
        (51.0, 0.5000),
        (51.0, 0.2500),
        (25.5, 1.0000),
    }

    n_fit = df_params_all["used_for_fit"].sum()
    n_extrapolation = len(df_params_all) - n_fit

    logger.info(f"Conditions used for RSM fit: {n_fit}")
    logger.info(f"Conditions for extrapolation: {n_extrapolation}")

    # R² computation for each condition
    r2_values = []
    for idx, row in df_params_all.iterrows():
        L0, C0, replicate_fits = row["L0"], row["C0"], row["replicate_fits"]

        N0_values = [
            fit["fit"]["N0"]
            for fit in replicate_fits
            if "fit" in fit and fit["fit"] is not None
        ]
        if not N0_values:
            continue
        N0_mean = np.mean(N0_values)

        r2_replicates = []
        for fit_data in replicate_fits:
            time_exp, biomass_exp = fit_data["time"], fit_data["biomass"]
            valid_mask = ~np.isnan(biomass_exp) & (biomass_exp > 0)
            time_clean, biomass_clean = time_exp[valid_mask], biomass_exp[valid_mask]
            if len(time_clean) < 4:
                continue

            # Use df_params_all (which contains all conditions) for t_lag lookup
            biomass_rsm = logistic_growth_rsm(
                time_clean, N0_mean, L0, C0, coeffs_mu_max, coeffs_Nmax, df_params_all
            )
            r2 = calculate_r2(biomass_clean, biomass_rsm)
            r2_replicates.append(r2)

        r2_values.append(
            {
                "L0": L0,
                "C0": C0,
                "R2_rsm_mean": np.mean(r2_replicates) if r2_replicates else np.nan,
                "R2_rsm_std": np.std(r2_replicates) if r2_replicates else np.nan,
                "used_for_fit": row["used_for_fit"],
            }
        )

    df_r2 = pd.DataFrame(r2_values)
    r2_path = output_dir / "rsm_extrapolation_r2.csv"
    df_r2.to_csv(r2_path, index=False, float_format="%.6f", sep=";")

    # ========================================================================
    # DISPLAY OF GLOBAL R² STATISTICS
    # ========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION - R² GLOBAL STATISTICS")
    logger.info(f"{'=' * 80}\n")

    # Extract all valid R² values (non-NaN)
    all_r2_values = df_r2["R2_rsm_mean"].values
    valid_r2 = all_r2_values[~np.isnan(all_r2_values)]

    if len(valid_r2) > 0:
        # Collect all experimental data and predictions for global R²
        all_biomass_exp = []
        all_biomass_pred = []

        for idx, row in df_params_all.iterrows():
            L0, C0, replicate_fits = row["L0"], row["C0"], row["replicate_fits"]

            N0_values = [
                fit["fit"]["N0"]
                for fit in replicate_fits
                if "fit" in fit and fit["fit"] is not None
            ]
            if not N0_values:
                continue
            N0_mean = np.mean(N0_values)

            for fit_data in replicate_fits:
                time_exp, biomass_exp = fit_data["time"], fit_data["biomass"]
                valid_mask = ~np.isnan(biomass_exp) & (biomass_exp > 0)
                time_clean, biomass_clean = (
                    time_exp[valid_mask],
                    biomass_exp[valid_mask],
                )
                if len(time_clean) < 4:
                    continue

                biomass_rsm = logistic_growth_rsm(
                    time_clean,
                    N0_mean,
                    L0,
                    C0,
                    coeffs_mu_max,
                    coeffs_Nmax,
                    df_params_all,
                )

                all_biomass_exp.extend(biomass_clean)
                all_biomass_pred.extend(biomass_rsm)

        # Global R² computation over all data points
        if len(all_biomass_exp) > 0:
            r2_global = calculate_r2(
                np.array(all_biomass_exp), np.array(all_biomass_pred)
            )
            logger.info(f"  R² global (all conditions):      {r2_global:.6f}")
        else:
            logger.info(f"  R² global (all conditions):      N/A (no valid data)")

        # R² statistics per condition
        r2_mean = np.mean(valid_r2)
        r2_std = np.std(valid_r2)
        r2_median = np.median(valid_r2)
        r2_min = np.min(valid_r2)
        r2_max = np.max(valid_r2)

        logger.info(f"  Mean R² (per condition):           {r2_mean:.6f}")
        logger.info(f"  R² standard deviation:            {r2_std:.6f}")
        logger.info(f"  Median R²:                        {r2_median:.6f}")
        logger.info(f"  Minimum R²:                       {r2_min:.6f}")
        logger.info(f"  Maximum R²:                       {r2_max:.6f}")
        logger.info(f"  Number of valid conditions:       {len(valid_r2)}")

        # Identify conditions with min and max R²
        idx_min = np.argmin(valid_r2)
        idx_max = np.argmax(valid_r2)

        # Retrieve corresponding rows in df_r2
        valid_r2_df = df_r2[~df_r2["R2_rsm_mean"].isna()]
        condition_min = valid_r2_df.iloc[idx_min]
        condition_max = valid_r2_df.iloc[idx_max]

        logger.info(f"\n  Condition with minimum R²:")
        logger.info(
            f"    L0 = {condition_min['L0']:.1f} µmol/m²/s, C0 = {condition_min['C0']:.4f}"
        )
        logger.info(
            f"    R² = {condition_min['R2_rsm_mean']:.6f} ± {condition_min['R2_rsm_std']:.6f}"
        )
        logger.info(
            f"    {'[USED FOR FIT]' if condition_min['used_for_fit'] else '[EXTRAPOLATION]'}"
        )

        logger.info(f"\n  Condition with maximum R²:")
        logger.info(
            f"    L0 = {condition_max['L0']:.1f} µmol/m²/s, C0 = {condition_max['C0']:.4f}"
        )
        logger.info(
            f"    R² = {condition_max['R2_rsm_mean']:.6f} ± {condition_max['R2_rsm_std']:.6f}"
        )
        logger.info(
            f"    {'[USED FOR FIT]' if condition_max['used_for_fit'] else '[EXTRAPOLATION]'}"
        )
    else:
        logger.warning("  No valid R² values found!")

    logger.info(f"\n{'=' * 80}\n")

    # Display t_lag values used for each condition
    logger.info(f"\n{'=' * 80}")
    logger.info("T_LAG VALUES USED FOR EACH CONDITION")
    logger.info(f"{'=' * 80}\n")
    logger.info(f"{'L0 (µmol/m²/s)':<18} {'C0':<10} {'t_lag (h)':<12} {'Status':<15}")
    logger.info("-" * 80)

    # Prepare data for CSV
    tlag_data = []
    for idx, row in df_params_all.iterrows():
        L0, C0 = row["L0"], row["C0"]
        t_lag_mean = row["t_lag_mean"]
        t_lag_std = row["t_lag_std"]
        used_for_fit = row["used_for_fit"]
        status = "FIT" if used_for_fit else "EXTRAPOLATION"

        # Check if this value was manually adjusted
        manually_adjusted = (
            (L0, C0) in manual_tlag_adjustments
            if "manual_tlag_adjustments" in locals()
            else False
        )

        logger.info(
            f"{L0:<18.1f} {C0:<10.4f} {t_lag_mean:<12.2f} {status:<15}{'[MANUAL]' if manually_adjusted else ''}"
        )

        tlag_data.append(
            {
                "L0": L0,
                "C0": C0,
                "t_lag_mean": t_lag_mean,
                "t_lag_std": t_lag_std,
                "used_for_RSM_fit": used_for_fit,
                "status": status,
                "manually_adjusted": manually_adjusted,
            }
        )

    # Save the CSV with t_lag values
    df_tlag = pd.DataFrame(tlag_data)
    tlag_path = output_dir / "t_lag_values_used.csv"
    df_tlag.to_csv(tlag_path, index=False, float_format="%.6f", sep=";")
    logger.info(f"\nT_lag values saved to: {tlag_path}")

    # Also save a txt file in simplified format (like fitting_plates.py)
    tlag_txt_path = output_dir / "t_lag_adjustments.txt"
    with open(tlag_txt_path, "w") as f:
        f.write("# t_lag values for each (L0, C0) condition\n")
        f.write("# Format: (L0 [µmol/m²/s], C0_factor): t_lag_hours\n\n")
        for _, row in df_params_all.iterrows():
            L0, C0 = row["L0"], row["C0"]
            t_lag_mean = row["t_lag_mean"]
            f.write(f"({L0:.1f}, {C0:.4f}): {t_lag_mean:.1f}\n")
    logger.info(f"T_lag adjustments saved to: {tlag_txt_path}")
    logger.info(f"{'=' * 80}\n")

    # Generation of the global figure (25 subplots)
    n_conditions = len(df_params_all)
    cols, rows = 5, 5

    # *** MODIFICATION: Sort data to have C0 ascending (from left to right) ***
    df_params_all = df_params_all.sort_values(
        ["L0", "C0"], ascending=[False, True]
    ).reset_index(drop=True)

    fig, axes = plt.subplots(rows, cols, figsize=(25, 25))
    colors_rep = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71"}

    for idx, row in df_params_all.iterrows():
        ax = axes[idx // cols, idx % cols]
        L0, C0, replicate_fits, used_for_fit = (
            row["L0"],
            row["C0"],
            row["replicate_fits"],
            row["used_for_fit"],
        )

        # Retrieve t_lag for this condition (for display)
        t_lag_mean = row["t_lag_mean"]

        # Apply a different background if the pair is in highlight_pairs
        """
        if (L0, C0) in highlight_pairs:
            ax.set_facecolor('white')
        else:
            ax.set_facecolor(mcolors.to_rgba('gainsboro', alpha=0.4))
        """

        # Apply grey background if R² < 0.9, otherwise white
        r2_row = df_r2[(df_r2["L0"] == L0) & (df_r2["C0"] == C0)].iloc[0]
        r2_rsm_value = r2_row["R2_rsm_mean"]

        if not np.isnan(r2_rsm_value) and r2_rsm_value >= 0.85:
            ax.set_facecolor("white")
        else:
            ax.set_facecolor(mcolors.to_rgba("silver", alpha=0.4))

        N0_values = [
            fit["fit"]["N0"]
            for fit in replicate_fits
            if "fit" in fit and fit["fit"] is not None
        ]
        if not N0_values:
            ax.axis("off")
            continue
        N0_mean = np.mean(N0_values)

        r2_row = df_r2[(df_r2["L0"] == L0) & (df_r2["C0"] == C0)].iloc[0]
        r2_rsm, r2_rsm_std = r2_row["R2_rsm_mean"], r2_row["R2_rsm_std"]

        # Experimental data
        t_max = 0
        for fit_data in replicate_fits:
            rep_name, time, biomass = (
                fit_data["replicate"],
                fit_data["time"],
                fit_data["biomass"],
            )
            t_max = max(t_max, np.nanmax(time))
            ax.plot(
                time, biomass, "o", color=colors_rep[rep_name], alpha=0.7, markersize=5
            )
            fit = fit_data["fit"]
            ax.plot(
                fit["time_fit"],
                fit["biomass_fit"],
                "-",
                color=colors_rep[rep_name],
                alpha=0.7,
            )

        # RSM curve
        if t_max > 0:
            time_rsm = np.linspace(0, t_max, 500)
            biomass_rsm = logistic_growth_rsm(
                time_rsm, N0_mean, L0, C0, coeffs_mu_max, coeffs_Nmax, df_params_all
            )
            color = "black" if used_for_fit else "mediumslateblue"
            ax.plot(
                time_rsm,
                biomass_rsm,
                color=color,
                linewidth=2.5,
                label="RSM (fit)" if used_for_fit else "RSM (extrap)",
                zorder=10,
            )

            # Log the t_lag value used
            if idx == 0 or idx % 5 == 0:  # Display every 5 to avoid overloading
                logger.debug(
                    f"  Condition L0={L0:.1f}, C0={C0:.4f}: using t_lag={t_lag_mean:.2f} h"
                )

        # Title
        fit_status = "fit" if used_for_fit else "extrapolation"
        ax.set_title(
            f"L0={L0:.1f}, C0={C0:.3f} [{fit_status}]\nR²(RSM)={r2_rsm:.3f}±{r2_rsm_std:.3f}",
            fontsize=14,
            fontweight="bold" if not used_for_fit else "normal",
            color="mediumslateblue" if not used_for_fit else "black",
        )
        # ax.set_ylim(0, 6e7)

        # Axes
        row_idx, col_idx = idx // cols, idx % cols
        if row_idx == rows - 1:
            ax.set_xlabel("Time (h)", fontsize=14)
        if col_idx == 0:
            ax.set_ylabel("Biomass (cells/mL)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

    plt.tight_layout()
    output_path = output_dir / "rsm_extrapolation_all_conditions_highlighted_bg.png"
    plt.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close()

    logger.info(f"Extrapolation figure saved with background highlight: {output_path}")
    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION EVALUATION COMPLETED")
    logger.info(f"{'=' * 80}\n")

    return df_r2


def evaluate_rsm_extrapolation_by_L0(
    df_params_selected,
    coeffs_mu_max,
    coeffs_Nmax,
    df_params_tlag,
    files_dict,
    conv_OD_to_cell,
    output_dir,
):
    """
    Evaluates the extrapolation capability of the RSM model with a layout grouped by L0.
    Creates a row of 5 plots, one for each L0 value.
    Each plot contains 5 curves (one per C0) with distinct colors.
    Replicates are differentiated by markers (circle, triangle, square).

    Args:
        df_params_selected: DataFrame with conditions used for RSM fit
        coeffs_mu_max: RSM coefficients for mu_max
        coeffs_Nmax: RSM coefficients for Nmax
        df_params_tlag: DataFrame with columns ['L0', 'C0', 't_lag_mean'] for t_lag lookup
        files_dict: dictionary of data files
        conv_OD_to_cell: conversion factor
        output_dir: output directory
    """
    # =========================================================================
    # Font size definition
    # =========================================================================
    FONT_TITLE = 23  # Title of each plot
    FONT_LABEL = 23  # Axis labels
    FONT_TICK = 20  # 19       # Graduations
    FONT_LEGEND = 19  # Legend
    FONT_SUPTITLE = 19  # Global title

    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION EVALUATION BY L0")
    logger.info(f"{'=' * 80}\n")

    # Use df_params_tlag which contains manual adjustments
    df_params_all = df_params_tlag.copy()
    logger.info(
        f"Using {len(df_params_all)} conditions (with manual t_lag adjustments applied)"
    )

    # Identify conditions used for the fit
    selected_pairs = set(zip(df_params_selected["L0"], df_params_selected["C0"]))
    df_params_all["used_for_fit"] = df_params_all.apply(
        lambda row: (row["L0"], row["C0"]) in selected_pairs, axis=1
    )

    # Sort data
    df_params_all = df_params_all.sort_values(
        ["L0", "C0"], ascending=[False, True]
    ).reset_index(drop=True)

    # Define colors by C0
    colors_C0 = {
        1.0000: "mediumseagreen",
        0.5000: "grey",
        0.2500: "tomato",
        0.1250: "teal",
        0.0625: "orange",
    }

    # Define markers by replicate
    markers_rep = {
        "A": "o",  # point
        "B": "^",  # triangle
        "C": "s",  # square
    }

    # L0 values
    L0_values = sorted(
        df_params_all["L0"].unique(), reverse=True
    )  # [170, 102, 51, 25.5, 11.9]

    # =========================================================================
    # COMPUTE COMMON LIMITS FOR THE Y-AXIS
    # =========================================================================
    y_min_global = float("inf")
    y_max_global = float("-inf")

    for L0 in L0_values:
        df_L0 = df_params_all[df_params_all["L0"] == L0]
        for idx_row, row in df_L0.iterrows():
            replicate_fits = row["replicate_fits"]
            for fit_data in replicate_fits:
                biomass = fit_data["biomass"]
                if len(biomass) > 0:
                    y_min_global = min(y_min_global, np.nanmin(biomass))
                    y_max_global = max(y_max_global, np.nanmax(biomass))

    # Add a 5% margin
    y_range = y_max_global - y_min_global
    y_min_plot = max(0, y_min_global - 0.05 * y_range)
    y_max_plot = y_max_global + 0.05 * y_range

    # Convert to 10^7 for display
    y_min_plot_scaled = y_min_plot / 1e7
    y_max_plot_scaled = y_max_plot / 1e7

    logger.info(f"Global Y-axis range: {y_min_plot:.2e} to {y_max_plot:.2e} cells/mL")
    logger.info(
        f"Scaled Y-axis range: {y_min_plot_scaled:.2f} to {y_max_plot_scaled:.2f} × 10^7 cells/mL"
    )

    # =========================================================================
    # CREATE THE FIGURE
    # =========================================================================
    fig, axes = plt.subplots(
        1, 5, figsize=(36, 7)
    )  # Increased from 32 to 36 for more space

    # For each L0 value, create a plot
    for idx_L0, L0 in enumerate(L0_values):
        ax = axes[idx_L0]

        # Filter data for this L0 value
        df_L0 = df_params_all[df_params_all["L0"] == L0]

        # For each C0 condition in this L0
        for idx_row, row in df_L0.iterrows():
            C0 = row["C0"]
            replicate_fits = row["replicate_fits"]
            used_for_fit = row["used_for_fit"]

            # Get the color for this C0 value
            color = colors_C0.get(C0, "black")

            # Compute mean N0
            N0_values = [
                fit["fit"]["N0"]
                for fit in replicate_fits
                if "fit" in fit and fit["fit"] is not None
            ]
            if not N0_values:
                continue
            N0_mean = np.mean(N0_values)

            # Plot experimental data for each replicate
            t_max = 0
            for fit_data in replicate_fits:
                rep_name = fit_data["replicate"]
                time = fit_data["time"]
                biomass = fit_data["biomass"]
                t_max = max(t_max, np.nanmax(time))

                # Marker based on replicate
                marker = markers_rep.get(rep_name, "o")

                # Plot experimental points (convert to 10^7)
                ax.plot(
                    time,
                    biomass / 1e7,
                    marker=marker,
                    color=color,
                    alpha=0.6,
                    markersize=9,
                    linestyle="none",
                )

            # Plot the RSM curve for this condition
            if t_max > 0:
                time_rsm = np.linspace(0, t_max, 500)
                biomass_rsm = logistic_growth_rsm(
                    time_rsm, N0_mean, L0, C0, coeffs_mu_max, coeffs_Nmax, df_params_all
                )

                # The RSM curve uses the same color as C0
                # Line style depending on whether used for fit or extrapolation
                linestyle = "-" if used_for_fit else "--"

                ax.plot(
                    time_rsm,
                    biomass_rsm / 1e7,
                    color=color,
                    linewidth=2.5,
                    linestyle=linestyle,
                    label=f"C0={C0:.4f}",
                    alpha=0.9,
                )

        # Plot configuration
        # Title with LaTeX notation and BOLD
        ax.set_title(
            rf"$L_{{0}}$ = {L0:.1f} µmol$_{{h\nu}}$ m$^{{-2}}$ s$^{{-1}}$",
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

        # Remove the offset text (1e7)
        ax.yaxis.get_offset_text().set_visible(False)

        # Hide Y-axis tick labels on all plots except the first
        if idx_L0 != 0:
            ax.set_yticklabels([])

        # ax.grid(True, alpha=0.3)
        ax.grid(False)
        ax.tick_params(labelsize=FONT_TICK)

    # =========================================================================
    # CREATE THE LEGEND
    # =========================================================================
    legend_elements = []

    # Add C0 colors
    for C0, color in sorted(colors_C0.items(), reverse=True):
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=f"$C_{{0}}$={C0:.4f}")
        )

    # Add a separator
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Add replicate markers
    for rep, marker in markers_rep.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                linestyle="none",
                markersize=8,
                label=f"Replicate {rep}",
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

    # Place the legend to the right of all plots (adjusted to avoid overlap)
    fig.legend(
        handles=legend_elements,
        loc="center right",
        fontsize=FONT_LEGEND,
        framealpha=0.9,
        bbox_to_anchor=(0.985, 0.5),
    )

    # Global title
    # fig.suptitle(r'RSM Model Evaluation: Grouped by $L_{0}$ (Light Intensity)', fontsize=FONT_SUPTITLE, fontweight='bold', y=1.02)

    # Adjust spacing to make room for the legend (reduces plot space)
    plt.tight_layout(rect=[0, 0, 0.88, 0.98])  # Reduced from 0.92 to 0.88

    output_path = output_dir / "rsm_extrapolation_by_L0.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"RSM extrapolation by L0 figure saved: {output_path}")
    logger.info(f"\n{'=' * 80}")
    logger.info("RSM EXTRAPOLATION BY L0 COMPLETED")
    logger.info(f"{'=' * 80}\n")


def plot_r2_heatmap_grid(
    df_params_all, coeffs_mu_max, coeffs_Nmax, df_params_tlag, output_dir
):
    """
    Generates a 5x5 heatmap (25 conditions) where each cell is divided into 4 sub-squares:
    - Top-left corner: R² replicate A
    - Top-right corner: R² replicate B
    - Bottom-left corner: R² replicate C
    - Bottom-right corner: R² RSM-ODE hybrid model

    Args:
        df_params_all: DataFrame with all 25 conditions
        coeffs_mu_max, coeffs_Nmax: RSM coefficients
        df_params_tlag: DataFrame with columns ['L0', 'C0', 't_lag_mean'] for t_lag lookup
        output_dir: output directory
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATING R² HEATMAP GRID (5x5 with 4 sub-squares each)")
    logger.info(f"{'=' * 80}\n")

    # Sort conditions by L0 (descending) and C0 (descending)
    df_sorted = df_params_all.sort_values(
        ["L0", "C0"], ascending=[False, False]
    ).reset_index(drop=True)

    # Create unique L0 and C0 values for the axes
    L0_values = sorted(
        df_sorted["L0"].unique(), reverse=True
    )  # [170, 102, 51, 25.5, 11.9]
    C0_values = sorted(
        df_sorted["C0"].unique(), reverse=False
    )  # [1.0, 0.5, 0.25, 0.125, 0.0625]

    # Create a 3D matrix: (5_L0 x 5_C0 x 4_sub_squares)
    # The 4 sub-squares are: [A, B, C, RSM]
    r2_matrix = np.full((len(L0_values), len(C0_values), 4), np.nan)

    # Fill the matrix with R² values
    for idx, row in df_sorted.iterrows():
        L0 = row["L0"]
        C0 = row["C0"]
        replicate_fits = row["replicate_fits"]

        # Find indices in the matrix
        i = L0_values.index(L0)
        j = C0_values.index(C0)

        # Extract mean N0
        N0_values = [
            fit["fit"]["N0"]
            for fit in replicate_fits
            if "fit" in fit and fit["fit"] is not None
        ]
        if not N0_values:
            continue
        N0_mean = np.mean(N0_values)

        # Compute R² for each replicate (A=0, B=1, C=2)
        rep_map = {"A": 0, "B": 1, "C": 2}
        for fit_data in replicate_fits:
            rep_name = fit_data["replicate"]
            if rep_name not in rep_map:
                continue

            # R² of the individual replicate fit
            if "fit" in fit_data and fit_data["fit"] is not None:
                r2_individual = fit_data["fit"]["r_squared"]
                r2_matrix[i, j, rep_map[rep_name]] = r2_individual

        # Compute R² of the RSM-ODE hybrid model (average over 3 replicates)
        r2_rsm_list = []
        for fit_data in replicate_fits:
            time_exp = fit_data["time"]
            biomass_exp = fit_data["biomass"]

            valid_mask = ~np.isnan(biomass_exp) & (biomass_exp > 0)
            time_clean = time_exp[valid_mask]
            biomass_clean = biomass_exp[valid_mask]

            if len(time_clean) < 4:
                continue

            # RSM prediction (uses df_params_all for t_lag lookup)
            biomass_rsm = logistic_growth_rsm(
                time_clean, N0_mean, L0, C0, coeffs_mu_max, coeffs_Nmax, df_params_all
            )

            r2 = calculate_r2(biomass_clean, biomass_rsm)
            r2_rsm_list.append(r2)

        if r2_rsm_list:
            r2_matrix[i, j, 3] = np.mean(r2_rsm_list)  # Indice 3 = RSM

    # Create the figure
    fig, ax = plt.subplots(figsize=(18, 16))

    # Create a colormap for R² values (0 = red, 1 = green)
    from matplotlib.colors import LinearSegmentedColormap

    # colors_list = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
    # cmap = LinearSegmentedColormap.from_list('r2_cmap', colors_list, N=256)
    cmap = plt.cm.inferno

    # Draw each cell with its 4 sub-squares
    cell_size = 1.0
    for i, L0 in enumerate(L0_values):
        for j, C0 in enumerate(C0_values):
            # Base position of cell (i, j)
            x_base = j * cell_size
            y_base = (
                len(L0_values) - 1 - i
            ) * cell_size  # Invert Y to have L0 max at the top

            # The 4 sub-squares: [A (TL), B (TR), C (BL), RSM (BR)]
            # TL = Top Left, TR = Top Right, BL = Bottom Left, BR = Bottom Right
            sub_positions = [
                (
                    x_base,
                    y_base + cell_size / 2,
                    cell_size / 2,
                    cell_size / 2,
                ),  # A: top-left
                (
                    x_base + cell_size / 2,
                    y_base + cell_size / 2,
                    cell_size / 2,
                    cell_size / 2,
                ),  # B: top-right
                (x_base, y_base, cell_size / 2, cell_size / 2),  # C: bottom-left
                (
                    x_base + cell_size / 2,
                    y_base,
                    cell_size / 2,
                    cell_size / 2,
                ),  # RSM: bottom-right
            ]

            sub_labels = ["A", "B", "C", "RSM-ODE"]

            for k, (x, y, w, h) in enumerate(sub_positions):
                r2_val = r2_matrix[i, j, k]

                if not np.isnan(r2_val):
                    color = cmap(r2_val)
                    rect = plt.Rectangle(
                        (x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.5
                    )
                    ax.add_patch(rect)

                    # Add text with the R² value
                    text_x = x + w / 2
                    text_y = y + h / 2

                    # Text color: black if R² > 0.5, white otherwise
                    text_color = "black" if r2_val > 0.5 else "white"

                    # Label + valeur
                    ax.text(
                        text_x,
                        text_y + 0.08,
                        sub_labels[k],
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=text_color,
                    )
                    ax.text(
                        text_x,
                        text_y - 0.08,
                        f"{r2_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                        fontweight="bold",
                    )
                else:
                    # Case where R² is not available (grey)
                    rect = plt.Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor="lightgray",
                        edgecolor="white",
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x + w / 2,
                        y + h / 2,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="darkgray",
                    )

            # Add separation lines between main cells
            rect_main = plt.Rectangle(
                (x_base, y_base),
                cell_size,
                cell_size,
                fill=False,
                edgecolor="white",
                linewidth=4,
            )
            ax.add_patch(rect_main)

    # Axes configuration
    ax.set_xlim(0, len(C0_values) * cell_size)
    ax.set_ylim(0, len(L0_values) * cell_size)
    ax.set_aspect("equal")

    # Axis labels
    ax.set_xticks([j * cell_size + cell_size / 2 for j in range(len(C0_values))])
    ax.set_xticklabels([f"{c:.4f}" for c in C0_values], fontsize=11, fontweight="bold")
    ax.set_xlabel("C₀ (dilution factor)", fontsize=14, fontweight="bold")

    ax.set_yticks([i * cell_size + cell_size / 2 for i in range(len(L0_values))])
    ax.set_yticklabels(
        [f"{L:.1f}" for L in reversed(L0_values)], fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("L₀ (µmol/m²/s)", fontsize=14, fontweight="bold")

    # Title
    ax.set_title(
        "R² Heatmap Grid: Individual Replicates (A, B, C) and RSM-ODE Hybrid Model\n"
        + "Each cell divided into 4 sub-squares: A (top-left), B (top-right), C (bottom-left), RSM (bottom-right)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=40)
    cbar.set_label("R² Value", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    # Save
    plt.tight_layout()
    output_path = output_dir / "r2_heatmap_grid_5x5.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved R² heatmap grid: {output_path}")

    # Compute and display statistics
    logger.info(f"\n{'=' * 80}")
    logger.info("R² STATISTICS FROM HEATMAP")
    logger.info(f"{'=' * 80}")

    for k, label in enumerate(
        ["Replicate A", "Replicate B", "Replicate C", "RSM-ODE Model"]
    ):
        r2_values = r2_matrix[:, :, k].flatten()
        r2_valid = r2_values[~np.isnan(r2_values)]

        if len(r2_valid) > 0:
            logger.info(f"\n{label}:")
            logger.info(f"  Mean R²: {np.mean(r2_valid):.4f} ± {np.std(r2_valid):.4f}")
            logger.info(f"  Min R²:  {np.min(r2_valid):.4f}")
            logger.info(f"  Max R²:  {np.max(r2_valid):.4f}")
            logger.info(f"  Median:  {np.median(r2_valid):.4f}")
            logger.info(f"  N:       {len(r2_valid)}")

    logger.info(f"\n{'=' * 80}\n")


def create_full_rsm_panel(df, output_dir, df_all_conditions=None):
    """
    Creates a complete panel of 8 plots for Nature Communications:
    - Row 1: 3D surfaces of mu_max and Nmax
    - Row 2: 2D cross-sections of mu_max and Nmax vs L0
    - Row 3: 4 sensitivity plots

    Parameters:
    -----------
    df : DataFrame
        Conditions used to calibrate the RSM surface (selected_conditions)
    output_dir : Path
        Output directory
    df_all_conditions : DataFrame, optional
        All experimental conditions (25 conditions)
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    # =========================================================================
    # GLOBAL FONT SIZE VARIABLES
    # =========================================================================
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

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    # Fit RSM surfaces for mu_max and Nmax
    predict_func_mu, coeffs_mu, r2_mu = fit_response_surface(df, "mu_max_mean")
    predict_func_N, coeffs_N, r2_N = fit_response_surface(df, "Nmax_mean")

    # Experimental domain
    if df_all_conditions is not None:
        L0_min, L0_max = df_all_conditions["L0"].min(), df_all_conditions["L0"].max()
        C0_min, C0_max = df_all_conditions["C0"].min(), df_all_conditions["C0"].max()
    else:
        L0_min, L0_max = df["L0"].min(), df["L0"].max()
        C0_min, C0_max = df["C0"].min(), df["C0"].max()

    # Grids for surfaces
    L0_range = np.linspace(L0_min, L0_max, 50)
    C0_range = np.linspace(C0_min, C0_max, 50)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # Predictions
    mu_grid = np.zeros_like(L0_grid)
    N_grid = np.zeros_like(L0_grid)
    for i in range(L0_grid.shape[0]):
        for j in range(L0_grid.shape[1]):
            mu_grid[i, j] = predict_func_mu(L0_grid[i, j], C0_grid[i, j])
            N_grid[i, j] = predict_func_N(L0_grid[i, j], C0_grid[i, j])

    # Calibration/extrapolation identification
    df_calib_keys = set(zip(df["L0"], df["C0"]))

    # Calculate global R² (over all conditions including extrapolation)
    if df_all_conditions is not None:
        # mu_max global R²
        mu_pred_all = np.array(
            [
                predict_func_mu(row["L0"], row["C0"])
                for _, row in df_all_conditions.iterrows()
            ]
        )
        mu_obs_all = df_all_conditions["mu_max_mean"].values
        ss_res_mu = np.sum((mu_obs_all - mu_pred_all) ** 2)
        ss_tot_mu = np.sum((mu_obs_all - np.mean(mu_obs_all)) ** 2)
        r2_mu_global = 1 - (ss_res_mu / ss_tot_mu) if ss_tot_mu > 0 else r2_mu

        # Nmax global R²
        N_pred_all = np.array(
            [
                predict_func_N(row["L0"], row["C0"])
                for _, row in df_all_conditions.iterrows()
            ]
        )
        N_obs_all = df_all_conditions["Nmax_mean"].values
        ss_res_N = np.sum((N_obs_all - N_pred_all) ** 2)
        ss_tot_N = np.sum((N_obs_all - np.mean(N_obs_all)) ** 2)
        r2_N_global = 1 - (ss_res_N / ss_tot_N) if ss_tot_N > 0 else r2_N
    else:
        r2_mu_global = r2_mu
        r2_N_global = r2_N

    # Specific colors for C0
    colors_C0_specific = {
        1.0000: "mediumseagreen",
        0.5000: "grey",
        0.2500: "tomato",
        0.1250: "teal",
        0.0625: "orange",
    }

    C0_unique = (
        sorted(df_all_conditions["C0"].unique())
        if df_all_conditions is not None
        else sorted(df["C0"].unique())
    )

    # =========================================================================
    # FIGURE CREATION
    # =========================================================================

    fig = plt.figure(figsize=(18, 18))

    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        3,
        4,
        height_ratios=[1.3, 0.9, 1.0],  # Row 3 slightly taller
        hspace=0.40,
        wspace=0.30,
        left=0.06,
        right=0.94,
        top=0.97,
        bottom=0.08,
    )  # More space at bottom for colorbars

    # =========================================================================
    # ROW 1: 3D SURFACES
    # =========================================================================

    # --- Surface 3D mu_max ---
    ax_3d_mu = fig.add_subplot(gs[0, 0:2], projection="3d")

    surf_mu = ax_3d_mu.plot_surface(
        L0_grid,
        C0_grid,
        mu_grid,
        cmap="viridis",
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Calibration/extrapolation points for mu_max
    if df_all_conditions is not None:
        df_calibration = df_all_conditions[
            df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()
        df_extrapolation = df_all_conditions[
            ~df_all_conditions.apply(
                lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
            )
        ].copy()

        for idx, row in df_calibration.iterrows():
            ax_3d_mu.scatter(
                row["L0"],
                row["C0"],
                row["mu_max_mean"],
                color="black",
                s=50,
                marker="o",
                alpha=0.9,
            )
            # Vertical error bars
            ax_3d_mu.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row["mu_max_mean"] - row["mu_max_std"],
                    row["mu_max_mean"] + row["mu_max_std"],
                ],
                color="black",
                linewidth=1.5,
                alpha=0.6,
            )

        for idx, row in df_extrapolation.iterrows():
            ax_3d_mu.scatter(
                row["L0"],
                row["C0"],
                row["mu_max_mean"],
                color="mediumslateblue",
                s=100,
                marker="o",
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            )
            # Vertical error bars
            ax_3d_mu.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row["mu_max_mean"] - row["mu_max_std"],
                    row["mu_max_mean"] + row["mu_max_std"],
                ],
                color="mediumslateblue",
                linewidth=1.5,
                alpha=0.6,
            )

    ax_3d_mu.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_mu.set_ylabel("$C_0$", fontsize=FONT_LABEL_3D, labelpad=10)
    ax_3d_mu.set_zlabel(
        r"$\mu_{\mathrm{max}}$ (h$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_mu.set_title(
        f"R² = {r2_mu:.3f} (calibration), R² = {r2_mu_global:.3f} (global)",
        fontsize=FONT_TITLE_3D,
        pad=10,
    )
    ax_3d_mu.tick_params(labelsize=FONT_TICK_3D)

    # Colorbar for mu_max
    cbar_mu = plt.colorbar(surf_mu, ax=ax_3d_mu, shrink=0.5, aspect=10, pad=0.1)
    cbar_mu.set_label(
        r"$\mu_{\mathrm{max}}$ (h$^{-1}$)",
        fontsize=FONT_COLORBAR,
        rotation=270,
        labelpad=20,
    )
    cbar_mu.ax.tick_params(labelsize=FONT_TICK_3D)

    # --- 3D Surface Nmax ---
    ax_3d_N = fig.add_subplot(gs[0, 2:4], projection="3d")

    surf_N = ax_3d_N.plot_surface(
        L0_grid,
        C0_grid,
        N_grid,
        cmap="plasma",
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Calibration/extrapolation points for Nmax
    if df_all_conditions is not None:
        for idx, row in df_calibration.iterrows():
            ax_3d_N.scatter(
                row["L0"],
                row["C0"],
                row["Nmax_mean"],
                color="black",
                s=50,
                marker="o",
                alpha=0.9,
            )
            # Vertical error bars
            ax_3d_N.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row["Nmax_mean"] - row["Nmax_std"],
                    row["Nmax_mean"] + row["Nmax_std"],
                ],
                color="black",
                linewidth=1.5,
                alpha=0.6,
            )

        for idx, row in df_extrapolation.iterrows():
            ax_3d_N.scatter(
                row["L0"],
                row["C0"],
                row["Nmax_mean"],
                color="mediumslateblue",
                s=100,
                marker="o",
                edgecolor="white",
                linewidth=1.5,
                alpha=0.9,
            )
            # Vertical error bars
            ax_3d_N.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [
                    row["Nmax_mean"] - row["Nmax_std"],
                    row["Nmax_mean"] + row["Nmax_std"],
                ],
                color="mediumslateblue",
                linewidth=1.5,
                alpha=0.6,
            )

    ax_3d_N.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_3D, labelpad=10
    )
    ax_3d_N.set_ylabel("$C_0$", fontsize=FONT_LABEL_3D, labelpad=10)
    ax_3d_N.set_zlabel(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ × $10^7$)",
        fontsize=FONT_LABEL_3D,
        labelpad=10,
    )
    ax_3d_N.set_title(
        f"R² = {r2_N:.3f} (calibration), R² = {r2_N_global:.3f} (global)",
        fontsize=FONT_TITLE_3D,
        pad=10,
    )
    ax_3d_N.tick_params(labelsize=FONT_TICK_3D)
    ax_3d_N.zaxis.get_offset_text().set_visible(False)

    # ADD THESE LINES to reduce the number of tick marks
    ax_3d_mu.xaxis.set_major_locator(MaxNLocator(6))  # Max 5 graduations on X
    ax_3d_mu.yaxis.set_major_locator(MaxNLocator(6))  # Max 4 graduations on Y
    ax_3d_mu.zaxis.set_major_locator(MaxNLocator(6))  # Max 5 graduations on Z

    # Colorbar for Nmax
    cbar_N = plt.colorbar(surf_N, ax=ax_3d_N, shrink=0.5, aspect=10, pad=0.1)
    cbar_N.set_label(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ × $10^7$)",
        fontsize=FONT_COLORBAR,
        rotation=270,
        labelpad=20,
    )
    cbar_N.ax.tick_params(labelsize=FONT_TICK_3D)
    cbar_N.ax.yaxis.get_offset_text().set_visible(False)

    ax_3d_N.xaxis.set_major_locator(MaxNLocator(6))
    ax_3d_N.yaxis.set_major_locator(MaxNLocator(6))
    ax_3d_N.zaxis.set_major_locator(MaxNLocator(6))

    # =========================================================================
    # ROW 2: 2D CROSS-SECTIONS
    # =========================================================================

    # --- mu_max vs L0 ---
    ax_2d_mu = fig.add_subplot(gs[1, 0:2])

    if df_all_conditions is not None:
        for C0_val in C0_unique:
            color = colors_C0_specific.get(C0_val, "black")

            df_C0 = df_all_conditions[df_all_conditions["C0"] == C0_val].copy()
            df_C0_calib = df_C0[
                df_C0.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
            ]
            df_C0_extrap = df_C0[
                ~df_C0.apply(
                    lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                )
            ]

            # Calibration points
            if len(df_C0_calib) > 0:
                ax_2d_mu.errorbar(
                    df_C0_calib["L0"],
                    df_C0_calib["mu_max_mean"],
                    yerr=df_C0_calib["mu_max_std"],
                    fmt="o",
                    color=color,
                    markersize=7,
                    ecolor=color,
                    elinewidth=2,
                    capsize=4,
                    capthick=2,
                    alpha=0.8,
                    zorder=5,
                )

            # Extrapolation points
            if len(df_C0_extrap) > 0:
                ax_2d_mu.errorbar(
                    df_C0_extrap["L0"],
                    df_C0_extrap["mu_max_mean"],
                    yerr=df_C0_extrap["mu_max_std"],
                    fmt="^",
                    color=color,
                    markersize=7,
                    ecolor=color,
                    elinewidth=2,
                    capsize=4,
                    capthick=2,
                    alpha=0.8,
                    zorder=5,
                )

            # RSM curve
            L0_curve = np.linspace(L0_min, L0_max, 100)
            mu_curve = np.array([predict_func_mu(L0, C0_val) for L0 in L0_curve])
            ax_2d_mu.plot(
                L0_curve,
                mu_curve,
                "-",
                color=color,
                linewidth=2,
                label=f"$C_0$ = {C0_val:.4f}",
                alpha=0.9,
            )

    ax_2d_mu.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_2D
    )
    ax_2d_mu.set_ylabel(r"$\mu_{\mathrm{max}}$ (h$^{-1}$)", fontsize=FONT_LABEL_2D)

    # ========================================================================
    # TWO SEPARATE LEGENDS
    # ========================================================================

    # LEGEND 1: C0 values (top left)
    handles_C0, labels_C0 = ax_2d_mu.get_legend_handles_labels()
    legend1 = ax_2d_mu.legend(
        handles_C0,
        labels_C0,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        loc="upper left",
        title="$C_0$ values",
        title_fontsize=FONT_LEGEND,
        frameon=False,
    )

    # Add the first legend to the axis (otherwise it will be replaced by the second)
    ax_2d_mu.add_artist(legend1)

    # LEGEND 2: Calibration/Extrapolation (bottom right)
    calib_marker = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="grey",
        markersize=8,
        linestyle="none",
        label="Calibration",
    )
    extrap_marker = Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="grey",
        markersize=10,
        linestyle="none",
        label="Extrapolation",
    )

    legend2 = ax_2d_mu.legend(
        handles=[calib_marker, extrap_marker],
        labels=["Calibration", "Extrapolation"],
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        loc="lower right",
        frameon=False,
    )

    # ax_2d_mu.grid(True, alpha=0.3)
    ax_2d_mu.grid(False)
    ax_2d_mu.tick_params(labelsize=FONT_TICK_2D)

    # --- Nmax vs L0 ---
    ax_2d_N = fig.add_subplot(gs[1, 2:4])

    if df_all_conditions is not None:
        for C0_val in C0_unique:
            color = colors_C0_specific.get(C0_val, "black")

            df_C0 = df_all_conditions[df_all_conditions["C0"] == C0_val].copy()
            df_C0_calib = df_C0[
                df_C0.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
            ]
            df_C0_extrap = df_C0[
                ~df_C0.apply(
                    lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1
                )
            ]

            # Calibration points
            if len(df_C0_calib) > 0:
                ax_2d_N.errorbar(
                    df_C0_calib["L0"],
                    df_C0_calib["Nmax_mean"],
                    yerr=df_C0_calib["Nmax_std"],
                    fmt="o",
                    color=color,
                    markersize=7,
                    ecolor=color,
                    elinewidth=2,
                    capsize=4,
                    capthick=2,
                    alpha=0.8,
                    zorder=5,
                )

            # Extrapolation points
            if len(df_C0_extrap) > 0:
                ax_2d_N.errorbar(
                    df_C0_extrap["L0"],
                    df_C0_extrap["Nmax_mean"],
                    yerr=df_C0_extrap["Nmax_std"],
                    fmt="^",
                    color=color,
                    markersize=7,
                    ecolor=color,
                    elinewidth=2,
                    capsize=4,
                    capthick=2,
                    alpha=0.8,
                    zorder=5,
                )

            # RSM curve
            L0_curve = np.linspace(L0_min, L0_max, 100)
            N_curve = np.array([predict_func_N(L0, C0_val) for L0 in L0_curve])
            ax_2d_N.plot(L0_curve, N_curve, "-", color=color, linewidth=2, alpha=0.9)

    ax_2d_N.set_xlabel(
        r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_2D
    )
    ax_2d_N.set_ylabel(
        r"$N_{\mathrm{max}}$ (cells mL$^{-1}$ × $10^7$)", fontsize=FONT_LABEL_2D
    )
    # ax_2d_N.set_title('N$_{max}$ vs L$_0$', fontsize=FONT_TITLE_2D, fontweight='bold')
    # ax_2d_N.grid(True, alpha=0.3)
    ax_2d_N.grid(False)
    ax_2d_N.tick_params(labelsize=FONT_TICK_2D)
    ax_2d_N.yaxis.get_offset_text().set_visible(False)

    # =========================================================================
    # ROW 3: SENSITIVITIES
    # =========================================================================

    # Derivative computation
    L0_sens = np.linspace(L0_min, L0_max, 100)
    C0_sens = np.linspace(C0_min, C0_max, 100)
    L0_mesh, C0_mesh = np.meshgrid(L0_sens, C0_sens)

    # Derivatives for mu_max and Nmax (from estimated coefficients, L0 normalized)
    # coeffs_mu = [β₀, K_L0, K_C0], coeffs_N = [β₀, K_L0, K_C0]
    L0_max_norm = 170.0
    beta0_mu, KL_mu, KC_mu = coeffs_mu[0], coeffs_mu[1], coeffs_mu[2]
    beta0_N, KL_N, KC_N = coeffs_N[0], coeffs_N[1], coeffs_N[2]

    L0_mesh_norm = L0_mesh / L0_max_norm

    dmu_dL0 = (
        beta0_mu
        * (KL_mu / L0_max_norm / (L0_mesh_norm + KL_mu / L0_max_norm) ** 2)
        * (C0_mesh / (C0_mesh + KC_mu))
    )
    dmu_dC0 = (
        beta0_mu
        * (L0_mesh_norm / (L0_mesh_norm + KL_mu / L0_max_norm))
        * (KC_mu / (C0_mesh + KC_mu) ** 2)
    )
    dN_dL0 = (
        beta0_N
        * (KL_N / L0_max_norm / (L0_mesh_norm + KL_N / L0_max_norm) ** 2)
        * (C0_mesh / (C0_mesh + KC_N))
    )
    dN_dC0 = (
        beta0_N
        * (L0_mesh_norm / (L0_mesh_norm + KL_N / L0_max_norm))
        * (KC_N / (C0_mesh + KC_N) ** 2)
    )

    # Common scales
    vmin_mu = min(dmu_dL0.min(), dmu_dC0.min())
    vmax_mu = max(dmu_dL0.max(), dmu_dC0.max())
    vmin_N = min(dN_dL0.min(), dN_dC0.min())
    vmax_N = max(dN_dL0.max(), dN_dC0.max())

    # C0 values for ticks
    c0_values = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
    c0_labels = ["1", "1/2", "1/4", "1/8", "1/16"]

    sensitivity_data = [
        (
            dmu_dL0,
            r"$\partial\mu_{\mathrm{max}}/\partial L_0$",
            vmin_mu,
            vmax_mu,
            "YlOrBr",
            "black",
        ),
        (
            dmu_dC0,
            r"$\partial\mu_{\mathrm{max}}/\partial C_0$",
            vmin_mu,
            vmax_mu,
            "YlOrBr",
            "black",
        ),
        (
            dN_dL0,
            r"$\partial N_{\mathrm{max}}/\partial L_0$",
            vmin_N,
            vmax_N,
            "YlGnBu",
            "black",
        ),
        (
            dN_dC0,
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

        # Filled contours
        cs = ax.contourf(L0_sens, C0_sens, field, levels=40, cmap=cmap, norm=norm)

        # Contour lines
        field_min = field.min()
        field_max = field.max()
        cont_levels = np.logspace(np.log10(field_min), np.log10(field_max), 6)[1:-1]

        cont = ax.contour(
            L0_sens,
            C0_sens,
            field,
            levels=cont_levels,
            colors=contour_color,
            linewidths=1.5,
            alpha=0.7,
        )

        labels = ax.clabel(
            cont,
            inline=True,
            fontsize=FONT_CONTOUR,
            fmt="%.1e",
            inline_spacing=15,
            colors=contour_color,
        )

        # Background for labels
        for label in labels:
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

        ax.set_title(title, fontsize=FONT_TITLE_SENS, fontweight="bold", pad=8)
        ax.set_yscale("log")
        ax.set_yticks(c0_values)
        ax.set_yticklabels(c0_labels)
        ax.set_xlim([L0_sens.min(), L0_sens.max()])
        ax.set_ylim([C0_sens.min(), C0_sens.max()])
        ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5, color="gray")

        ax.set_xlabel(
            r"$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)", fontsize=FONT_LABEL_SENS
        )
        if idx == 0:
            ax.set_ylabel(r"$C_0$", fontsize=FONT_LABEL_SENS)

        ax.tick_params(labelsize=FONT_TICK_SENS)

    # Colorbars for sensitivities (horizontal at bottom, no overlap)
    # Colorbar for mu_max (panels 0 and 1) - left
    cbar_ax1 = fig.add_axes([0.08, 0.02, 0.38, 0.015])  # [left, bottom, width, height]
    norm_mu_sens = LogNorm(vmin=vmin_mu, vmax=vmax_mu)
    sm_mu = plt.cm.ScalarMappable(norm=norm_mu_sens, cmap="YlOrBr")
    sm_mu.set_array([])
    cbar1 = plt.colorbar(sm_mu, cax=cbar_ax1, orientation="horizontal", format="%.0e")
    cbar1.set_label(
        r"$\mu_{\mathrm{max}}$ sensitivity (h$^{-1}$)", fontsize=FONT_COLORBAR
    )
    cbar1.ax.tick_params(labelsize=FONT_TICK_SENS)

    # Colorbar for N_max (panels 2 and 3) - right
    cbar_ax2 = fig.add_axes([0.54, 0.02, 0.38, 0.015])  # [left, bottom, width, height]
    norm_N_sens = LogNorm(vmin=vmin_N, vmax=vmax_N)
    sm_N = plt.cm.ScalarMappable(norm=norm_N_sens, cmap="YlGnBu")
    sm_N.set_array([])
    cbar2 = plt.colorbar(sm_N, cax=cbar_ax2, orientation="horizontal", format="%.0e")
    cbar2.set_label(
        r"$N_{\mathrm{max}}$ sensitivity (cells mL$^{-1}$)", fontsize=FONT_COLORBAR
    )
    cbar2.ax.tick_params(labelsize=FONT_TICK_SENS)

    # =========================================================================
    # SAVE
    # =========================================================================

    output_path = output_dir / "RSM_full_panel.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()

    logger.info(f"Saved full RSM panel: {output_path}")
    logger.info(f"mu_max R² = {r2_mu:.4f}, Nmax R² = {r2_N:.4f}")

    return fig


def main():
    """
    Main function with the option to filter conditions.

    IMPORTANT CONFIGURATION:
    - To analyze ALL conditions: selected_conditions = None
    - To filter certain conditions: selected_conditions = [(L0_1, C0_1), (L0_2, C0_2), ...]

    Examples of available (L0, C0) pairs:
    - L0 can be: 11.9, 25.5, 51.0, 102.0, 170.0 (µmol/m²/s)
    - C0 can be: 1.0, 0.5, 0.25, 0.125, 0.0625 (dilution factor)
    """

    # ==================================================================================
    # FILTERING CONFIGURATION - MODIFY HERE TO SELECT CONDITIONS
    # ==================================================================================

    # Option 1: Analyze ALL conditions (default)
    # selected_conditions = None

    # Option 2: same configuration used for logistic + 2nd order polynomial
    selected_conditions = [
        (170.0, 1.0000),
        (170.0, 0.5000),
        # (170.0, 0.2500), # removing to validate (170.0, 0.2500),
        (170.0, 0.125),
        ##(170.0, 0.0625), # can be used instead of (170, 0.125) to estimate the model
        (102.0, 1.0000),
        # (102.0, 0.5000), # removing to validate (102.0, 0.5000),
        (102.0, 0.2500),
        # (51.0, 1.0000),  # removing to validate (51.0, 1.0000),
        # (51.0, 0.5000),
        (51.0, 0.2500),
        (25.5, 1),
    ]

    # ==================================================================================
    # MANUAL ADJUSTMENT OF T_LAG VALUES (optional)
    # ==================================================================================
    # If some automatically estimated t_lag values are not satisfactory,
    # you can manually correct them here.
    # Format: {(L0, C0): new_t_lag_value}
    # Example: {(102.0, 0.5): 15.0, (51.0, 1.0): 20.0}

    manual_tlag_adjustments = {
        # (170, 1): 25,
        (170, 0.250): 20,
        (102, 1): 15,
        (102.0, 0.5): 15.0,
        (51.0, 1.0): 15.0,
        (51.0, 0.5): 10.0,
        (51.0, 0.25): 5,
        (51.0, 0.125): 0,
        (51.0, 0.0625): -5,
        (25.5, 0.250): 0,
        (25.5, 0.125): -20,
        (25.5, 0.0625): 0,
        (11.9, 1): 0,
        (11.9, 0.5): 5,
        (11.9, 0.250): -20,
        (11.9, 0.125): -10,
        (11.9, 0.0625): -10,
    }

    # ==================================================================================

    conv_OD_to_cell = 4.77e6  # 4.46e6

    # Dictionary: L0_factor -> filepath
    files_dict = {
        1.000: "all_data/data_exp_Chlamy_07-07-25.csv",
        0.6: "all_data/data_exp_Chlamy_01-07-25.csv",
        0.3: "all_data/data_exp_Chlamy_17-02-25.csv",
        0.15: "all_data/data_exp_Chlamy_04-11-24.csv",
        0.07: "all_data/data_exp_Chlamy_16-09-24.csv",
    }

    output_dir = pathlib.Path("results_erlen")
    output_dir.mkdir(exist_ok=True)

    logger.info(f"\n{'=' * 80}")
    logger.info("RESPONSE SURFACE METHODOLOGY - ERLEN DATA WITH REPLICATES")
    if selected_conditions is not None:
        logger.info(
            f"MODE: FILTERED ANALYSIS ({len(selected_conditions)} conditions selected)"
        )
    else:
        logger.info("MODE: FULL ANALYSIS (all conditions)")
    logger.info(f"{'=' * 80}\n")

    logger.info("Extracting growth parameters from logistic fits with replicates...")

    # STEP 1: Extract ALL 25 conditions (to have all t_lag values)
    logger.info("\n1. Extracting ALL 25 conditions to get t_lag values...")
    df_params_all_25 = extract_parameters_with_replicates(
        files_dict,
        conv_OD_to_cell,
        selected_conditions=None,  # All conditions
    )
    logger.info(
        f"   Extracted {len(df_params_all_25)} conditions total (all t_lag values available)"
    )

    # STEP 2: Filter for RSM fitting (if necessary)
    if selected_conditions is not None:
        logger.info(
            f"\n2. Filtering to {len(selected_conditions)} selected conditions for RSM fitting..."
        )
        df_params_for_rsm = extract_parameters_with_replicates(
            files_dict, conv_OD_to_cell, selected_conditions=selected_conditions
        )
        logger.info(
            f"   Using {len(df_params_for_rsm)} conditions for RSM surface fitting"
        )
    else:
        logger.info("\n2. Using all conditions for RSM fitting (no filtering)...")
        df_params_for_rsm = df_params_all_25

    logger.info(f"\nSuccessfully extracted parameters:")
    logger.info(f"  - Total conditions (with t_lag): {len(df_params_all_25)}")
    logger.info(f"  - Conditions for RSM fit: {len(df_params_for_rsm)}")

    # Apply manual t_lag adjustments (if defined)
    if manual_tlag_adjustments:
        df_params_all_25 = apply_manual_tlag_adjustments(
            df_params_all_25, manual_tlag_adjustments
        )

    # Display conditions used for RSM if filtering is active
    if selected_conditions is not None:
        logger.info("\n" + "=" * 80)
        logger.info("CONDITIONS USED FOR RSM FITTING:")
        logger.info("=" * 80)
        for _, row in df_params_for_rsm.iterrows():
            logger.info(f"  L0={row['L0']:.1f} µmol/m²/s, C0={row['C0']:.4f}")
        logger.info("=" * 80 + "\n")

    # Check that we have enough points for RSM
    if len(df_params_for_rsm) < 6:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: Not enough data points for RSM analysis!")
        logger.error(f"Found: {len(df_params_for_rsm)} conditions")
        logger.error(f"Required: at least 6 conditions for RSM model (3 parameters)")
        logger.error(f"{'!' * 80}\n")
        logger.error("Available conditions:")
        for _, row in df_params_for_rsm.iterrows():
            logger.error(f"  L0={row['L0']:.1f}, C0={row['C0']:.4f}")
        return

    # df_params = conditions used to fit the RSM surfaces for mu_max and Nmax
    # df_params_all_25 = all 25 conditions with their t_lag values (used for lookup)
    df_params = df_params_for_rsm

    # Save the results
    csv_path = output_dir / "growth_parameters_rsm_fit.csv"
    df_params.drop(columns=["replicate_fits"]).to_csv(
        csv_path, index=False, float_format="%.6f", sep=";"
    )
    logger.info(f"Saved RSM fit parameters to: {csv_path}")

    # Also save all parameters (25 conditions with t_lag) with calibration/extrapolation type
    calib_keys = set(zip(df_params["C0"], df_params["L0"]))
    df_params_all_25_export = df_params_all_25.drop(columns=["replicate_fits"]).copy()
    df_params_all_25_export["type"] = df_params_all_25_export.apply(
        lambda row: (
            "calibration" if (row["C0"], row["L0"]) in calib_keys else "extrapolation"
        ),
        axis=1,
    )
    csv_path_all = output_dir / "growth_parameters_all_25_conditions.csv"
    df_params_all_25_export.to_csv(
        csv_path_all, index=False, float_format="%.6f", sep=";"
    )
    logger.info(f"Saved all 25 conditions (with t_lag) to: {csv_path_all}")

    logger.info(f"\n{'=' * 80}")
    logger.info("PARAMETER STATISTICS")
    logger.info(f"{'=' * 80}")
    for param in ["mu_max", "Nmax", "t_lag"]:
        logger.info(f"\n{param}_mean:")
        logger.info(
            f"  Overall Mean ± Std: {df_params[f'{param}_mean'].mean():.4e} ± {df_params[f'{param}_mean'].std():.4e}"
        )
        logger.info(
            f"  Range: [{df_params[f'{param}_mean'].min():.4e}, {df_params[f'{param}_mean'].max():.4e}]"
        )
        logger.info(
            f"  Avg StdDev across replicates: {df_params[f'{param}_std'].mean():.4e}"
        )

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATING RSM 3D SURFACES")
    logger.info(f"{'=' * 80}\n")

    # Fit the RSM surfaces and store the coefficients (except t_lag)
    coeffs_dict = {}
    for response_var in ["mu_max", "Nmax"]:  # We remove 't_lag' from this loop
        logger.info(f"Processing {response_var}...")
        coeffs = plot_3d_surface(
            df_params, response_var, output_dir, df_all_conditions=df_params_all_25
        )
        coeffs_dict[response_var] = coeffs

    # =========================================================================
    # EXPORT RSM COEFFICIENTS TO YAML (format compatible with fitting_plates.py)
    # =========================================================================
    # Coefficients for mu_max: [beta_0 (mu_max_ref), K_L0 (k_l), K_C0 (k_c)]
    if coeffs_dict["mu_max"] is not None:
        growth_rate_params = {
            "mu_max_ref": float(coeffs_dict["mu_max"][0]),
            "k_l": float(coeffs_dict["mu_max"][1]),
            "k_c": float(coeffs_dict["mu_max"][2]),
        }
        growth_rate_path = output_dir / "parameters_growth_rate.yaml"
        with open(growth_rate_path, "w") as f:
            yaml.dump(growth_rate_params, f)
        logger.info(f"Saved growth rate parameters to {growth_rate_path}")

    # Coefficients for Nmax: [beta_0 (n_max_ref), K_L0 (k_l), K_C0 (k_c)]
    if coeffs_dict["Nmax"] is not None:
        steady_state_params = {
            "n_max_ref": float(coeffs_dict["Nmax"][0]),
            "k_l": float(coeffs_dict["Nmax"][1]),
            "k_c": float(coeffs_dict["Nmax"][2]),
        }
        steady_state_path = output_dir / "parameters_steady_state.yaml"
        with open(steady_state_path, "w") as f:
            yaml.dump(steady_state_params, f)
        logger.info(f"Saved steady state parameters to {steady_state_path}")

    fig = create_full_rsm_panel(
        df_params, output_dir, df_all_conditions=df_params_all_25
    )

    logger.info(f"\n{'=' * 80}")
    logger.info("RSM COEFFICIENTS SUMMARY")
    logger.info(f"{'=' * 80}\n")
    for var_name, coeffs in coeffs_dict.items():
        logger.info(f"\n{var_name} coefficients:")

        # Get parameter names from ModelConfig
        if var_name == "mu_max":
            param_names = ModelConfig.mu_max_param_names()
        elif var_name == "Nmax":
            param_names = ModelConfig.Nmax_param_names()
        else:
            raise ValueError(f"Unknown response variable: {var_name}")

        # Display each coefficient with its name
        for i, (name, value) in enumerate(zip(param_names, coeffs)):
            logger.info(f"  {name}: {value:.6e}")

    logger.info("\nGenerating fits overview...")
    plot_fits_overview(df_params, output_dir)

    logger.info("\nGenerating RSM-logistic fits overview...")
    plot_fits_overview_rsm(
        df_params,
        coeffs_dict["mu_max"],
        coeffs_dict["Nmax"],
        df_params_all_25,  # Pass ALL 25 conditions for t_lag lookup
        output_dir,
    )

    # Evaluate RSM extrapolation on all 25 conditions
    logger.info("\nEvaluating RSM extrapolation on all 25 conditions...")
    df_r2_extrapolation = evaluate_rsm_extrapolation(
        df_params,  # Conditions used for RSM fit (for marking)
        coeffs_dict["mu_max"],
        coeffs_dict["Nmax"],
        df_params_all_25,  # Pass ALL 25 conditions for t_lag lookup
        files_dict,
        conv_OD_to_cell,
        output_dir,
    )

    _ = evaluate_rsm_extrapolation_by_L0(
        df_params,  # Conditions used for RSM fit (for marking)
        coeffs_dict["mu_max"],
        coeffs_dict["Nmax"],
        df_params_all_25,  # Pass ALL 25 conditions for t_lag lookup
        files_dict,
        conv_OD_to_cell,
        output_dir,
    )

    # Generate the R² heatmap in a 5x5 grid
    logger.info("\nGenerating R² heatmap grid (5x5 with 4 sub-squares)...")
    # Use df_params_all_25 already extracted (all 25 conditions)
    plot_r2_heatmap_grid(
        df_params_all_25,
        coeffs_dict["mu_max"],
        coeffs_dict["Nmax"],
        df_params_all_25,  # Pass ALL 25 conditions for t_lag lookup
        output_dir,
    )

    # ========================================================================
    # R² STATISTICAL SUMMARY (new format)
    # ========================================================================
    logger.info("")
    if df_r2_extrapolation is not None and len(df_r2_extrapolation) > 0:
        print_r2_statistics_summary(
            df_r2_extrapolation,
            df_params_all_25,
            coeffs_dict["mu_max"],
            coeffs_dict["Nmax"],
        )

    logger.info(f"\n{'=' * 80}")
    logger.info("RSM ANALYSIS COMPLETED!")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"\nAnalysis mode: {'FILTERED' if selected_conditions is not None else 'FULL'}"
    )
    logger.info(f"Number of conditions analyzed: {len(df_params)}")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(
        "  - growth_parameters_with_replicates.csv (includes individual A, B, C values)"
    )
    logger.info(
        "  - RSM_3D_mu_max.png, RSM_3D_Nmax.png (3D surface plots with error bars)"
    )
    logger.info("  - RSM_full_panel.png (combined panel with RSM surfaces)")
    logger.info("  - logistic_fits_calibration.png (fits for all replicates)")
    logger.info("  - RSM_ODE_calibration.png (RSM-integrated logistic model)")
    logger.info("  - parameters_growth_rate.yaml (mu_max surface coefficients)")
    logger.info("  - parameters_steady_state.yaml (Nmax surface coefficients)")
    logger.info(
        "  - rsm_extrapolation_all_conditions.png (RSM fits on all 25 conditions)"
    )
    logger.info(
        "  - rsm_extrapolation_r2.csv (R² values for fit and extrapolation conditions)"
    )
    logger.info(
        "  - r2_heatmap_grid_5x5.png (R² heatmap grid: replicates A/B/C and RSM-ODE)"
    )

    if selected_conditions is not None:
        logger.info(f"\n{'=' * 80}")
        logger.info("NOTE: Filtered analysis was performed")
        logger.info("Conditions analyzed:")
        for L0, C0 in selected_conditions:
            logger.info(f"  - L0={L0:.1f} µmol/m²/s, C0={C0:.4f}")
        logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
