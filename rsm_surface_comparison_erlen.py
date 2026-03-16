"""
Testing multiple mathematical models for mu_max and N_max RSM surfaces

This script tests different functions:
1. Monod (product of two Monod) - current model
2. Mix Monod x Haldane
3. Linear inverse
4. Polynomial inverse
5. Binding/competition

For each function, generates 2 figures (mu_max and N_max) with 3 panels:
- Experimental data with error bars
- 3D RSM surface
- Contour map
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# MODEL DEFINITIONS
# ================================================================================


class RSMModels:
    """
    RSM model library for mu_max and N_max

    Calibration data used for initialization:
    - mu_max: [0.062, 0.210] h^-1, median=0.147
    - N_max: [1.01e7, 5.92e7] cells/mL, median=3.58e7
    - L0: [0.07, 1.0] (factors), median=0.3
    - C0: [0.0625, 1.0], median=0.5
    """

    # =========================================================================
    # SEPARABLE MODELS: y = y_max * f(L0) * g(C0)
    # =========================================================================

    # MODEL 1: MONOD
    @staticmethod
    def monod(L0, C0, params):
        return params[0] * (L0 / (L0 + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def monod_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, np.median(L0), np.median(C0)]

    @staticmethod
    def monod_bounds():
        # K_L0 >= 0.059 (= 10/170 umol) to match fitting_erlen.py bounds in factor space
        return ([0, 0.059, 0.001], [np.inf, 5.9, 10])

    @staticmethod
    def monod_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.3e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3e}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3e}}}$"
        )

    # MODEL 2: MONOD_HALDANE
    @staticmethod
    def monod_haldane(L0, C0, params):
        haldane_term = L0 / (L0 + params[1] + L0**2 / params[2])
        monod_term = C0 / (C0 + params[3])
        return params[0] * haldane_term * monod_term

    @staticmethod
    def monod_haldane_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, np.median(L0), 10, np.median(C0)]

    @staticmethod
    def monod_haldane_bounds():
        return ([0, 0.01, 0.1, 0.01], [np.inf, 5, 100, 5])

    @staticmethod
    def monod_haldane_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.1f} + L_0^2/{params[2]:.0f}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[3]:.3f}}}$"
        )

    # MODEL 3: POLYNOMIAL_INVERSE
    @staticmethod
    def polynomial_inverse(L0, C0, params):
        return params[0] / (1 + params[1] * L0 + params[2] * C0 + params[3] * C0 * L0)

    @staticmethod
    def polynomial_inverse_initial_params(L0, C0, y):
        return [np.max(y) * 1.5, 0.5, 0.5, 0.1]

    @staticmethod
    def polynomial_inverse_bounds():
        return ([0, 0.001, 0.01, -10], [np.inf, 10, 3, 20])

    @staticmethod
    def polynomial_inverse_latex(params, param_name):
        return (
            f"${param_name} = \\frac{{{params[0]:.2e}}}"
            f"{{1 + {params[1]:.4f} L_0 + {params[2]:.3f} C_0 + {params[3]:.5f} L_0 C_0}}$"
        )

    # MODEL 4: HILL_DOUBLE
    @staticmethod
    def hill_double(L0, C0, params):
        term_L0 = (L0 ** params[3]) / (L0 ** params[3] + params[1] ** params[3])
        term_C0 = (C0 ** params[4]) / (C0 ** params[4] + params[2] ** params[4])
        return params[0] * term_L0 * term_C0

    @staticmethod
    def hill_double_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, np.median(L0), np.median(C0), 1.5, 1.5]

    @staticmethod
    def hill_double_bounds():
        return ([0, 0.01, 0.01, 0.3, 0.3], [np.inf, 5, 5, 4, 4])

    @staticmethod
    def hill_double_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{L_0^{{{params[3]:.2f}}}}}{{L_0^{{{params[3]:.2f}}} + {params[1]:.1f}^{{{params[3]:.2f}}}}} \\times "
            f"\\frac{{C_0^{{{params[4]:.2f}}}}}{{C_0^{{{params[4]:.2f}}} + {params[2]:.3f}^{{{params[4]:.2f}}}}}$"
        )

    # MODEL 5: EXPONENTIAL_SATURATION
    @staticmethod
    def exponential_saturation(L0, C0, params):
        term_L0 = 1 - np.exp(-params[1] * L0)
        term_C0 = 1 - np.exp(-params[2] * C0)
        return params[0] * term_L0 * term_C0

    @staticmethod
    def exponential_saturation_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, 2.0, 1.0]

    @staticmethod
    def exponential_saturation_bounds():
        return ([0, 0.1, 0.1], [np.inf, 50, 10])

    @staticmethod
    def exponential_saturation_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"(1 - e^{{-{params[1]:.4f} L_0}}) \\times "
            f"(1 - e^{{-{params[2]:.3f} C_0}})$"
        )

    # =========================================================================
    # NON-SEPARABLE MODELS: L0 x C0 interactions
    # =========================================================================

    # MODEL 6: MONOD_CROSS_CONSTANTS
    @staticmethod
    def monod_cross_constants(L0, C0, params):
        K_L = params[1] + params[2] / (C0 + 0.01)
        K_C = params[3] + params[4] / (L0 + 0.01)
        term_L0 = L0 / (L0 + K_L)
        term_C0 = C0 / (C0 + K_C)
        return params[0] * term_L0 * term_C0

    @staticmethod
    def monod_cross_constants_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, 0.5, 0.1, 0.4, 0.01]

    @staticmethod
    def monod_cross_constants_bounds():
        return ([0, 0.01, 0, 0.01, 0], [np.inf, 5, 5, 2, 1])

    @staticmethod
    def monod_cross_constants_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.1f} + {params[2]:.1f}/C_0}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[3]:.3f} + {params[4]:.3f}/L_0}}$"
        )

    # MODEL 7: MONOD_INTERACTION
    @staticmethod
    def monod_interaction(L0, C0, params):
        term_L0 = L0 / (L0 + params[1])
        term_C0 = C0 / (C0 + params[2])
        interaction = 1 + (params[3] * L0 * C0) / (1 + params[4] * L0 * C0)
        return params[0] * term_L0 * term_C0 * interaction

    @staticmethod
    def monod_interaction_initial_params(L0, C0, y):
        return [np.max(y) * 1.0, np.median(L0), np.median(C0), 0.1, 1]

    @staticmethod
    def monod_interaction_bounds():
        return ([0, 0.01, 0.01, 0, 0.01], [np.inf, 5, 5, 10, 100])

    @staticmethod
    def monod_interaction_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{L_0}}{{L_0 + {params[1]:.1f}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3f}}} \\times "
            f"(1 + \\frac{{{params[3]:.4f} L_0 C_0}}{{1 + {params[4]:.1f} L_0 C_0}})$"
        )

    # MODEL 8: GAUSSIAN_2D
    @staticmethod
    def gaussian_2d(L0, C0, params):
        term1 = params[3] * (L0 - params[1]) ** 2
        term2 = params[4] * (C0 - params[2]) ** 2
        term_cross = params[5] * (L0 - params[1]) * (C0 - params[2])
        exponent = -(term1 + term2 + term_cross)
        return params[0] * np.exp(exponent)

    @staticmethod
    def gaussian_2d_initial_params(L0, C0, y):
        return [np.max(y), np.max(L0) * 0.8, np.max(C0) * 0.8, 1.0, 5, 1.0]

    @staticmethod
    def gaussian_2d_bounds():
        return ([0, 0, 0, 0.1, 0.1, -50], [np.inf, np.inf, np.inf, 50, 50, 50])

    @staticmethod
    def gaussian_2d_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\exp(-[{params[3]:.5f}(L_0-{params[1]:.1f})^2 + "
            f"{params[4]:.3f}(C_0-{params[2]:.2f})^2 + "
            f"{params[5]:.4f}(L_0-{params[1]:.1f})(C_0-{params[2]:.2f})])$"
        )

    # MODEL 9: STEELE (Photo-inhibition)
    @staticmethod
    def steele(L0, C0, params):
        """Steele model with photo-inhibition (separable)"""
        beta_0, L_opt, K_C = params
        term_L0 = (L0 / L_opt) * np.exp(1 - L0 / L_opt)
        term_C0 = C0 / (C0 + K_C)
        return beta_0 * term_L0 * term_C0

    @staticmethod
    def steele_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, np.max(L0) * 0.7, np.median(C0)]

    @staticmethod
    def steele_bounds():
        return ([0, 0.01, 0.01], [np.inf, 5, 5])

    @staticmethod
    def steele_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{L_0}}{{{params[1]:.1f}}} e^{{1 - L_0/{params[1]:.1f}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3f}}}$"
        )

    # MODEL 10: MONOD_BETA0_MODULATION (NON-separable)
    @staticmethod
    def monod_beta0_modulation(L0, C0, params):
        """Monod with beta0 modulated by C0 (NON-separable)"""
        beta_1, beta_2, beta_3, K_L, K_C = params
        beta_0_C0 = beta_1 + beta_2 * C0 / (C0 + beta_3)
        term_L0 = L0 / (L0 + K_L)
        term_C0 = C0 / (C0 + K_C)
        return beta_0_C0 * term_L0 * term_C0

    @staticmethod
    def monod_beta0_modulation_initial_params(L0, C0, y):
        return [
            np.max(y) * 0.8,
            np.max(y) * 0.4,
            np.median(C0),
            np.median(L0),
            np.median(C0),
        ]

    @staticmethod
    def monod_beta0_modulation_bounds():
        return ([0, 0, 0.01, 0.01, 0.01], [np.inf, np.inf, 2, 5, 5])

    @staticmethod
    def monod_beta0_modulation_latex(params, param_name):
        return (
            f"${param_name} = ({params[0]:.2e} + \\frac{{{params[1]:.2e} C_0}}{{C_0 + {params[2]:.3f}}}) \\times "
            f"\\frac{{L_0}}{{L_0 + {params[3]:.1f}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[4]:.3f}}}$"
        )

    # MODEL 11: LOGISTIC_BILINEAR (NON-separable)
    @staticmethod
    def logistic_bilinear(L0, C0, params):
        """Bilinear logistic with sigmoid transitions (NON-separable)"""
        beta_0, k1, L_thresh, k2, C_thresh = params
        term_L0 = 1 / (1 + np.exp(-k1 * (L0 - L_thresh)))
        term_C0 = 1 / (1 + np.exp(-k2 * (C0 - C_thresh)))
        return beta_0 * term_L0 * term_C0

    @staticmethod
    def logistic_bilinear_initial_params(L0, C0, y):
        return [np.max(y) * 1.2, 5.0, np.median(L0), 5, np.median(C0)]

    @staticmethod
    def logistic_bilinear_bounds():
        return ([0, 0.1, 0.01, 0.1, 0.01], [np.inf, 100, 5, 50, 2])

    @staticmethod
    def logistic_bilinear_latex(params, param_name):
        return (
            f"${param_name} = {params[0]:.2e} \\times "
            f"\\frac{{1}}{{1 + e^{{-{params[1]:.3f}(L_0-{params[2]:.1f})}}}} \\times "
            f"\\frac{{1}}{{1 + e^{{-{params[3]:.3f}(C_0-{params[4]:.3f})}}}}$"
        )


# ================================================================================
# MODEL DICTIONARY
# ================================================================================

MODEL_LIBRARY = {
    "Monod": {
        "func": RSMModels.monod,
        "initial": RSMModels.monod_initial_params,
        "bounds": RSMModels.monod_bounds,
        "latex": RSMModels.monod_latex,
        "description": "Monod double (separable)",
        "separable": True,
    },
    "Monod_Haldane": {
        "func": RSMModels.monod_haldane,
        "initial": RSMModels.monod_haldane_initial_params,
        "bounds": RSMModels.monod_haldane_bounds,
        "latex": RSMModels.monod_haldane_latex,
        "description": "Monod x Haldane (separable)",
        "separable": True,
    },
    "Polynomial_Inverse": {
        "func": RSMModels.polynomial_inverse,
        "initial": RSMModels.polynomial_inverse_initial_params,
        "bounds": RSMModels.polynomial_inverse_bounds,
        "latex": RSMModels.polynomial_inverse_latex,
        "description": "Polynome inverse (separable)",
        "separable": True,
    },
    "Hill_Double": {
        "func": RSMModels.hill_double,
        "initial": RSMModels.hill_double_initial_params,
        "bounds": RSMModels.hill_double_bounds,
        "latex": RSMModels.hill_double_latex,
        "description": "Hill double (separable)",
        "separable": True,
    },
    "Exponential_Saturation": {
        "func": RSMModels.exponential_saturation,
        "initial": RSMModels.exponential_saturation_initial_params,
        "bounds": RSMModels.exponential_saturation_bounds,
        "latex": RSMModels.exponential_saturation_latex,
        "description": "Saturation exponentielle (separable)",
        "separable": True,
    },
    "Monod_Cross_Constants": {
        "func": RSMModels.monod_cross_constants,
        "initial": RSMModels.monod_cross_constants_initial_params,
        "bounds": RSMModels.monod_cross_constants_bounds,
        "latex": RSMModels.monod_cross_constants_latex,
        "description": "Monod constantes croisees (NON-separable)",
        "separable": False,
    },
    "Monod_Interaction": {
        "func": RSMModels.monod_interaction,
        "initial": RSMModels.monod_interaction_initial_params,
        "bounds": RSMModels.monod_interaction_bounds,
        "latex": RSMModels.monod_interaction_latex,
        "description": "Monod avec interaction (NON-separable)",
        "separable": False,
    },
    "Gaussian_2D": {
        "func": RSMModels.gaussian_2d,
        "initial": RSMModels.gaussian_2d_initial_params,
        "bounds": RSMModels.gaussian_2d_bounds,
        "latex": RSMModels.gaussian_2d_latex,
        "description": "Gaussienne 2D (NON-separable)",
        "separable": False,
    },
    "Steele": {
        "func": RSMModels.steele,
        "initial": RSMModels.steele_initial_params,
        "bounds": RSMModels.steele_bounds,
        "latex": RSMModels.steele_latex,
        "description": "Steele photo-inhibition (separable)",
        "separable": True,
    },
    "Monod_Beta0_Modulation": {
        "func": RSMModels.monod_beta0_modulation,
        "initial": RSMModels.monod_beta0_modulation_initial_params,
        "bounds": RSMModels.monod_beta0_modulation_bounds,
        "latex": RSMModels.monod_beta0_modulation_latex,
        "description": "Monod beta0 module (NON-separable)",
        "separable": False,
    },
    "Logistic_Bilinear": {
        "func": RSMModels.logistic_bilinear,
        "initial": RSMModels.logistic_bilinear_initial_params,
        "bounds": RSMModels.logistic_bilinear_bounds,
        "latex": RSMModels.logistic_bilinear_latex,
        "description": "Logistique bilineaire (NON-separable)",
        "separable": False,
    },
}


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================


def calculate_r2(y_true, y_pred):
    """Compute the coefficient of determination R2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def read_csv_with_replicates(filepath, conv_OD_to_cell=4.77e6):
    """
    Read a CSV file containing technical replicates A, B, C for each condition.
    IDENTICAL TO ORIGINAL CODE

    Returns:
        dict: {condition_index: {'Time': array, 'A': array, 'B': array, 'C': array}}
    """
    df = pd.read_csv(filepath, sep=";")

    # Extract time
    time = df["Time (s)"].values / 3600

    data = {}

    # For each condition (0, 1, 2, 3, 4)
    for cond_idx in range(5):
        replicate_data = {
            "Time": time,
            "A": df[f"OD {cond_idx}A"].values * conv_OD_to_cell,
            "B": df[f"OD {cond_idx}B"].values * conv_OD_to_cell,
            "C": df[f"OD {cond_idx}C"].values * conv_OD_to_cell,
        }
        data[cond_idx] = replicate_data

    return data


def logistic_growth(t, N0, Nmax, mu_max, t_lag):
    """Logistic growth model"""
    return Nmax / (1 + (Nmax / N0 - 1) * np.exp(-mu_max * (t - t_lag)))


def estimate_growth_parameters(time, biomass):
    """
    Estimate growth parameters from data.
    IDENTICAL TO ORIGINAL CODE

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

    # Estimate mu_max from the exponential phase
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

        # Compute R2 on experimental points
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


def fit_rsm_surface(df, response_var, model_name):
    """
    Fit an RSM surface with a specific model

    Parameters:
    -----------
    df : DataFrame
        Calibration data with columns L0, C0 and response_var
    response_var : str
        Response variable name (with _mean, e.g.: 'mu_max_mean')
    model_name : str
        Model name to use (must be in MODEL_LIBRARY)

    Returns:
    --------
    predict_func : function
        Function to predict new values
    coeffs : array
        Fitted coefficients
    r2 : float
        R2 of the fit
    n_obs : int
        Number of calibration observations
    n_params : int
        Number of model parameters
    rss : float
        Residual sum of squares (RSS)
    """
    L0 = df["L0"].values
    C0 = df["C0"].values
    y = df[response_var].values

    model = MODEL_LIBRARY[model_name]
    model_func = model["func"]
    initial_params = model["initial"](L0, C0, y)
    bounds = model["bounds"]()

    # Wrapper for curve_fit
    def model_to_fit(X, *params):
        L0_vals, C0_vals = X
        return model_func(L0_vals, C0_vals, np.array(params))

    try:
        popt, pcov = curve_fit(
            model_to_fit, (L0, C0), y, p0=initial_params, bounds=bounds, maxfev=10000
        )

        coeffs = popt
        y_pred = model_func(L0, C0, coeffs)
        r2 = calculate_r2(y, y_pred)
        n_obs = len(y)
        n_params = len(coeffs)
        rss = np.sum((y - y_pred) ** 2)

        def predict(L0_new, C0_new):
            return model_func(L0_new, C0_new, coeffs)

        logger.info(f"  Fit {model_name} successful: R² = {r2:.4f}")
        return predict, coeffs, r2, n_obs, n_params, rss

    except Exception as e:
        logger.warning(f"Fit {model_name} failed: {e}")
        # Return a function that predicts the mean
        mean_val = np.mean(y)
        n_obs = len(y)
        n_params = len(initial_params)
        rss = np.sum((y - mean_val) ** 2)

        def predict(L0_new, C0_new):
            return mean_val

        return predict, initial_params, 0.0, n_obs, n_params, rss


def plot_rsm_comparison(df, df_all, response_var, model_name, output_dir):
    """
    Generate a figure with 3 panels to compare an RSM model

    Parameters:
    -----------
    df : DataFrame
        Calibration data
    df_all : DataFrame
        All data (calibration + extrapolation)
    response_var : str
        Response variable ('mu_max' or 'Nmax')
    model_name : str
        Model name to use
    output_dir : Path
        Output directory
    """
    response_mean = f"{response_var}_mean"
    response_std = f"{response_var}_std"

    # Determine the LaTeX label
    if response_var == "mu_max":
        param_latex = "\\mu_{\\mathrm{max}}"
        unit = "h$^{-1}$"
        cmap_choice = "viridis"
    else:  # Nmax
        param_latex = "N_{\\mathrm{max}}"
        unit = "cells/mL"
        cmap_choice = "plasma"

    # Fit the RSM surface
    predict_func, coeffs, r2_calib, n_calib, n_params, rss_calib = fit_rsm_surface(
        df, response_mean, model_name
    )

    # Compute global R2 on ALL conditions (calibration + extrapolation)
    y_all = df_all[response_mean].values
    y_pred_all = np.array(
        [predict_func(row["L0"], row["C0"]) for _, row in df_all.iterrows()]
    )
    r2_global = calculate_r2(y_all, y_pred_all)
    n_global = len(y_all)
    rss_global = np.sum((y_all - y_pred_all) ** 2)

    logger.info(
        f"  Model {model_name}, {response_var}: R²_calib={r2_calib:.4f}, R²_global={r2_global:.4f}"
    )

    # Create figure
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

    # -------------------------------------------------------------------------
    # PANEL 1: Experimental data with error bars
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")

    # Colors by L0
    L0_unique = sorted(df["L0"].unique())
    colors_L0 = plt.cm.winter(np.linspace(0, 1, len(L0_unique)))
    L0_to_color = {L0: colors_L0[i] for i, L0 in enumerate(L0_unique)}

    for L0_val in L0_unique:
        df_L0 = df[df["L0"] == L0_val]

        for _, row in df_L0.iterrows():
            ax1.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                c=[L0_to_color[L0_val]],
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
                color=L0_to_color[L0_val],
                linewidth=2,
                alpha=0.6,
            )

    ax1.set_xlabel("$L_0$ (factor)", fontsize=11, labelpad=10)
    ax1.set_ylabel("$C_0$ (dilution factor)", fontsize=11, labelpad=10)
    ax1.set_zlabel(f"${param_latex}$ ({unit})", fontsize=11, labelpad=10)
    ax1.set_title("Experimental data with error bars", fontsize=12, pad=20)

    # -------------------------------------------------------------------------
    # PANEL 2: RSM Surface
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    # Extended grid over the full experimental domain
    L0_min, L0_max = df_all["L0"].min(), df_all["L0"].max()
    C0_min, C0_max = df_all["C0"].min(), df_all["C0"].max()

    L0_range = np.linspace(L0_min, L0_max, 50)
    C0_range = np.linspace(C0_min, C0_max, 50)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # Prediction
    response_grid = np.zeros_like(L0_grid)
    for i in range(L0_grid.shape[0]):
        for j in range(L0_grid.shape[1]):
            response_grid[i, j] = predict_func(L0_grid[i, j], C0_grid[i, j])

    # Surface
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
    df_calib_keys = set(zip(df["L0"], df["C0"]))

    df_calibration = df_all[
        df_all.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
    ].copy()
    df_extrapolation = df_all[
        ~df_all.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
    ].copy()

    logger.info(
        f"  Model {model_name}, {response_var}: {len(df_calibration)} calibration points, {len(df_extrapolation)} extrapolation points"
    )

    # Calibration points (black, circles)
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

    # Extrapolation points (purple blue, circles)
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

    ax2.set_xlabel("$L_0$ (factor)", fontsize=11, labelpad=10)
    ax2.set_ylabel("$C_0$ (dilution factor)", fontsize=11, labelpad=10)
    ax2.set_zlabel(f"${param_latex}$ ({unit})", fontsize=11, labelpad=10)
    ax2.set_title(
        f"RSM Surface\nR² calib = {r2_calib:.3f} | R² global = {r2_global:.3f}",
        fontsize=12,
        pad=20,
    )

    # Add legend if there are extrapolation points
    if len(df_extrapolation) > 0:
        ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # -------------------------------------------------------------------------
    # PANEL 3: Contour map
    # -------------------------------------------------------------------------
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

    ax3.set_xlabel("$L_0$ (factor)", fontsize=11)
    ax3.set_ylabel("$C_0$ (dilution factor)", fontsize=11)
    ax3.set_title("Contour map", fontsize=12)

    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label(f"${param_latex}$ ({unit})", rotation=270, labelpad=25, fontsize=11)

    # -------------------------------------------------------------------------
    # PANEL 4: Model equation
    # -------------------------------------------------------------------------
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # Get LaTeX equation
    equation_latex = MODEL_LIBRARY[model_name]["latex"](coeffs, param_latex)

    ax_text.text(
        0.5,
        0.5,
        equation_latex,
        ha="center",
        va="center",
        fontsize=13,
        family="serif",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.7),
    )

    # Save
    filename = f"RSM_3D_{response_var}_{model_name}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {filename}")

    # Return metrics for the summary table
    return r2_calib, r2_global, n_calib, n_global, n_params, rss_calib, rss_global


def plot_logistic_fits_overview(df_params, output_dir):
    """
    Display an overview of logistic fits for all conditions.
    Grid layout: rows = L0 (bottom=smallest, top=largest),
                 cols = C0 (left=smallest, right=largest).
    """
    # Build sorted axes for L0 (rows) and C0 (columns)
    L0_values = sorted(df_params["L0"].unique())  # ascending
    C0_values = sorted(df_params["C0"].unique())  # ascending
    nrows = len(L0_values)
    ncols = len(C0_values)

    # Map values to grid indices (L0 reversed so smallest is at bottom)
    L0_to_row = {v: nrows - 1 - i for i, v in enumerate(L0_values)}
    C0_to_col = {v: i for i, v in enumerate(C0_values)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    # Track which cells are filled
    filled = set()

    colors_rep = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71"}

    for _, row in df_params.iterrows():
        L0 = row["L0"]
        C0 = row["C0"]

        r = L0_to_row.get(L0)
        c = C0_to_col.get(C0)
        if r is None or c is None:
            continue

        ax = axes[r, c]
        filled.add((r, c))

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

        # Title with parameters
        title_text = (
            f"L0={L0:.3f}, C0={C0:.4f}\n"
            f"mumax={row['mu_max_mean']:.3f}+/-{row['mu_max_std']:.3f} h-1\n"
            f"Nmax={row['Nmax_mean']:.2e}+/-{row['Nmax_std']:.2e} cells/mL"
        )

        ax.set_title(title_text, fontsize=8)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_ylabel("N (cells/mL)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Add row / column labels and hide empty axes
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in filled:
                axes[r, c].axis("off")
        # Row label (L0) on the left
        axes[r, 0].set_ylabel(
            f"L0={L0_values[nrows - 1 - r]:.3f}\nN (cells/mL)", fontsize=9
        )
    for c in range(ncols):
        # Column label (C0) on top
        axes[0, c].set_title(
            f"C0={C0_values[c]:.4f}\n" + axes[0, c].get_title(), fontsize=8
        )

    plt.tight_layout()
    output_path = output_dir / "logistic_fits_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: logistic_fits_overview.png")


# ================================================================================
# DATA LOADING
# ================================================================================


def load_and_process_data(files_dict, conv_OD_to_cell, selected_conditions=None):
    """
    Load data and fit logistic curves
    USES THE SAME STRUCTURE AS THE ORIGINAL CODE

    IMPORTANT: Loads ALL conditions, then filters at the end

    Returns:
    --------
    df_params : DataFrame
        Parameters for calibration conditions
    df_params_all : DataFrame
        Parameters for all conditions (calibration + extrapolation)
    """
    results = []

    # DO NOT FILTER during loading - load ALL conditions
    logger.info("Loading ALL conditions (no filtering during data loading)")

    for L0_factor, filepath in files_dict.items():
        logger.info(f"\nProcessing file: {filepath} (L0_factor={L0_factor})")

        # Read data with replicates (IDENTICAL TO ORIGINAL CODE)
        data = read_csv_with_replicates(filepath, conv_OD_to_cell)

        # For each C0 condition - LOAD ALL without filtering
        for cond_idx in range(5):
            C0_factor = 1.0 / (2**cond_idx)  # 1.0, 0.5, 0.25, 0.125, 0.0625

            replicate_params = {"mu_max": [], "Nmax": [], "t_lag": [], "r_squared": []}
            replicate_values = {"A": {}, "B": {}, "C": {}}
            replicate_fits = []

            # Fit each replicate A, B, C (IDENTICAL TO ORIGINAL CODE)
            for rep_name in ["A", "B", "C"]:
                time = data[cond_idx]["Time"]
                biomass = data[cond_idx][rep_name]

                params = estimate_growth_parameters(time, biomass)

                if params and params["r_squared"] > 0.7:
                    replicate_params["mu_max"].append(params["mu_max"])
                    replicate_params["Nmax"].append(params["Nmax"])
                    replicate_params["t_lag"].append(params["t_lag"])
                    replicate_params["r_squared"].append(params["r_squared"])

                    # Store individual values
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
                    # Store NaN if the fit failed
                    replicate_values[rep_name]["mu_max"] = np.nan
                    replicate_values[rep_name]["Nmax"] = np.nan
                    replicate_values[rep_name]["t_lag"] = np.nan
                    if params is None or params["r_squared"] <= 0.7:
                        logger.warning(
                            f"  L0={L0_factor}, C0={C0_factor}, Rep {rep_name}: Fitting failed or R² too low"
                        )

            # Compute mean and std if at least 2 replicates succeeded
            if len(replicate_params["mu_max"]) >= 2:
                results.append(
                    {
                        "L0": L0_factor,
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
                    f"  ✓ L0={L0_factor:.3f}, C0={C0_factor:.4f}: "
                    f"mu_max={np.mean(replicate_params['mu_max']):.3f}±{np.std(replicate_params['mu_max'], ddof=1):.3f} h⁻¹, "
                    f"Nmax={np.mean(replicate_params['Nmax']):.2e}±{np.std(replicate_params['Nmax'], ddof=1):.2e}"
                )
            else:
                logger.warning(
                    f"  ✗ L0={L0_factor:.3f}, C0={C0_factor:.4f}: Not enough successful fits ({len(replicate_params['mu_max'])} replicates)"
                )

    # Create DataFrame with ALL conditions
    df_params_all = pd.DataFrame(results)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Total conditions loaded: {len(df_params_all)}")
    logger.info(f"{'=' * 80}")

    # NOW filter for calibration conditions if specified
    if selected_conditions is not None:
        selected_keys = set(selected_conditions)
        df_params = df_params_all[
            df_params_all.apply(
                lambda row: (row["L0"], row["C0"]) in selected_keys, axis=1
            )
        ].copy()

        logger.info(f"\n{'=' * 80}")
        logger.info("FILTERING FOR CALIBRATION")
        logger.info(f"{'=' * 80}")
        logger.info(f"Calibration conditions: {len(df_params)}")
        logger.info(f"Extrapolation conditions: {len(df_params_all) - len(df_params)}")
        logger.info(f"{'=' * 80}\n")
    else:
        df_params = df_params_all.copy()

    return df_params, df_params_all


# ================================================================================
# MAIN FUNCTION
# ================================================================================


def main():
    """
    Main function that tests all models
    """
    logger.info("=" * 80)
    logger.info("COMPARISON OF RSM MODELS FOR μ_max AND N_max")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------

    # Data files
    files_dict = {
        1.000: "all_data/data_exp_Chlamy_07-07-25.csv",
        0.6: "all_data/data_exp_Chlamy_01-07-25.csv",
        0.3: "all_data/data_exp_Chlamy_17-02-25.csv",
        0.15: "all_data/data_exp_Chlamy_04-11-24.csv",
        0.07: "all_data/data_exp_Chlamy_16-09-24.csv",
    }

    conv_OD_to_cell = 4.77e6

    # Calibration conditions (others will be used for extrapolation)
    # L0 and C0 are factors (L0 in [0.07, 1.0], C0 in [0.0625, 1.0])
    selected_conditions = [
        (1.0, 1.0000),
        (1.0, 0.5000),
        # (1.0, 0.2500), # removing to validate
        (1.0, 0.125),
        ##(1.0, 0.0625), # can be used instead of (1.0, 0.125)
        (0.6, 1.0000),
        # (0.6, 0.5000), # removing to validate
        (0.6, 0.2500),
        # (0.3, 1.0000),  # removing to validate
        # (0.3, 0.5000),
        (0.3, 0.2500),
        (0.15, 1),
    ]

    # Output directory
    output_dir = pathlib.Path("surfaces_comparison/erlen")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------

    logger.info("\nLoading and processing data...")
    df_params, df_params_all = load_and_process_data(
        files_dict, conv_OD_to_cell, selected_conditions
    )

    # Save logistic fits
    logger.info("\nGenerating logistic fits overview...")
    plot_logistic_fits_overview(df_params_all, output_dir)

    # -------------------------------------------------------------------------
    # TEST EACH MODEL
    # -------------------------------------------------------------------------

    logger.info("\nTesting all models...")
    logger.info("-" * 80)

    results_summary = []

    for model_name, model_info in MODEL_LIBRARY.items():
        logger.info(f"\nModel: {model_name} - {model_info['description']}")
        logger.info("-" * 80)

        # Tester pour μ_max
        logger.info("  Fitting μ_max...")
        (
            r2_mu_calib,
            r2_mu_global,
            n_mu_calib,
            n_mu_global,
            k_mu,
            rss_mu_calib,
            rss_mu_global,
        ) = plot_rsm_comparison(
            df_params, df_params_all, "mu_max", model_name, output_dir
        )

        # Tester pour N_max
        logger.info("  Fitting N_max...")
        (
            r2_N_calib,
            r2_N_global,
            n_N_calib,
            n_N_global,
            k_N,
            rss_N_calib,
            rss_N_global,
        ) = plot_rsm_comparison(
            df_params, df_params_all, "Nmax", model_name, output_dir
        )

        # Calcul AIC, AICc et BIC (sur donnees de calibration)
        # AIC  = n * ln(RSS/n) + 2*k
        # AICc = AIC + 2*k*(k+1) / (n - k - 1)   (correction petit echantillon)
        # BIC  = n * ln(RSS/n) + k * ln(n)
        aic_mu = n_mu_calib * np.log(rss_mu_calib / n_mu_calib) + 2 * k_mu
        bic_mu = n_mu_calib * np.log(rss_mu_calib / n_mu_calib) + k_mu * np.log(
            n_mu_calib
        )
        aicc_mu = (
            aic_mu + 2 * k_mu * (k_mu + 1) / (n_mu_calib - k_mu - 1)
            if n_mu_calib - k_mu - 1 > 0
            else np.inf
        )

        aic_N = n_N_calib * np.log(rss_N_calib / n_N_calib) + 2 * k_N
        bic_N = n_N_calib * np.log(rss_N_calib / n_N_calib) + k_N * np.log(n_N_calib)
        aicc_N = (
            aic_N + 2 * k_N * (k_N + 1) / (n_N_calib - k_N - 1)
            if n_N_calib - k_N - 1 > 0
            else np.inf
        )

        results_summary.append(
            {
                "Model": model_name,
                "Description": model_info["description"],
                "n_params": k_mu,
                "R2_mumax_calib": r2_mu_calib,
                "R2_mumax_global": r2_mu_global,
                "AIC_mumax": aic_mu,
                "AICc_mumax": aicc_mu,
                "BIC_mumax": bic_mu,
                "R2_Nmax_calib": r2_N_calib,
                "R2_Nmax_global": r2_N_global,
                "AIC_Nmax": aic_N,
                "AICc_Nmax": aicc_N,
                "BIC_Nmax": bic_N,
            }
        )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    df_results = pd.DataFrame(results_summary)
    logger.info("\n" + df_results.to_string(index=False))

    # Save summary
    df_results.to_csv(
        output_dir / "model_comparison_summary.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    logger.info(f"\nResults saved to: {output_dir / 'model_comparison_summary.csv'}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"\nAll figures saved in: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - logistic_fits_overview.png")
    for model_name in MODEL_LIBRARY.keys():
        logger.info(f"  - RSM_3D_mu_max_{model_name}.png")
        logger.info(f"  - RSM_3D_Nmax_{model_name}.png")
    logger.info("  - model_comparison_summary.csv")


if __name__ == "__main__":
    main()
