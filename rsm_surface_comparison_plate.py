"""
Testing multiple mathematical models for mu_max and N_max RSM surfaces — PLATE data

Data source:
  - results_plates/plates_growth_rates_measured.csv  (C0, L0, mu_max, type)
  - results_plates/plates_steady_states_measured.csv (C0, L0, N_max, type)

L0 values are normalized factors in [0, 1] (L0=1 corresponds to 170 µmol/m²/s).

For each model generates 2 figures (mu_max and N_max) with 3 panels:
- Experimental data
- 3D RSM surface
- Contour map

Outputs saved to: surfaces_comparison/plate/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# MODEL DEFINITIONS
# ================================================================================


class RSMModels:
    """
    RSM model library for mu_max and N_max

    Calibration data used for initialization:
    - mu_max: [0.043, 0.163] h^-1, median ~0.09
    - N_max: [4.4e6, 8.9e7] cells/mL
    - L0: [0.07, 1.0] (normalized factor), median=0.3
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
        return ([0, 0.001, 0.001], [np.inf, 5.0, 5.0])

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
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3f} + L_0^2/{params[2]:.1f}}} \\times "
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
            f"\\frac{{L_0^{{{params[3]:.2f}}}}}{{L_0^{{{params[3]:.2f}}} + {params[1]:.3f}^{{{params[3]:.2f}}}}} \\times "
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
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3f} + {params[2]:.3f}/C_0}} \\times "
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
            f"\\frac{{L_0}}{{L_0 + {params[1]:.3f}}} \\times "
            f"\\frac{{C_0}}{{C_0 + {params[2]:.3f}}} \\times "
            f"(1 + \\frac{{{params[3]:.4f} L_0 C_0}}{{1 + {params[4]:.3f} L_0 C_0}})$"
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
            f"\\exp(-[{params[3]:.5f}(L_0-{params[1]:.3f})^2 + "
            f"{params[4]:.3f}(C_0-{params[2]:.3f})^2 + "
            f"{params[5]:.4f}(L_0-{params[1]:.3f})(C_0-{params[2]:.3f})])$"
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
            f"\\frac{{L_0}}{{{params[1]:.3f}}} e^{{1 - L_0/{params[1]:.3f}}} \\times "
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
            f"\\frac{{L_0}}{{L_0 + {params[3]:.3f}}} \\times "
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
            f"\\frac{{1}}{{1 + e^{{-{params[1]:.3f}(L_0-{params[2]:.3f})}}}} \\times "
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
    """Compute the coefficient of determination R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def load_plate_data(growth_rates_file, steady_states_file):
    """
    Load pre-processed plate data from CSV files produced by fitting_plates.py.

    L0 is already a normalized factor in [0, 1] (L0=1 → 170 µmol/m²/s).

    Returns
    -------
    df_calib : DataFrame
        Calibration conditions only (type == 'calibration')
    df_all : DataFrame
        All conditions (calibration + extrapolation)
    """
    df_mu = pd.read_csv(growth_rates_file, sep=";")
    df_N = pd.read_csv(steady_states_file, sep=";")

    df = pd.merge(df_mu, df_N, on=["C0", "L0", "type"])

    # Rename to _mean convention used by fitting functions
    df = df.rename(columns={"mu_max": "mu_max_mean", "N_max": "Nmax_mean"})

    # No replicate std available from pre-processed data
    df["mu_max_std"] = np.nan
    df["Nmax_std"] = np.nan

    df_all = df.reset_index(drop=True)
    df_calib = df[df["type"] == "calibration"].reset_index(drop=True)

    logger.info(
        f"Loaded {len(df_all)} conditions total "
        f"({len(df_calib)} calibration, {len(df_all) - len(df_calib)} extrapolation)"
    )
    logger.info(
        f"L0 range: [{df_all['L0'].min()}, {df_all['L0'].max()}]  "
        f"(normalized factor, 1 = 170 µmol/m²/s)"
    )
    logger.info(f"C0 range: [{df_all['C0'].min()}, {df_all['C0'].max()}]")

    return df_calib, df_all


def plot_logistic_fits_overview(df_all, output_dir):
    """
    Overview of measured mu_max and N_max across the (L0, C0) space.

    Layout: rows = L0 values (bottom = smallest), columns = response variable.
    Left column: mu_max vs C0, Right column: N_max vs C0.
    Calibration and extrapolation points are distinguished.
    """
    L0_values = sorted(df_all["L0"].unique())  # ascending
    nrows = len(L0_values)
    ncols = 2  # mu_max | N_max

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 3 * nrows), sharex=False, sharey=False
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    colors = {"calibration": "#2980B9", "extrapolation": "#E74C3C"}
    markers = {"calibration": "o", "extrapolation": "s"}

    for row_idx, L0_val in enumerate(reversed(L0_values)):
        r = row_idx  # top row = largest L0
        df_L0 = df_all[df_all["L0"] == L0_val]

        for col_idx, (response, label, unit) in enumerate(
            [
                ("mu_max_mean", r"$\mu_\mathrm{max}$", "h$^{-1}$"),
                ("Nmax_mean", r"$N_\mathrm{max}$", "cells/mL"),
            ]
        ):
            ax = axes[r, col_idx]

            for typ in ["calibration", "extrapolation"]:
                subset = df_L0[df_L0["type"] == typ]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["C0"],
                    subset[response],
                    c=colors[typ],
                    marker=markers[typ],
                    s=50,
                    alpha=0.85,
                    label=typ.capitalize(),
                )

            ax.set_xlabel("$C_0$ (dilution factor)", fontsize=9)
            ax.set_ylabel(f"{label} ({unit})", fontsize=9)
            ax.set_title(f"L0 = {L0_val:.2f}  |  {label}", fontsize=9)
            ax.grid(True, alpha=0.3)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="best")

    plt.suptitle(
        "Measured growth parameters — plate 96-well\n"
        "(L0 normalized: 1 = 170 µmol/m²/s)",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()

    output_path = output_dir / "logistic_fits_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: logistic_fits_overview.png")


def fit_rsm_surface(df, response_mean, model_name):
    """
    Fit an RSM surface with a specific model.

    Parameters
    ----------
    df : DataFrame
        Calibration data with columns L0, C0 and response_mean
    response_mean : str
        Response variable name (e.g. 'mu_max_mean')
    model_name : str
        Model name (must be in MODEL_LIBRARY)

    Returns
    -------
    predict_func, coeffs, r2, n_obs, n_params, rss
    """
    L0 = df["L0"].values
    C0 = df["C0"].values
    y = df[response_mean].values

    model = MODEL_LIBRARY[model_name]
    model_func = model["func"]

    # For Monod_Haldane, use the parameters already fitted by fitting_plates.py
    # (loaded from YAML files) to guarantee identical results.
    # No re-fitting is done: the YAML coefficients are used directly.
    #
    # fitting_plates.py parameter order: [ref, k_c, k_l, k_i]
    # monod_haldane() parameter order:   [ref, k_l, k_i, k_c]   ← reordered here
    if model_name == "Monod_Haldane":
        is_mu_max = "mu_max" in response_mean
        yaml_path = pathlib.Path("results_plates") / (
            "parameters_growth_rate.yaml"
            if is_mu_max
            else "parameters_steady_state.yaml"
        )
        if yaml_path.exists():
            with open(yaml_path) as f:
                p = yaml.safe_load(f)
            if is_mu_max:
                # YAML keys: mu_max_ref, k_c, k_l, k_i → monod_haldane order: [ref, k_l, k_i, k_c]
                coeffs = np.array([p["mu_max_ref"], p["k_l"], p["k_i"], p["k_c"]])
            else:
                # YAML keys: n_max_ref, k_c, k_l, k_i → monod_haldane order: [ref, k_l, k_i, k_c]
                coeffs = np.array([p["n_max_ref"], p["k_l"], p["k_i"], p["k_c"]])

            logger.info(
                f"  Monod_Haldane: using fitting_plates.py params from {yaml_path.name}"
            )
            y_pred = model_func(L0, C0, coeffs)
            r2 = calculate_r2(y, y_pred)
            n_obs = len(y)
            n_params = len(coeffs)
            rss = np.sum((y - y_pred) ** 2)

            def predict(L0_new, C0_new, _coeffs=coeffs):
                return model_func(L0_new, C0_new, _coeffs)

            logger.info(
                f"  Monod_Haldane successful: R² = {r2:.4f} (identical to fitting_plates.py)"
            )
            return predict, coeffs, r2, n_obs, n_params, rss
        else:
            logger.warning(
                f"  YAML not found at {yaml_path}, falling back to generic p0"
            )
            initial_params = (
                [np.max(y), np.mean(L0), np.mean(L0) * 10, np.mean(C0)]
                if is_mu_max
                else [np.max(y) * 1.5, 0.2, 5.0, 0.3]
            )
            bounds = (
                ([0.01, 0.001, 0.001, 0.001], [1.0, 1000.0, 10000.0, 10.0])
                if is_mu_max
                else ([0, 0.001, 0.5, 0], [np.inf, 10.0, 100.0, 10.0])
            )
    else:
        initial_params = model["initial"](L0, C0, y)
        bounds = model["bounds"]()

    def model_to_fit(X, *params):
        L0_vals, C0_vals = X
        return model_func(L0_vals, C0_vals, np.array(params))

    try:
        popt, _ = curve_fit(
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
        mean_val = np.mean(y)
        n_obs = len(y)
        n_params = len(initial_params)
        rss = np.sum((y - mean_val) ** 2)

        def predict(L0_new, C0_new):
            return mean_val

        return predict, initial_params, 0.0, n_obs, n_params, rss


def plot_rsm_comparison(df, df_all, response_var, model_name, output_dir):
    """
    Generate a figure with 3 panels + equation to compare an RSM model.

    Parameters
    ----------
    df : DataFrame
        Calibration data (type == 'calibration')
    df_all : DataFrame
        All data (calibration + extrapolation)
    response_var : str
        'mu_max' or 'Nmax'
    model_name : str
        Model name to use
    output_dir : Path
        Output directory

    Returns
    -------
    r2_calib, r2_global, n_calib, n_global, n_params, rss_calib, rss_global
    """
    response_mean = f"{response_var}_mean"
    response_std = f"{response_var}_std"

    if response_var == "mu_max":
        param_latex = "\\mu_{\\mathrm{max}}"
        unit = "h$^{-1}$"
        cmap_choice = "viridis"
    else:
        param_latex = "N_{\\mathrm{max}}"
        unit = "cells/mL"
        cmap_choice = "plasma"

    predict_func, coeffs, r2_calib, n_calib, n_params, rss_calib = fit_rsm_surface(
        df, response_mean, model_name
    )

    y_all = df_all[response_mean].values
    y_pred_all = np.array(
        [predict_func(row["L0"], row["C0"]) for _, row in df_all.iterrows()]
    )
    r2_global = calculate_r2(y_all, y_pred_all)
    n_global = len(y_all)
    rss_global = np.sum((y_all - y_pred_all) ** 2)

    logger.info(
        f"  {model_name}, {response_var}: "
        f"R²_calib={r2_calib:.4f}, R²_global={r2_global:.4f}"
    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
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

    # --- Panel 1: Experimental data ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")

    L0_unique = sorted(df_all["L0"].unique())
    colors_L0 = plt.cm.winter(np.linspace(0, 1, len(L0_unique)))
    L0_to_color = {L0: colors_L0[i] for i, L0 in enumerate(L0_unique)}

    for L0_val in L0_unique:
        subset = df_all[df_all["L0"] == L0_val]
        for _, row in subset.iterrows():
            ax1.scatter(
                row["L0"],
                row["C0"],
                row[response_mean],
                c=[L0_to_color[L0_val]],
                s=50,
                alpha=0.8,
            )
            std_val = row[response_std]
            if not np.isnan(std_val) and std_val > 0:
                ax1.plot(
                    [row["L0"], row["L0"]],
                    [row["C0"], row["C0"]],
                    [row[response_mean] - std_val, row[response_mean] + std_val],
                    color=L0_to_color[L0_val],
                    linewidth=1.5,
                    alpha=0.6,
                )

    ax1.set_xlabel("$L_0$ (factor)", fontsize=11, labelpad=10)
    ax1.set_ylabel("$C_0$ (dilution factor)", fontsize=11, labelpad=10)
    ax1.set_zlabel(f"${param_latex}$ ({unit})", fontsize=11, labelpad=10)
    ax1.set_title("Experimental data", fontsize=12, pad=20)

    # --- Panel 2: RSM Surface ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    L0_min, L0_max = df_all["L0"].min(), df_all["L0"].max()
    C0_min, C0_max = df_all["C0"].min(), df_all["C0"].max()

    L0_range = np.linspace(L0_min, L0_max, 50)
    C0_range = np.linspace(C0_min, C0_max, 50)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    response_grid = np.zeros_like(L0_grid)
    for i in range(L0_grid.shape[0]):
        for j in range(L0_grid.shape[1]):
            response_grid[i, j] = predict_func(L0_grid[i, j], C0_grid[i, j])

    ax2.plot_surface(
        L0_grid,
        C0_grid,
        response_grid,
        cmap=cmap_choice,
        alpha=0.7,
        edgecolor="none",
        antialiased=True,
    )

    # Identify calibration vs extrapolation
    df_calib_keys = set(zip(df["L0"], df["C0"]))
    df_calibration = df_all[
        df_all.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
    ].copy()
    df_extrapolation = df_all[
        ~df_all.apply(lambda row: (row["L0"], row["C0"]) in df_calib_keys, axis=1)
    ].copy()

    logger.info(
        f"  {model_name}, {response_var}: "
        f"{len(df_calibration)} calibration, {len(df_extrapolation)} extrapolation"
    )

    for idx, row in df_calibration.iterrows():
        ax2.scatter(
            row["L0"],
            row["C0"],
            row[response_mean],
            color="black",
            s=50,
            marker="o",
            alpha=0.9,
            label="Calibration" if idx == df_calibration.index[0] else "",
        )
        std_val = row[response_std]
        if not np.isnan(std_val) and std_val > 0:
            ax2.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [row[response_mean] - std_val, row[response_mean] + std_val],
                color="black",
                linewidth=2,
                alpha=0.7,
            )

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
        std_val = row[response_std]
        if not np.isnan(std_val) and std_val > 0:
            ax2.plot(
                [row["L0"], row["L0"]],
                [row["C0"], row["C0"]],
                [row[response_mean] - std_val, row[response_mean] + std_val],
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

    if len(df_extrapolation) > 0:
        ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # --- Panel 3: Contour map ---
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
    ax3.clabel(contour_lines, inline=True, fontsize=8, fmt="%.3f")

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

    # --- Panel 4: Model equation ---
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

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

    filename = f"RSM_3D_{response_var}_{model_name}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {filename}")

    return r2_calib, r2_global, n_calib, n_global, n_params, rss_calib, rss_global


# ================================================================================
# MAIN FUNCTION
# ================================================================================


def main():
    logger.info("=" * 80)
    logger.info("COMPARISON OF RSM MODELS FOR μ_max AND N_max — PLATE 96-WELL")
    logger.info("L0 normalized: factor in [0, 1]  (1 = 170 µmol/m²/s)")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    base_dir = pathlib.Path(".")

    growth_rates_file = base_dir / "results_plates" / "plates_growth_rates_measured.csv"
    steady_states_file = (
        base_dir / "results_plates" / "plates_steady_states_measured.csv"
    )

    output_dir = pathlib.Path("surfaces_comparison") / "plate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    logger.info("\nLoading pre-processed plate data...")
    df_calib, df_all = load_plate_data(growth_rates_file, steady_states_file)

    # -------------------------------------------------------------------------
    # OVERVIEW FIGURE
    # -------------------------------------------------------------------------
    logger.info("\nGenerating data overview...")
    plot_logistic_fits_overview(df_all, output_dir)

    # -------------------------------------------------------------------------
    # TEST EACH MODEL
    # -------------------------------------------------------------------------
    logger.info("\nTesting all models...")
    logger.info("-" * 80)

    results_summary = []

    for model_name, model_info in MODEL_LIBRARY.items():
        logger.info(f"\nModel: {model_name} — {model_info['description']}")
        logger.info("-" * 80)

        logger.info("  Fitting mu_max...")
        (
            r2_mu_calib,
            r2_mu_global,
            n_mu_calib,
            n_mu_global,
            k_mu,
            rss_mu_calib,
            rss_mu_global,
        ) = plot_rsm_comparison(df_calib, df_all, "mu_max", model_name, output_dir)

        logger.info("  Fitting Nmax...")
        (
            r2_N_calib,
            r2_N_global,
            n_N_calib,
            n_N_global,
            k_N,
            rss_N_calib,
            rss_N_global,
        ) = plot_rsm_comparison(df_calib, df_all, "Nmax", model_name, output_dir)

        # AIC, AICc, BIC on calibration data
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
    logger.info(f"\nAll outputs saved in: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - logistic_fits_overview.png")
    for model_name in MODEL_LIBRARY.keys():
        logger.info(f"  - RSM_3D_mu_max_{model_name}.png")
        logger.info(f"  - RSM_3D_Nmax_{model_name}.png")
    logger.info("  - model_comparison_summary.csv")


if __name__ == "__main__":
    main()
