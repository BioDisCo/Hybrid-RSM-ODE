"""Hybrid RSM-ODE model applied to Kambe et al. (2022) photoautotrophic data.

Applies the same hybrid RSM-ODE framework used for our C. reinhardtii
mixotrophic data to the published Monoraphidium sp. photoautotrophic dataset.

Pipeline (mirrors fitting_erlen.py):
  1. Load 4 * 4 = 16 growth curves (4 light * 4 nutrient conditions)
     via data_import.read_csv_data_erlen_Kambe
  2. Fit logistic growth (with lag phase) to each curve individually
     -> per-condition (mu_max, N_max, t_lag, N0)
  3. Fit Double Monod RSM surfaces (same model as for our erlen data):
       mu_max(L0, C0) = beta0 * [L0/(L0 + K_L)] * [C0/(C0 + K_C)]
       N_max(L0, C0)  = beta0 * [L0/(L0 + K_L)] * [C0/(C0 + K_C)]

  4. Fit full time series from RSM-ODE for all 16 conditions
  5. Compute per-condition and global R^2
  6. Save plot and parameter CSVs

Light intensities (umol m^-2 s^-1): [96.8, 184.4, 386.7, 1034]
Nutrient factors (C0):              [1.0,  0.5,   0.25,  0.125]

Reference: Kambe et al. (2022), Algal Research.
"""

import os
import logging
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from data_import import read_csv_data_erlen_Kambe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

# -- constants -----------------------------------------------------------------

# Real light intensities (umol m^-2 s^-1).
# Derived from L0_factor * L0_REF_KAMBE where L0_factor=1 maps to the
# Highest light level: Ein=1034 umol m^-2 s^-1 (= 2.92 umol/s / 0.002826 m^2).
L0_REF_KAMBE = 1034.0  # umol m^-2 s^-1

DATA_FOLDER = "data_Kambe"
FILENAMES = [
    "data-Ein-0.274.csv",
    "data-Ein-0.521.csv",
    "data-Ein-1.09.csv",
    "data-Ein-2.92.csv",
]

OUTPUT_DIR = pathlib.Path("results_Monoraphidium_data")


# -- RSM model (Double Monod, identical to fitting_erlen.py) ------------------

class ModelConfig:
    """
    RSM surfaces (both Double Monod):
    - mu_max: beta0 * [L0/(L0 + K_L)] * [C0/(C0 + K_C)]
    - N_max:  beta0 * [L0/(L0 + K_L)] * [C0/(C0 + K_C)]
    L0 is normalised to factor space [0,1] before fitting.
    """

    @staticmethod
    def mu_max_model(L0_fac, C0, params):
        """Double Monod: beta0 * L0/(L0 + K_L) * C0/(C0 + K_C)."""
        return params[0] * (L0_fac / (L0_fac + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def mu_max_initial_params(L0_fac, C0, y):
        return [np.max(y) * 1.2, np.median(L0_fac), np.median(C0)]

    @staticmethod
    def mu_max_bounds():
        # K_L in factor space; K_C in [0, 1]
        return ([0, 0.01, 0.001], [np.inf, 5.0, 5.0])

    @staticmethod
    def mu_max_latex(params):
        K_L_real = params[1] * L0_REF_KAMBE
        return (f"$\\mu_{{\\max}} = {params[0]:.3e} \\times "
                f"\\frac{{L_0}}{{L_0 + {K_L_real:.0f}\\,\\mu mol\\,m^{{-2}}s^{{-1}}}} \\times "
                f"\\frac{{C_0}}{{C_0 + {params[2]:.3f}}}$")

    @staticmethod
    def Nmax_model(L0_fac, C0, params):
        return params[0] * (L0_fac / (L0_fac + params[1])) * (C0 / (C0 + params[2]))

    @staticmethod
    def Nmax_initial_params(L0_fac, C0, y):
        return [np.max(y) * 1.2, np.median(L0_fac), np.median(C0)]

    @staticmethod
    def Nmax_bounds():
        # K_L in factor space [0, 1]; K_C in [0, 1]
        return ([0, 0.01, 0.001], [np.inf, 5.0, 5.0])

    @staticmethod
    def Nmax_latex(params):
        K_L_real = params[1] * L0_REF_KAMBE
        return (f"$N_{{\\max}} = {params[0]:.3e} \\times "
                f"\\frac{{L_0}}{{L_0 + {K_L_real:.0f}}} \\times "
                f"\\frac{{C_0}}{{C_0 + {params[2]:.3f}}}$")


# -- logistic growth model -----------------------------------------------------

def logistic_growth(t, N0, Nmax, mu_max, t_lag):
    """Logistic growth with lag phase."""
    return Nmax / (1 + ((Nmax - N0) / N0) * np.exp(-mu_max * (t - t_lag)))


def logistic_growth_rsm(t, N0, L0, C0, coeffs_mu_max, coeffs_Nmax, t_lag):
    """Logistic ODE driven by RSM-predicted mu_max and N_max (Double Monod).
    L0 is in real umol/m^2/s; normalised internally to factor space for RSM."""
    L0_fac = L0 / L0_REF_KAMBE
    mu_max = ModelConfig.mu_max_model(L0_fac, C0, coeffs_mu_max)
    Nmax   = ModelConfig.Nmax_model(L0_fac, C0, coeffs_Nmax)
    return logistic_growth(t, N0, Nmax, mu_max, t_lag)


# -- R^2 -----------------------------------------------------------------------

def calculate_r2(y_true, y_pred, x_true=None, x_pred=None):
    """Robust R^2 - handles differing lengths via interpolation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if x_true is None and x_pred is None and len(y_true) == len(y_pred):
        y_t, y_p = y_true, y_pred
    else:
        if x_true is None:
            x_true = np.linspace(0.0, 1.0, len(y_true))
        else:
            x_true = np.asarray(x_true, dtype=float)
        if x_pred is None:
            x_pred = np.linspace(0.0, 1.0, len(y_pred))
        else:
            x_pred = np.asarray(x_pred, dtype=float)

        xmin = max(np.min(x_true), np.min(x_pred))
        xmax = min(np.max(x_true), np.max(x_pred))
        mask = (x_true >= xmin) & (x_true <= xmax)
        if np.sum(mask) < 2:
            return 0.0
        x_t_common = x_true[mask]
        y_t_common = y_true[mask]
        try:
            f = interp1d(x_pred, y_pred, kind='linear', bounds_error=True)
            y_p_common = f(x_t_common)
        except ValueError:
            return 0.0
        y_t, y_p = y_t_common, y_p_common

    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if np.sum(valid) < 2:
        return 0.0
    y_t, y_p = y_t[valid], y_p[valid]
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# -- data loading --------------------------------------------------------------

def load_kambe_data(data_folder=DATA_FOLDER):
    """
    Load all 16 Kambe conditions via read_csv_data_erlen_Kambe.

    Returns a list of records, each with:
        L0   - real light intensity (umol m^-2 s^-1)
        C0   - dimensionless nutrient factor [0.125, 0.25, 0.5, 1.0]
        time - np.ndarray (hours)
        od   - np.ndarray (OD730, mean across replicates)
    """
    records = []
    for fname in FILENAMES:
        filepath = os.path.join(data_folder, fname)
        experiments = read_csv_data_erlen_Kambe(filepath)
        for exp in experiments.values():
            records.append({
                'L0':  exp['L0_factor'] * L0_REF_KAMBE,
                'C0':  exp['C0_factor'],
                'time': np.array(exp['Time']),
                'od':   np.array(exp['Mean']),
            })
    logger.info(f"Loaded {len(records)} conditions from {data_folder}/")
    return records


# -- individual logistic fits per condition ------------------------------------

def estimate_growth_parameters(time, od):
    """
    Fit logistic growth (with lag) to a single OD curve.
    Returns dict {N0, Nmax, mu_max, t_lag, r_squared} or None on failure.
    Mirrors estimate_growth_parameters() from fitting_erlen.py.
    """
    time = np.asarray(time)
    od   = np.asarray(od)
    valid = ~np.isnan(od) & (od > 0)
    t, y  = time[valid], od[valid]
    if len(t) < 4:
        return None

    N0_guess   = y[0]
    Nmax_guess = np.max(y)

    # Rough mu_max from log-linear phase
    growth_rates = []
    for i in range(1, len(y) - 1):
        if N0_guess * 1.5 < y[i] < Nmax_guess * 0.7:
            dt = t[i + 1] - t[i - 1]
            if dt > 0:
                growth_rates.append((np.log(y[i + 1]) - np.log(y[i - 1])) / dt)
    mu_guess = max(growth_rates) if growth_rates else 0.05

    p0 = [N0_guess, Nmax_guess,   mu_guess,       t[0]]
    lb = [N0_guess * 0.5, Nmax_guess * 0.7, 1e-4, 0]
    ub = [N0_guess * 2.0, Nmax_guess * 2.5, 1.0,  t[-1] * 0.3]

    try:
        popt, _ = curve_fit(
            logistic_growth, t, y, p0=p0, bounds=(lb, ub), maxfev=5000
        )
        N0_fit, Nmax_fit, mu_max_fit, t_lag_fit = popt
        r2 = calculate_r2(y, logistic_growth(t, *popt))
        return dict(N0=N0_fit, Nmax=Nmax_fit, mu_max=mu_max_fit, t_lag=t_lag_fit,
                    r_squared=r2)
    except Exception as e:
        logger.warning(f"Fitting failed: {e}")
        return None


def fit_all_conditions(records):
    """
    Run individual logistic fits for all 16 conditions.
    Adds 'fit' key to each record. Returns (records, df_params).
    df_params columns: L0, C0, mu_max, Nmax, t_lag, N0, r_squared_individual.
    """
    rows = []
    for rec in records:
        fit = estimate_growth_parameters(rec['time'], rec['od'])
        rec['fit'] = fit
        if fit is not None:
            rows.append({
                'L0':    rec['L0'],
                'C0':    rec['C0'],
                'mu_max': fit['mu_max'],
                'Nmax':   fit['Nmax'],
                't_lag':  fit['t_lag'],
                'N0':     fit['N0'],
                'r_squared_individual': fit['r_squared'],
            })
            logger.info(
                f"  L0={rec['L0']:7.1f} umol/m^2/s  C0={rec['C0']:.4f}  "
                f"mu_max={fit['mu_max']:.4f} h^-1  Nmax={fit['Nmax']:.3f} OD  "
                f"t_lag={fit['t_lag']:.1f} h  R^2={fit['r_squared']:.3f}"
            )
        else:
            logger.warning(f"  L0={rec['L0']:.1f}  C0={rec['C0']:.4f}: individual fit FAILED")

    return records, pd.DataFrame(rows)


# -- RSM surface fitting -------------------------------------------------------

def fit_rsm_mu_max(df_params):
    """
    Fit Double Monod to mu_max: beta0 * [L0_fac/(L0_fac+K_L)] * [C0/(C0+K_C)].
    L0 is normalised to [0,1] factors before fitting. Returns [beta0, K_L, K_C]
    in factor space.
    """
    L0_fac = df_params['L0'].values / L0_REF_KAMBE
    C0     = df_params['C0'].values
    y      = df_params['mu_max'].values
    valid  = np.isfinite(y) & (y > 0)
    L0v, C0v, yv = L0_fac[valid], C0[valid], y[valid]

    def model(XY, beta0, K_L, K_C):
        L, C = XY
        return beta0 * (L / (L + K_L)) * (C / (C + K_C))

    lb, ub = ModelConfig.mu_max_bounds()
    popt, _ = curve_fit(
        model, (L0v, C0v), yv,
        p0=ModelConfig.mu_max_initial_params(L0v, C0v, yv),
        bounds=(lb, ub),
        maxfev=10000,
    )
    return popt


def fit_rsm_nmax(df_params):
    """
    Fit Double Monod surface to Nmax: beta0 * [L0_fac/(L0_fac+K_L)] * [C0/(C0+K_C)].
    L0 is normalised to [0,1] factors before fitting. Returns [beta0, K_L, K_C]
    in factor space.
    """
    L0_fac = df_params['L0'].values / L0_REF_KAMBE
    C0     = df_params['C0'].values
    y      = df_params['Nmax'].values
    valid  = np.isfinite(y) & (y > 0)
    L0v, C0v, yv = L0_fac[valid], C0[valid], y[valid]

    def model(XY, beta0, K_L, K_C):
        L, C = XY
        return beta0 * (L / (L + K_L)) * (C / (C + K_C))

    lb, ub = ModelConfig.Nmax_bounds()
    popt, _ = curve_fit(
        model, (L0v, C0v), yv,
        p0=ModelConfig.Nmax_initial_params(L0v, C0v, yv),
        bounds=(lb, ub),
        maxfev=10000,
    )
    return popt


# -- R^2_param (parameter-space R^2) ------------------------------------------

def compute_r2_param(df_params, coeffs_mu_max, coeffs_Nmax):
    """
    R^2_param: R^2 in parameter space (paper's Methods).

    Compares per-condition fitted parameters (theta_bar, from individual logistic fits)
    against RSM surface predictions (theta_hat) at the same (L0, C0) conditions:

        R^2_param = 1 - sum(theta_bar - theta_hat)^2 / sum(theta_bar - mean(theta_bar))^2

    Computed separately for mu_max and Nmax.
    Returns dict {'r2_param_mu_max': float, 'r2_param_Nmax': float}.
    """
    L0_fac = df_params['L0'].values / L0_REF_KAMBE
    C0     = df_params['C0'].values

    # --- mu_max (Double Monod surface) ----------------------------------------
    theta_bar_mu = df_params['mu_max'].values
    theta_hat_mu = ModelConfig.mu_max_model(L0_fac, C0, coeffs_mu_max)
    ss_res_mu = np.sum((theta_bar_mu - theta_hat_mu) ** 2)
    ss_tot_mu = np.sum((theta_bar_mu - np.mean(theta_bar_mu)) ** 2)
    r2_mu = float(1.0 - ss_res_mu / ss_tot_mu) if ss_tot_mu > 1e-12 else 0.0

    # --- Nmax (Double Monod surface) ------------------------------------------
    theta_bar_nmax = df_params['Nmax'].values
    theta_hat_nmax = ModelConfig.Nmax_model(L0_fac, C0, coeffs_Nmax)
    ss_res_nmax = np.sum((theta_bar_nmax - theta_hat_nmax) ** 2)
    ss_tot_nmax = np.sum((theta_bar_nmax - np.mean(theta_bar_nmax)) ** 2)
    r2_nmax = float(1.0 - ss_res_nmax / ss_tot_nmax) if ss_tot_nmax > 1e-12 else 0.0

    return {'r2_param_mu_max': r2_mu, 'r2_param_Nmax': r2_nmax}


# -- RSM-ODE predictions -------------------------------------------------------

def compute_rsm_predictions(records, coeffs_mu_max, coeffs_Nmax):
    """
    For each condition, predict OD using RSM-ODE and compute R^2.
    t_lag and N0 are taken from the individual fits (same convention as
    fitting_erlen.py: t_lag is fixed at the per-condition empirical value).

    Returns (df_r2, global_r2).
    """
    rows = []
    all_exp, all_pred = [], []

    for rec in records:
        if rec['fit'] is None:
            continue
        L0, C0   = rec['L0'], rec['C0']
        t_lag    = rec['fit']['t_lag']
        N0       = rec['fit']['N0']
        t_exp    = rec['time']
        od_exp   = rec['od']

        valid = ~np.isnan(od_exp) & (od_exp > 0)
        t_clean, od_clean = t_exp[valid], od_exp[valid]
        if len(t_clean) < 2:
            continue

        od_rsm = logistic_growth_rsm(t_clean, N0, L0, C0, coeffs_mu_max, coeffs_Nmax, t_lag)
        r2 = calculate_r2(od_clean, od_rsm)

        all_exp.extend(od_clean)
        all_pred.extend(od_rsm)

        rows.append({'L0': L0, 'C0': C0, 'R2_rsm': r2})
        logger.info(f"  L0={L0:7.1f}  C0={C0:.4f}  R^2(RSM-ODE) = {r2:.4f}")

    global_r2 = calculate_r2(np.array(all_exp), np.array(all_pred))
    df_r2 = pd.DataFrame(rows)

    n_pts = len(all_exp)
    passing = df_r2[df_r2['R2_rsm'] >= 0.85]
    n_good = len(passing)
    logger.info(f"\nGlobal R^2 (RSM-ODE, {n_pts} data points): {global_r2:.4f}")
    logger.info(f"Conditions with R^2 >= 0.85: {n_good}/{len(df_r2)}")
    for _, row in passing.iterrows():
        logger.info(f"  PASS  L0={row['L0']:7.1f}  C0={row['C0']:.4f}  R^2={row['R2_rsm']:.4f}")

    return df_r2, global_r2


# -- visualization -------------------------------------------------------------

COLORS_C0 = {
    1.0000: 'mediumseagreen',
    0.5000: 'grey',
    0.2500: 'tomato',
    0.1250: 'teal',
}


def plot_growth_curves(records, coeffs_mu_max, coeffs_Nmax, df_r2, global_r2, output_dir):
    """
    4-panel figure (one panel per L0): experimental data vs RSM-ODE prediction.
    """
    L0_values = sorted(set(r['L0'] for r in records))
    C0_values = sorted(set(r['C0'] for r in records), reverse=True)

    fig, axes = plt.subplots(1, len(L0_values), figsize=(5 * len(L0_values), 5))
    fig.suptitle(
        r'RSM-ODE fits ($\mathit{Monoraphidium\ sp.}$)'
        f'  |  Global $R^2$ = {global_r2:.3f}',
        fontsize=12, fontweight='bold'
    )

    t_sim = np.linspace(0, 1300, 500)

    for ax, L0 in zip(axes, L0_values):
        for C0 in C0_values:
            color = COLORS_C0.get(C0, 'grey')

            rec = next(
                (r for r in records if abs(r['L0'] - L0) < 5 and abs(r['C0'] - C0) < 0.01),
                None
            )
            if rec is None:
                continue

            # Experimental
            valid = ~np.isnan(rec['od']) & (rec['od'] > 0)
            ax.plot(rec['time'][valid], rec['od'][valid],
                    'o', color=color, markersize=4, alpha=0.7, zorder=5)

            # RSM-ODE prediction
            if rec['fit'] is not None:
                od_rsm = logistic_growth_rsm(
                    t_sim, rec['fit']['N0'], L0, C0,
                    coeffs_mu_max, coeffs_Nmax, rec['fit']['t_lag']
                )
                r2_row = df_r2[(abs(df_r2['L0'] - L0) < 5) & (abs(df_r2['C0'] - C0) < 0.01)]
                r2_val = r2_row['R2_rsm'].values[0] if len(r2_row) > 0 else np.nan
                ax.plot(t_sim, od_rsm, '-', color=color, linewidth=2,
                        label=f'$C_0={C0:.3f}$  $R^2={r2_val:.2f}$')

        ax.set_title(f'$L_0 = {L0:.0f}\\;\\mu\\mathrm{{mol}}_{{h\\nu}}\\;\\mathrm{{m}}^{{-2}}\\;\\mathrm{{s}}^{{-1}}$',
                     fontsize=11)
        ax.set_xlabel('Time (h)')
        ax.set_xlim([0, 1300])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper left')

    axes[0].set_ylabel('OD (730 nm)')
    plt.tight_layout()

    out_path = output_dir / 'hybrid_rsm_ode_Monoraphidium_data.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved: {out_path}")
    return fig


def plot_individual_fits(records, output_dir):
    """
    4-panel figure showing individual per-condition logistic fits vs data.
    Uses the locally fitted (mu_max, N_max, t_lag, N0) - no RSM involved.
    """
    L0_values = sorted(set(r['L0'] for r in records))
    C0_values = sorted(set(r['C0'] for r in records), reverse=True)

    fig, axes = plt.subplots(1, len(L0_values), figsize=(5 * len(L0_values), 5))
    fig.suptitle(
        r'Individual logistic fits ($\mathit{Monoraphidium\ sp.}$)',
        fontsize=12, fontweight='bold'
    )

    t_sim = np.linspace(0, 1300, 500)

    for ax, L0 in zip(axes, L0_values):
        for C0 in C0_values:
            color = COLORS_C0.get(C0, 'grey')

            rec = next(
                (r for r in records if abs(r['L0'] - L0) < 5 and abs(r['C0'] - C0) < 0.01),
                None
            )
            if rec is None:
                continue

            # Experimental
            valid = ~np.isnan(rec['od']) & (rec['od'] > 0)
            ax.plot(rec['time'][valid], rec['od'][valid],
                    'o', color=color, markersize=4, alpha=0.7, zorder=5)

            # Individual logistic fit
            if rec['fit'] is not None:
                f = rec['fit']
                od_local = logistic_growth(t_sim, f['N0'], f['Nmax'], f['mu_max'], f['t_lag'])
                ax.plot(t_sim, od_local, '-', color=color, linewidth=2,
                        label=f"$C_0={C0:.3f}$  $R^2={f['r_squared']:.2f}$")

        ax.set_title(f'$L_0 = {L0:.0f}\\;\\mu\\mathrm{{mol}}_{{h\\nu}}\\;\\mathrm{{m}}^{{-2}}\\;\\mathrm{{s}}^{{-1}}$',
                     fontsize=11)
        ax.set_xlabel('Time (h)')
        ax.set_xlim([0, 1300])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='upper left')

    axes[0].set_ylabel('OD (730 nm)')
    plt.tight_layout()

    out_path = output_dir / 'individual_logistic_fits_Monoraphidium_data.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved: {out_path}")
    return fig


def plot_rsm_surfaces(df_params, coeffs_mu_max, coeffs_Nmax, output_dir):
    """
    2-panel 3D surface plot for mu_max and N_max RSM surfaces.
    Equation style matches fitting_erlen.py RSM_3D figures.
    """
    from matplotlib.gridspec import GridSpec

    # Grids in factor space for evaluation, real umol/m^2/s for display
    L0_real_grid = np.linspace(50, 1100, 60)
    L0_fac_grid  = L0_real_grid / L0_REF_KAMBE
    C0_grid      = np.linspace(0.05, 1.1, 60)
    LL_real, CC  = np.meshgrid(L0_real_grid, C0_grid)
    LL_fac,  _   = np.meshgrid(L0_fac_grid,  C0_grid)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[5, 1],
                  hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax_eq1 = fig.add_subplot(gs[1, 0])
    ax_eq2 = fig.add_subplot(gs[1, 1])

    # Axis label strings reused in both panels
    xlabel_latex = r'$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$'
    ylabel_latex = r'$C_0$'

    # --- mu_max panel ---------------------------------------------------------
    ZZ_mu = ModelConfig.mu_max_model(LL_fac, CC, coeffs_mu_max)
    ax1.plot_surface(LL_real, CC, ZZ_mu, alpha=0.7, cmap='viridis')
    ax1.scatter(df_params['L0'], df_params['C0'], df_params['mu_max'],
                c='black', s=40, zorder=6)
    ax1.set_xlabel(xlabel_latex, labelpad=10)
    ax1.set_ylabel(ylabel_latex, labelpad=6)
    ax1.set_zlabel(r'$\mu_{\max}\ (\mathrm{h}^{-1})$', labelpad=6)
    ax1.set_title(r'$\mu_{\max}\ (\mathrm{h}^{-1})$', fontsize=11)

    # Equation — style from fitting_erlen.py (wheat box, \cdot, :.3e)
    K_L_mu_real = coeffs_mu_max[1] * L0_REF_KAMBE
    eq_mu = (r"$\mu_{\mathrm{max}}$ "
             r"$= " + f"{coeffs_mu_max[0]:.3e}" +
             r" \cdot \frac{L_0}{L_0 + " + f"{K_L_mu_real:.3e}" +
             r"} \cdot \frac{C_0}{C_0 + " + f"{coeffs_mu_max[2]:.3e}" +
             r"}$")
    ax_eq1.axis('off')
    ax_eq1.text(0.5, 0.5, eq_mu, fontsize=10, ha='center', va='center',
                transform=ax_eq1.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8))

    # --- N_max panel ----------------------------------------------------------
    ZZ_nmax = ModelConfig.Nmax_model(LL_fac, CC, coeffs_Nmax)
    ax2.plot_surface(LL_real, CC, ZZ_nmax, alpha=0.7, cmap='plasma')
    ax2.scatter(df_params['L0'], df_params['C0'], df_params['Nmax'],
                c='black', s=40, zorder=6)
    ax2.set_xlabel(xlabel_latex, labelpad=10)
    ax2.set_ylabel(ylabel_latex, labelpad=6)
    ax2.set_zlabel(r'$N_{\max}\ (\mathrm{OD}\ (730\ \mathrm{nm}))$', labelpad=6)
    ax2.set_title(r'$N_{\max}\ (\mathrm{OD}\ (730\ \mathrm{nm}))$', fontsize=11)

    # Equation — style from fitting_erlen.py (wheat box, \cdot, :.3e)
    K_L_nmax_real = coeffs_Nmax[1] * L0_REF_KAMBE
    eq_nmax = (r"$N_{\mathrm{max}}$ "
               r"$= " + f"{coeffs_Nmax[0]:.3e}" +
               r" \cdot \frac{L_0}{L_0 + " + f"{K_L_nmax_real:.3e}" +
               r"} \cdot \frac{C_0}{C_0 + " + f"{coeffs_Nmax[2]:.3e}" +
               r"}$")
    ax_eq2.axis('off')
    ax_eq2.text(0.5, 0.5, eq_nmax, fontsize=10, ha='center', va='center',
                transform=ax_eq2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8))

    out_path = output_dir / 'rsm_surfaces.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved: {out_path}")
    return fig


# -- main ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("HYBRID RSM-ODE - KAMBE ET AL. (2022) PHOTOAUTOTROPHIC DATA")
    logger.info("Monoraphidium sp.   BG-11 medium   4 * 4 = 16 conditions")
    logger.info("=" * 70)

    # --- Step 1: Load --------------------------------------------------------
    logger.info("\n--- Step 1: Loading Kambe data ---")
    records = load_kambe_data()

    # --- Step 2: Individual logistic fits ------------------------------------
    logger.info("\n--- Step 2: Individual logistic fits per condition ---")
    records, df_params = fit_all_conditions(records)
    df_params.to_csv(OUTPUT_DIR / 'individual_fits_Monoraphidium_data.csv', index=False, sep=';')
    logger.info(f"  -> {len(df_params)} successful fits  "
                f"(mean R^2={df_params['r_squared_individual'].mean():.3f})")

    # --- Step 3: RSM surface fitting -----------------------------------------
    logger.info("\n--- Step 3: Fitting RSM surfaces ---")
    logger.info("  mu_max: Double Monod  beta0 * L0/(L0+K_L) * C0/(C0+K_C)")
    logger.info("  N_max:  Double Monod  beta0 * L0/(L0+K_L) * C0/(C0+K_C)")
    coeffs_mu_max = fit_rsm_mu_max(df_params)
    coeffs_Nmax   = fit_rsm_nmax(df_params)
    logger.info(f"  mu_max: {ModelConfig.mu_max_latex(coeffs_mu_max)}")
    logger.info(f"  N_max: {ModelConfig.Nmax_latex(coeffs_Nmax)}")

    r2_param = compute_r2_param(df_params, coeffs_mu_max, coeffs_Nmax)
    logger.info(f"  R^2_param (mu_max): {r2_param['r2_param_mu_max']:.4f}")
    logger.info(f"  R^2_param (N_max): {r2_param['r2_param_Nmax']:.4f}")

    # --- Step 4: RSM-ODE predictions and R^2 ----------------------------------
    logger.info("\n--- Step 4: RSM-ODE predictions ---")
    df_r2, global_r2 = compute_rsm_predictions(records, coeffs_mu_max, coeffs_Nmax)
    df_r2.to_csv(OUTPUT_DIR / 'r2_rsm_ode_Monoraphidium_data.csv', index=False, sep=';')

    # --- Save RSM parameters -------------------------------------------------
    df_rsm_mu = pd.DataFrame({
        'parameter': ['beta0', 'K_L (umol/m^2/s)', 'K_C'],
        'mu_max': coeffs_mu_max,
    })
    df_rsm_nmax = pd.DataFrame({
        'parameter': ['beta0', 'K_L (umol/m^2/s)', 'K_C'],
        'Nmax': coeffs_Nmax,
    })
    pd.concat([df_rsm_mu, df_rsm_nmax], axis=0).to_csv(
        OUTPUT_DIR / 'rsm_surface_params_Monoraphidium_data.csv', index=False, sep=';'
    )
    logger.info(f"  RSM parameters saved to {OUTPUT_DIR}/rsm_surface_params_Monoraphidium_data.csv")

    # --- Step 5: Figures -----------------------------------------------------
    logger.info("\n--- Step 5: Generating figures ---")
    plot_individual_fits(records, OUTPUT_DIR)
    plot_growth_curves(records, coeffs_mu_max, coeffs_Nmax, df_r2, global_r2, OUTPUT_DIR)
    plot_rsm_surfaces(df_params, coeffs_mu_max, coeffs_Nmax, OUTPUT_DIR)

    # --- Save R^2_param --------------------------------------------------------
    pd.DataFrame([{
        'metric': 'R2_param_mu_max',
        'value':  r2_param['r2_param_mu_max'],
        'description': 'R^2 in parameter space (mu_max): local fits vs RSM surface',
    }, {
        'metric': 'R2_param_Nmax',
        'value':  r2_param['r2_param_Nmax'],
        'description': 'R^2 in parameter space (Nmax): local fits vs RSM surface',
    }, {
        'metric': 'R2_global_RSM_ODE',
        'value':  global_r2,
        'description': 'Global R^2 on OD time series (RSM-ODE vs data, all conditions)',
    }]).to_csv(OUTPUT_DIR / 'r2_summary_Monoraphidium_data.csv', index=False, sep=';')

    logger.info("\n" + "=" * 70)
    logger.info(f"R^2_param (mu_max):             {r2_param['r2_param_mu_max']:.4f}")
    logger.info(f"R^2_param (N_max):             {r2_param['r2_param_Nmax']:.4f}")
    logger.info(f"GLOBAL R^2 (RSM-ODE, all conditions): {global_r2:.4f}")
    logger.info(f"Results in: {OUTPUT_DIR}/")
    logger.info("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
