"""Scaling analysis between plate and erlen RSM surfaces.

Loads fitted surface parameters from YAML files and generates comparison plots:
1. Superimposed 3D surfaces for mu_max and N_max
2. N_max scaling analysis (multiplicative constant)
3. mu_max scaling analysis (light/nutrient coordinate transformation)

Models used:
- Erlen: Double Monod (3 params)
    f(L0, C0) = f_ref × [L0/(L0 + k_l)] × [C0/(C0 + k_c)]
    L0 in µmol/m²/s (actual values)

- Plates: Haldane with photoinhibition (4 params)
    f(C, L) = f_ref × [C/(k_c + C)] × [L/(k_l + L + L²/k_i)]
    L0 in factors (0-1 range, multiply by L0_REF=170 for actual values)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize_scalar, minimize
import yaml
import pathlib
import logging
import matplotlib as mpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = pathlib.Path("results_scaling")
OUTPUT_DIR.mkdir(exist_ok=True)

PLATE_DIR = pathlib.Path("results_plates")
ERLEN_DIR = pathlib.Path("results_erlen")

# Grid resolution for surface plots
N_GRID = 50

# Reference light intensity for plates (to convert factors to actual values)
L0_REF = 170.0  # µmol/m²/s

# Domain for plotting (in actual L0 units for erlen compatibility)
C0_MIN, C0_MAX = 0.0625, 1.0
L0_MIN, L0_MAX = 11.9, 170.0  # µmol/m²/s


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def double_monod_erlen(L0, C0, f_ref, k_l, k_c):
    """
    Double Monod model for ERLEN data.

    Formula: f = f_ref × [L0/(L0 + k_l)] × [C0/(C0 + k_c)]

    Note: Erlen model has L0 as first argument (different from plates!)

    Parameters:
    -----------
    L0 : array-like
        Light intensity in µmol/m²/s (actual values, not factors)
    C0 : array-like
        Nutrient concentration factor
    f_ref : float
        Reference maximum value (mu_max_ref or n_max_ref)
    k_l : float
        Half-saturation constant for light (in µmol/m²/s)
    k_c : float
        Half-saturation constant for nutrients

    Returns:
    --------
    array-like : Model prediction
    """
    return f_ref * (L0 / (L0 + k_l)) * (C0 / (C0 + k_c))


def haldane_plate(C0, L0_factor, f_ref, k_c, k_l, k_i):
    """
    Haldane model with photoinhibition for PLATE data.

    Formula: f = f_ref × [C/(k_c + C)] × [L/(k_l + L + L²/k_i)]

    Note: Plates use L0 as factors (0-1), not actual values!

    Parameters:
    -----------
    C0 : array-like
        Nutrient concentration factor
    L0_factor : array-like
        Light intensity as factor (0-1 range)
    f_ref : float
        Reference maximum value (mu_max_ref or n_max_ref)
    k_c : float
        Half-saturation constant for nutrients
    k_l : float
        Half-saturation constant for light (in factor units)
    k_i : float
        Photoinhibition constant (higher = less inhibition)

    Returns:
    --------
    array-like : Model prediction
    """
    L = L0_factor
    return f_ref * (C0 / (k_c + C0)) * (L / (k_l + L + L**2 / k_i))


# ============================================================================
# LOAD PARAMETERS FROM YAML
# ============================================================================
def load_parameters():
    """Load RSM surface parameters from YAML files."""

    # Plate parameters
    plate_gr_path = PLATE_DIR / "parameters_growth_rate.yaml"
    plate_ss_path = PLATE_DIR / "parameters_steady_state.yaml"

    # Erlen parameters
    erlen_gr_path = ERLEN_DIR / "parameters_growth_rate.yaml"
    erlen_ss_path = ERLEN_DIR / "parameters_steady_state.yaml"

    params = {}

    # Load plate growth rate
    with open(plate_gr_path) as f:
        params['plate_gr'] = yaml.safe_load(f)
    logger.info(f"Loaded plate growth rate params from {plate_gr_path}")

    # Load plate steady state
    with open(plate_ss_path) as f:
        params['plate_ss'] = yaml.safe_load(f)
    logger.info(f"Loaded plate steady state params from {plate_ss_path}")

    # Load erlen growth rate
    with open(erlen_gr_path) as f:
        params['erlen_gr'] = yaml.safe_load(f)
    logger.info(f"Loaded erlen growth rate params from {erlen_gr_path}")

    # Load erlen steady state
    with open(erlen_ss_path) as f:
        params['erlen_ss'] = yaml.safe_load(f)
    logger.info(f"Loaded erlen steady state params from {erlen_ss_path}")

    return params


def compute_surfaces(params, C0_grid, L0_grid):
    """
    Compute all surfaces using the appropriate models.

    Parameters:
    -----------
    params : dict
        Loaded parameters
    C0_grid : ndarray
        2D grid of C0 values
    L0_grid : ndarray
        2D grid of L0 values in µmol/m²/s (actual values)

    Returns:
    --------
    dict : Dictionary with computed surfaces
    """
    # Convert L0 to factors for plate model
    L0_factor_grid = L0_grid / L0_REF

    surfaces = {}

    # ===== ERLEN surfaces (Double Monod) =====
    # Note: erlen model uses (L0, C0) order
    surfaces['mu_erlen'] = double_monod_erlen(
        L0_grid, C0_grid,
        params['erlen_gr']['mu_max_ref'],
        params['erlen_gr']['k_l'],
        params['erlen_gr']['k_c']
    )

    surfaces['n_erlen'] = double_monod_erlen(
        L0_grid, C0_grid,
        params['erlen_ss']['n_max_ref'],
        params['erlen_ss']['k_l'],
        params['erlen_ss']['k_c']
    )

    # ===== PLATE surfaces (Haldane with photoinhibition) =====
    # Check if k_i exists in params (Haldane model)
    if 'k_i' in params['plate_gr']:
        surfaces['mu_plate'] = haldane_plate(
            C0_grid, L0_factor_grid,
            params['plate_gr']['mu_max_ref'],
            params['plate_gr']['k_c'],
            params['plate_gr']['k_l'],
            params['plate_gr']['k_i']
        )
    else:
        # Fallback to simple double Monod if k_i not available
        logger.warning("k_i not found for plate growth rate, using simple Monod")
        surfaces['mu_plate'] = params['plate_gr']['mu_max_ref'] * \
            (C0_grid / (params['plate_gr']['k_c'] + C0_grid)) * \
            (L0_factor_grid / (params['plate_gr']['k_l'] + L0_factor_grid))

    if 'k_i' in params['plate_ss']:
        surfaces['n_plate'] = haldane_plate(
            C0_grid, L0_factor_grid,
            params['plate_ss']['n_max_ref'],
            params['plate_ss']['k_c'],
            params['plate_ss']['k_l'],
            params['plate_ss']['k_i']
        )
    else:
        # Fallback to simple double Monod if k_i not available
        logger.warning("k_i not found for plate steady state, using simple Monod")
        surfaces['n_plate'] = params['plate_ss']['n_max_ref'] * \
            (C0_grid / (params['plate_ss']['k_c'] + C0_grid)) * \
            (L0_factor_grid / (params['plate_ss']['k_l'] + L0_factor_grid))

    return surfaces


# ============================================================================
# FIGURE 1: SUPERIMPOSED 3D SURFACES (mu_max and N_max)
# ============================================================================
def plot_superimposed_surfaces(params):
    """
    Create a figure with 2 3D subplots:
    - Left: mu_max surfaces (plate vs erlen)
    - Right: N_max surfaces (plate vs erlen)
    """

    # Create meshgrid (L0 in actual values) - same order as fitting_erlen.py
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # Compute all surfaces
    surfaces = compute_surfaces(params, C0_grid, L0_grid)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 7))

    # ===== Subplot 1: mu_max =====
    ax1 = fig.add_subplot(121, projection='3d')

    # Plate surface
    ax1.plot_surface(
        L0_grid, C0_grid, surfaces['mu_plate'],
        color='turquoise', alpha=0.6, edgecolor='none'
    )

    # Erlen surface
    ax1.plot_surface(
        L0_grid, C0_grid, surfaces['mu_erlen'],
        color='seagreen', alpha=0.6, edgecolor='none'
    )

    ax1.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=12, labelpad=10)
    ax1.set_ylabel('$C_0$', fontsize=12, labelpad=10)
    ax1.set_zlabel('$\\mu_{max}$ (h⁻¹)', fontsize=12, labelpad=10)
    ax1.set_title('Growth Rate $\\mu_{max}(C_0, L_0)$\nPlate: Haldane | Erlen: Double Monod', fontsize=12, pad=20)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='turquoise', alpha=0.6, label='Plate (Haldane)'),
        Patch(facecolor='seagreen', alpha=0.6, label='Erlen (Monod)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # ===== Subplot 2: N_max =====
    ax2 = fig.add_subplot(122, projection='3d')

    # Plate surface
    ax2.plot_surface(
        L0_grid, C0_grid, surfaces['n_plate'],
        color='darkviolet', alpha=0.6, edgecolor='none'
    )

    # Erlen surface
    ax2.plot_surface(
        L0_grid, C0_grid, surfaces['n_erlen'],
        color='orange', alpha=0.6, edgecolor='none'
    )

    ax2.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=12, labelpad=10)
    ax2.set_ylabel('$C_0$', fontsize=12, labelpad=10)
    ax2.set_zlabel('$N_{max}$ (cells/mL)', fontsize=12, labelpad=10)
    ax2.set_title('Steady State $N_{max}(C_0, L_0)$\nPlate: Haldane | Erlen: Double Monod', fontsize=12, pad=20)
    legend_elements_nmax = [
        Patch(facecolor='darkviolet', alpha=0.6, label='Plate (Haldane)'),
        Patch(facecolor='orange', alpha=0.6, label='Erlen (Monod)')
    ]
    ax2.legend(handles=legend_elements_nmax, loc='upper left', fontsize=10)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'surfaces_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved superimposed surfaces plot: {output_path}")
    plt.close()

    return fig


# ============================================================================
# FIGURE 2: N_max SCALING (MULTIPLICATIVE CONSTANT)
# ============================================================================
def plot_nmax_scaling(params):
    """
    Find optimal multiplicative constant alpha such that:
    alpha × N_max_plate(C0, L0) ≈ N_max_erlen(C0, L0)

    Generate figure showing original and scaled surfaces with formula below.
    """

    # Create meshgrid - same order as fitting_erlen.py
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)

    # Compute surfaces
    surfaces = compute_surfaces(params, C0_grid, L0_grid)
    n_plate = surfaces['n_plate']
    n_erlen = surfaces['n_erlen']

    # Flatten for optimization
    n_plate_flat = n_plate.flatten()
    n_erlen_flat = n_erlen.flatten()

    # Optimize alpha to minimize ||alpha * n_plate - n_erlen||²
    def objective(alpha):
        return np.sum((alpha * n_plate_flat - n_erlen_flat)**2)

    result = minimize_scalar(objective, bounds=(0.001, 100.0), method='bounded')
    alpha_opt = result.x

    # Calculate R² for scaled surface
    n_plate_scaled_flat = alpha_opt * n_plate_flat
    ss_res = np.sum((n_erlen_flat - n_plate_scaled_flat)**2)
    ss_tot = np.sum((n_erlen_flat - np.mean(n_erlen_flat))**2)
    r2_scaled = 1 - (ss_res / ss_tot)

    # RMSE
    rmse = np.sqrt(np.mean((n_erlen_flat - n_plate_scaled_flat)**2))

    logger.info(f"\nN_max Scaling Analysis:")
    logger.info(f"  Optimal alpha = {alpha_opt:.4f}")
    logger.info(f"  R² (scaled) = {r2_scaled:.4f}")
    logger.info(f"  RMSE = {rmse:.2e} cells/mL")

    # Create figure with GridSpec for 3 plots + formula below
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.3, wspace=0.35)

    # ===== Subplot 1: Original surfaces =====
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot_surface(
        L0_grid, C0_grid, n_plate,
        color='darkviolet', alpha=0.6, edgecolor='none'
    )
    ax1.plot_surface(
        L0_grid, C0_grid, n_erlen,
        color='orange', alpha=0.6, edgecolor='none'
    )

    ax1.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11, labelpad=8)
    ax1.set_ylabel('$C_0$', fontsize=11, labelpad=8)
    ax1.set_zlabel('$N_{max}$', fontsize=11, labelpad=8)
    ax1.set_title('Original Surfaces', fontsize=12, pad=15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkviolet', alpha=0.6, label='Plate'),
        Patch(facecolor='orange', alpha=0.6, label='Erlen')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # ===== Subplot 2: Scaled plate vs erlen =====
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    n_plate_scaled = alpha_opt * n_plate

    ax2.plot_surface(
        L0_grid, C0_grid, n_plate_scaled,
        color='darkviolet', alpha=0.6, edgecolor='none'
    )
    ax2.plot_surface(
        L0_grid, C0_grid, n_erlen,
        color='orange', alpha=0.6, edgecolor='none'
    )

    ax2.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11, labelpad=8)
    ax2.set_ylabel('$C_0$', fontsize=11, labelpad=8)
    ax2.set_zlabel('$N_{max}$', fontsize=11, labelpad=8)
    ax2.set_title(f'Scaled Plate vs Erlen\nR² = {r2_scaled:.4f}', fontsize=12, pad=15)

    legend_elements_scaled = [
        Patch(facecolor='darkviolet', alpha=0.6, label='Scaled Plate'),
        Patch(facecolor='orange', alpha=0.6, label='Erlen')
    ]
    ax2.legend(handles=legend_elements_scaled, loc='upper left', fontsize=9)

    # ===== Subplot 3: Residuals (heatmap) =====
    ax3 = fig.add_subplot(gs[0, 2])

    residuals = n_erlen - n_plate_scaled
    residuals_percent = 100 * residuals / (n_erlen + 1e-10)  # Avoid division by zero

    vmax = np.max(np.abs(residuals_percent))
    im = ax3.contourf(L0_grid, C0_grid, residuals_percent, levels=20,
                      cmap='PuOr', vmin=-vmax, vmax=vmax)
    ax3.contour(L0_grid, C0_grid, residuals_percent, levels=[0], colors='black', linewidths=2)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Relative error (%)', fontsize=11)

    ax3.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11)
    ax3.set_ylabel('$C_0$', fontsize=11)
    ax3.set_title('Relative error: (Erlen - Scaled Plate) / Erlen', fontsize=12)

    # ===== Formula box centered below the 3 plots =====
    ax_formula = fig.add_subplot(gs[1, :])
    ax_formula.axis('off')

    formula_text = (
        f"$N_{{max}}^{{erlen}}(C_0, L_0) \\approx \\alpha \\times N_{{max}}^{{plate}}(C_0, L_0)$\n\n"
        f"$\\alpha = {alpha_opt:.4f}$      "
        f"$R^2 = {r2_scaled:.4f}$      "
        f"$RMSE = {rmse:.2e}$ cells/mL"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8)
    ax_formula.text(0.5, 0.5, formula_text, transform=ax_formula.transAxes,
                    fontsize=14, ha='center', va='center', bbox=props)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'nmax_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved N_max scaling plot: {output_path}")
    plt.close()

    return alpha_opt, r2_scaled


# ============================================================================
# FIGURE 3: mu_max SCALING (LIGHT + AMPLITUDE)
# ============================================================================
def plot_mumax_scaling(params):
    """
    mu_max scaling using Light + Amplitude strategy.

    Layout: 3 panels + formula below
    - Panel 1: Original surfaces (plate vs erlen)
    - Panel 2: Scaled plate vs erlen with s_L and z
    - Panel 3: Residuals heatmap
    - Below: Transformation formula

    Transformation: μ_erlen(C0, L0) ≈ z × μ_plate(C0, s_L × L0_factor)
    """

    # Create meshgrid - same order as fitting_erlen.py
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)
    L0_factor_grid = L0_grid / L0_REF

    # Compute surfaces
    surfaces = compute_surfaces(params, C0_grid, L0_grid)
    mu_plate = surfaces['mu_plate']
    mu_erlen = surfaces['mu_erlen']

    # Flatten for optimization
    mu_erlen_flat = mu_erlen.flatten()
    C0_flat = C0_grid.flatten()
    L0_factor_flat = L0_factor_grid.flatten()

    # Plate parameters
    p_gr = params['plate_gr']
    has_ki = 'k_i' in p_gr

    # Function to compute scaled plate surface
    def mu_plate_scaled(C0_s, L0_factor_s, z=1.0):
        if has_ki:
            return z * haldane_plate(C0_s, L0_factor_s, p_gr['mu_max_ref'],
                                     p_gr['k_c'], p_gr['k_l'], p_gr['k_i'])
        else:
            return z * p_gr['mu_max_ref'] * (C0_s / (p_gr['k_c'] + C0_s)) * \
                   (L0_factor_s / (p_gr['k_l'] + L0_factor_s))

    # Optimize Light + Amplitude scaling
    def objective_light_amp(scales):
        s_L, z = scales
        mu_pred = mu_plate_scaled(C0_flat, L0_factor_flat * s_L, z)
        return np.sum((mu_erlen_flat - mu_pred)**2)

    result = minimize(objective_light_amp, x0=[1.0, 1.0],
                     bounds=[(0.01, 100.0), (0.1, 10.0)], method='L-BFGS-B')
    s_L_opt, z_opt = result.x

    # Compute scaled surface
    mu_plate_scaled_grid = mu_plate_scaled(C0_grid, L0_factor_grid * s_L_opt, z_opt)

    # Calculate R²
    mu_scaled_flat = mu_plate_scaled_grid.flatten()
    ss_res = np.sum((mu_erlen_flat - mu_scaled_flat)**2)
    ss_tot = np.sum((mu_erlen_flat - np.mean(mu_erlen_flat))**2)
    r2_scaled = 1 - (ss_res / ss_tot)

    # RMSE
    rmse = np.sqrt(np.mean((mu_erlen_flat - mu_scaled_flat)**2))

    logger.info(f"\nmu_max Scaling Analysis (Light + Amplitude):")
    logger.info(f"  s_L = {s_L_opt:.4f} (light scaling factor)")
    logger.info(f"  z = {z_opt:.4f} (amplitude scaling factor)")
    logger.info(f"  R² = {r2_scaled:.4f}")
    logger.info(f"  RMSE = {rmse:.6f} h⁻¹")

    # =========================================================================
    # Create figure with GridSpec for 3 plots + formula below
    # =========================================================================
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.3, wspace=0.35)

    # ===== Subplot 1: Original surfaces =====
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    ax1.plot_surface(
        L0_grid, C0_grid, mu_plate,
        color='turquoise', alpha=0.6, edgecolor='none'
    )
    ax1.plot_surface(
        L0_grid, C0_grid, mu_erlen,
        color='seagreen', alpha=0.6, edgecolor='none'
    )

    ax1.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11, labelpad=8)
    ax1.set_ylabel('$C_0$', fontsize=11, labelpad=8)
    ax1.set_zlabel('$\\mu_{max}$ (h⁻¹)', fontsize=11, labelpad=8)
    ax1.set_title('Original Surfaces', fontsize=12, pad=15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='turquoise', alpha=0.6, label='Plate (Haldane)'),
        Patch(facecolor='seagreen', alpha=0.6, label='Erlen (Monod)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # ===== Subplot 2: Scaled plate vs erlen =====
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    ax2.plot_surface(
        L0_grid, C0_grid, mu_plate_scaled_grid,
        color='turquoise', alpha=0.6, edgecolor='none'
    )
    ax2.plot_surface(
        L0_grid, C0_grid, mu_erlen,
        color='seagreen', alpha=0.6, edgecolor='none'
    )

    ax2.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11, labelpad=8)
    ax2.set_ylabel('$C_0$', fontsize=11, labelpad=8)
    ax2.set_zlabel('$\\mu_{max}$ (h⁻¹)', fontsize=11, labelpad=8)
    ax2.set_title(f'Scaled Plate vs Erlen\nR² = {r2_scaled:.4f}', fontsize=12, pad=15)

    legend_elements_scaled = [
        Patch(facecolor='turquoise', alpha=0.6, label='Scaled Plate'),
        Patch(facecolor='seagreen', alpha=0.6, label='Erlen')
    ]
    ax2.legend(handles=legend_elements_scaled, loc='upper left', fontsize=9)

    # ===== Subplot 3: Residuals (heatmap) =====
    ax3 = fig.add_subplot(gs[0, 2])

    residuals = mu_erlen - mu_plate_scaled_grid
    residuals_percent = 100 * residuals / (mu_erlen + 1e-10)  # Avoid division by zero

    # Symmetric color scale centered on 0
    vmax = np.max(np.abs(residuals_percent))
    im = ax3.contourf(L0_grid, C0_grid, residuals_percent, levels=20,
                      cmap='BrBG', vmin=-vmax, vmax=vmax)
    ax3.contour(L0_grid, C0_grid, residuals_percent, levels=[0], colors='black', linewidths=2)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Relative error (%)', fontsize=11)

    ax3.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=11)
    ax3.set_ylabel('$C_0$', fontsize=11)
    ax3.set_title('Relative error: (Erlen - Scaled Plate) / Erlen', fontsize=12)

    # ===== Formula box centered below the 3 plots =====
    ax_formula = fig.add_subplot(gs[1, :])
    ax_formula.axis('off')

    formula_text = (
        f"$\\mu_{{max}}^{{erlen}}(C_0, L_0) \\approx z \\times \\mu_{{max}}^{{plate}}(C_0, s_L \\times L_0)$\n\n"
        f"$s_L = {s_L_opt:.4f}$      "
        f"$z = {z_opt:.4f}$      "
        f"$R^2 = {r2_scaled:.4f}$      "
        f"$RMSE = {rmse:.6f}$ h⁻¹"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8)
    ax_formula.text(0.5, 0.5, formula_text, transform=ax_formula.transAxes,
                    fontsize=14, ha='center', va='center', bbox=props)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'mumax_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved mu_max scaling plot: {output_path}")
    plt.close()

    return {'s_L': s_L_opt, 'z': z_opt, 'r2': r2_scaled}


# ============================================================================
# SAVE SCALING PARAMETERS TO YAML
# ============================================================================
def save_scaling_parameters(alpha_nmax, r2_nmax, mu_scaling_results):
    """Save all scaling parameters to a YAML file."""

    scaling_params = {
        'nmax_scaling': {
            'alpha': float(alpha_nmax),
            'r2': float(r2_nmax),
            'description': 'N_max_erlen ≈ alpha × N_max_plate'
        },
        'mumax_scaling': {
            's_L': float(mu_scaling_results['s_L']),
            'z': float(mu_scaling_results['z']),
            'r2': float(mu_scaling_results['r2']),
            'description': 'mu_max_erlen ≈ z × mu_max_plate(C0, s_L × L0)'
        }
    }

    output_path = OUTPUT_DIR / 'scaling_parameters.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(scaling_params, f, default_flow_style=False)

    logger.info(f"Saved scaling parameters to {output_path}")


# ============================================================================
# INDIVIDUAL FIGURES
# ============================================================================
def plot_individual_figures(params):
    """
    Generate individual figures for each plot in nmax_scaling and mumax_scaling.

    Generates 6 figures:
    1. nmax_original.png - Original N_max surfaces (plate vs erlen)
    2. nmax_scaled.png - Scaled N_max surfaces
    3. nmax_residuals.png - N_max residuals heatmap
    4. mumax_original.png - Original mu_max surfaces (plate vs erlen)
    5. mumax_scaled.png - Scaled mu_max surfaces
    6. mumax_residuals.png - mu_max residuals heatmap
    """
    from matplotlib.patches import Patch

    # Output directory (same as main figures)
    individual_dir = OUTPUT_DIR

    # Create meshgrid - same order as fitting_erlen.py
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)
    L0_factor_grid = L0_grid / L0_REF

    # Compute surfaces
    surfaces = compute_surfaces(params, C0_grid, L0_grid)
    n_plate = surfaces['n_plate']
    n_erlen = surfaces['n_erlen']
    mu_plate = surfaces['mu_plate']
    mu_erlen = surfaces['mu_erlen']

    # =========================================================================
    # Compute scaling parameters (same as in plot_nmax_scaling and plot_mumax_scaling)
    # =========================================================================

    # N_max scaling: alpha
    n_plate_flat = n_plate.flatten()
    n_erlen_flat = n_erlen.flatten()

    def objective_nmax(alpha):
        return np.sum((alpha * n_plate_flat - n_erlen_flat)**2)

    result_nmax = minimize_scalar(objective_nmax, bounds=(0.001, 100.0), method='bounded')
    alpha_opt = result_nmax.x
    n_plate_scaled = alpha_opt * n_plate

    # N_max R²
    n_plate_scaled_flat = alpha_opt * n_plate_flat
    ss_res_n = np.sum((n_erlen_flat - n_plate_scaled_flat)**2)
    ss_tot_n = np.sum((n_erlen_flat - np.mean(n_erlen_flat))**2)
    r2_nmax = 1 - (ss_res_n / ss_tot_n)

    # mu_max scaling: s_L, z
    mu_erlen_flat = mu_erlen.flatten()
    C0_flat = C0_grid.flatten()
    L0_factor_flat = L0_factor_grid.flatten()

    p_gr = params['plate_gr']
    has_ki = 'k_i' in p_gr

    def mu_plate_scaled_func(C0_s, L0_factor_s, z=1.0):
        if has_ki:
            return z * haldane_plate(C0_s, L0_factor_s, p_gr['mu_max_ref'],
                                     p_gr['k_c'], p_gr['k_l'], p_gr['k_i'])
        else:
            return z * p_gr['mu_max_ref'] * (C0_s / (p_gr['k_c'] + C0_s)) * \
                   (L0_factor_s / (p_gr['k_l'] + L0_factor_s))

    def objective_mumax(scales):
        s_L, z = scales
        mu_pred = mu_plate_scaled_func(C0_flat, L0_factor_flat * s_L, z)
        return np.sum((mu_erlen_flat - mu_pred)**2)

    result_mumax = minimize(objective_mumax, x0=[1.0, 1.0],
                           bounds=[(0.01, 100.0), (0.1, 10.0)], method='L-BFGS-B')
    s_L_opt, z_opt = result_mumax.x
    mu_plate_scaled_grid = mu_plate_scaled_func(C0_grid, L0_factor_grid * s_L_opt, z_opt)

    # mu_max R²
    mu_scaled_flat = mu_plate_scaled_grid.flatten()
    ss_res_mu = np.sum((mu_erlen_flat - mu_scaled_flat)**2)
    ss_tot_mu = np.sum((mu_erlen_flat - np.mean(mu_erlen_flat))**2)
    r2_mumax = 1 - (ss_res_mu / ss_tot_mu)

    # =========================================================================
    # Common plot settings
    # =========================================================================
    # Font sizes for 3D individual figures
    FONT_LABEL_3D = 26
    FONT_TITLE_3D = 26
    FONT_TICK_3D = 26
    FONT_LEGEND_3D = 26

    # Font sizes for 2D residuals figures
    FONT_LABEL_2D = 22
    FONT_TITLE_2D = 20
    FONT_TICK_2D = 22
    FONT_CBAR = 22
    FONT_CBAR_TICK = 22

    # Legends for N_max figures (darkviolet/orange)
    legend_nmax_original = [
        Patch(facecolor='darkviolet', alpha=0.6, label='Plate'),
        Patch(facecolor='orange', alpha=0.6, label='Erlen')
    ]
    legend_nmax_scaled = [
        Patch(facecolor='darkviolet', alpha=0.6, label='Scaled Plate'),
        Patch(facecolor='orange', alpha=0.6, label='Erlen')
    ]

    # Legends for mu_max figures (seagreen/turquoise)
    legend_mumax_original = [
        Patch(facecolor='turquoise', alpha=0.6, label='Plate'),
        Patch(facecolor='seagreen', alpha=0.6, label='Erlen')
    ]
    legend_mumax_scaled = [
        Patch(facecolor='turquoise', alpha=0.6, label='Scaled Plate'),
        Patch(facecolor='seagreen', alpha=0.6, label='Erlen')
    ]

    # =========================================================================
    # Common axis limits for paired figures
    # =========================================================================
    # X and Y limits (same for all 3D figures)
    xlim = (L0_MIN, L0_MAX)
    ylim = (C0_MIN, C0_MAX)

    # Z limits for N_max figures (original and scaled share same scale)
    z_nmax_min = min(n_plate.min(), n_erlen.min(), n_plate_scaled.min())
    z_nmax_max = max(n_plate.max(), n_erlen.max(), n_plate_scaled.max())
    zlim_nmax = (z_nmax_min * 0.95, z_nmax_max * 1.05)  # Add 5% margin
    # 4 evenly spaced ticks for N_max z-axis
    zticks_nmax = np.linspace(zlim_nmax[0], zlim_nmax[1], 4)

    # Z limits for mu_max figures (original and scaled share same scale)
    z_mumax_min = min(mu_plate.min(), mu_erlen.min(), mu_plate_scaled_grid.min())
    z_mumax_max = max(mu_plate.max(), mu_erlen.max(), mu_plate_scaled_grid.max())
    zlim_mumax = (z_mumax_min * 0.95, z_mumax_max * 1.05)  # Add 5% margin
    # 4 evenly spaced ticks for mu_max z-axis
    zticks_mumax = np.linspace(zlim_mumax[0], zlim_mumax[1], 4)

    # Custom formatters for z-axis tick labels
    # N_max: divide by 1e7 and show 1 decimal place
    def format_nmax(x, pos):
        return f'{x/1e7:.1f}'
    formatter_nmax = FuncFormatter(format_nmax)

    # mu_max: 2 decimal places
    def format_mumax(x, pos):
        return f'{x:.2f}'
    formatter_mumax = FuncFormatter(format_mumax)

    # Compute residuals for later use
    residuals_n = n_erlen - n_plate_scaled
    residuals_n_percent = 100 * residuals_n / (n_erlen + 1e-10)
    residuals_mu = mu_erlen - mu_plate_scaled_grid
    residuals_mu_percent = 100 * residuals_mu / (mu_erlen + 1e-10)

    # =========================================================================
    # FIGURE 1: N_max Original Surfaces
    # =========================================================================
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.plot_surface(L0_grid, C0_grid, n_plate,
                     color='darkviolet', alpha=0.6, edgecolor='none')
    ax1.plot_surface(L0_grid, C0_grid, n_erlen,
                     color='orange', alpha=0.6, edgecolor='none')

    # Axes labels with increased labelpad
    ax1.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax1.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    # Z-axis label horizontal above the axis
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel('$N_{max}$ (cells/mL)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax1.set_title('$N_{max}$ - Original Surfaces\nPlate vs Erlen', fontsize=FONT_TITLE_3D, pad=20)

    # Common axis limits for N_max figures
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim_nmax)

    # Reduce number of ticks to avoid overlapping
    ax1.set_xticks([25, 75, 125, 175])
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_zticks(zticks_nmax)
    ax1.zaxis.set_major_formatter(formatter_nmax)
    # Add ×10⁷ at top of z-axis
    ax1.text2D(0.92, 0.85, r'×10$^7$', transform=ax1.transAxes, fontsize=FONT_TICK_3D)
    ax1.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax1.legend(handles=legend_nmax_original, loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    fig1.savefig(individual_dir / 'nmax_original.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'nmax_original.png'}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: N_max Scaled Surfaces
    # =========================================================================
    fig2 = plt.figure(figsize=(12, 12))
    ax2 = fig2.add_subplot(111, projection='3d')

    ax2.plot_surface(L0_grid, C0_grid, n_plate_scaled,
                     color='darkviolet', alpha=0.6, edgecolor='none')
    ax2.plot_surface(L0_grid, C0_grid, n_erlen,
                     color='orange', alpha=0.6, edgecolor='none')

    # Axes labels with increased labelpad
    ax2.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax2.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    # Z-axis label horizontal above the axis
    ax2.zaxis.set_rotate_label(False)
    ax2.set_zlabel('$N_{max}$ (cells/mL)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax2.set_title(f'$N_{{max}}$ - Scaled Plate vs Erlen\n$\\alpha$ = {alpha_opt:.4f}, R² = {r2_nmax:.4f}',
                  fontsize=FONT_TITLE_3D, pad=20)

    # Common axis limits for N_max figures
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim_nmax)

    # Reduce number of ticks to avoid overlapping
    ax2.set_xticks([25, 75, 125, 175])
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_zticks(zticks_nmax)
    ax2.zaxis.set_major_formatter(formatter_nmax)
    # Add ×10⁷ at top of z-axis
    ax2.text2D(0.92, 0.85, r'×10$^7$', transform=ax2.transAxes, fontsize=FONT_TICK_3D)
    ax2.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax2.legend(handles=legend_nmax_scaled, loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    fig2.savefig(individual_dir / 'nmax_scaled.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'nmax_scaled.png'}")
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3: N_max Residuals Heatmap
    # =========================================================================
    fig3 = plt.figure(figsize=(10, 10))
    ax3 = fig3.add_subplot(111)

    vmax_n = np.max(np.abs(residuals_n_percent))
    im3 = ax3.contourf(L0_grid, C0_grid, residuals_n_percent, levels=20,
                       cmap='PuOr', vmin=-vmax_n, vmax=vmax_n)
    ax3.contour(L0_grid, C0_grid, residuals_n_percent, levels=[0], colors='black', linewidths=2)

    # Horizontal colorbar below the plot
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.15, aspect=40)
    cbar3.set_label('Relative error (%)', fontsize=FONT_CBAR, labelpad=10)
    cbar3.ax.tick_params(labelsize=FONT_CBAR_TICK)

    ax3.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_2D)
    ax3.set_ylabel('$C_0$', fontsize=FONT_LABEL_2D)
    #ax3.set_title(f'$N_{{max}}$ Relative error: (Erlen - Scaled Plate) / Erlen\n$\\alpha$ = {alpha_opt:.4f}', fontsize=FONT_TITLE_2D)
    ax3.tick_params(labelsize=FONT_TICK_2D)

    plt.tight_layout()
    fig3.savefig(individual_dir / 'nmax_residuals.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'nmax_residuals.png'}")
    plt.close(fig3)

    # =========================================================================
    # FIGURE 4: mu_max Original Surfaces
    # =========================================================================
    fig4 = plt.figure(figsize=(12, 12))
    ax4 = fig4.add_subplot(111, projection='3d')

    ax4.plot_surface(L0_grid, C0_grid, mu_plate,
                     color='turquoise', alpha=0.6, edgecolor='none')
    ax4.plot_surface(L0_grid, C0_grid, mu_erlen,
                     color='seagreen', alpha=0.6, edgecolor='none')

    # Axes labels with increased labelpad
    ax4.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax4.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    # Z-axis label horizontal above the axis
    ax4.zaxis.set_rotate_label(False)
    #ax4.set_zlabel('$\\mu_{max}$ (h$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax4.set_title('$\\mu_{max}$ - Original Surfaces\nPlate (Haldane) vs Erlen (Monod)', fontsize=FONT_TITLE_3D, pad=20)

    # Common axis limits for mu_max figures
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_zlim(zlim_mumax)

    # Reduce number of ticks to avoid overlapping
    ax4.set_xticks([25, 75, 125, 175])
    ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax4.set_zticks(zticks_mumax)
    ax4.zaxis.set_major_formatter(formatter_mumax)
    ax4.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax4.tick_params(axis='z', pad=18)  # Extra padding for z-axis ticks
    ax4.legend(handles=legend_mumax_original, loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    fig4.savefig(individual_dir / 'mumax_original.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'mumax_original.png'}")
    plt.close(fig4)

    # =========================================================================
    # FIGURE 5: mu_max Scaled Surfaces
    # =========================================================================
    fig5 = plt.figure(figsize=(12, 12))
    ax5 = fig5.add_subplot(111, projection='3d')

    ax5.plot_surface(L0_grid, C0_grid, mu_plate_scaled_grid,
                     color='turquoise', alpha=0.6, edgecolor='none')
    ax5.plot_surface(L0_grid, C0_grid, mu_erlen,
                     color='seagreen', alpha=0.6, edgecolor='none')

    # Axes labels with increased labelpad
    ax5.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax5.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    # Z-axis label horizontal above the axis
    ax5.zaxis.set_rotate_label(False)
    #ax5.set_zlabel('$\\mu_{max}$ (h$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax5.set_title(f'$\\mu_{{max}}$ - Scaled Plate vs Erlen\n$s_L$ = {s_L_opt:.4f}, $z$ = {z_opt:.4f}, R² = {r2_mumax:.4f}',
                  fontsize=FONT_TITLE_3D, pad=20)

    # Common axis limits for mu_max figures
    ax5.set_xlim(xlim)
    ax5.set_ylim(ylim)
    ax5.set_zlim(zlim_mumax)

    # Reduce number of ticks to avoid overlapping
    ax5.set_xticks([25, 75, 125, 175])
    ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax5.set_zticks(zticks_mumax)
    ax5.zaxis.set_major_formatter(formatter_mumax)
    ax5.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax5.tick_params(axis='z', pad=18)  # Extra padding for z-axis ticks
    ax5.legend(handles=legend_mumax_scaled, loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    fig5.savefig(individual_dir / 'mumax_scaled.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'mumax_scaled.png'}")
    plt.close(fig5)

    # =========================================================================
    # FIGURE 6: mu_max Residuals Heatmap
    # =========================================================================
    fig6 = plt.figure(figsize=(10, 10))
    ax6 = fig6.add_subplot(111)

    vmax_mu = np.max(np.abs(residuals_mu_percent))
    im6 = ax6.contourf(L0_grid, C0_grid, residuals_mu_percent, levels=20,
                       cmap='BrBG', vmin=-vmax_mu, vmax=vmax_mu)
    ax6.contour(L0_grid, C0_grid, residuals_mu_percent, levels=[0], colors='black', linewidths=2)

    # Horizontal colorbar below the plot
    cbar6 = plt.colorbar(im6, ax=ax6, orientation='horizontal', pad=0.15, aspect=40)
    cbar6.set_label('Relative error (%)', fontsize=FONT_CBAR, labelpad=10)
    cbar6.ax.tick_params(labelsize=FONT_CBAR_TICK)

    ax6.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_2D)
    ax6.set_ylabel('$C_0$', fontsize=FONT_LABEL_2D)
    #ax6.set_title(f'$\\mu_{{max}}$ Relative error: (Erlen - Scaled Plate) / Erlen\n$s_L$ = {s_L_opt:.4f}, $z$ = {z_opt:.4f}', fontsize=FONT_TITLE_2D)
    ax6.tick_params(labelsize=FONT_TICK_2D)

    plt.tight_layout()
    fig6.savefig(individual_dir / 'mumax_residuals.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {individual_dir / 'mumax_residuals.png'}")
    plt.close(fig6)

    logger.info(f"All individual figures saved in: {individual_dir}/")

    return {
        'alpha': alpha_opt,
        'r2_nmax': r2_nmax,
        's_L': s_L_opt,
        'z': z_opt,
        'r2_mumax': r2_mumax
    }


# ============================================================================
# MAIN
# ============================================================================
def plot_scaled_plate_vs_erlen_data(params):
    """
    Generate a figure with 2 side-by-side 3D plots showing the scaled plate surface
    for mu_max and N_max, overlaid with experimental erlen data points from
    results_erlen/growth_parameters_all_25_conditions.csv.

    Calibration points are shown in black, extrapolation points in mediumslateblue.
    Error bars show standard deviation across replicates A, B, C.
    """
    from matplotlib.patches import Patch

    # =========================================================================
    # Load experimental erlen data
    # =========================================================================
    csv_path = ERLEN_DIR / "growth_parameters_all_25_conditions.csv"
    df = pd.read_csv(csv_path, sep=';')
    logger.info(f"Loaded {len(df)} conditions from {csv_path}")

    # =========================================================================
    # Compute scaled plate surfaces (same as plot_individual_figures)
    # =========================================================================
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)
    L0_factor_grid = L0_grid / L0_REF

    surfaces = compute_surfaces(params, C0_grid, L0_grid)
    n_plate = surfaces['n_plate']
    mu_plate = surfaces['mu_plate']
    n_erlen = surfaces['n_erlen']
    mu_erlen = surfaces['mu_erlen']

    # N_max scaling: alpha
    n_plate_flat = n_plate.flatten()
    n_erlen_flat = n_erlen.flatten()

    def objective_nmax(alpha):
        return np.sum((alpha * n_plate_flat - n_erlen_flat)**2)

    result_nmax = minimize_scalar(objective_nmax, bounds=(0.001, 100.0), method='bounded')
    alpha_opt = result_nmax.x
    n_plate_scaled = alpha_opt * n_plate

    # mu_max scaling: s_L, z
    mu_erlen_flat = mu_erlen.flatten()
    C0_flat = C0_grid.flatten()
    L0_factor_flat = L0_factor_grid.flatten()

    p_gr = params['plate_gr']
    has_ki = 'k_i' in p_gr

    def mu_plate_scaled_func(C0_s, L0_factor_s, z=1.0):
        if has_ki:
            return z * haldane_plate(C0_s, L0_factor_s, p_gr['mu_max_ref'],
                                     p_gr['k_c'], p_gr['k_l'], p_gr['k_i'])
        else:
            return z * p_gr['mu_max_ref'] * (C0_s / (p_gr['k_c'] + C0_s)) * \
                   (L0_factor_s / (p_gr['k_l'] + L0_factor_s))

    def objective_mumax(scales):
        s_L, z = scales
        mu_pred = mu_plate_scaled_func(C0_flat, L0_factor_flat * s_L, z)
        return np.sum((mu_erlen_flat - mu_pred)**2)

    result_mumax = minimize(objective_mumax, x0=[1.0, 1.0],
                           bounds=[(0.01, 100.0), (0.1, 10.0)], method='L-BFGS-B')
    s_L_opt, z_opt = result_mumax.x
    mu_plate_scaled_grid = mu_plate_scaled_func(C0_grid, L0_factor_grid * s_L_opt, z_opt)

    # =========================================================================
    # Compute R² on experimental points (calibration and global)
    # =========================================================================
    # Separate calibration and extrapolation data
    df_calib = df[df['type'] == 'calibration']
    df_extrap = df[df['type'] == 'extrapolation']

    p_ss = params['plate_ss']
    has_ki_ss = 'k_i' in p_ss

    def n_plate_scaled_at(C0_v, L0_v):
        L0_factor_v = L0_v / L0_REF
        if has_ki_ss:
            return alpha_opt * haldane_plate(C0_v, L0_factor_v, p_ss['n_max_ref'],
                                             p_ss['k_c'], p_ss['k_l'], p_ss['k_i'])
        else:
            return alpha_opt * p_ss['n_max_ref'] * (C0_v / (p_ss['k_c'] + C0_v)) * \
                   (L0_factor_v / (p_ss['k_l'] + L0_factor_v))

    def mu_plate_scaled_at(C0_v, L0_v):
        L0_factor_v = L0_v / L0_REF
        return mu_plate_scaled_func(C0_v, L0_factor_v * s_L_opt, z_opt)

    # mu_max R² on calibration points
    mu_obs_calib = df_calib['mu_max_mean'].values
    mu_pred_calib = mu_plate_scaled_at(df_calib['C0'].values, df_calib['L0'].values)
    r2_mu_calib = 1 - np.sum((mu_obs_calib - mu_pred_calib)**2) / np.sum((mu_obs_calib - np.mean(mu_obs_calib))**2)

    # mu_max R² global (all points)
    mu_obs_all = df['mu_max_mean'].values
    mu_pred_all = mu_plate_scaled_at(df['C0'].values, df['L0'].values)
    r2_mu_global = 1 - np.sum((mu_obs_all - mu_pred_all)**2) / np.sum((mu_obs_all - np.mean(mu_obs_all))**2)

    # N_max R² on calibration points
    n_obs_calib = df_calib['Nmax_mean'].values
    n_pred_calib = n_plate_scaled_at(df_calib['C0'].values, df_calib['L0'].values)
    r2_n_calib = 1 - np.sum((n_obs_calib - n_pred_calib)**2) / np.sum((n_obs_calib - np.mean(n_obs_calib))**2)

    # N_max R² global (all points)
    n_obs_all = df['Nmax_mean'].values
    n_pred_all = n_plate_scaled_at(df['C0'].values, df['L0'].values)
    r2_n_global = 1 - np.sum((n_obs_all - n_pred_all)**2) / np.sum((n_obs_all - np.mean(n_obs_all))**2)

    logger.info(f"  mu_max: R² calib = {r2_mu_calib:.4f}, R² global = {r2_mu_global:.4f}")
    logger.info(f"  N_max:  R² calib = {r2_n_calib:.4f}, R² global = {r2_n_global:.4f}")

    # =========================================================================
    # Plot settings (same as plot_individual_figures)
    # =========================================================================
    FONT_LABEL_3D = 26
    FONT_TITLE_3D = 26
    FONT_TICK_3D = 26
    FONT_LEGEND_3D = 20

    xlim = (L0_MIN, L0_MAX)
    ylim = (C0_MIN, C0_MAX)

    # N_max formatter
    def format_nmax(x, pos):
        return f'{x/1e7:.1f}'
    formatter_nmax = FuncFormatter(format_nmax)

    # mu_max formatter
    def format_mumax(x, pos):
        return f'{x:.2f}'
    formatter_mumax = FuncFormatter(format_mumax)

    # =========================================================================
    # FIGURE: 2 side-by-side 3D plots
    # =========================================================================
    fig = plt.figure(figsize=(26, 12))

    # --- mu_max subplot ---
    ax_mu = fig.add_subplot(121, projection='3d')

    ax_mu.plot_surface(L0_grid, C0_grid, mu_plate_scaled_grid,
                       color='turquoise', alpha=0.5, edgecolor='none')

    # Calibration points (black)
    ax_mu.scatter(df_calib['L0'].values, df_calib['C0'].values, df_calib['mu_max_mean'].values,
                  color='black', s=80, marker='o', alpha=0.9, zorder=5, label='Calibration')
    # Error bars for calibration
    for _, row in df_calib.iterrows():
        ax_mu.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                   [row['mu_max_mean'] - row['mu_max_std'], row['mu_max_mean'] + row['mu_max_std']],
                   color='black', linewidth=1.5, alpha=0.7)

    # Extrapolation points (mediumslateblue)
    ax_mu.scatter(df_extrap['L0'].values, df_extrap['C0'].values, df_extrap['mu_max_mean'].values,
                  color='mediumslateblue', s=100, marker='o', edgecolor='white', linewidth=1.5,
                  alpha=0.9, zorder=5, label='Extrapolation')
    # Error bars for extrapolation
    for _, row in df_extrap.iterrows():
        ax_mu.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                   [row['mu_max_mean'] - row['mu_max_std'], row['mu_max_mean'] + row['mu_max_std']],
                   color='mediumslateblue', linewidth=1.5, alpha=0.7)

    ax_mu.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_mu.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_mu.zaxis.set_rotate_label(False)
    ax_mu.set_zlabel(r'$\mu_{max}$ (h$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax_mu.set_title(f'$\\mu_{{max}}$ - Scaled Plate vs Erlen Data\n'
                    f'R² = {r2_mu_calib:.3f} (calibration), R² = {r2_mu_global:.3f} (global)',
                    fontsize=FONT_TITLE_3D, pad=20)
    ax_mu.set_xlim(xlim)
    ax_mu.set_ylim(ylim)
    ax_mu.set_xticks([25, 75, 125, 175])
    ax_mu.set_yticks([0.25, 0.5, 0.75, 1.0])
    z_mumax_min = min(mu_plate_scaled_grid.min(), df['mu_max_mean'].min() - df['mu_max_std'].max())
    z_mumax_max = max(mu_plate_scaled_grid.max(), df['mu_max_mean'].max() + df['mu_max_std'].max())
    zlim_mumax = (z_mumax_min * 0.95, z_mumax_max * 1.05)
    ax_mu.set_zlim(zlim_mumax)
    zticks_mumax = np.linspace(zlim_mumax[0], zlim_mumax[1], 4)
    ax_mu.set_zticks(zticks_mumax)
    ax_mu.zaxis.set_major_formatter(formatter_mumax)
    ax_mu.tick_params(labelsize=FONT_TICK_3D, pad=8)
    legend_mu = [
        Patch(facecolor='turquoise', alpha=0.5, label='Scaled Plate'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=10, label='Calibration'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumslateblue',
                   markeredgecolor='white', markersize=10, label='Extrapolation'),
    ]
    ax_mu.legend(handles=legend_mu, loc='upper left', fontsize=FONT_LEGEND_3D)

    # --- N_max subplot ---
    ax_n = fig.add_subplot(122, projection='3d')

    ax_n.plot_surface(L0_grid, C0_grid, n_plate_scaled,
                      color='darkviolet', alpha=0.5, edgecolor='none')

    # Calibration points (black)
    ax_n.scatter(df_calib['L0'].values, df_calib['C0'].values, df_calib['Nmax_mean'].values,
                 color='black', s=80, marker='o', alpha=0.9, zorder=5, label='Calibration')
    # Error bars for calibration
    for _, row in df_calib.iterrows():
        ax_n.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                  [row['Nmax_mean'] - row['Nmax_std'], row['Nmax_mean'] + row['Nmax_std']],
                  color='black', linewidth=1.5, alpha=0.7)

    # Extrapolation points (mediumslateblue)
    ax_n.scatter(df_extrap['L0'].values, df_extrap['C0'].values, df_extrap['Nmax_mean'].values,
                 color='mediumslateblue', s=100, marker='o', edgecolor='white', linewidth=1.5,
                 alpha=0.9, zorder=5, label='Extrapolation')
    # Error bars for extrapolation
    for _, row in df_extrap.iterrows():
        ax_n.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                  [row['Nmax_mean'] - row['Nmax_std'], row['Nmax_mean'] + row['Nmax_std']],
                  color='mediumslateblue', linewidth=1.5, alpha=0.7)

    ax_n.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_n.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_n.zaxis.set_rotate_label(False)
    ax_n.set_zlabel('$N_{max}$ (cells/mL)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax_n.set_title(f'$N_{{max}}$ - Scaled Plate vs Erlen Data\n'
                   f'R² = {r2_n_calib:.3f} (calibration), R² = {r2_n_global:.3f} (global)',
                   fontsize=FONT_TITLE_3D, pad=20)
    ax_n.set_xlim(xlim)
    ax_n.set_ylim(ylim)
    ax_n.set_xticks([25, 75, 125, 175])
    ax_n.set_yticks([0.25, 0.5, 0.75, 1.0])
    z_nmax_min = min(n_plate_scaled.min(), df['Nmax_mean'].min() - df['Nmax_std'].max())
    z_nmax_max = max(n_plate_scaled.max(), df['Nmax_mean'].max() + df['Nmax_std'].max())
    zlim_nmax = (z_nmax_min * 0.95, z_nmax_max * 1.05)
    ax_n.set_zlim(zlim_nmax)
    zticks_nmax = np.linspace(zlim_nmax[0], zlim_nmax[1], 4)
    ax_n.set_zticks(zticks_nmax)
    ax_n.zaxis.set_major_formatter(formatter_nmax)
    ax_n.text2D(0.92, 0.85, r'×10$^7$', transform=ax_n.transAxes, fontsize=FONT_TICK_3D)
    ax_n.tick_params(labelsize=FONT_TICK_3D, pad=8)
    legend_n = [
        Patch(facecolor='darkviolet', alpha=0.5, label='Scaled Plate'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=10, label='Calibration'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumslateblue',
                   markeredgecolor='white', markersize=10, label='Extrapolation'),
    ]
    ax_n.legend(handles=legend_n, loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'scaled_plate_vs_erlen_data.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_plate_erlen_points_comparison(params):
    """
    Generate a figure with 2 side-by-side 3D plots comparing experimental erlen
    points (green) vs plate experimental points transformed by the scaling formula (blue).
    No surfaces are displayed.

    Also generates a CSV with per-condition R² local (relative error),
    and global R² for calibration, extrapolation, and combined sets.
    """
    import csv as csv_module

    # =========================================================================
    # Load experimental data
    # =========================================================================
    # Erlen data
    df_erlen = pd.read_csv(ERLEN_DIR / "growth_parameters_all_25_conditions.csv", sep=';')
    logger.info(f"Loaded {len(df_erlen)} erlen conditions")

    # Plate data (L0 is a factor, multiply by L0_REF for actual µmol/m²/s)
    df_plate_mu = pd.read_csv(PLATE_DIR / "plates_growth_rates_measured.csv", sep=';')
    df_plate_n = pd.read_csv(PLATE_DIR / "plates_steady_states_measured.csv", sep=';')
    df_plate_mu['L0_actual'] = df_plate_mu['L0'] * L0_REF
    df_plate_n['L0_actual'] = df_plate_n['L0'] * L0_REF
    logger.info(f"Loaded {len(df_plate_mu)} plate mu_max points, {len(df_plate_n)} plate N_max points")

    # =========================================================================
    # Compute scaling parameters (same as other functions)
    # =========================================================================
    L0_range = np.linspace(L0_MIN, L0_MAX, N_GRID)
    C0_range = np.linspace(C0_MIN, C0_MAX, N_GRID)
    L0_grid, C0_grid = np.meshgrid(L0_range, C0_range)
    L0_factor_grid = L0_grid / L0_REF

    surfaces = compute_surfaces(params, C0_grid, L0_grid)
    n_plate_surf = surfaces['n_plate'].flatten()
    n_erlen_surf = surfaces['n_erlen'].flatten()
    mu_erlen_surf = surfaces['mu_erlen'].flatten()
    C0_flat = C0_grid.flatten()
    L0_factor_flat = L0_factor_grid.flatten()

    # N_max: alpha
    def objective_nmax(alpha):
        return np.sum((alpha * n_plate_surf - n_erlen_surf)**2)
    alpha_opt = minimize_scalar(objective_nmax, bounds=(0.001, 100.0), method='bounded').x

    # mu_max: s_L, z
    p_gr = params['plate_gr']
    has_ki = 'k_i' in p_gr

    def mu_plate_scaled_func(C0_s, L0_factor_s, z=1.0):
        if has_ki:
            return z * haldane_plate(C0_s, L0_factor_s, p_gr['mu_max_ref'],
                                     p_gr['k_c'], p_gr['k_l'], p_gr['k_i'])
        else:
            return z * p_gr['mu_max_ref'] * (C0_s / (p_gr['k_c'] + C0_s)) * \
                   (L0_factor_s / (p_gr['k_l'] + L0_factor_s))

    def objective_mumax(scales):
        s_L, z = scales
        mu_pred = mu_plate_scaled_func(C0_flat, L0_factor_flat * s_L, z)
        return np.sum((mu_erlen_surf - mu_pred)**2)

    result_mumax = minimize(objective_mumax, x0=[1.0, 1.0],
                           bounds=[(0.01, 100.0), (0.1, 10.0)], method='L-BFGS-B')
    s_L_opt, z_opt = result_mumax.x

    logger.info(f"  Scaling parameters: alpha={alpha_opt:.4f}, s_L={s_L_opt:.4f}, z={z_opt:.4f}")

    # =========================================================================
    # Transform plate points using the scaled plate MODEL
    # mu_max: z * haldane_plate(C0, s_L * L0_factor)
    # N_max:  alpha * haldane_plate(C0, L0_factor) (no coordinate shift)
    # =========================================================================
    p_ss = params['plate_ss']
    has_ki_ss = 'k_i' in p_ss

    def eval_scaled_mu(C0_v, L0_factor_v):
        if has_ki:
            return z_opt * haldane_plate(C0_v, s_L_opt * L0_factor_v,
                                         p_gr['mu_max_ref'], p_gr['k_c'], p_gr['k_l'], p_gr['k_i'])
        else:
            L = s_L_opt * L0_factor_v
            return z_opt * p_gr['mu_max_ref'] * (C0_v / (p_gr['k_c'] + C0_v)) * \
                   (L / (p_gr['k_l'] + L))

    def eval_scaled_nmax(C0_v, L0_factor_v):
        if has_ki_ss:
            return alpha_opt * haldane_plate(C0_v, L0_factor_v,
                                             p_ss['n_max_ref'], p_ss['k_c'], p_ss['k_l'], p_ss['k_i'])
        else:
            L = L0_factor_v
            return alpha_opt * p_ss['n_max_ref'] * (C0_v / (p_ss['k_c'] + C0_v)) * \
                   (L / (p_ss['k_l'] + L))

    df_plate_mu['mu_max_transformed'] = eval_scaled_mu(
        df_plate_mu['C0'].values, df_plate_mu['L0'].values)
    df_plate_n['N_max_transformed'] = eval_scaled_nmax(
        df_plate_n['C0'].values, df_plate_n['L0'].values)

    # =========================================================================
    # Match plate and erlen conditions for R² computation
    # =========================================================================
    # Round for matching
    erlen_keys = {(round(r['C0'], 4), round(r['L0'], 1)): r
                  for _, r in df_erlen.iterrows()}

    comparison_rows = []
    for _, row_plate_mu in df_plate_mu.iterrows():
        key = (round(row_plate_mu['C0'], 4), round(row_plate_mu['L0_actual'], 1))
        if key in erlen_keys:
            row_erlen = erlen_keys[key]
            # Find matching plate N_max row
            mask_n = ((df_plate_n['C0'] - row_plate_mu['C0']).abs() < 1e-4) & \
                     ((df_plate_n['L0'] - row_plate_mu['L0']).abs() < 1e-4)
            if mask_n.sum() > 0:
                row_plate_n = df_plate_n[mask_n].iloc[0]
            else:
                continue

            comparison_rows.append({
                'C0': key[0],
                'L0': key[1],
                'mu_max_erlen': row_erlen['mu_max_mean'],
                'mu_max_plate_transformed': row_plate_mu['mu_max_transformed'],
                'mu_max_error_rel_pct': 100 * abs(row_erlen['mu_max_mean'] - row_plate_mu['mu_max_transformed']) / row_erlen['mu_max_mean'],
                'N_max_erlen': row_erlen['Nmax_mean'],
                'N_max_plate_transformed': row_plate_n['N_max_transformed'],
                'N_max_error_rel_pct': 100 * abs(row_erlen['Nmax_mean'] - row_plate_n['N_max_transformed']) / row_erlen['Nmax_mean'],
                'type': row_erlen['type'],
            })

    df_comp = pd.DataFrame(comparison_rows).sort_values(['C0', 'L0'])

    # =========================================================================
    # Compute global R² values
    # =========================================================================
    def compute_r2(observed, predicted):
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    df_calib = df_comp[df_comp['type'] == 'calibration']
    df_extrap = df_comp[df_comp['type'] == 'extrapolation']

    r2_mu_calib = compute_r2(df_calib['mu_max_erlen'].values, df_calib['mu_max_plate_transformed'].values)
    r2_mu_extrap = compute_r2(df_extrap['mu_max_erlen'].values, df_extrap['mu_max_plate_transformed'].values)
    r2_mu_global = compute_r2(df_comp['mu_max_erlen'].values, df_comp['mu_max_plate_transformed'].values)

    r2_n_calib = compute_r2(df_calib['N_max_erlen'].values, df_calib['N_max_plate_transformed'].values)
    r2_n_extrap = compute_r2(df_extrap['N_max_erlen'].values, df_extrap['N_max_plate_transformed'].values)
    r2_n_global = compute_r2(df_comp['N_max_erlen'].values, df_comp['N_max_plate_transformed'].values)

    logger.info(f"  mu_max R²: calib={r2_mu_calib:.4f}, extrap={r2_mu_extrap:.4f}, global={r2_mu_global:.4f}")
    logger.info(f"  N_max  R²: calib={r2_n_calib:.4f}, extrap={r2_n_extrap:.4f}, global={r2_n_global:.4f}")

    # =========================================================================
    # Save CSV
    # =========================================================================
    csv_path = OUTPUT_DIR / "plate_erlen_points_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.writer(f, delimiter=';')
        writer.writerow(["C0", "L0", "type",
                         "mu_max_erlen", "mu_max_plate_transformed", "mu_max_error_rel_pct",
                         "N_max_erlen", "N_max_plate_transformed", "N_max_error_rel_pct"])
        for _, row in df_comp.iterrows():
            writer.writerow([f"{row['C0']:.4f}", f"{row['L0']:.1f}", row['type'],
                             f"{row['mu_max_erlen']:.6e}", f"{row['mu_max_plate_transformed']:.6e}",
                             f"{row['mu_max_error_rel_pct']:.2f}",
                             f"{row['N_max_erlen']:.6e}", f"{row['N_max_plate_transformed']:.6e}",
                             f"{row['N_max_error_rel_pct']:.2f}"])
        # Summary R² rows
        writer.writerow([])
        writer.writerow(["# Global R2 summary", "", "",
                         "mu_max_R2", "", "",
                         "N_max_R2", "", ""])
        writer.writerow(["calibration", "", "",
                         f"{r2_mu_calib:.6f}", "", "",
                         f"{r2_n_calib:.6f}", "", ""])
        writer.writerow(["extrapolation", "", "",
                         f"{r2_mu_extrap:.6f}", "", "",
                         f"{r2_n_extrap:.6f}", "", ""])
        writer.writerow(["global", "", "",
                         f"{r2_mu_global:.6f}", "", "",
                         f"{r2_n_global:.6f}", "", ""])
    logger.info(f"  Saved comparison CSV: {csv_path}")

    # =========================================================================
    # Plot settings
    # =========================================================================
    FONT_LABEL_3D = 26
    FONT_TITLE_3D = 26
    FONT_TICK_3D = 26
    FONT_LEGEND_3D = 20

    xlim = (L0_MIN, L0_MAX)
    ylim = (C0_MIN, C0_MAX)

    def format_nmax(x, pos):
        return f'{x/1e7:.1f}'
    formatter_nmax = FuncFormatter(format_nmax)

    def format_mumax(x, pos):
        return f'{x:.2f}'
    formatter_mumax = FuncFormatter(format_mumax)

    # =========================================================================
    # FIGURE: 2 side-by-side 3D scatter plots
    # =========================================================================
    fig = plt.figure(figsize=(26, 12))

    # Separate erlen and plate by type
    df_e_calib = df_erlen[df_erlen['type'] == 'calibration']
    df_e_extrap = df_erlen[df_erlen['type'] == 'extrapolation']
    df_p_mu_calib = df_plate_mu[df_plate_mu['type'] == 'calibration']
    df_p_mu_extrap = df_plate_mu[df_plate_mu['type'] == 'extrapolation']
    df_p_n_calib = df_plate_n[df_plate_n['type'] == 'calibration']
    df_p_n_extrap = df_plate_n[df_plate_n['type'] == 'extrapolation']

    # --- mu_max subplot ---
    ax_mu = fig.add_subplot(121, projection='3d')

    # Plate transformed: calibration (blue circles) and extrapolation (blue triangles)
    ax_mu.scatter(df_p_mu_calib['L0_actual'].values, df_p_mu_calib['C0'].values,
                  df_p_mu_calib['mu_max_transformed'].values,
                  color='teal', s=60, marker='o', alpha=0.7, zorder=4,
                  label=f'Plate calib. (n={len(df_p_mu_calib)})')
    ax_mu.scatter(df_p_mu_extrap['L0_actual'].values, df_p_mu_extrap['C0'].values,
                  df_p_mu_extrap['mu_max_transformed'].values,
                  color='teal', s=60, marker='^', edgecolor='white', linewidth=0.8,
                  alpha=0.7, zorder=4,
                  label=f'Plate extrap. (n={len(df_p_mu_extrap)})')

    # Erlen: calibration (green circles) with error bars
    ax_mu.scatter(df_e_calib['L0'].values, df_e_calib['C0'].values,
                  df_e_calib['mu_max_mean'].values,
                  color='tomato', s=100, marker='o', alpha=0.9, zorder=5,
                  label=f'Erlen calib. (n={len(df_e_calib)})')
    for _, row in df_e_calib.iterrows():
        ax_mu.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                   [row['mu_max_mean'] - row['mu_max_std'], row['mu_max_mean'] + row['mu_max_std']],
                   color='tomato', linewidth=1.5, alpha=0.7)

    # Erlen: extrapolation (green triangles) with error bars
    ax_mu.scatter(df_e_extrap['L0'].values, df_e_extrap['C0'].values,
                  df_e_extrap['mu_max_mean'].values,
                  color='tomato', s=100, marker='^', edgecolor='white', linewidth=1,
                  alpha=0.9, zorder=5,
                  label=f'Erlen extrap. (n={len(df_e_extrap)})')
    for _, row in df_e_extrap.iterrows():
        ax_mu.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                   [row['mu_max_mean'] - row['mu_max_std'], row['mu_max_mean'] + row['mu_max_std']],
                   color='tomato', linewidth=1.5, alpha=0.7)

    ax_mu.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_mu.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_mu.zaxis.set_rotate_label(False)
    ax_mu.set_zlabel(r'$\mu_{max}$ (h$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax_mu.set_title(f'$\\mu_{{max}}$ - Plate Transformed vs Erlen\n'
                    f'R² = {r2_mu_calib:.3f} (calib), {r2_mu_extrap:.3f} (extrap), {r2_mu_global:.3f} (global)',
                    fontsize=FONT_TITLE_3D, pad=20)
    ax_mu.set_xlim(xlim)
    ax_mu.set_ylim(ylim)
    ax_mu.set_xticks([25, 75, 125, 175])
    ax_mu.set_yticks([0.25, 0.5, 0.75, 1.0])
    all_mu = np.concatenate([df_plate_mu['mu_max_transformed'].values, df_erlen['mu_max_mean'].values])
    zlim_mu = (all_mu.min() * 0.90, all_mu.max() * 1.10)
    ax_mu.set_zlim(zlim_mu)
    ax_mu.set_zticks(np.linspace(zlim_mu[0], zlim_mu[1], 4))
    ax_mu.zaxis.set_major_formatter(formatter_mumax)
    ax_mu.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax_mu.legend(loc='upper left', fontsize=FONT_LEGEND_3D)

    # --- N_max subplot ---
    ax_n = fig.add_subplot(122, projection='3d')

    # Plate transformed: calibration (blue circles) and extrapolation (blue triangles)
    ax_n.scatter(df_p_n_calib['L0_actual'].values, df_p_n_calib['C0'].values,
                 df_p_n_calib['N_max_transformed'].values,
                 color='teal', s=60, marker='o', alpha=0.7, zorder=4,
                 label=f'Plate calib. (n={len(df_p_n_calib)})')
    ax_n.scatter(df_p_n_extrap['L0_actual'].values, df_p_n_extrap['C0'].values,
                 df_p_n_extrap['N_max_transformed'].values,
                 color='teal', s=60, marker='^', edgecolor='white', linewidth=0.8,
                 alpha=0.7, zorder=4,
                 label=f'Plate extrap. (n={len(df_p_n_extrap)})')

    # Erlen: calibration (green circles) with error bars
    ax_n.scatter(df_e_calib['L0'].values, df_e_calib['C0'].values,
                 df_e_calib['Nmax_mean'].values,
                 color='tomato', s=100, marker='o', alpha=0.9, zorder=5,
                 label=f'Erlen calib. (n={len(df_e_calib)})')
    for _, row in df_e_calib.iterrows():
        ax_n.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                  [row['Nmax_mean'] - row['Nmax_std'], row['Nmax_mean'] + row['Nmax_std']],
                  color='tomato', linewidth=1.5, alpha=0.7)

    # Erlen: extrapolation (green triangles) with error bars
    ax_n.scatter(df_e_extrap['L0'].values, df_e_extrap['C0'].values,
                 df_e_extrap['Nmax_mean'].values,
                 color='mediumseagreen', s=100, marker='^', edgecolor='white', linewidth=1,
                 alpha=0.9, zorder=5,
                 label=f'Erlen extrap. (n={len(df_e_extrap)})')
    for _, row in df_e_extrap.iterrows():
        ax_n.plot([row['L0'], row['L0']], [row['C0'], row['C0']],
                  [row['Nmax_mean'] - row['Nmax_std'], row['Nmax_mean'] + row['Nmax_std']],
                  color='mediumseagreen', linewidth=1.5, alpha=0.7)

    ax_n.set_xlabel(r'$L_0$ (µmol$_{h\nu}$ m$^{-2}$ s$^{-1}$)', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_n.set_ylabel('$C_0$', fontsize=FONT_LABEL_3D, labelpad=30)
    ax_n.zaxis.set_rotate_label(False)
    ax_n.set_zlabel('$N_{max}$ (cells/mL)', fontsize=FONT_LABEL_3D, labelpad=30, rotation=90)
    ax_n.set_title(f'$N_{{max}}$ - Plate Transformed vs Erlen\n'
                   f'R² = {r2_n_calib:.3f} (calib), {r2_n_extrap:.3f} (extrap), {r2_n_global:.3f} (global)',
                   fontsize=FONT_TITLE_3D, pad=20)
    ax_n.set_xlim(xlim)
    ax_n.set_ylim(ylim)
    ax_n.set_xticks([25, 75, 125, 175])
    ax_n.set_yticks([0.25, 0.5, 0.75, 1.0])
    all_n = np.concatenate([df_plate_n['N_max_transformed'].values, df_erlen['Nmax_mean'].values])
    zlim_n = (all_n.min() * 0.90, all_n.max() * 1.10)
    ax_n.set_zlim(zlim_n)
    ax_n.set_zticks(np.linspace(zlim_n[0], zlim_n[1], 4))
    ax_n.zaxis.set_major_formatter(formatter_nmax)
    ax_n.text2D(0.92, 0.85, r'×10$^7$', transform=ax_n.transAxes, fontsize=FONT_TICK_3D)
    ax_n.tick_params(labelsize=FONT_TICK_3D, pad=8)
    ax_n.legend(loc='upper left', fontsize=FONT_LEGEND_3D)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'plate_erlen_points_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Main function."""

    print("=" * 80)
    print("SCALING ANALYSIS: PLATE vs ERLEN RSM SURFACES")
    print("=" * 80)
    print("\nModels:")
    print("  - Plates: Haldane with photoinhibition")
    print("    f = f_ref × [C/(k_c+C)] × [L/(k_l+L+L²/k_i)]")
    print("  - Erlen: Double Monod (no photoinhibition)")
    print("    f = f_ref × [L/(L+k_l)] × [C/(C+k_c)]")

    # Load parameters
    print("\nLoading RSM parameters...")
    params = load_parameters()

    # Print loaded parameters
    print("\n--- Plate Parameters (Haldane) ---")
    print(f"  mu_max: mu_max_ref={params['plate_gr']['mu_max_ref']:.4e}, "
          f"k_c={params['plate_gr']['k_c']:.4f}, k_l={params['plate_gr']['k_l']:.4f}", end="")
    if 'k_i' in params['plate_gr']:
        print(f", k_i={params['plate_gr']['k_i']:.4f}")
    else:
        print(" (no k_i)")

    print(f"  N_max:  n_max_ref={params['plate_ss']['n_max_ref']:.4e}, "
          f"k_c={params['plate_ss']['k_c']:.4f}, k_l={params['plate_ss']['k_l']:.4f}", end="")
    if 'k_i' in params['plate_ss']:
        print(f", k_i={params['plate_ss']['k_i']:.4f}")
    else:
        print(" (no k_i)")

    print("\n--- Erlen Parameters (Double Monod) ---")
    print(f"  mu_max: mu_max_ref={params['erlen_gr']['mu_max_ref']:.4e}, "
          f"k_c={params['erlen_gr']['k_c']:.4f}, k_l={params['erlen_gr']['k_l']:.2f}")
    print(f"  N_max:  n_max_ref={params['erlen_ss']['n_max_ref']:.4e}, "
          f"k_c={params['erlen_ss']['k_c']:.4f}, k_l={params['erlen_ss']['k_l']:.2f}")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    print("\n1. Superimposed surfaces (mu_max and N_max)...")
    plot_superimposed_surfaces(params)

    print("\n2. N_max scaling analysis...")
    alpha_nmax, r2_nmax = plot_nmax_scaling(params)

    print("\n3. mu_max scaling analysis (Light + Amplitude)...")
    mu_scaling_results = plot_mumax_scaling(params)

    print("\n4. Individual figures...")
    plot_individual_figures(params)

    print("\n5. Scaled plate vs erlen experimental data...")
    plot_scaled_plate_vs_erlen_data(params)

    print("\n6. Plate vs erlen points comparison...")
    plot_plate_erlen_points_comparison(params)

    # Save scaling parameters
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_scaling_parameters(alpha_nmax, r2_nmax, mu_scaling_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nN_max Scaling:")
    print(f"  N_max_erlen ≈ {alpha_nmax:.4f} × N_max_plate")
    print(f"  R² = {r2_nmax:.4f}")

    print(f"\nmu_max Scaling (Light + Amplitude):")
    print(f"  mu_max_erlen ≈ {mu_scaling_results['z']:.4f} × mu_max_plate(C0, {mu_scaling_results['s_L']:.4f} × L0)")
    print(f"  R² = {mu_scaling_results['r2']:.4f}")

    print(f"\nOutput files saved in: {OUTPUT_DIR}/")
    print("  - surfaces_comparison.png")
    print("  - nmax_scaling.png")
    print("  - mumax_scaling.png")
    print("  - scaling_parameters.yaml")
    print("  - nmax_original.png")
    print("  - nmax_scaled.png")
    print("  - nmax_residuals.png")
    print("  - mumax_original.png")
    print("  - mumax_scaled.png")
    print("  - mumax_residuals.png")
    print("  - scaled_plate_vs_erlen_data.png")
    print("  - plate_erlen_points_comparison.png")
    print("  - plate_erlen_points_comparison.csv")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
