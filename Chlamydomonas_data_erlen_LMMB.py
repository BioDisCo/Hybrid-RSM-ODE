"""
Parameter Estimation for Chlamydomonas Growth Data
Adapted from Kambe et al. 2022 fitting methodology
Includes MR and Full model fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, curve_fit
import warnings
import os
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Conversion factor
conv_OD_to_cell = 4.77e6  # cells/mL per OD

# Physical constants
K = 5.08e-9  # extinction coefficient [ml cm^-1 cell^-1]
dep = 1.1  # culture depth [cm]
vol = 50  # culture volume [ml]

# Light conditions (μmol photons m^-2 s^-1)
Ein_values = np.array([170, 102, 51, 25.5, 11.9])  # μmol m^-2 s^-1
light_area = 0.01
Ein = Ein_values * light_area * 3600  # Convert to μmol hour^-1

# Medium concentrations
C0_values = np.array([1, 1/2, 1/4, 1/8, 1/16])

# Initial biomass
N0 = 69000 # cells/ml

# Files dictionary
files_dict = {
    1.000: "all_data/data_exp_Chlamy_07-07-25.csv",
    0.6: "all_data/data_exp_Chlamy_01-07-25.csv",
    0.3: "all_data/data_exp_Chlamy_17-02-25.csv",
    0.15: "all_data/data_exp_Chlamy_04-11-24.csv",
    0.07: "all_data/data_exp_Chlamy_16-09-24.csv"
}

# Reference parameters (will be estimated)
PARAMS_REF = {
    'mu': 0.2,  # h^-1 - initial guess
    'lambda_L': 1e-6,  # μmol s^-1 cell^-1 - initial guess
    'Gamma': 1e8,  # cells/ml - initial guess
    'xi': 0.01,  # initial guess
    'alpha': 1e-8  # cell^-1 - initial guess
}

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def lightpercell(L, N, dep, vol, K):
    """
    Calculate light flux per cell
    
    Args:
        L: incident light flux [μmol hour^-1]
        N: cell concentration [cells/ml]
        dep: culture depth [cm]
        vol: culture volume [ml]
        K: extinction coefficient [ml cm^-1 cell^-1]
    
    Returns:
        Light per cell [μmol hour^-1 cell^-1]
    """
    if N <= 0:
        return 0
    cell_conc = N / vol
    LPC = L * (1 - 10**(-K * cell_conc * dep)) / N
    return LPC


def MR_model(t, y, mu, lambda_L, Ein_val, dep, vol, K, Gamma):
    """
    Medium-Rich model (eq 3.3 from Kambe)
    
    dN/dt = μ * (L/(λ_L + L)) * (1 - N/Γ) * N
    """
    N = y[0]
    L = lightpercell(Ein_val, N, dep, vol, K)
    dNdt = mu * (L / (lambda_L + L)) * (1 - N/Gamma) * N
    return [dNdt]


def Full_model(t, y, alpha, mu, lambda_L, Ein_val, dep, vol, K, Gamma, xi):
    """
    Full model with medium concentration (eq 4.1 from Kambe)
    
    dN/dt = μ * (C/(ξ + C)) * (L/(λ_L + L)) * (1 - N/Γ) * N
    dC/dt = -α * dN/dt
    """
    N, C = y[0], y[1]
    L = lightpercell(Ein_val, N, dep, vol, K)
    dNdt = mu * (C / (xi + C)) * (L / (lambda_L + L)) * (1 - N/Gamma) * N
    dCdt = -alpha * dNdt
    return [dNdt, dCdt]


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def read_csv_with_replicates(filepath, conv_OD_to_cell=4.77e6):
    """
    Lit un fichier CSV contenant des réplicats techniques A, B, C pour chaque condition.
    
    Args:
        filepath: path to CSV file
        conv_OD_to_cell: conversion factor from OD to cells/mL
    
    Returns:
        dict: {condition_index: {'Time': array, 'A': array, 'B': array, 'C': array}}
    """
    df = pd.read_csv(filepath, sep=';')
    
    # Extract time (convert seconds to hours)
    time = df['Time (s)'].values / 3600
    
    data = {}
    
    # For each condition (0, 1, 2, 3, 4 corresponding to C0 = 1, 1/2, 1/4, 1/8, 1/16)
    for cond_idx in range(5):
        replicate_data = {
            'Time': time,
            'A': df[f'OD {cond_idx}A'].values * conv_OD_to_cell,
            'B': df[f'OD {cond_idx}B'].values * conv_OD_to_cell,
            'C': df[f'OD {cond_idx}C'].values * conv_OD_to_cell
        }
        data[cond_idx] = replicate_data
    
    return data


def load_all_data(files_dict, conv_OD_to_cell=4.77e6):
    """
    Load all experimental data from multiple files
    
    Returns:
        dict: {(Ein_value, C0_value): {'time': array, 'replicates': [array_A, array_B, array_C]}}
    """
    all_data = {}
    
    logger.info("\n" + "="*80)
    logger.info("LOADING EXPERIMENTAL DATA")
    logger.info("="*80)
    
    for L0_factor, filepath in files_dict.items():
        Ein_val = L0_factor * 170  # L0 max = 170
        
        logger.info(f"\nReading file: {filepath}")
        logger.info(f"  Ein = {Ein_val:.1f} μmol m^-2 s^-1")
        
        if not os.path.exists(filepath):
            logger.warning(f"  ⚠ File not found: {filepath}")
            continue
        
        data = read_csv_with_replicates(filepath, conv_OD_to_cell)
        
        # Pour chaque condition C0
        for cond_idx in range(5):
            C0_val = C0_values[cond_idx]
            
            all_data[(Ein_val, C0_val)] = {
                'time': data[cond_idx]['Time'],
                'replicates': [
                    data[cond_idx]['A'],
                    data[cond_idx]['B'],
                    data[cond_idx]['C']
                ],
                'mean': np.mean([data[cond_idx]['A'],
                                data[cond_idx]['B'],
                                data[cond_idx]['C']], axis=0),
                'std': np.std([data[cond_idx]['A'],
                              data[cond_idx]['B'],
                              data[cond_idx]['C']], axis=0, ddof=1)
            }
            
            logger.info(f"  ✓ Loaded condition: Ein={Ein_val:.1f}, C0={C0_val:.3f}")
    
    logger.info(f"\n✓ Total conditions loaded: {len(all_data)}")
    logger.info("="*80 + "\n")
    
    return all_data


def organize_data_for_fitting(all_data):
    """
    Organize data for parameter estimation
    
    Returns:
        time_data: common time points
        biomass_array: shape (n_conditions, n_timepoints)
        conditions_list: list of (Ein_idx, C0_idx) tuples
    """
    # Sort conditions by (Ein, C0)
    sorted_conditions = sorted(all_data.keys())
    
    # Get time data (assume all have same time points)
    time_data = all_data[sorted_conditions[0]]['time']
    
    biomass_list = []
    conditions_list = []
    
    for Ein_val, C0_val in sorted_conditions:
        biomass_list.append(all_data[(Ein_val, C0_val)]['mean'])
        
        # Find indices
        Ein_idx = np.argmin(np.abs(Ein_values - Ein_val))
        C0_idx = np.argmin(np.abs(C0_values - C0_val))
        conditions_list.append((Ein_idx, C0_idx))
    
    biomass_array = np.array(biomass_list)
    
    return time_data, biomass_array, conditions_list


# ============================================================================
# SIMPLE GROWTH PARAMETER ESTIMATION
# ============================================================================

def logistic_growth(t, N0, mu_max, Nmax):
    """Logistic growth equation"""
    return Nmax / (1 + (Nmax/N0 - 1) * np.exp(-mu_max * t))


def estimate_growth_parameters(time, biomass):
    """
    Estimate simple growth parameters from data
    
    Returns:
        dict with mu_max, Nmax, N0, r_squared
    """
    # Remove NaN and invalid values
    valid = ~np.isnan(biomass) & (biomass > 0)
    if valid.sum() < 5:
        return None
    
    t_fit = time[valid]
    N_fit = biomass[valid]
    
    # Initial guesses
    N0_guess = N_fit[0]
    Nmax_guess = np.max(N_fit)
    mu_max_guess = 0.05
    
    try:
        popt, _ = curve_fit(
            logistic_growth,
            t_fit,
            N_fit,
            p0=[N0_guess, mu_max_guess, Nmax_guess],
            bounds=([N0_guess*0.1, 0.001, Nmax_guess*0.5],
                   [N0_guess*10, 0.5, Nmax_guess*2]),
            maxfev=5000
        )
        
        N0_fit, mu_max_fit, Nmax_fit = popt
        
        # Calculate R²
        N_pred = logistic_growth(t_fit, *popt)
        ss_res = np.sum((N_fit - N_pred)**2)
        ss_tot = np.sum((N_fit - np.mean(N_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'mu_max': mu_max_fit,
            'Nmax': Nmax_fit,
            'N0': N0_fit,
            'r_squared': r_squared,
            't_lag': 0  # Could be estimated separately
        }
    except:
        return None


# ============================================================================
# MR MODEL PARAMETER ESTIMATION
# ============================================================================

def estimate_MR_parameters(all_data, verbose=True):
    """
    Estimate μ, λ_L, and Γ using data with C0 = 1 (all light intensities)
    
    Args:
        all_data: dictionary from load_all_data()
    
    Returns:
        mu_opt, lambda_L_opt, Gamma_opt
    """
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("STEP 1: ESTIMATING MR MODEL PARAMETERS (C0 = 1)")
        logger.info("="*70)
    
    # Select data with C0 = 1
    data_C0_1 = {k: v for k, v in all_data.items() if k[1] == 1.0}
    
    if len(data_C0_1) == 0:
        raise ValueError("No data found with C0 = 1")
    
    logger.info(f"Using {len(data_C0_1)} light conditions with C0 = 1")
    
    def objective(params):
        mu, lambda_L, Gamma = params
        
        if mu <= 0 or lambda_L <= 0 or Gamma <= 0:
            return 1e12
        
        total_error = 0
        n_points = 0
        
        for (Ein_val, C0_val), data_dict in data_C0_1.items():
            time_exp = data_dict['time']
            biomass_exp = data_dict['mean']
            
            # Remove NaN
            valid = ~np.isnan(biomass_exp) & (biomass_exp > 0)
            if valid.sum() < 5:
                continue
            
            t_fit = time_exp[valid]
            N_exp = biomass_exp[valid]
            
            # Initial condition
            N0_cond = N_exp[0]
            
            # Find Ein index
            Ein_idx = np.argmin(np.abs(Ein_values - Ein_val))
            Ein_hour = Ein[Ein_idx]
            
            try:
                sol = solve_ivp(
                    MR_model,
                    [t_fit[0], t_fit[-1]],
                    [N0_cond],
                    args=(mu, lambda_L, Ein_hour, dep, vol, K, Gamma),
                    t_eval=t_fit,
                    method='LSODA',
                    rtol=1e-8,
                    atol=1e-10
                )
                
                if sol.success:
                    N_sim = sol.y[0, :]
                    error = np.sum((N_sim - N_exp)**2)
                    total_error += error
                    n_points += len(t_fit)
                else:
                    return 1e12
            except:
                return 1e12
        
        return total_error / max(n_points, 1)
    
    # Initial guess and bounds
    x0 = [0.05, 1e-6, 1e8]
    bounds = [(0.01, 0.3), (1e-8, 1e-4), (1e7, 1e9)]
    
    if verbose:
        logger.info("Optimizing with differential_evolution...")
        logger.info(f"Initial: μ={x0[0]:.3f}, λ_L={x0[1]:.2e}, Γ={x0[2]:.2e}")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=150,
        popsize=15,
        seed=42,
        atol=1e-6,
        tol=1e-6,
        workers=1
    )
    
    mu_opt, lambda_L_opt, Gamma_opt = result.x
    
    if verbose:
        logger.info(f"\n✓ Optimized parameters:")
        logger.info(f"  μ = {mu_opt:.4f} h⁻¹")
        logger.info(f"  λ_L = {lambda_L_opt:.4e} μmol h⁻¹ cell⁻¹")
        logger.info(f"  Γ = {Gamma_opt:.4e} cells/ml")
        logger.info(f"  MSE = {result.fun:.4e}")
    
    return mu_opt, lambda_L_opt, Gamma_opt


# ============================================================================
# FULL MODEL PARAMETER ESTIMATION
# ============================================================================

def estimate_Full_parameters(all_data, mu, lambda_L, Gamma, verbose=True):
    """
    Estimate α and ξ for Full model using all conditions
    
    Args:
        all_data: dictionary from load_all_data()
        mu, lambda_L, Gamma: parameters from MR model
    
    Returns:
        alpha_opt, xi_opt
    """
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("STEP 2: ESTIMATING FULL MODEL PARAMETERS (α, ξ)")
        logger.info("="*70)
    
    def objective(params):
        alpha, xi = params
        
        if alpha <= 0 or xi <= 0:
            return 1e12
        
        total_error = 0
        n_points = 0
        
        for (Ein_val, C0_val), data_dict in all_data.items():
            time_exp = data_dict['time']
            biomass_exp = data_dict['mean']
            
            # Remove NaN
            valid = ~np.isnan(biomass_exp) & (biomass_exp > 0)
            if valid.sum() < 5:
                continue
            
            t_fit = time_exp[valid]
            N_exp = biomass_exp[valid]
            
            # Initial conditions
            N0_cond = N_exp[0]
            
            # Find indices
            Ein_idx = np.argmin(np.abs(Ein_values - Ein_val))
            Ein_hour = Ein[Ein_idx]
            
            try:
                sol = solve_ivp(
                    Full_model,
                    [t_fit[0], t_fit[-1]],
                    [N0_cond, C0_val],
                    args=(alpha, mu, lambda_L, Ein_hour, dep, vol, K, Gamma, xi),
                    t_eval=t_fit,
                    method='LSODA',
                    rtol=1e-8,
                    atol=1e-10
                )
                
                if sol.success:
                    N_sim = sol.y[0, :]
                    error = np.sum((N_sim - N_exp)**2)
                    total_error += error
                    n_points += len(t_fit)
                else:
                    return 1e12
            except:
                return 1e12
        
        return total_error / max(n_points, 1)
    
    # Initial guess and bounds
    x0 = [1e-8, 0.01]
    bounds = [(1e-10, 1e-6), (0.001, 0.1)]
    
    if verbose:
        logger.info("Optimizing with differential_evolution...")
        logger.info(f"Initial: α={x0[0]:.2e}, ξ={x0[1]:.3f}")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=150,
        popsize=15,
        seed=42,
        atol=1e-6,
        tol=1e-6,
        workers=1
    )
    
    alpha_opt, xi_opt = result.x
    
    if verbose:
        logger.info(f"\n✓ Optimized parameters:")
        logger.info(f"  α = {alpha_opt:.2e} ml cell⁻¹")
        logger.info(f"  ξ = {xi_opt:.4f}")
        logger.info(f"  MSE = {result.fun:.4e}")
    
    # ----------------------------------------------------------------------
    #  GLOBAL R² COMPUTATION AFTER OPTIMIZATION
    # ----------------------------------------------------------------------
    all_obs = []
    all_pred = []

    for (Ein_val, C0_val), data_dict in all_data.items():
        time_exp = data_dict['time']
        biomass_exp = data_dict['mean']

        valid = ~np.isnan(biomass_exp) & (biomass_exp > 0)
        if valid.sum() < 5:
            continue

        t_fit = time_exp[valid]
        N_exp = biomass_exp[valid]

        N0_cond = N_exp[0]
        Ein_idx = np.argmin(np.abs(Ein_values - Ein_val))
        Ein_hour = Ein[Ein_idx]

        try:
            sol = solve_ivp(
                Full_model,
                [t_fit[0], t_fit[-1]],
                [N0_cond, C0_val],
                args=(alpha_opt, mu, lambda_L, Ein_hour, dep, vol, K, Gamma, xi_opt),
                t_eval=t_fit,
                method='LSODA',
                rtol=1e-8,
                atol=1e-10
            )
            if sol.success:
                N_sim = sol.y[0, :]

                all_obs.append(N_exp)
                all_pred.append(N_sim)

        except:
            pass  # unstable conditions are ignored

    # Concatenate into a single vector
    if len(all_obs) > 0:
        all_obs = np.concatenate(all_obs)
        all_pred = np.concatenate(all_pred)

        ss_res = np.sum((all_obs - all_pred)**2)
        ss_tot = np.sum((all_obs - np.mean(all_obs))**2)
        R2_global = 1 - ss_res / ss_tot

        logger.info("\n" + "-"*70)
        logger.info(f"GLOBAL R² of Full Model fit: {R2_global:.4f}")
        logger.info("-"*70 + "\n")
    else:
        R2_global = float('nan')
        logger.info("⚠ Cannot compute global R² (not enough data).")

    return alpha_opt, xi_opt, R2_global


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_MR_model_results(all_data, mu, lambda_L, Gamma, output_dir='results_erlen'):
    """
    Plot MR model fits for C0 = 1 conditions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select data with C0 = 1
    data_C0_1 = {k: v for k, v in all_data.items() if k[1] == 1.0}
    sorted_keys = sorted(data_C0_1.keys())
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_keys)))
    
    for idx, (L0_val, C0_val) in enumerate(sorted_keys):
        data_dict = data_C0_1[(L0_val, C0_val)]
        time_exp = data_dict['time']
        biomass_mean = data_dict['mean']
        biomass_std = data_dict['std']
        
        # Remove NaN
        valid = ~np.isnan(biomass_mean)
        t_plot = time_exp[valid]
        N_plot = biomass_mean[valid]
        N_std = biomass_std[valid]
        
        # Simulation
        N0_cond = N_plot[0]
        L0_idx = np.argmin(np.abs(Ein_values - L0_val))  # Ein_values can remain for lookup
        L0_hour = Ein[L0_idx]  # value used for simulation
        
        t_sim = np.linspace(t_plot[0], t_plot[-1], 200)
        
        sol = solve_ivp(
            MR_model,
            [t_plot[0], t_plot[-1]],
            [N0_cond],
            args=(mu, lambda_L, L0_hour, dep, vol, K, Gamma),
            t_eval=t_sim,
            method='LSODA'
        )
        
        # Plot experimental data (points) without legend
        ax.errorbar(t_plot, N_plot, yerr=N_std, fmt='o', color=colors[idx],
                   markersize=6, capsize=3, alpha=0.6)
        
        # Plot model (line) with legend
        label = rf'$L_0={L0_val:.1f}\ \mu$mol/m²/s'
        ax.plot(sol.t, sol.y[0, :], '-', color=colors[idx], linewidth=2.5,
                alpha=0.8, label=label)
    
    ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Biomass (cells/mL)', fontsize=14, fontweight='bold')
    ax.set_title('MR Model Fit (C₀ = 1)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'LMMB_MR_model_fit.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {filepath}")
    plt.close()


def plot_Full_model_results(all_data, mu, lambda_L, Gamma, alpha, xi, R2_global=float('nan'), output_dir='results_erlen'):
    """
    Plot Full model fits for all conditions organized by light intensity.
    Displays 5 graphs in a row (one per L₀ value) with replicates shown as different markers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors per C0
    colors_C0 = {
        1.0000: 'mediumseagreen',
        0.5000: 'grey',
        0.2500: 'tomato',
        0.1250: 'teal',
        0.0625: 'orange'
    }
    
    # Define markers per replicate
    markers_rep = {
        'A': 'o',      # circle
        'B': '^',      # triangle
        'C': 's'       # square
    }
    
    # Get unique Ein values (L₀) sorted in descending order
    L0_unique = sorted(set([k[0] for k in all_data.keys()]), reverse=True)
    
    # Create figure with 1 row and 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(32, 6))
    
    # For each L₀ value, create a plot
    for idx_L0, L0_val in enumerate(L0_unique):
        ax = axes[idx_L0]
        
        # Get all C0 values for this L₀
        conditions = [(L0_val, C0) for C0 in C0_values if (L0_val, C0) in all_data]
        
        for _, C0_val in conditions:
            data_dict = all_data[(L0_val, C0_val)]
            time_exp = data_dict['time']
            replicates = data_dict['replicates']
            
            # Get color for this C0 value
            color = colors_C0.get(C0_val, 'black')
            
            # Plot experimental data for each replicate
            replicate_names = ['A', 'B', 'C']
            t_max = 0
            N0_values = []
            
            for rep_idx, (rep_name, biomass_rep) in enumerate(zip(replicate_names, replicates)):
                # Remove NaN
                valid = ~np.isnan(biomass_rep)
                t_plot = time_exp[valid]
                N_plot = biomass_rep[valid]
                
                if len(t_plot) > 0:
                    t_max = max(t_max, t_plot[-1])
                    N0_values.append(N_plot[0])
                
                # Marker according to replicate
                marker = markers_rep.get(rep_name, 'o')
                
                # Plot experimental points
                ax.plot(t_plot, N_plot, marker=marker, color=color, alpha=0.6,
                       markersize=6, linestyle='none')
            
            # Compute mean N0 for simulation
            if len(N0_values) > 0:
                N0_mean = np.mean(N0_values)
                
                # Plot Full model curve for this condition
                Ein_idx_lookup = np.argmin(np.abs(Ein_values - L0_val))
                Ein_hour = Ein[Ein_idx_lookup]
                
                t_sim = np.linspace(0, t_max, 500)
                
                sol = solve_ivp(
                    Full_model,
                    [0, t_max],
                    [N0_mean, C0_val],
                    args=(alpha, mu, lambda_L, Ein_hour, dep, vol, K, Gamma, xi),
                    t_eval=t_sim,
                    method='LSODA'
                )
                
                # Plot continuous curve with the same color as C0
                ax.plot(sol.t, sol.y[0, :], color=color, linewidth=2.5,
                       linestyle='-', label=f'C₀={C0_val:.4f}', alpha=0.9)
        
        # Configuration du graphique
        ax.set_title(f'$L_0$ = {L0_val:.1f} µmol/m²/s', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (h)', fontsize=14)
        if idx_L0 == 0:
            ax.set_ylabel('Biomass (cells/mL)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    
    # Create legend outside the plots (on the right)
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add C0 colors
    for C0, color in sorted(colors_C0.items(), reverse=True):
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2.5,
                                     label=f'C₀={C0:.4f}'))
    
    # Add a separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Add replicate markers
    for rep, marker in markers_rep.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray',
                                     linestyle='none', markersize=8,
                                     label=f'Replicate {rep}'))
    
    # Place legend to the right of all plots
    fig.legend(handles=legend_elements, loc='center right', fontsize=11,
              framealpha=0.9, bbox_to_anchor=(0.98, 0.5))
    
    fig.suptitle(rf'LMMB model fit  |  Global $R^{{2}}$ = {R2_global:.3f}', fontsize=18, fontweight='bold', y=1.02)
    
    # Adjust spacing to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    
    filepath = os.path.join(output_dir, 'LMMB_full_model_fit.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {filepath}")
    plt.close()


# ============================================================================
# RESULTS TABLE
# ============================================================================

def create_results_table(mu, lambda_L, Gamma, alpha, xi, output_dir='results_erlen'):
    """
    Create summary table of estimated parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("PARAMETER ESTIMATION SUMMARY")
    logger.info("="*70)
    
    results = {
        'Parameter': ['μ', 'λ_L', 'Γ', 'ξ', 'α'],
        'Value': [
            f'{mu:.4f}',
            f'{lambda_L:.4e}',
            f'{Gamma:.4e}',
            f'{xi:.4f}',
            f'{alpha:.2e}'
        ],
        'Unit': ['h⁻¹', 'μmol h⁻¹ cell⁻¹', 'cells/ml', '-', 'ml cell⁻¹']
    }
    
    df = pd.DataFrame(results)
    logger.info("\n" + df.to_string(index=False))
    
    filepath = os.path.join(output_dir, 'LMMB_estimated_parameters.csv')
    df.to_csv(filepath, index=False, sep=';', encoding='utf-8-sig')
    logger.info(f"\n✓ Saved: {filepath}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    logger.info("\n" + "="*80)
    logger.info(" CHLAMYDOMONAS GROWTH MODEL - PARAMETER ESTIMATION ")
    logger.info(" Based on Kambe et al. 2022 methodology ")
    logger.info("="*80)
    
    # Load all experimental data
    all_data = load_all_data(files_dict, conv_OD_to_cell)
    
    if len(all_data) == 0:
        logger.error("ERROR: No data loaded. Please check file paths.")
        return
    
    # Estimate MR model parameters (using C0 = 1 data)
    mu, lambda_L, Gamma = estimate_MR_parameters(all_data, verbose=True)
    
    # Estimate Full model parameters (using all data)
    alpha, xi, R2_global = estimate_Full_parameters(all_data, mu, lambda_L, Gamma, verbose=True)

    # Create results table
    df_results = create_results_table(mu, lambda_L, Gamma, alpha, xi)

    # Generate figures
    plot_Full_model_results(all_data, mu, lambda_L, Gamma, alpha, xi, R2_global=R2_global)
    
    logger.info("\n" + "="*80)
    logger.info(" ANALYSIS COMPLETE ")
    logger.info("="*80)
    logger.info("\nGenerated files in 'output' directory:")
    logger.info("  1. MR_model_fit_C0_1.png - MR model fit (C₀ = 1)")
    logger.info("  2. Full_model_fit_all_conditions.png - Full model fit (all conditions)")
    logger.info("  3. estimated_parameters.csv - Parameter estimates")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()


