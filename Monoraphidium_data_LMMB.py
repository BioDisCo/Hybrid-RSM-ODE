"""
Reproduction of Kambe et al. 2022 fitting
Includes data generation, parameter estimation, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import warnings
import os
import matplotlib as mpl
from data_import import read_csv_data_erlen_Kambe
warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

# ============================================================================
# CONSTANTS (from Kambe et al. 2022)
# ============================================================================

ODN = 30.10e6  # cells per OD
J = 0.1529
K = J / ODN  # extinction coefficient [ml cm^-1 cell^-1]

dep = 3.7  # culture depth [cm]
vol = 200  # culture volume [ml]

OD0 = 0.025
N0 = OD0 * ODN * vol  # initial biomass [cells]

# Light conditions
ein = np.array([96.8, 184.4, 386.7, 1034])  # [μE s^-1 m^-2]
light_area = 0.002826  # [m^2]
Ein = ein * light_area * 3600  # [μE hour^-1]

# Medium concentrations (order matches CSV columns: mean 0, mean 1, mean 2, mean 3)
C0 = np.array([1, 1/2, 1/4, 1/8])

# Reference parameters from Table 2
PARAMS_REF = {
    'mu': 0.194,
    'lambda_L': 1900e-9,
    'Gamma_OD': 16.6,
    'xi': 0.012,
    'alpha': 8.7e-12
}

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def lightpercell(L, N, dep, vol, K, ODN):
    """Calculate light flux per cell"""
    if N <= 0:
        return 0
    cell_conc = N / vol
    LPC = L * (1 - 10**(-K * cell_conc * dep)) / N
    return LPC


def MR_model(t, y, mu, lambda_L, Ein_val, dep, vol, K, Gamma, ODN):
    """Medium-Rich model (eq 3.3)"""
    N = y[0]
    L = lightpercell(Ein_val, N, dep, vol, K, ODN)
    dNdt = mu * (L / (lambda_L + L)) * (1 - N/Gamma) * N
    return [dNdt]


def Full_model(t, y, alpha, mu, lambda_L, Ein_val, dep, vol, K, Gamma, xi, ODN):
    """Full model with medium concentration (eq 4.1)"""
    N, C = y[0], y[1]
    L = lightpercell(Ein_val, N, dep, vol, K, ODN)
    dNdt = mu * (C / (xi + C)) * (L / (lambda_L + L)) * (1 - N/Gamma) * N
    dCdt = -alpha * dNdt
    return [dNdt, dCdt]

# ============================================================================
# DATA LOADING (from CSV files)
# ============================================================================

def read_Kambe_data(data_folder='data_Kambe'):
    """
    Load experimental data from CSV files using data_import.read_csv_data_erlen_Kambe.

    Final od_data array shape: (16, n_timepoints)
    - Index formula: data_idx = light_idx * 4 + C0_idx
    - light_idx 0..3 → Ein = [0.274, 0.521, 1.09, 2.92] μE/s
    - C0_idx 0..3 → C0 = [1, 1/2, 1/4, 1/8]
    """
    filenames = [
        "data-Ein-0.274.csv",
        "data-Ein-0.521.csv",
        "data-Ein-1.09.csv",
        "data-Ein-2.92.csv"
    ]

    C0_values = [1.0, 0.5, 0.25, 0.125]

    od_data_list = []
    time_data = None

    for fname in filenames:
        filepath = os.path.join(data_folder, fname)
        experiments = read_csv_data_erlen_Kambe(filepath)

        # Sort experiments by C0_factor descending (1, 0.5, 0.25, 0.125)
        sorted_exps = sorted(experiments.values(), key=lambda e: e['C0_factor'], reverse=True)

        if time_data is None:
            time_data = np.array(sorted_exps[0]['Time'])

        for c0_val in C0_values:
            for exp in sorted_exps:
                if abs(exp['C0_factor'] - c0_val) < 1e-3:
                    od_data_list.append(np.array(exp['Mean']))
                    break

    od_data = np.array(od_data_list)
    print(f"Loaded data: {od_data.shape[0]} conditions, {len(time_data)} time points")

    return time_data, od_data

# ============================================================================
# PARAMETER ESTIMATION
# ============================================================================

def estimate_MR_parameters(time_data, od_data, verbose=True):
    """
    Estimate μ, λ_L, and Γ using data with C0 = 1
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: ESTIMATING MR MODEL PARAMETERS")
        print("="*70)
    
    # Data for C0 = 1 (mean 0 column in each file)
    # With C0 = [1, 1/2, 1/4, 1/8], C0=1 corresponds to C0_idx=0
    # For C0_idx = 0, the indices are: light_idx * 4 + 0
    indices_C0_1 = [0, 4, 8, 12]  # Ein indices: [0.274, 0.521, 1.09, 2.92]
    light_indices = [i // 4 for i in indices_C0_1]  # correctly map to light: [0, 1, 2, 3]
    
    od_data_C0_1 = od_data[indices_C0_1, :]
    
    def objective(params):
        mu, lambda_L, Gamma_OD = params
        Gamma = Gamma_OD * ODN * vol
        
        if mu <= 0 or lambda_L <= 0 or Gamma <= 0:
            return 1e12
        
        total_error = 0
        n_points = 0
        
        for idx, light_idx in enumerate(light_indices):
            # Time truncation for high light (as in MATLAB)
            if light_idx == 2:  # Ein = 1.09
                t_max = 690
            elif light_idx == 3:  # Ein = 2.92
                t_max = 306
            else:
                t_max = 1200
            
            mask = time_data <= t_max
            t_fit = time_data[mask]
            od_exp = od_data_C0_1[idx, mask]
            
            valid = ~np.isnan(od_exp) & (od_exp > 0)
            t_fit = t_fit[valid]
            od_exp = od_exp[valid]
            
            if len(t_fit) < 3:
                continue
            
            try:
                sol = solve_ivp(
                    MR_model,
                    [t_fit[0], t_fit[-1]],
                    [N0],
                    args=(mu, lambda_L, Ein[light_idx], dep, vol, K, Gamma, ODN),
                    t_eval=t_fit,
                    method='LSODA',
                    rtol=1e-8,
                    atol=1e-10
                )
                
                if sol.success:
                    od_sim = sol.y[0, :] / (ODN * vol)
                    error = np.sum((od_sim - od_exp)**2)
                    total_error += error
                    n_points += len(t_fit)
                else:
                    return 1e12
            except:
                return 1e12
        
        return total_error / max(n_points, 1)
    
    # Initial guess
    x0 = [PARAMS_REF['mu'], PARAMS_REF['lambda_L'], PARAMS_REF['Gamma_OD']]
    bounds = [(0.1, 0.35), (1000e-9, 5000e-9), (14, 20)]
    
    if verbose:
        print("Optimizing with differential_evolution...")
        print(f"Initial: μ={x0[0]:.3f}, λ_L={x0[1]:.2e}, Γ_OD={x0[2]:.1f}")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=200,
        popsize=15,
        seed=42,
        atol=1e-6,
        tol=1e-6
    )
    
    mu_opt, lambda_L_opt, Gamma_OD_opt = result.x
    Gamma_opt = Gamma_OD_opt * ODN * vol
    
    if verbose:
        print(f"\nOptimized parameters:")
        print(f"  μ = {mu_opt:.4f} h⁻¹ (ref: {PARAMS_REF['mu']})")
        print(f"  λ_L = {lambda_L_opt:.4e} μE s⁻¹ cell⁻¹ (ref: {PARAMS_REF['lambda_L']:.2e})")
        print(f"  Γ_OD = {Gamma_OD_opt:.2f} (ref: {PARAMS_REF['Gamma_OD']})")
        print(f"  MSE = {result.fun:.4e}")
    
    return mu_opt, lambda_L_opt, Gamma_opt


def estimate_Full_parameters(time_data, od_data, mu, lambda_L, Gamma, verbose=True):
    """
    Estimate α and ξ for Full model
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: ESTIMATING FULL MODEL PARAMETERS")
        print("="*70)
    
    def objective(params):
        alpha, xi = params
        
        if alpha <= 0 or xi <= 0:
            return 1e12
        
        total_error = 0
        n_points = 0
        
        for light_idx in range(4):
            for C0_idx in range(4):
                data_idx = light_idx * 4 + C0_idx
                
                od_exp = od_data[data_idx, :]
                valid = ~np.isnan(od_exp) & (od_exp > 0)
                t_fit = time_data[valid]
                od_exp = od_exp[valid]
                
                if len(t_fit) < 3:
                    continue
                
                y0 = [N0, C0[C0_idx]]
                
                try:
                    sol = solve_ivp(
                        Full_model,
                        [t_fit[0], t_fit[-1]],
                        y0,
                        args=(alpha, mu, lambda_L, Ein[light_idx], dep, vol, K, Gamma, xi, ODN),
                        t_eval=t_fit,
                        method='LSODA',
                        rtol=1e-8,
                        atol=1e-10
                    )
                    
                    if sol.success:
                        od_sim = sol.y[0, :] / (ODN * vol)
                        error = np.sum((od_sim - od_exp)**2)
                        total_error += error
                        n_points += len(t_fit)
                    else:
                        return 1e12
                except:
                    return 1e12
        
        return total_error / max(n_points, 1)
    
    x0 = [PARAMS_REF['alpha'], PARAMS_REF['xi']]
    bounds = [(5e-12, 12e-12), (0.005, 0.025)]
    
    if verbose:
        print("Optimizing with differential_evolution...")
        print(f"Initial: α={x0[0]:.2e}, ξ={x0[1]:.3f}")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=150,
        popsize=15,
        seed=42,
        atol=1e-6,
        tol=1e-6
    )
    
    alpha_opt, xi_opt = result.x
    
    if verbose:
        print(f"\nOptimized parameters:")
        print(f"  α = {alpha_opt:.2e} cell⁻¹ (ref: {PARAMS_REF['alpha']:.2e})")
        print(f"  ξ = {xi_opt:.4f} (ref: {PARAMS_REF['xi']})")
        print(f"  MSE = {result.fun:.4e}")
    
    return alpha_opt, xi_opt
    

def compute_global_R2(time_data, od_data, mu, lambda_L, Gamma, alpha, xi):
    """
    Compute global R² over ALL 16 datasets for the Full model.
    """
    ss_res = 0.0
    ss_tot = 0.0

    for light_idx in range(4):
        for C0_idx in range(4):

            data_idx = light_idx * 4 + C0_idx

            od_exp = od_data[data_idx, :]
            valid = ~np.isnan(od_exp) & (od_exp > 0)

            t_fit = time_data[valid]
            od_exp = od_exp[valid]

            if len(t_fit) < 3:
                continue

            y0 = [N0, C0[C0_idx]]

            sol = solve_ivp(
                Full_model,
                [t_fit[0], t_fit[-1]],
                y0,
                args=(alpha, mu, lambda_L, Ein[light_idx], dep, vol, K, Gamma, xi, ODN),
                t_eval=t_fit,
                method='LSODA'
            )

            if not sol.success:
                continue

            od_sim = sol.y[0, :] / (ODN * vol)

            ss_res += np.sum((od_exp - od_sim)**2)
            ss_tot += np.sum((od_exp - np.mean(od_exp))**2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res/ss_tot

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(time_data, od_data, mu, lambda_L, Gamma, alpha, xi):
    """
    Create combined visualization: Full model (row 1) and MR model (row 2)
    """
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    print("\n" + "="*70)
    print("GENERATING FIGURE")
    print("="*70)

    simtime = np.linspace(0, 1200, 300)

    fig = plt.figure(figsize=(28, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3)

    # ---- Row 1: Full Model (all conditions) ----
    colors_C0 = {
        1.0000: 'mediumseagreen',
        0.5000: 'grey',
        0.2500: 'tomato',
        0.1250: 'teal'
    }

    # Sort Ein in descending order
    Ein_sorted_indices = np.argsort(Ein)[::-1]

    for idx_graph, light_idx in enumerate(Ein_sorted_indices):
        ax = fig.add_subplot(gs[0, idx_graph])

        for C0_idx in range(4):
            data_idx = light_idx * 4 + C0_idx
            C0_val = C0[C0_idx]

            color = colors_C0.get(C0_val, 'black')

            # Simulation
            y0 = [N0, C0_val]
            sol = solve_ivp(
                Full_model,
                [0, 1200],
                y0,
                args=(alpha, mu, lambda_L, Ein[light_idx], dep, vol, K, Gamma, xi, ODN),
                t_eval=simtime,
                method='LSODA'
            )

            # Convert to biomass (cells/mL)
            biomass_sim = sol.y[0, :] / vol
            ax.plot(sol.t, biomass_sim, color=color, linewidth=2.5, alpha=0.9,
                   linestyle='-', label=f'$C_0={C0_val:.4f}$')

            # Experimental data - convert to cells/mL
            od_exp = od_data[data_idx, :]
            valid = ~np.isnan(od_exp)
            biomass_exp = od_exp[valid] * ODN
            ax.plot(time_data[valid], biomass_exp, marker='o', color=color,
                   linestyle='none', markersize=6, alpha=0.6)

        ein_val = ein[light_idx]
        ax.set_title(f'$L_0$ = {ein_val:.1f} $\\mu mol_{{h\\nu}}$ $m^{{-2}}$ $s^{{-1}}$', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (h)', fontsize=14)
        if idx_graph == 0:
            ax.set_ylabel('Cell density (cell mL$^{-1}$)', fontsize=14)
        ax.set_xlim([0, 1200])
        ax.set_ylim([0, 20 * ODN])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Legend inside the last subplot (lowest Ein)
    legend_elements = []
    for C0_val in sorted(colors_C0.keys(), reverse=True):
        color = colors_C0[C0_val]
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2.5,
                                     label=f'$C_0={C0_val:.4f}$'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
              framealpha=0.9)

    # ---- Row 2: MR Model (C0 = 1) ----
    ax_mr = fig.add_subplot(gs[1, :])

    colors_mr = ['teal', 'tomato', 'grey', 'mediumseagreen']
    Ein_labels = [f'$L_0$ = {ein[i]:.1f} $\\mu mol_{{h\\nu}}$ $m^{{-2}}$ $s^{{-1}}$' for i in range(4)]

    indices_C0_1 = [0, 4, 8, 12]
    light_indices = [i // 4 for i in indices_C0_1]

    for idx, light_idx in enumerate(light_indices):
        sol = solve_ivp(
            MR_model,
            [0, 1200],
            [N0],
            args=(mu, lambda_L, Ein[light_idx], dep, vol, K, Gamma, ODN),
            t_eval=simtime,
            method='LSODA'
        )

        od_sim = sol.y[0, :] / (ODN * vol)
        ax_mr.plot(sol.t, od_sim, color=colors_mr[idx], linestyle='-', linewidth=2.5, alpha=0.8)

        # Experimental data
        od_exp = od_data[indices_C0_1[idx], :]
        valid = ~np.isnan(od_exp)
        ax_mr.scatter(time_data[valid], od_exp[valid], c=colors_mr[idx],
                   s=60, marker='o', linewidths=0.5,
                   label=Ein_labels[idx], alpha=0.7)

    ax_mr.set_xlabel('Time (h)', fontsize=16, fontweight='bold')
    ax_mr.set_ylabel('OD (730 nm)', fontsize=16, fontweight='bold')
    ax_mr.set_title('MR Model ($C_0 = 1$)', fontsize=18, fontweight='bold')
    ax_mr.set_xlim([0, 1200])
    ax_mr.set_ylim([0, 20])
    ax_mr.legend(fontsize=11, ncol=2, loc='upper left')
    ax_mr.grid(True, alpha=0.3, linestyle='--')
    ax_mr.tick_params(labelsize=14)



    plt.savefig('results_Monoraphidium_data/fitting_LMMB.png', dpi=300, bbox_inches='tight')
    print("Saved: fitting_Kambe_data.png")

    return fig

# ============================================================================
# RESULTS TABLE
# ============================================================================

def create_results_table(mu, lambda_L, Gamma, alpha, xi):
    """
    Create comparison table with reference values
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Table 2)")
    print("="*70)
    
    Gamma_OD = Gamma / (ODN * vol)
    
    results = {
        'Parameter': ['μ', 'λ_L', 'Γ', 'Γ_OD', 'ξ', 'α'],
        'Estimated': [
            f'{mu:.4f}',
            f'{lambda_L:.4e}',
            f'{Gamma:.4e}',
            f'{Gamma_OD:.2f}',
            f'{xi:.4f}',
            f'{alpha:.2e}'
        ],
        'Reference': [
            f'{PARAMS_REF["mu"]:.4f}',
            f'{PARAMS_REF["lambda_L"]:.4e}',
            f'{PARAMS_REF["Gamma_OD"]*ODN*vol:.4e}',
            f'{PARAMS_REF["Gamma_OD"]:.2f}',
            f'{PARAMS_REF["xi"]:.4f}',
            f'{PARAMS_REF["alpha"]:.2e}'
        ],
        'Unit': ['h⁻¹', 'μE s⁻¹ cell⁻¹', 'cells', '-', '-', 'cell⁻¹'],
        'Rel_Error_%': [
            f'{abs(mu - PARAMS_REF["mu"])/PARAMS_REF["mu"]*100:.1f}',
f'{abs(lambda_L - PARAMS_REF["lambda_L"])/PARAMS_REF["lambda_L"]*100:.1f}',
            f'{abs(Gamma_OD - PARAMS_REF["Gamma_OD"])/PARAMS_REF["Gamma_OD"]*100:.1f}',
            f'{abs(Gamma_OD - PARAMS_REF["Gamma_OD"])/PARAMS_REF["Gamma_OD"]*100:.1f}',
            f'{abs(xi - PARAMS_REF["xi"])/PARAMS_REF["xi"]*100:.1f}',
            f'{abs(alpha - PARAMS_REF["alpha"])/PARAMS_REF["alpha"]*100:.1f}'
        ]
    }
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    df.to_csv('results_Monoraphidium_data/parameters_comparison_paper_vs_reproduction.csv', index=False, sep=';', encoding='utf-8-sig')
    print("\n✓ Saved: parameters_comparison.csv")
    
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print(" KAMBE ET AL. 2022 - PARAMETER ESTIMATION ")
    print(" A parametric logistic equation with light flux and ")
    print(" medium concentration for cultivation planning of microalgae ")
    print("="*70)
    
    os.makedirs('results_Monoraphidium_data', exist_ok=True)

    # Load experimental data
    time_data, od_data = read_Kambe_data('data_Kambe')
    
    # Estimate MR model parameters
    mu, lambda_L, Gamma = estimate_MR_parameters(time_data, od_data)
    
    # Estimate Full model parameters
    alpha, xi = estimate_Full_parameters(time_data, od_data, mu, lambda_L, Gamma)
    
    # Compute global R²
    R2_global = compute_global_R2(time_data, od_data, mu, lambda_L, Gamma, alpha, xi)
    print(f"\nGlobal R² (Full Model, all conditions) = {R2_global:.4f}")
    
    # Create results table
    df_results = create_results_table(mu, lambda_L, Gamma, alpha, xi)
    
    # Generate figure
    fig = plot_results(time_data, od_data, mu, lambda_L, Gamma, alpha, xi)

    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE ")
    print("="*70)
    print("\nGenerated files in results_Monoraphidium_data/:")
    print("  1. fitting_Kambe_data.png - Full model + MR model fits")
    print("  2. parameters_comparison.csv - Parameter estimates vs reference")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
