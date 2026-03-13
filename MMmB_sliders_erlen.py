import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from data_import import read_csv_data_erlen


# -------------------- Loading t_lag adjustments --------------------
def load_tlag_adjustments(filepath="results_erlen/t_lag_adjustments.txt"):
    """
    Load t_lag values from a text file.

    Expected format:
    (L0, C0): t_lag_hours

    Returns:
        dict: {(L0, C0_factor): t_lag_hours}
    """
    tlag_dict = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    parts = line.split(':')
                    if len(parts) != 2:
                        continue
                    key_str = parts[0].strip().strip('()')
                    t_lag = float(parts[1].strip())
                    l0_str, c0_str = key_str.split(',')
                    l0 = float(l0_str.strip())
                    c0 = float(c0_str.strip())
                    tlag_dict[(l0, c0)] = t_lag
                except (ValueError, IndexError):
                    continue
        return tlag_dict
    except FileNotFoundError:
        print(f"t_lag file not found: {filepath}. Proceeding without t_lag adjustments.")
        return {}

TLAG_ADJUSTMENTS = load_tlag_adjustments()


def system(t, y, params):
    """
    Model:
    dN/dt = mu_max * C(t)/(C(t) + K_C) * L(N(t))/(L(N(t)) + K_L) * N(t) - m * N(t)

    with:
    - m = m_nat + beta * L_0
    - C(t) = C0 - (N - N0)
    - L(N(t)) = L0 * exp(-K * N(t) * depth_culture)
    """
    N = y[0]
    
    N0 = params['N0']
    C0 = params['C0']
    L0 = params['L0']
    depth_culture = params['depth_culture']
    K = params['K']
    mu_max = params['mu_max']
    K_C = params['K_C']
    K_L = params['K_L']
    m_nat = params['m_nat']
    beta = params['beta']
    
    # Compute C(t)
    C = C0 - (N - N0)
    C = max(C, 0)  # Prevent negative values
    #C = C0 - (growth - N0)
    
    # Compute L(N(t))
    L = L0 * np.exp(-K * N * depth_culture)
    
    # Compute m
    m = m_nat + beta * L0
    #m = m_nat + beta * L
    
    # Growth term
    fC = C / (C + K_C)
    fL = L / (L + K_L)
    growth = mu_max * fC * fL * N
    
    # Mortality term
    mortality = m * N
    
    dN = growth - mortality
    
    return [dN]


def plot_with_sliders():
    # Import experimental data from CSV files
    factor = 4.77e6  # OD to cell count conversion

    # Erlen experiment files: (L0 value, L0_factor, filepath)
    erlen_experiments = [
        (11.9, 0.07, "all_data/data_exp_Chlamy_16-09-24.csv"),
        (25.5, 0.15, "all_data/data_exp_Chlamy_21-10-24.csv"),
        (51, 0.3, "all_data/data_exp_Chlamy_17-02-25.csv"),
        (102, 0.6, "all_data/data_exp_Chlamy_01-07-25.csv"),
        (170, 1.0, "all_data/data_exp_Chlamy_07-07-25.csv"),
    ]

    conditions = []
    for L0, L0_factor, filepath in erlen_experiments:
        exp_data = read_csv_data_erlen(filepath, conv_OD_to_cell=factor)

        # Sort experiments by C0_factor descending (1.0, 0.5, 0.25, 0.125, 0.0625)
        sorted_keys = sorted(
            exp_data.keys(),
            key=lambda k: -float(k.split("C0x")[1].split("_")[0])
        )

        times = np.array(exp_data[sorted_keys[0]]["Time"])
        data = {}
        std_data = {}
        for idx, key in enumerate(sorted_keys):
            data[idx + 1] = np.array(exp_data[key]["Mean"])
            std_data[idx + 1] = np.array(exp_data[key]["Std"])

        conditions.append({
            'L0': L0,
            'L0_factor': L0_factor,
            'times': times,
            'data': data,
            'std': std_data,
        })
    
    # C0 factors for each replicate
    C0_factors = {1: 1.0, 2: 1/2, 3: 1/4, 4: 1/8, 5: 1/16}
    
    # Reference initial parameters
    init_params = {
        'L0': 170,              # Reference light intensity (µmol/m²/s)
        'C0': 0.6357e8,         # Reference initial nutrient concentration
        'N0': 6.9e4,            # Initial population
        'depth_culture': 0.01,        # Fixed L parameter
        'K': 0.00000601,        # Extinction coefficient (mL cell⁻¹ m⁻¹)
        'mu_max': 0.2849,       # Maximum growth rate (h⁻¹)
        'K_C': 0.047e8,         # Monod constant for nutrients (dimensionless)
        'K_L': 31,              # Monod constant for light (µmol/m²/s)
        'm_nat': 0.0036,        # Natural mortality (h⁻¹)
        'beta': 0.00015,        # Photo-inhibition coefficient (h⁻¹ (µmol/m²/s)⁻¹)
    }
    
    t_span = (0, 250)
    t_eval = np.linspace(*t_span, 500)
    t_eval = np.linspace(*t_span, 500)
    
    # Create the plot grid (5 rows × 5 columns)
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.30, top=0.92, hspace=0.55, wspace=0.25)
    
    # Experimental data colours
    # j=1→C0×1.0, j=2→C0×0.5, j=3→C0×0.25, j=4→C0×0.125, j=5→C0×0.0625
    colors_exp = ['mediumseagreen', 'grey', 'tomato', 'teal', 'orange']
    model_lines = []
    
    # Initialise subplots
    for i, cond in enumerate(conditions):
        for j in range(1, 6):  # Replicates 1 to 5
            ax = axes[4-i, 5-j]

            # Experimental data with error bars
            ax.errorbar(cond['times'], cond['data'][j], yerr=cond['std'][j],
                       fmt='none', ecolor=colors_exp[j-1], capsize=2,
                       elinewidth=0.8, alpha=0.4)
            ax.scatter(cond['times'], cond['data'][j],
                      color=colors_exp[j-1], s=30, alpha=0.7,
                      label='Exp data')

            # Model line (empty at initialisation)
            line, = ax.plot([], [], 'k-', lw=2, label='Model')

            # R² annotation:
            # Top-left subplot (i==4, j==5): R² at upper right, lowered
            # (legend occupies lower right); all others: lower right
            if i == 4 and j == 5:
                r2_text = ax.text(0.98, 0.90, '', transform=ax.transAxes,
                                 verticalalignment='top', horizontalalignment='right',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                 fontsize=8, fontweight='bold')
            else:
                r2_text = ax.text(0.98, 0.08, '', transform=ax.transAxes,
                                 verticalalignment='bottom', horizontalalignment='right',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                 fontsize=8, fontweight='bold')

            model_lines.append({'line': line, 'ax': ax, 'i': i, 'j': j, 'r2_text': r2_text})

            # Axis configuration
            ax.set_xlim([0, t_span[1]])
            max_val = np.nanmax(cond['data'][j])
            ax.set_ylim([0, max_val * 1.2])
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

            # Titles and labels
            if j == 5:
                ax.set_ylabel(f'$L_0 = {cond["L0"]}$\n$N$ (cells mL$^{{-1}}$)', fontsize=9, fontweight='bold')
            if i == 0:
                ax.set_xlabel('Time (h)', fontsize=9)
            if i == 4:
                ax.set_title(f'$C_0 \\times {C0_factors[j]:.3f}$', fontsize=10, fontweight='bold')

            # Small legend on top-left subplot only
            if i == 4 and j == 5:
                ax.legend(fontsize=7, loc='lower right')
    
    # Sliders for MMmB model parameters
    axcolor = "lightgoldenrodyellow"
    slider_defs = [
        ("L0",       10,   500,  init_params['L0'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0",       1e6,  1e8,  init_params['C0'],
         r"$C_0\ (\mathrm{dimensionless})$"),
        ("mu_max",   0.01, 0.5,  init_params['mu_max'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("K_C",      1e5,  1e8,  init_params['K_C'],
         r"$K_C\ (\mathrm{dimensionless})$"),
        ("K",        1e-8, 1e-4, init_params['K'],
         r"$K\ (\mathrm{mL}\ \mathrm{cell}^{-1}\ \mathrm{m}^{-1})$"),
        ("K_L",      1,    500,  init_params['K_L'],
         r"$K_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("m_nat",    0.0,  0.2,  init_params['m_nat'],
         r"$m\ (\mathrm{h}^{-1})$"),
        ("beta",     0.0,  5e-2, init_params['beta'],
         r"$\beta\ (\mathrm{h}^{-1}\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})^{-1})$"),
    ]

    sliders = {}
    spacing = 0.022
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([0.15, 0.22 - idx*spacing, 0.70, 0.015], facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax, valinit=val, valstep=(vmax-vmin)/1000, color='black')
    
    # Global title showing R² and condition count
    title_text = fig.suptitle('Global R²: Calculating...  |  Conditions R² ≥ 0.85 : .../25',
                              fontsize=11, fontweight='bold')
    
    # Update function
    def update(val):
        # Read slider values
        L0_ref = sliders['L0'].val
        C0_ref = sliders['C0'].val
        
        base_params = {
            'N0': init_params['N0'],
            'depth_culture': init_params['depth_culture'],
            'mu_max': sliders['mu_max'].val,
            'K_C': sliders['K_C'].val,
            'K': sliders['K'].val,
            'K_L': sliders['K_L'].val,
            'm_nat': sliders['m_nat'].val,
            'beta': sliders['beta'].val,
        }
        
        # Variables for global MSE and R² computation
        total_squared_error = 0
        total_points = 0
        all_y_exp = []
        all_y_pred = []
        n_good_conditions = 0

        # Update each curve
        for ml in model_lines:
            i = ml['i']  # Condition index (row)
            j = ml['j']  # Replicate index (column)

            # Apply scaling factors
            L0_cond = L0_ref * conditions[i]['L0_factor']
            C0_cond = C0_ref * C0_factors[j]
            
            params = base_params.copy()
            params['L0'] = L0_cond
            params['C0'] = C0_cond
            
            try:
                y0 = [params['N0']]
                sol = solve_ivp(system, t_span, y0, args=(params,), t_eval=t_eval, method='LSODA')
                
                if sol.success:
                    N_model = sol.y[0]

                    # Apply t_lag time shift
                    L0_actual = conditions[i]['L0']
                    C0_factor_val = C0_factors[j]
                    t_lag = TLAG_ADJUSTMENTS.get((L0_actual, C0_factor_val), 0.0)
                    t_shifted = sol.t + t_lag
                    ml['line'].set_data(t_shifted, N_model)

                    # Compute error for this curve
                    times_exp = conditions[i]['times']
                    N_exp = conditions[i]['data'][j]

                    # Interpolate model at experimental time points (with time shift)
                    N_model_interp = np.interp(times_exp, t_shifted, N_model)
                    
                    # Compute squared error
                    valid_mask = ~np.isnan(N_exp)
                    squared_errors = (N_exp[valid_mask] - N_model_interp[valid_mask]) ** 2
                    total_squared_error += np.sum(squared_errors)
                    total_points += np.sum(valid_mask)
                    
                    # Compute per-subplot R²
                    y_exp_local = N_exp[valid_mask]
                    y_pred_local = N_model_interp[valid_mask]
                    y_mean_local = np.mean(y_exp_local)
                    ss_res_local = np.sum((y_exp_local - y_pred_local) ** 2)
                    ss_tot_local = np.sum((y_exp_local - y_mean_local) ** 2)
                    
                    if ss_tot_local > 0:
                        r2_local = 1 - (ss_res_local / ss_tot_local)
                        ml['r2_text'].set_text(f'R² = {r2_local:.4f}')
                        if r2_local >= 0.85:
                            n_good_conditions += 1
                    else:
                        ml['r2_text'].set_text('R² = N/A')
                    
                    # Collect data for global R²
                    all_y_exp.extend(N_exp[valid_mask])
                    all_y_pred.extend(N_model_interp[valid_mask])
                    
                    # Expand ylim if model exceeds current range
                    max_N = np.nanmax(N_model)
                    current_ylim = ml['ax'].get_ylim()
                    if max_N > current_ylim[1] * 0.9:
                        ml['ax'].set_ylim([0, max_N * 1.2])
                        
            except Exception as e:
                print(f"Error for condition L0={conditions[i]['L0']}, Rep{j}: {e}")
                ml['r2_text'].set_text('R² = Error')
        
        # Compute global MSE and R²
        if total_points > 0:
            global_mse = total_squared_error / total_points

            # Compute R²
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)

            # R² = 1 - (SS_res / SS_tot)
            # SS_res = sum of squared residuals
            # SS_tot = total sum of squares (variance of experimental data)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred) ** 2)
            ss_tot = np.sum((all_y_exp - y_mean) ** 2)
            
            if ss_tot > 0:
                global_r2 = 1 - (ss_res / ss_tot)
                title_text.set_text(f'Global R² = {global_r2:.6f}  |  Conditions R² ≥ 0.85 : {n_good_conditions}/25')
            else:
                title_text.set_text(f'Global R² = N/A  |  Conditions R² ≥ 0.85 : {n_good_conditions}/25')
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    for s in sliders.values():
        s.on_changed(update)
    
    # Initial render
    update(None)
    
    plt.show()


if __name__ == "__main__":
    plot_with_sliders()


