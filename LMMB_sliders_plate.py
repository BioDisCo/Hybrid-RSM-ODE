"""
Interactive visualisation script with sliders for the LMMB model (Kambe).
Uses the same structure and layout as DMB_sliders_plate.py
with three figures: test, calibration, and validation

MODEL: Kambe Full Model (Kambe et al. 2022)
dN/dt = mu * (C/(xi_c + C)) * (L/(lambda_L + L)) * (1 - N/Gamma) * N
dC/dt = -alpha * dN/dt
"""

import glob
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

import data_import
from data_import import Experiments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TIME_HOURS = 500

# -------------------- Loading t_lag --------------------
def load_tlag_adjustments(filepath="results_plates/t_lag_adjustments.txt"):
    """
    Load t_lag values from a text file.

    Expected format:
    (L0, C0): t_lag_hours

    Returns:
        dict: {(L0_factor, C0_factor): t_lag_hours}
    """
    tlag_dict = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                try:
                    # Parse line: (L0, C0): t_lag
                    parts = line.split(':')
                    if len(parts) != 2:
                        continue

                    # Extract (L0, C0)
                    key_str = parts[0].strip()
                    t_lag = float(parts[1].strip())

                    # Convert string tuple to float tuple
                    key_str = key_str.strip('()')
                    l0_str, c0_str = key_str.split(',')
                    l0 = float(l0_str.strip())
                    c0 = float(c0_str.strip())

                    tlag_dict[(l0, c0)] = t_lag

                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid line: {line} ({e})")
                    continue

        logger.info(f"Loaded {len(tlag_dict)} t_lag values from {filepath}")
        return tlag_dict

    except FileNotFoundError:
        logger.warning(f"t_lag file not found: {filepath}. Proceeding without t_lag adjustments.")
        return {}

# Load t_lag values at startup
TLAG_ADJUSTMENTS = load_tlag_adjustments()

# ==================== Kambe model ====================
def lightpercell(L, N, dep, vol, K):
    """
    Calculate light flux per cell (from Kambe methodology)

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


def kambe_full_model(t, y, params):
    """
    Full model with medium concentration (eq 4.1 from Kambe et al. 2022)

    Variables:
    - N: biomass concentration (cells/mL)
    - C: medium concentration (dimensionless, relative to C0=1)

    Equations:
    dN/dt = mu * (C/(xi_c + C)) * (L/(lambda_L + L)) * (1 - N/Gamma) * N
    dC/dt = -alpha * dN/dt

    with:
    - mu: maximum growth rate (h^-1)
    - lambda_L: light half-saturation constant (μmol h^-1 cell^-1)
    - Gamma: maximum cell density (cells/mL)
    - xi_c: nutrient half-saturation constant (dimensionless)
    - alpha: nutrient consumption coefficient (ml cell^-1)
    - L: light per cell, calculated from incident light and self-shading
    """
    N, C = y

    # Parameters
    mu = params['mu']
    lambda_L = params['lambda_L']
    Gamma = params['Gamma']
    xi_c = params['xi_c']
    alpha = params['alpha']
    Ein_val = params['Ein_val']
    dep = params['dep']
    vol = params['vol']
    K = params['K']

    # Prevent negative values
    N = max(N, 1e-10)
    C = max(C, 0)

    # Compute light per cell
    L = lightpercell(Ein_val, N, dep, vol, K)

    # Differential equations
    dNdt = mu * (C / (xi_c + C)) * (L / (lambda_L + L)) * (1 - N/Gamma) * N
    dCdt = -alpha * dNdt

    return [dNdt, dCdt]


# ==================== Data loading ====================
def load_intermediate_data_with_timeseries(
    conv_OD_plate_to_OD_erlen: float = 6.01,
    conv_OD_to_cell: float = 4.77e6
) -> dict[tuple[float, float], dict]:
    """Load intermediate data with validation time series."""
    date_to_l0 = {
        "01_07_2025": 0.6,
        "04_11_2024": 0.15,
        "07_07_2025": 1.0,
        "17_02_2025": 0.3,
    }
    intermediate_files = sorted(glob.glob("all_data/final_corrected_data/replicate_OD_intermediate_dilution_*.csv"))
    intermediate_data = {}
    for fp in intermediate_files:
        filename = pathlib.Path(fp).name
        date = None
        for d, l0 in date_to_l0.items():
            if d in filename:
                date = d
                l0_value = l0
                break
        if date is None:
            continue
        try:
            df = pd.read_csv(fp, sep=";")
            time_col = df.columns[0]
            time = df[time_col].values
            c0_values = set()
            for col in df.columns[1:]:
                if "C0 = " in col:
                    c0_str = col.split("C0 = ")[1].split(" ")[0]
                    try:
                        c0_values.add(float(c0_str))
                    except ValueError:
                        pass
            for c0 in sorted(c0_values):
                rep_cols = [col for col in df.columns[1:] if f"C0 = {c0}" in col]
                if not rep_cols:
                    continue
                replicates = []
                all_values = []
                for col in rep_cols:
                    values = df[col].values * conv_OD_plate_to_OD_erlen * conv_OD_to_cell
                    replicates.append({
                        "Time": time.tolist(),
                        "Value": values.tolist()
                    })
                    all_values.append(values)
                mean_values = np.nanmean(all_values, axis=0)
                key = (c0, l0_value)
                intermediate_data[key] = {
                    'time': time,
                    'mean': mean_values,
                    'replicates': replicates
                }
        except Exception as e:
            logger.warning(f"Failed to load intermediate data from {fp}: {e}")
    return intermediate_data


# ==================== test figure ====================
def plot_with_sliders_test():
    """Interactive figure with sliders showing only the 7 specific test conditions."""
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Load experimental data
    all_plate_files = sorted(glob.glob("all_data/hand_cleaned/replicates_OD_*.csv"))
    plate_files = all_plate_files

    experiments_plate: Experiments = {}
    for fp in plate_files:
        try:
            new_exp = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell
            )
            experiments_plate.update(new_exp)
            logger.info(f"Loaded {fp}")
        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")

    intermediate_data_ts = load_intermediate_data_with_timeseries(
        conv_OD_plate_to_OD_erlen, conv_OD_to_cell
    )

    # Group conditions
    all_conditions = {}
    for exp_name, exp_data in experiments_plate.items():
        key = (exp_data["C0_factor"], exp_data["L0_factor"])
        if key not in all_conditions:
            all_conditions[key] = {"type": "experiment", "data": exp_data}
    for (c0, l0), data_dict in intermediate_data_ts.items():
        key = (c0, l0)
        if key not in all_conditions:
            all_conditions[key] = {"type": "intermediate",
                                   "time": data_dict['time'],
                                   "mean": data_dict['mean'],
                                   "replicates": data_dict['replicates']}

    # Filter for the 7 specific test conditions
    test_pairs = [
        (1.0, 1.0),
        (0.5, 1.0),
        (0.125, 1.0),
        (1.0, 0.6),
        (0.25, 0.6),
        (0.25, 0.3),
        (1.0, 0.15)
    ]

    test_conditions = {k: v for k, v in all_conditions.items() if k in test_pairs}
    # Sort by decreasing L0, then increasing C0
    sorted_keys = sorted(test_conditions.keys(), key=lambda x: (-x[1], x[0]))

    n_conditions = len(sorted_keys)
    if n_conditions == 0:
        logger.error("No test conditions found!")
        return

    # Initial Kambe model parameters
    init_params = {
        'mu': 0.153,
        'lambda_L': 0.012e-5,
        'Gamma': 0.64e8,
        'xi_c': 0.011,
        'alpha': 0.016e-6,
        'L0_ref': 170,
        'dep': 0.5,      # culture depth [cm] - plate
        'vol': 0.2,      # culture volume [ml] - plate
        'K': 1e-8        # extinction coefficient [ml cm^-1 cell^-1]
    }

    # Simulation configuration
    light_area = 0.000032  # m² (plate well area)
    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(0, MAX_TIME_HOURS, 1000)

    # Create grid layout for 7 subplots (2 rows x 4 columns)
    fig = plt.figure(figsize=(16, 8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, left=0.05, right=0.95,
                  top=0.88, bottom=0.30, hspace=0.3, wspace=0.3)

    plot_elements = []

    # Create subplots for the 7 conditions
    for idx, (c0, l0) in enumerate(sorted_keys):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        condition = test_conditions[(c0, l0)]
        # Extract data
        if condition["type"] == "experiment":
            exp_data = condition["data"]
            time = np.array(exp_data["Time"])
            concentration = np.array(exp_data["Mean"])
            replicates = exp_data["replicates"]
        else:
            time = condition["time"]
            concentration = condition["mean"]
            replicates = condition["replicates"]

        # Plot replicates
        for replicate in replicates:
            ax.scatter(replicate["Time"], replicate["Value"],
                      color='grey', s=8, alpha=0.15)

        # Plot mean
        ax.scatter(time, concentration, color='orange', s=15, alpha=0.7,
                  label='Exp data')

        # Model line
        line, = ax.plot([], [], 'black', linewidth=1.2, label='Model')

        r2_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                          fontsize=7, verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.85,
                                   edgecolor='gray', linewidth=0.5))

        plot_elements.append({
            'ax': ax,
            'line': line,
            'r2_text': r2_text,
            'c0': c0,
            'l0': l0,
            'time_exp': time,
            'conc_exp': concentration
        })

        ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=8, pad=3)
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.set_xlim(0, MAX_TIME_HOURS)
        ax.tick_params(axis='both', labelsize=6)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

    # Disable unused axis (position 7 in a 2x4 grid)
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis('off')

    fig.text(0.5, 0.25, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'test (7 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/7',
        fontsize=11, fontweight='bold')

    # ==================== Sliders ====================
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("mu",       0.01,  1.0,   init_params['mu'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("lambda_L", 1e-8,  1e-5,  init_params['lambda_L'],
         r"$\lambda_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{h}^{-1}\ \mathrm{cell}^{-1})$"),
        ("Gamma",    1e6,   1e8,   init_params['Gamma'],
         r"$\Gamma\ (\mathrm{cells}\ \mathrm{mL}^{-1})$"),
        ("xi_c",     0.001, 0.5,   init_params['xi_c'],
         r"$\xi_C\ (\mathrm{dimensionless})$"),
        ("alpha",    1e-9,  1e-6,  init_params['alpha'],
         r"$\alpha\ (\mathrm{mL}\ \mathrm{cell}^{-1})$"),
        ("L0_ref",   10,    500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",   0.1,   10.0,  1.0,
         r"$C_0\ (\mathrm{dimensionless})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.23 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # ==================== Update function ====================
    def update(val):
        all_y_exp = []
        all_y_pred = []
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        mu = sliders['mu'].val
        lambda_L = sliders['lambda_L'].val
        Gamma = sliders['Gamma'].val
        xi_c = sliders['xi_c'].val
        alpha = sliders['alpha'].val
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val

        for elem in plot_elements:
            # Compute Ein_val for this condition
            L0_cond = L0_ref * elem['l0']
            Ein_val = L0_cond * light_area * 3600  # Convert µmol/m²/s to µmol/h

            # Initial nutrient concentration
            C0_cond = C0_ref * elem['c0']

            # ODE parameters
            params = {
                'mu': mu,
                'lambda_L': lambda_L,
                'Gamma': Gamma,
                'xi_c': xi_c,
                'alpha': alpha,
                'Ein_val': Ein_val,
                'dep': init_params['dep'],
                'vol': init_params['vol'],
                'K': init_params['K']
            }

            try:
                # Initial conditions: [N0, C0]
                N0 = elem['conc_exp'][0] if elem['conc_exp'][0] > 0 else 1e5
                y0 = [N0, C0_cond]

                sol = solve_ivp(kambe_full_model, t_span, y0, args=(params,),
                              t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)

                if sol.success:
                    N_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, N_model)

                    # Interpolate at experimental time points
                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    N_interp = np.interp(time_exp, t_shifted, N_model)

                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = N_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                        # Count subplots with R² >= 0.85
                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        # Background colour
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local >= 0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(N_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val * 1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred) ** 2)
            ss_tot = np.sum((all_y_exp - y_mean) ** 2)
            global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            r2_val = f'{global_r2:.6f}' if not np.isnan(global_r2) else 'N/A'
            title_text.set_text(
                f'test (7 conditions)  |  Global R² = {r2_val}  |  '
                f'Conditions R² ≥ 0.85 (green) : {n_good_r2}/{n_total_graphs}'
            )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()


# ==================== calibration figure ====================
def plot_with_sliders_calibration():
    """Interactive figure with sliders showing only C0 = 1, 0.5, 0.25, 0.125, 0.0625 for all L0 values."""
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Load experimental data
    all_plate_files = sorted(glob.glob("all_data/hand_cleaned/replicates_OD_*.csv"))
    plate_files = all_plate_files

    experiments_plate: Experiments = {}
    for fp in plate_files:
        try:
            new_exp = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell
            )
            experiments_plate.update(new_exp)
        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")

    intermediate_data_ts = load_intermediate_data_with_timeseries(
        conv_OD_plate_to_OD_erlen, conv_OD_to_cell
    )

    # Group conditions
    all_conditions = {}
    for exp_name, exp_data in experiments_plate.items():
        key = (exp_data["C0_factor"], exp_data["L0_factor"])
        if key not in all_conditions:
            all_conditions[key] = {"type": "experiment", "data": exp_data}
    for (c0, l0), data_dict in intermediate_data_ts.items():
        key = (c0, l0)
        if key not in all_conditions:
            all_conditions[key] = {"type": "intermediate",
                                   "time": data_dict['time'],
                                   "mean": data_dict['mean'],
                                   "replicates": data_dict['replicates']}

    # Filter for C0 = 1, 0.5, 0.25, 0.125, 0.0625
    calibration_conditions = {k: v for k, v in all_conditions.items()
                          if k[0] in [1.0, 0.5, 0.25, 0.125, 0.0625]}

    # Organise as grid: C0 in columns (ascending), L0 in rows (descending via reversed)
    c0_values = sorted(set(k[0] for k in calibration_conditions.keys()))
    l0_values = sorted(set(k[1] for k in calibration_conditions.keys()))

    n_c0 = len(c0_values)
    n_l0 = len(l0_values)

    # Initial Kambe model parameters
    init_params = {
        'mu': 0.153,
        'lambda_L': 0.012e-5,
        'Gamma': 0.64e8,
        'xi_c': 0.008,
        'alpha': 0.016e-6,
        'L0_ref': 170,
        'dep': 0.5,      # culture depth [cm] - plate
        'vol': 0.2,      # culture volume [ml] - plate
        'K': 1e-8        # extinction coefficient [ml cm^-1 cell^-1]
    }

    # Simulation configuration
    light_area = 0.000032
    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(0, MAX_TIME_HOURS, 1000)

    fig = plt.figure(figsize=(1.2*n_c0, 2*n_l0 + 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_l0, n_c0, figure=fig, left=0.05, right=0.95,
                  top=0.95, bottom=0.30, hspace=0.3, wspace=0.3)

    axes = np.empty((n_l0, n_c0), dtype=object)
    for i in range(n_l0):
        for j in range(n_c0):
            axes[i, j] = fig.add_subplot(gs[i, j])

    plot_elements = []

    # Create subplots
    for i, l0 in enumerate(reversed(l0_values)):
        for j, c0 in enumerate(c0_values):
            ax = axes[i, j]
            key = (c0, l0)
            if key in calibration_conditions:
                condition = calibration_conditions[key]

                # Extract data
                if condition["type"] == "experiment":
                    exp_data = condition["data"]
                    time = np.array(exp_data["Time"])
                    concentration = np.array(exp_data["Mean"])
                    replicates = exp_data["replicates"]
                else:
                    time = condition["time"]
                    concentration = condition["mean"]
                    replicates = condition["replicates"]

                # Plot replicates
                for replicate in replicates:
                    ax.scatter(replicate["Time"], replicate["Value"],
                              color='grey', s=8, alpha=0.15)

                # Plot mean
                ax.scatter(time, concentration, color='orange', s=15, alpha=0.7,
                          label='Exp data')

                # Model line
                line, = ax.plot([], [], 'black', linewidth=1.2, label='Model')

                r2_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                                  fontsize=5.5, verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.85,
                                           edgecolor='gray', linewidth=0.5))

                plot_elements.append({
                    'ax': ax,
                    'line': line,
                    'r2_text': r2_text,
                    'c0': c0,
                    'l0': l0,
                    'time_exp': time,
                    'conc_exp': concentration
                })

                ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=6, pad=2)
                ax.grid(True, alpha=0.15, linewidth=0.3)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.tick_params(axis='y', labelsize=5)
                if i < n_l0 - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=5)
                if i == 0 and j == 0:
                    ax.legend(fontsize=5, loc='lower right', framealpha=0.8)
            else:
                ax.axis('off')

    fig.text(0.5, 0.27, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'test (25 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/25',
        fontsize=11, fontweight='bold')

    # ==================== Sliders ====================
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("mu",       0.01,  1.0,   init_params['mu'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("lambda_L", 1e-8,  1e-5,  init_params['lambda_L'],
         r"$\lambda_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{h}^{-1}\ \mathrm{cell}^{-1})$"),
        ("Gamma",    1e6,   1e8,   init_params['Gamma'],
         r"$\Gamma\ (\mathrm{cells}\ \mathrm{mL}^{-1})$"),
        ("xi_c",     0.001, 0.5,   init_params['xi_c'],
         r"$\xi_C\ (\mathrm{dimensionless})$"),
        ("alpha",    1e-9,  1e-6,  init_params['alpha'],
         r"$\alpha\ (\mathrm{mL}\ \mathrm{cell}^{-1})$"),
        ("L0_ref",   10,    500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",   0.1,   10.0,  1.0,
         r"$C_0\ (\mathrm{dimensionless})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.25 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # ==================== Update function ====================
    def update(val):
        all_y_exp = []
        all_y_pred = []
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        mu = sliders['mu'].val
        lambda_L = sliders['lambda_L'].val
        Gamma = sliders['Gamma'].val
        xi_c = sliders['xi_c'].val
        alpha = sliders['alpha'].val
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val

        for elem in plot_elements:
            L0_cond = L0_ref * elem['l0']
            Ein_val = L0_cond * light_area * 3600
            C0_cond = C0_ref * elem['c0']

            params = {
                'mu': mu,
                'lambda_L': lambda_L,
                'Gamma': Gamma,
                'xi_c': xi_c,
                'alpha': alpha,
                'Ein_val': Ein_val,
                'dep': init_params['dep'],
                'vol': init_params['vol'],
                'K': init_params['K']
            }

            try:
                N0 = elem['conc_exp'][0] if elem['conc_exp'][0] > 0 else 1e5
                y0 = [N0, C0_cond]

                sol = solve_ivp(kambe_full_model, t_span, y0, args=(params,),
                              t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)

                if sol.success:
                    N_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, N_model)

                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    N_interp = np.interp(time_exp, t_shifted, N_model)
                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = N_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local >= 0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(N_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val * 1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred) ** 2)
            ss_tot = np.sum((all_y_exp - y_mean) ** 2)
            global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            r2_val = f'{global_r2:.6f}' if not np.isnan(global_r2) else 'N/A'
            title_text.set_text(
                f'test (25 conditions)  |  Global R² = {r2_val}  |  '
                f'Conditions R² ≥ 0.85 (green) : {n_good_r2}/{n_total_graphs}'
            )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()


# ==================== validation figure ====================
def plot_with_sliders_validation():
    """Interactive figure with sliders showing all conditions."""
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Load experimental data
    all_plate_files = sorted(glob.glob("all_data/hand_cleaned/replicates_OD_*.csv"))
    plate_files = all_plate_files

    experiments_plate: Experiments = {}
    for fp in plate_files:
        try:
            new_exp = data_import.read_csv_data_plate_hand_cleaned(
                fp,
                conv_OD_plate_to_OD_erlen=conv_OD_plate_to_OD_erlen,
                conv_OD_to_cell=conv_OD_to_cell
            )
            experiments_plate.update(new_exp)
        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")

    intermediate_data_ts = load_intermediate_data_with_timeseries(
        conv_OD_plate_to_OD_erlen, conv_OD_to_cell
    )

    # Group all conditions
    all_conditions = {}
    for exp_name, exp_data in experiments_plate.items():
        key = (exp_data["C0_factor"], exp_data["L0_factor"])
        if key not in all_conditions:
            all_conditions[key] = {"type": "experiment", "data": exp_data}
    for (c0, l0), data_dict in intermediate_data_ts.items():
        key = (c0, l0)
        if key not in all_conditions:
            all_conditions[key] = {"type": "intermediate",
                                   "time": data_dict['time'],
                                   "mean": data_dict['mean'],
                                   "replicates": data_dict['replicates']}

    # Organise as grid: C0 in columns (ascending), L0 in rows (descending via reversed)
    c0_values = sorted(set(k[0] for k in all_conditions.keys()))
    l0_values = sorted(set(k[1] for k in all_conditions.keys()))

    n_c0 = len(c0_values)
    n_l0 = len(l0_values)

    # Initial Kambe model parameters
    init_params = {
        'mu': 0.167,
        'lambda_L': 0.012e-5,
        'Gamma': 0.64e8,
        'xi_c': 0.008,
        'alpha': 0.013e-6,
        'L0_ref': 170,
        'dep': 0.5,      # culture depth [cm] - plate
        'vol': 0.2,      # culture volume [ml] - plate
        'K': 1e-8        # extinction coefficient [ml cm^-1 cell^-1]
    }

    # Simulation configuration
    light_area = 0.000032
    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(0, MAX_TIME_HOURS, 1000)

    fig = plt.figure(figsize=(1.2*n_c0, 2*n_l0 + 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_l0, n_c0, figure=fig, left=0.05, right=0.95,
                  top=0.95, bottom=0.30, hspace=0.3, wspace=0.3)

    axes = np.empty((n_l0, n_c0), dtype=object)
    for i in range(n_l0):
        for j in range(n_c0):
            axes[i, j] = fig.add_subplot(gs[i, j])

    plot_elements = []

    # Create subplots
    for i, l0 in enumerate(reversed(l0_values)):
        for j, c0 in enumerate(c0_values):
            ax = axes[i, j]
            key = (c0, l0)
            if key in all_conditions:
                condition = all_conditions[key]

                # Extract data
                if condition["type"] == "experiment":
                    exp_data = condition["data"]
                    time = np.array(exp_data["Time"])
                    concentration = np.array(exp_data["Mean"])
                    replicates = exp_data["replicates"]
                else:
                    time = condition["time"]
                    concentration = condition["mean"]
                    replicates = condition["replicates"]

                # Plot replicates
                for replicate in replicates:
                    ax.scatter(replicate["Time"], replicate["Value"],
                              color='grey', s=8, alpha=0.15)

                # Plot mean
                ax.scatter(time, concentration, color='orange', s=15, alpha=0.7,
                          label='Exp data')

                # Model line
                line, = ax.plot([], [], 'black', linewidth=1.2, label='Model')

                r2_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                                  fontsize=5.5, verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.85,
                                           edgecolor='gray', linewidth=0.5))

                plot_elements.append({
                    'ax': ax,
                    'line': line,
                    'r2_text': r2_text,
                    'c0': c0,
                    'l0': l0,
                    'time_exp': time,
                    'conc_exp': concentration
                })

                ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=6, pad=2)
                ax.grid(True, alpha=0.15, linewidth=0.3)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.tick_params(axis='y', labelsize=5)
                if i < n_l0 - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=5)
                if i == 0 and j == 0:
                    ax.legend(fontsize=5, loc='lower right', framealpha=0.8)
            else:
                ax.axis('off')

    fig.text(0.5, 0.27, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'Validation (85 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/85',
        fontsize=11, fontweight='bold')

    # ==================== Sliders ====================
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("mu",       0.01,  1.0,   init_params['mu'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("lambda_L", 1e-8,  1e-5,  init_params['lambda_L'],
         r"$\lambda_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{h}^{-1}\ \mathrm{cell}^{-1})$"),
        ("Gamma",    1e6,   1e8,   init_params['Gamma'],
         r"$\Gamma\ (\mathrm{cells}\ \mathrm{mL}^{-1})$"),
        ("xi_c",     0.001, 0.5,   init_params['xi_c'],
         r"$\xi_C\ (\mathrm{dimensionless})$"),
        ("alpha",    1e-9,  1e-6,  init_params['alpha'],
         r"$\alpha\ (\mathrm{mL}\ \mathrm{cell}^{-1})$"),
        ("L0_ref",   10,    500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",   0.1,   10.0,  1.0,
         r"$C_0\ (\mathrm{dimensionless})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.25 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # ==================== Update function ====================
    def update(val):
        all_y_exp = []
        all_y_pred = []
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        mu = sliders['mu'].val
        lambda_L = sliders['lambda_L'].val
        Gamma = sliders['Gamma'].val
        xi_c = sliders['xi_c'].val
        alpha = sliders['alpha'].val
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val

        for elem in plot_elements:
            L0_cond = L0_ref * elem['l0']
            Ein_val = L0_cond * light_area * 3600
            C0_cond = C0_ref * elem['c0']

            params = {
                'mu': mu,
                'lambda_L': lambda_L,
                'Gamma': Gamma,
                'xi_c': xi_c,
                'alpha': alpha,
                'Ein_val': Ein_val,
                'dep': init_params['dep'],
                'vol': init_params['vol'],
                'K': init_params['K']
            }

            try:
                N0 = elem['conc_exp'][0] if elem['conc_exp'][0] > 0 else 1e5
                y0 = [N0, C0_cond]

                sol = solve_ivp(kambe_full_model, t_span, y0, args=(params,),
                              t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)

                if sol.success:
                    N_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, N_model)

                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    N_interp = np.interp(time_exp, t_shifted, N_model)
                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = N_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local >= 0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(N_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val * 1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred) ** 2)
            ss_tot = np.sum((all_y_exp - y_mean) ** 2)
            global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            r2_val = f'{global_r2:.6f}' if not np.isnan(global_r2) else 'N/A'
            title_text.set_text(
                f'Validation (85 conditions)  |  Global R² = {r2_val}  |  '
                f'Conditions R² ≥ 0.85 (green) : {n_good_r2}/{n_total_graphs}'
            )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()


# ==================== MAIN ====================
if __name__ == "__main__":
    # Display the three figures in order
    logger.info("="*60)
    logger.info("Generating test figure (7 specific conditions)")
    logger.info("Conditions: (1,1), (0.5,1), (0.125,1), (1,0.6), (0.25,0.6), (0.25,0.3), (1,0.15)")
    logger.info("="*60)
    plot_with_sliders_test()

    logger.info("="*60)
    logger.info("Generating calibration figure (C0=1, 0.5, 0.25, 0.125, 0.0625)")
    logger.info("="*60)
    plot_with_sliders_calibration()

    logger.info("="*60)
    logger.info("Generating validation figure (all conditions)")
    logger.info("="*60)
    plot_with_sliders_validation()


"""
Reference initialisation values:

    init_params = {
        'mu': 0.12,
        'lambda_L': 0.012e-5,
        'Gamma': 0.87e8,
        'xi_c': 0.011,
        'alpha': 0.014e-6,
        'L0_ref': 170,
        'dep': 0.5,      # culture depth [cm] - plate
        'vol': 0.2,      # culture volume [ml] - plate
        'K': 1e-8        # extinction coefficient [ml cm^-1 cell^-1]
    }
"""
