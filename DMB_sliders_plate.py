"""
Interactive visualisation script with sliders for
the Droop-Light model (Martínez 2020) on microplate data
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

# -------------------- Loading t_lag adjustments --------------------
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

# -------------------- Droop-Light model --------------------
def droop_light_system(t, y, params):
    """
    Light-limited Droop model (Martínez 2020) - BATCH MODE

    Variables :
    - x: biomass concentration (cells mL⁻¹)
    - q: internal nutrient quota (cell⁻¹ mL)
    - s: external nutrient concentration (eq-cells mL⁻¹)
    """
    x, q, s = y
    # Parameters
    mu_max = params['mu_max']
    Q0 = params['Q0']
    K_L = params['K_L']
    L0 = params['L0']
    K = params['K']
    K_bg = params['K_bg']
    L = params['L']
    rho_max = params['rho_max']
    K_C = params['K_C']
    Q_L = params['Q_L']

    # Prevent negative values
    x = max(x, 1e-10)
    q = max(q, Q0)
    s = max(s, 0)

    # 1. μ_I(t,x)
    if x > 0 and L0 > 0:
        I_out = L0 * np.exp(-(K * x + K_bg) * L)
        if I_out < L0:
            mu_I = (mu_max / ((K * x + K_bg) * L)) * np.log((K_L + L0) / (K_L + I_out))
        else:
            mu_I = mu_max * L0 / (K_L + L0)
    else:
        mu_I = 0

    # 2. μ_P(q) (Droop)
    mu_P = mu_max * (1 - Q0 / q) if q > Q0 else 0

    # 3. μ = min(μ_I, μ_P)
    mu = min(mu_I, mu_P)

    # 4. Nutrient uptake ρ(q,s)
    rho = rho_max * (s / (K_C + s)) * ((Q_L - q) / (Q_L - Q0)) if q < Q_L else 0

    dx_dt = mu * x
    dq_dt = rho - mu * q
    ds_dt = -rho * x

    return [dx_dt, dq_dt, ds_dt]


# -------------------- Data loading --------------------
def load_intermediate_data_with_timeseries(
    conv_OD_plate_to_OD_erlen: float = 6.01,
    conv_OD_to_cell: float = 4.77e6
) -> dict[tuple[float, float], dict]:
    """Load intermediate data with complete time series."""
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
                    # Convert all values without truncation
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


# -------------------- Interactive visualisation - test figure --------------------
def plot_with_sliders_calibration():
    """Interactive figure with sliders showing only the 7 specific calibration conditions."""
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Experimental data - INCLUDE all data, including 16_09_2024
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

    # FILTER for the 7 specific calibration conditions
    calibration_pairs = [
        (1.0, 1.0),
        (0.5, 1.0),
        (0.125, 1.0),
        (1.0, 0.6),
        (0.25, 0.6),
        (0.25, 0.3),
        (1.0, 0.15)
    ]

    calibration_conditions = {k: v for k, v in all_conditions.items() if k in calibration_pairs}

    # Sort conditions for consistent display
    sorted_keys = sorted(calibration_conditions.keys(), key=lambda x: (-x[1], -x[0]))

    n_conditions = len(sorted_keys)

    logger.info(f"[CALIBRATION] Number of conditions: {n_conditions}")
    logger.info(f"[CALIBRATION] Conditions (C0, L0): {sorted_keys}")

    # Initial Droop-Light model parameters (ORIGINAL VALUES)
    init_params = {
        'L0_ref': 170,
        'C0_ref': 0.93e7,
        'mu_max': 0.302,
        'Q0': 0.132,
        'K_L': 58,
        'K': 0.00004115,
        'K_bg': 0,
        'L': 0.005,
        'rho_max': 0.19,
        'K_C': 0.247e8,
        'Q_L': 0.65,
        'Q0_init': 0.13
    }

    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(*t_span, 500)

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

        condition = calibration_conditions[(c0, l0)]
        if condition["type"] == "experiment":
            time = np.array(condition["data"]["Time"])
            concentration = np.array(condition["data"]["Mean"])
            replicates = condition["data"]["replicates"]
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
            'conc_exp': concentration,
            'condition_type': condition["type"]
        })
        ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=8, pad=3)
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.set_xlim(0, MAX_TIME_HOURS)
        ax.tick_params(axis='both', labelsize=6)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

    # Disable unused axes (position 7 in a 2x4 grid)
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis('off')

    fig.text(0.5, 0.25, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'Calibration (7 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/7',
        fontsize=11, fontweight='bold')

    # -------------------- Sliders --------------------
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("L0_ref",  10,     500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",  1e6,    1e8,   init_params['C0_ref'],
         r"$C_0$ (eq-cells mL⁻¹)"),
        ("mu_max",  0.01,   0.5,   init_params['mu_max'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("Q0",      0.01,   0.5,   init_params['Q0'],
         r"$Q_0\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_L",     1,      200,   init_params['K_L'],
         r"$K_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("K",       1e-9,   1e-4,  init_params['K'],
         r"$K\ (\mathrm{cell}^{-1}\ \mathrm{mL}\ \mathrm{m}^{-1})$"),
        ("rho_max", 0.01,   1.0,   init_params['rho_max'],
         r"$\rho_{\max}\ (\mathrm{h}^{-1}\ \mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_C",     1e6,    1e8,   init_params['K_C'],
         r"$K_C$ (eq-cells mL⁻¹)"),
        ("Q_L",     0.5,    5.0,   init_params['Q_L'],
         r"$Q_L\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("Q0_init", 0.1,    2.0,   init_params['Q0_init'],
         r"$Q(t{=}0)\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.23 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # -------------------- Update function --------------------
    def update(val):
        all_y_exp = []
        all_y_pred = []

        # Counters for subplots with good R²
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val

        base_params = {
            'mu_max': sliders['mu_max'].val,
            'Q0': sliders['Q0'].val,
            'K_L': sliders['K_L'].val,
            'K': sliders['K'].val,
            'K_bg': init_params['K_bg'],
            'L': init_params['L'],
            'rho_max': sliders['rho_max'].val,
            'K_C': sliders['K_C'].val,
            'Q_L': sliders['Q_L'].val,
        }

        for elem in plot_elements:
            # Apply scaling factors
            L0_cond = L0_ref * elem['l0']
            C0_cond = C0_ref * elem['c0']
            Q0_init_cond = sliders['Q0_init'].val

            params = base_params.copy()
            params['L0'] = L0_cond

            try:
                x0 = elem['conc_exp'][0] if not np.isnan(elem['conc_exp'][0]) else 1e5
                y0 = [x0, Q0_init_cond, C0_cond]

                sol = solve_ivp(
                    droop_light_system, t_span, y0, args=(params,),
                    t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10
                )

                if sol.success:
                    x_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, x_model)

                    # Local R²
                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    x_interp = np.interp(time_exp, t_shifted, x_model)
                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = x_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot>0 else np.nan

                        # Count subplots with R² >= 0.85
                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        # Background colour
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local>=0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(x_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val*1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred)**2)
            ss_tot = np.sum((all_y_exp - y_mean)**2)
            global_r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

            # Build title with global R² and subplot counter
            r2_val = f'{global_r2:.6f}' if not np.isnan(global_r2) else 'N/A'
            title_text.set_text(
                f'Calibration (7 conditions)  |  Global R² = {r2_val}  |  '
                f'Conditions R² ≥ 0.85 (green) : {n_good_r2}/{n_total_graphs}'
            )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()


# -------------------- Interactive visualisation - calibration figure --------------------
def plot_with_sliders_filtered():
    """Interactive figure with sliders showing only C0 = 1, 0.5, 0.25, 0.125, 0.0625 for all L0 values."""
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Experimental data
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

    # Filter to keep only specific C0 values
    target_c0 = [1.0, 0.5, 0.25, 0.125, 0.0625]
    filtered_conditions = {k: v for k, v in all_conditions.items() if k[0] in target_c0}

    unique_c0 = sorted(set(c0 for c0, l0 in filtered_conditions.keys()))
    unique_l0 = sorted(set(l0 for c0, l0 in filtered_conditions.keys()))
    n_c0, n_l0 = len(unique_c0), len(unique_l0)

    logger.info(f"[FILTERED] Unique C0 factors: {unique_c0}")
    logger.info(f"[FILTERED] Unique L0 factors: {unique_l0}")
    logger.info(f"[FILTERED] Grid dimensions: {n_l0} rows × {n_c0} columns")

    # Initial Droop-Light model parameters
    init_params = {
        'L0_ref': 170,
        'C0_ref': 0.93e7,
        'mu_max': 0.302,
        'Q0': 0.132,
        'K_L': 58,
        'K': 0.00004115,
        'K_bg': 0,
        'L': 0.005,
        'rho_max': 0.19,
        'K_C': 0.247e8,
        'Q_L': 0.65,
        'Q0_init': 0.13
    }

    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(*t_span, 500)

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
    for i, l0 in enumerate(reversed(unique_l0)):
        for j, c0 in enumerate(unique_c0):
            ax = axes[i, j]
            key = (c0, l0)
            if key in filtered_conditions:
                condition = filtered_conditions[key]
                if condition["type"] == "experiment":
                    time = np.array(condition["data"]["Time"])
                    concentration = np.array(condition["data"]["Mean"])
                    replicates = condition["data"]["replicates"]
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
                    'conc_exp': concentration,
                    'condition_type': condition["type"]
                })
                ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=6, pad=2)
                ax.grid(True, alpha=0.15, linewidth=0.3)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j > 0: ax.tick_params(labelleft=False)
                else: ax.tick_params(axis='y', labelsize=5)
                if i < n_l0-1: ax.tick_params(labelbottom=False)
                else: ax.tick_params(axis='x', labelsize=5)
                if i == 0 and j == 0: ax.legend(fontsize=5, loc='lower right',
                                               framealpha=0.8)
            else:
                ax.axis('off')

    fig.text(0.5, 0.27, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'Calibration (25 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/25',
        fontsize=11, fontweight='bold')

    # -------------------- Sliders --------------------
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("L0_ref",  10,     500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",  1e6,    1e7,   init_params['C0_ref'],
         r"$C_0$ (eq-cells mL⁻¹)"),
        ("mu_max",  0.01,   0.5,   init_params['mu_max'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("Q0",      0.01,   0.5,   init_params['Q0'],
         r"$Q_0\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_L",     1,      200,   init_params['K_L'],
         r"$K_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("K",       0.1e-8, 1e-4,  init_params['K'],
         r"$K\ (\mathrm{cell}^{-1}\ \mathrm{mL}\ \mathrm{m}^{-1})$"),
        ("rho_max", 0.01,   1.0,   init_params['rho_max'],
         r"$\rho_{\max}\ (\mathrm{h}^{-1}\ \mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_C",     1e6,    1e8,   init_params['K_C'],
         r"$K_C$ (eq-cells mL⁻¹)"),
        ("Q_L",     0.5,    5.0,   init_params['Q_L'],
         r"$Q_L\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("Q0_init", 0.1,    2.0,   init_params['Q0_init'],
         r"$Q(t{=}0)\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.25 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # -------------------- Update function --------------------
    def update(val):
        all_y_exp = []
        all_y_pred = []

        # Counters for subplots with good R²
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val
        Q0_init = sliders['Q0_init'].val

        base_params = {
            'mu_max': sliders['mu_max'].val,
            'Q0': sliders['Q0'].val,
            'K_L': sliders['K_L'].val,
            'K': sliders['K'].val,
            'K_bg': init_params['K_bg'],
            'L': init_params['L'],
            'rho_max': sliders['rho_max'].val,
            'K_C': sliders['K_C'].val,
            'Q_L': sliders['Q_L'].val,
        }

        for elem in plot_elements:
            # Apply scaling factors
            L0_cond = L0_ref * elem['l0']
            C0_cond = C0_ref * elem['c0']

            params = base_params.copy()
            params['L0'] = L0_cond

            x0 = elem['conc_exp'][0] if elem['conc_exp'][0] > 0 else 1e5
            y0 = [x0, Q0_init, C0_cond]

            try:
                sol = solve_ivp(droop_light_system, t_span, y0, args=(params,),
                                t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
                if sol.success:
                    x_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, x_model)

                    # Local R²
                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    x_interp = np.interp(time_exp, t_shifted, x_model)
                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = x_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot>0 else np.nan

                        # Count subplots with R² >= 0.85
                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        # Background colour
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local>=0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(x_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val*1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred)**2)
            ss_tot = np.sum((all_y_exp - y_mean)**2)
            global_r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

            # Build title with global R² and subplot counter
            r2_val = f'{global_r2:.6f}' if not np.isnan(global_r2) else 'N/A'
            title_text.set_text(
                f'Calibration (25 conditions)  |  Global R² = {r2_val}  |  '
                f'Conditions R² ≥ 0.85 (green) : {n_good_r2}/{n_total_graphs}'
            )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()


# -------------------- Interactive visualisation - COMPLETE FIGURE --------------------
def plot_with_sliders():
    conv_OD_to_cell = 4.77e6
    conv_OD_plate_to_OD_erlen = 6.01

    # Experimental data
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

    unique_c0 = sorted(set(c0 for c0, l0 in all_conditions.keys()))
    unique_l0 = sorted(set(l0 for c0, l0 in all_conditions.keys()))
    n_c0, n_l0 = len(unique_c0), len(unique_l0)

    logger.info(f"Unique C0 factors: {unique_c0}")
    logger.info(f"Unique L0 factors: {unique_l0}")
    logger.info(f"Grid dimensions: {n_l0} rows × {n_c0} columns")

    # Initial Droop-Light model parameters
    init_params = {
        'L0_ref': 170,
        'C0_ref': 0.106e8,
        'mu_max': 0.287,
        'Q0': 0.132,
        'K_L': 58,
        'K': 0.00004121,
        'K_bg': 0,
        'L': 0.005,
        'rho_max': 0.19,
        'K_C': 0.247e8,
        'Q_L': 0.65,
        'Q0_init': 0.16
    }

    t_span = (0, MAX_TIME_HOURS)
    t_eval = np.linspace(*t_span, 500)

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
    for i, l0 in enumerate(reversed(unique_l0)):
        for j, c0 in enumerate(unique_c0):
            ax = axes[i, j]
            key = (c0, l0)
            if key in all_conditions:
                condition = all_conditions[key]
                if condition["type"] == "experiment":
                    time = np.array(condition["data"]["Time"])
                    concentration = np.array(condition["data"]["Mean"])
                    replicates = condition["data"]["replicates"]
                else:
                    time = condition["time"]
                    concentration = condition["mean"]
                    replicates = condition["replicates"]

                # Plot ALL replicate data without truncation
                for replicate in replicates:
                    ax.scatter(replicate["Time"], replicate["Value"],
                              color='grey', s=8, alpha=0.15)

                # Plot ALL mean data without truncation
                ax.scatter(time, concentration, color='orange', s=15, alpha=0.7,
                          label='Exp data')

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
                    'conc_exp': concentration,
                    'condition_type': condition["type"]
                })
                ax.set_title(f"$C_0 \\times {c0:.2f}$, $L_0 \\times {l0:.2f}$", fontsize=6, pad=2)
                ax.grid(True, alpha=0.15, linewidth=0.3)
                ax.set_xlim(0, MAX_TIME_HOURS)
                if j > 0: ax.tick_params(labelleft=False)
                else: ax.tick_params(axis='y', labelsize=5)
                if i < n_l0-1: ax.tick_params(labelbottom=False)
                else: ax.tick_params(axis='x', labelsize=5)
                if i == 0 and j == 0: ax.legend(fontsize=5, loc='lower right',
                                               framealpha=0.8)
            else:
                ax.axis('off')

    fig.text(0.5, 0.27, "Time (h)", ha="center", fontsize=10, weight='bold')
    fig.text(0.02, 0.6, "Cell density (cell mL$^{-1}$)", va="center",
            rotation="vertical", fontsize=10, weight='bold')
    title_text = fig.suptitle(
        'Validation (85 conditions)  |  Global R² = Calculating...  |  Conditions R² ≥ 0.85 (green) : -/85',
        fontsize=11, fontweight='bold')

    # Sliders
    axcolor = "lightgoldenrodyellow"
    slider_height = 0.015
    slider_left = 0.15
    slider_width = 0.70
    spacing = 0.023

    slider_defs = [
        ("L0_ref",  10,     500,   init_params['L0_ref'],
         r"$L_0\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("C0_ref",  1e6,    1e8,   init_params['C0_ref'],
         r"$C_0$ (eq-cells mL⁻¹)"),
        ("mu_max",  0.01,   0.5,   init_params['mu_max'],
         r"$\mu_{\max}\ (\mathrm{h}^{-1})$"),
        ("Q0",      0.01,   0.5,   init_params['Q0'],
         r"$Q_0\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_L",     1,      200,   init_params['K_L'],
         r"$K_L\ (\mu\mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1})$"),
        ("K",       0.1e-8, 1e-4,  init_params['K'],
         r"$K\ (\mathrm{cell}^{-1}\ \mathrm{mL}\ \mathrm{m}^{-1})$"),
        ("rho_max", 0.01,   1.0,   init_params['rho_max'],
         r"$\rho_{\max}\ (\mathrm{h}^{-1}\ \mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("K_C",     1e6,    1e8,   init_params['K_C'],
         r"$K_C$ (eq-cells mL⁻¹)"),
        ("Q_L",     0.5,    5.0,   init_params['Q_L'],
         r"$Q_L\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
        ("Q0_init", 0.1,    2.0,   init_params['Q0_init'],
         r"$Q(t{=}0)\ (\mathrm{cell}^{-1}\ \mathrm{mL})$"),
    ]

    sliders = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(slider_defs):
        ax_slider = plt.axes([slider_left, 0.25 - idx*spacing, slider_width, slider_height],
                             facecolor=axcolor)
        sliders[name] = Slider(ax_slider, label, vmin, vmax,
                              valinit=val, valstep=(vmax-vmin)/1000, color='black')

    # -------------------- Update function --------------------
    def update(val):
        all_y_exp = []
        all_y_pred = []

        # Counters for subplots with good R²
        n_good_r2 = 0
        n_total_graphs = len(plot_elements)

        # Read slider values
        L0_ref = sliders['L0_ref'].val
        C0_ref = sliders['C0_ref'].val
        Q0_init = sliders['Q0_init'].val

        base_params = {
            'mu_max': sliders['mu_max'].val,
            'Q0': sliders['Q0'].val,
            'K_L': sliders['K_L'].val,
            'K': sliders['K'].val,
            'K_bg': init_params['K_bg'],
            'L': init_params['L'],
            'rho_max': sliders['rho_max'].val,
            'K_C': sliders['K_C'].val,
            'Q_L': sliders['Q_L'].val,
        }

        for elem in plot_elements:
            # Apply scaling factors
            L0_cond = L0_ref * elem['l0']  # apply L0 factor
            C0_cond = C0_ref * elem['c0']  # apply C0 factor

            params = base_params.copy()
            params['L0'] = L0_cond

            x0 = elem['conc_exp'][0] if elem['conc_exp'][0] > 0 else 1e5
            y0 = [x0, Q0_init, C0_cond]

            try:
                sol = solve_ivp(droop_light_system, t_span, y0, args=(params,),
                                t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
                if sol.success:
                    x_model = sol.y[0]

                    # Apply t_lag time shift
                    t_lag = TLAG_ADJUSTMENTS.get((elem['l0'], elem['c0']), 0.0)
                    t_shifted = sol.t + t_lag
                    elem['line'].set_data(t_shifted, x_model)

                    # Local R²
                    time_exp = elem['time_exp']
                    conc_exp = elem['conc_exp']
                    x_interp = np.interp(time_exp, t_shifted, x_model)
                    valid_mask = ~np.isnan(conc_exp)

                    if np.sum(valid_mask) > 1:
                        y_exp_local = conc_exp[valid_mask]
                        y_pred_local = x_interp[valid_mask]
                        y_mean_local = np.mean(y_exp_local)
                        ss_res = np.sum((y_exp_local - y_pred_local) ** 2)
                        ss_tot = np.sum((y_exp_local - y_mean_local) ** 2)
                        r2_local = 1 - (ss_res / ss_tot) if ss_tot>0 else np.nan

                        # Count subplots with R² >= 0.85
                        if not np.isnan(r2_local) and r2_local >= 0.85:
                            n_good_r2 += 1

                        elem['r2_text'].set_text(
                            f'R²={r2_local:.2f}' if not np.isnan(r2_local) else 'R²=N/A'
                        )
                        # Background colour
                        elem['r2_text'].get_bbox_patch().set_facecolor(
                            '#c8f7c5' if r2_local>=0.85 else '#f7c5c5'
                        )

                        all_y_exp.extend(y_exp_local)
                        all_y_pred.extend(y_pred_local)

                    # Dynamically adjust ylim
                    max_val = max(np.nanmax(x_model), np.nanmax(conc_exp))
                    elem['ax'].set_ylim([0, max_val*1.2])

            except Exception as e:
                logger.error(f"Error for C0={elem['c0']}, L0={elem['l0']}: {e}")
                elem['r2_text'].set_text('R²=Err')

        # Global R²
        if len(all_y_exp) > 0:
            all_y_exp = np.array(all_y_exp)
            all_y_pred = np.array(all_y_pred)
            y_mean = np.mean(all_y_exp)
            ss_res = np.sum((all_y_exp - all_y_pred)**2)
            ss_tot = np.sum((all_y_exp - y_mean)**2)
            global_r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

            # Build title with global R² and subplot counter
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


if __name__ == "__main__":
    # Display the test figure (7 specific conditions)
    logger.info("="*60)
    logger.info("Generating CALIBRATION figure (7 specific conditions)")
    logger.info("Conditions: (1,1), (0.5,1), (0.125,1), (1,0.6), (0.25,0.6), (0.25,0.3), (1,0.15)")
    logger.info("="*60)
    plot_with_sliders_calibration()

    # Display the calibration figure (C0 = 1, 0.5, 0.25, 0.125, 0.0625)
    logger.info("="*60)
    logger.info("Generating FILTERED figure (C0=1, 0.5, 0.25, 0.125, 0.0625)")
    logger.info("="*60)
    plot_with_sliders_filtered()

    # Display the validation figure
    logger.info("="*60)
    logger.info("Generating COMPLETE figure (all conditions)")
    logger.info("="*60)
    plot_with_sliders()
