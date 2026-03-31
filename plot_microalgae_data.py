"""
Combined Figure with Exponential Phase Zooms:
Row 1: Monoraphidium sp. data (4 graphs)
Row 2: Chlamydomonas reinhardtii data (5 graphs)
Row 3: Exponential phase zooms (2 graphs, each spanning 2 columns)
  - Left: Kambe L0=386.7, t=0-60h
  - Right: Chlamy L0=102, t=0-60h
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import os
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS FOR KAMBE DATA (Row 1)
# ============================================================================

# Kambe constants
ODN_kambe = 30.10e6  # cells per OD

# Light conditions for Kambe
ein_kambe = np.array([96.8, 184.4, 386.7, 1034])  # [μE s^-1 m^-2]
light_area_kambe = 0.002826  # [m^2]
Ein_kambe = ein_kambe * light_area_kambe * 3600  # [μE hour^-1]

# Medium concentrations for Kambe
C0_kambe = np.array([1, 1 / 2, 1 / 4, 1 / 8])

# ============================================================================
# CONSTANTS FOR CHLAMYDOMONAS DATA (Row 2)
# ============================================================================

# Chlamydomonas constants
conv_OD_to_cell = 4.77e6  # cells/mL per OD

# Medium concentrations for Chlamydomonas (matching Kambe: 4 values)
C0_values_chlamy = np.array([1, 1 / 2, 1 / 4, 1 / 8])

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def read_Kambe_data(data_folder="data_Kambe"):
    """Load experimental data from Kambe CSV files"""
    filenames = [
        "data-Ein-0.274.csv",
        "data-Ein-0.521.csv",
        "data-Ein-1.09.csv",
        "data-Ein-2.92.csv",
    ]

    od_data_list = []
    time_data = None

    for fname in filenames:
        path = os.path.join(data_folder, fname)
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue

        df = pd.read_csv(path, sep=";")

        if time_data is None:
            time_data = df["Time (hour)"].values

        # mean columns for C0 = 1, 1/2, 1/4, 1/8
        for c in ["mean 0", "mean 1", "mean 2", "mean 3"]:
            od_data_list.append(df[c].values)

    od_data = np.array(od_data_list)
    print(
        f"✓ Loaded Kambe data: {od_data.shape[0]} conditions, {len(time_data)} time points"
    )

    return time_data, od_data


def read_csv_with_replicates(filepath, conv_OD_to_cell=4.77e6):
    """Read Chlamydomonas CSV file with replicates A, B, C (only first 4 conditions)"""
    df = pd.read_csv(filepath, sep=";")

    # Extract time (convert seconds to hours)
    time = df["Time (s)"].values / 3600

    data = {}

    # For each condition (0, 1, 2, 3 corresponding to C0 = 1, 1/2, 1/4, 1/8)
    # Note: condition 4 (C0 = 1/16) is excluded
    for cond_idx in range(4):  # Changed from range(5) to range(4)
        replicate_data = {
            "Time": time,
            "A": df[f"OD {cond_idx}A"].values * conv_OD_to_cell,
            "B": df[f"OD {cond_idx}B"].values * conv_OD_to_cell,
            "C": df[f"OD {cond_idx}C"].values * conv_OD_to_cell,
        }
        data[cond_idx] = replicate_data

    return data


def load_Chlamy_data(files_dict, conv_OD_to_cell=4.77e6):
    """Load all Chlamydomonas experimental data from multiple files"""
    all_data = {}

    print("\n✓ Loading Chlamydomonas data...")

    for L0_factor, filepath in files_dict.items():
        Ein_val = L0_factor * 170  # L0 max = 170

        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue

        data = read_csv_with_replicates(filepath, conv_OD_to_cell)

        # For each C0 condition (now only 4 conditions: 0, 1, 2, 3)
        for cond_idx in range(4):  # Changed from range(5) to range(4)
            C0_val = C0_values_chlamy[cond_idx]

            all_data[(Ein_val, C0_val)] = {
                "time": data[cond_idx]["Time"],
                "replicates": [
                    data[cond_idx]["A"],
                    data[cond_idx]["B"],
                    data[cond_idx]["C"],
                ],
            }

    print(f"✓ Loaded Chlamydomonas data: {len(all_data)} conditions")

    return all_data


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================


def create_combined_figure(time_kambe, od_kambe, all_data_chlamy):
    """
    Create combined figure with:
    - Row 1 (4 graphs): Kambe data
    - Row 2 (5 graphs): Chlamydomonas data
    - Row 3 (2 graphs): Exponential phase zooms
    """

    # ========================================================================
    # GLOBAL FONT SETTINGS (journal-quality)
    # ========================================================================
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "axes.titlesize": 24,  # 22,
            "axes.labelsize": 22,  # 20,
            "xtick.labelsize": 20,  # 18,
            "ytick.labelsize": 20,  # 18,
            "legend.fontsize": 18,
            "figure.titlesize": 22,
        }
    )

    # ========================================================================
    # LIGHT UNIT (Nature-compliant notation)
    # ========================================================================
    LIGHT_UNIT = r"$\mu\mathrm{mol}_{h\nu}\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}$"

    # ========================================================================
    # FIGURE & GRID
    # ========================================================================
    fig = plt.figure(figsize=(24, 18))

    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.35,
        wspace=0.25,
        left=0.05,
        right=0.95,
        top=0.94,
        bottom=0.05,
    )

    # ========================================================================
    # ROW 1: KAMBE DATA
    # ========================================================================
    print("\n✓ Plotting Row 1: Kambe data...")

    colors_C0_kambe = {
        1.0000: "mediumseagreen",
        0.5000: "grey",
        0.2500: "tomato",
        0.1250: "teal",
    }

    Ein_sorted_indices = np.argsort(Ein_kambe)[::-1]

    for idx_graph in range(4):
        ax = fig.add_subplot(gs[0, idx_graph])
        light_idx = Ein_sorted_indices[idx_graph]

        t_max_this_graph = 0
        for C0_idx in range(4):
            data_idx = light_idx * 4 + C0_idx
            od_exp = od_kambe[data_idx, :]
            valid = ~np.isnan(od_exp)
            if np.any(valid):
                t_max_this_graph = max(t_max_this_graph, np.max(time_kambe[valid]))

        t_max_this_graph += 100

        for C0_idx in range(4):
            data_idx = light_idx * 4 + C0_idx
            C0_val = C0_kambe[C0_idx]
            color = colors_C0_kambe.get(C0_val, "black")

            od_exp = od_kambe[data_idx, :]
            valid = ~np.isnan(od_exp)
            biomass_exp = od_exp[valid] * ODN_kambe

            ax.plot(
                time_kambe[valid],
                biomass_exp,
                marker="o",
                linestyle="none",
                color=color,
                markersize=9,
                alpha=0.6,
            )

        ein_val = ein_kambe[light_idx]
        ax.set_title(rf"$L_0$ = {ein_val:.1f} {LIGHT_UNIT}", pad=18)
        ax.set_xlabel("Time (h)")
        if idx_graph == 0:
            ax.set_ylabel(r"Biomass (cells mL$^{-1}$ × $10^{8}$)", labelpad=20)
        ax.set_xlim([0, t_max_this_graph])
        ax.set_ylim([0, 20 * ODN_kambe])
        # ax.grid(True, alpha=0.3)
        ax.grid(False)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_visible(False)

    # ========================================================================
    # ROW 2: CHLAMYDOMONAS DATA
    # ========================================================================
    print("✓ Plotting Row 2: Chlamydomonas data...")

    colors_C0_chlamy = colors_C0_kambe
    markers_rep = {"A": "o", "B": "^", "C": "s"}

    L0_unique_all = sorted(set(k[0] for k in all_data_chlamy.keys()), reverse=True)
    L0_unique = L0_unique_all[:4]

    y_max_global = 0
    for L0_val in L0_unique:
        for C0_val in C0_values_chlamy:
            if (L0_val, C0_val) not in all_data_chlamy:
                continue
            data = all_data_chlamy[(L0_val, C0_val)]
            for biomass_rep in data["replicates"]:
                y_max_global = max(y_max_global, np.nanmax(biomass_rep))

    y_max_global *= 1.1

    for idx_L0, L0_val in enumerate(L0_unique):
        ax = fig.add_subplot(gs[1, idx_L0])
        t_max_this_graph = 0

        for C0_val in C0_values_chlamy:
            if (L0_val, C0_val) not in all_data_chlamy:
                continue

            data = all_data_chlamy[(L0_val, C0_val)]
            time_exp = data["time"]
            color = colors_C0_chlamy[C0_val]

            for rep_name, biomass_rep in zip(["A", "B", "C"], data["replicates"]):
                valid = ~np.isnan(biomass_rep)
                t_plot = time_exp[valid]
                N_plot = biomass_rep[valid]

                if len(t_plot) > 0:
                    t_max_this_graph = max(t_max_this_graph, t_plot[-1])

                ax.plot(
                    t_plot,
                    N_plot,
                    marker=markers_rep[rep_name],
                    linestyle="none",
                    color=color,
                    alpha=0.6,
                    markersize=9,
                )

        ax.set_title(rf"$L_0$ = {L0_val:.1f} {LIGHT_UNIT}", pad=18)
        ax.set_xlabel("Time (h)")
        if idx_L0 == 0:
            ax.set_ylabel(r"Biomass (cells mL$^{-1}$ × $10^{7}$)", labelpad=20)
        ax.set_xlim([0, t_max_this_graph + 10])
        if idx_L0 == 0:
            ax.set_xticks([0, 50, 100, 150])
        ax.set_ylim([0, y_max_global])
        # ax.grid(True, alpha=0.3)
        ax.grid(False)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_visible(False)

    # ========================================================================
    # LEGEND
    # ========================================================================
    legend_elements = []

    for C0 in sorted(colors_C0_chlamy.keys(), reverse=True):
        legend_elements.append(
            Line2D(
                [0], [0], color=colors_C0_chlamy[C0], lw=3, label=rf"$C_0 = {C0:.4g}$"
            )
        )

    legend_elements.append(Line2D([0], [0], color="none", label=""))

    for rep, marker in markers_rep.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                color="gray",
                markersize=10,
                label=f"Rep {rep}",
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        framealpha=0.9,
    )

    # ========================================================================
    # ROW 3: EXPONENTIAL ZOOMS
    # ========================================================================
    print("✓ Plotting Row 3: Exponential phase zooms...")

    # --- ZOOM 1: Kambe ---
    ax_zoom1 = fig.add_subplot(gs[2, 0:2])
    Ein_sorted_indices = np.argsort(Ein_kambe)[::-1]
    light_idx_zoom1 = Ein_sorted_indices[1]

    # Storage for biomass values at t=56h
    biomass_at_56h = {}

    for C0_idx in range(4):
        data_idx = light_idx_zoom1 * 4 + C0_idx
        C0_val = C0_kambe[C0_idx]
        color = colors_C0_kambe[C0_val]

        od_exp = od_kambe[data_idx, :]
        valid = ~np.isnan(od_exp)
        t = time_kambe[valid]
        N = od_exp[valid] * ODN_kambe

        # Find biomass at t=56h (closest time point)
        idx_56h = np.argmin(np.abs(t - 56))
        biomass_at_56h[C0_val] = N[idx_56h]

        mask = t <= 60
        ax_zoom1.plot(
            t[mask],
            N[mask],
            marker="o",
            linestyle="none",
            color=color,
            markersize=10,
            alpha=0.7,
        )

    ax_zoom1.set_title(rf"Zoom $L_0$ = 386.7 {LIGHT_UNIT}", pad=18)
    ax_zoom1.set_xlabel("Time (h)")
    ax_zoom1.set_ylabel("Biomass (cells mL$^{-1}$ × $10^{7}$)")
    ax_zoom1.set_xlim([0, 60])
    # ax_zoom1.grid(True, alpha=0.3)
    ax_zoom1.grid(False)
    ax_zoom1.yaxis.get_offset_text().set_visible(False)

    # Calculate and display percentage increase for Zoom 1
    if 0.125 in biomass_at_56h and 1.0 in biomass_at_56h:
        N_C0_min = biomass_at_56h[0.125]
        N_C0_max = biomass_at_56h[1.0]
        percent_increase_zoom1 = ((N_C0_max - N_C0_min) / N_C0_min) * 100
        print(f"\n📊 ZOOM 1 (Photoautotrophy - L0=386.7):")
        print(f"   Biomass at t=56h for C0=1/8: {N_C0_min:.2e} cells/mL")
        print(f"   Biomass at t=56h for C0=1:   {N_C0_max:.2e} cells/mL")
        print(f"   ➤ Percentage increase: {percent_increase_zoom1:.1f}%")

    # --- ZOOM 2: Chlamydomonas ---
    ax_zoom2 = fig.add_subplot(gs[2, 2:4])
    L0_val_zoom2 = 102.0

    # Storage for biomass values at t≈56.983h for each replicate
    biomass_at_57h_replicates = {C0: [] for C0 in C0_values_chlamy}

    for C0_val in C0_values_chlamy:
        if (L0_val_zoom2, C0_val) not in all_data_chlamy:
            continue

        data = all_data_chlamy[(L0_val_zoom2, C0_val)]
        color = colors_C0_chlamy[C0_val]

        for rep_name, biomass_rep in zip(["A", "B", "C"], data["replicates"]):
            valid = ~np.isnan(biomass_rep)
            t = data["time"][valid]
            N = biomass_rep[valid]

            # Find biomass at t≈56.983h (closest time point)
            idx_57h = np.argmin(np.abs(t - 56.983333333333334))
            biomass_at_57h_replicates[C0_val].append(N[idx_57h])

            mask = t <= 60
            ax_zoom2.plot(
                t[mask],
                N[mask],
                marker=markers_rep[rep_name],
                linestyle="none",
                color=color,
                markersize=10,
                alpha=0.7,
            )

    ax_zoom2.set_title(rf"Zoom $L_0$ = 102 {LIGHT_UNIT}", pad=18)
    ax_zoom2.set_xlabel("Time (h)")
    # ax_zoom2.set_ylabel('Biomass (cells mL$^{-1}$ × $10^{7}$)')
    ax_zoom2.set_xlim([0, 60])
    # ax_zoom2.grid(True, alpha=0.3)
    ax_zoom2.grid(False)
    ax_zoom2.yaxis.get_offset_text().set_visible(False)

    # Calculate and display percentage increase for Zoom 2
    if 0.125 in biomass_at_57h_replicates and 1.0 in biomass_at_57h_replicates:
        # Calculate mean for each C0
        N_C0_min_replicates = biomass_at_57h_replicates[0.125]
        N_C0_max_replicates = biomass_at_57h_replicates[1.0]

        if len(N_C0_min_replicates) > 0 and len(N_C0_max_replicates) > 0:
            N_C0_min_mean = np.mean(N_C0_min_replicates)
            N_C0_max_mean = np.mean(N_C0_max_replicates)
            percent_increase_zoom2 = (
                (N_C0_max_mean - N_C0_min_mean) / N_C0_min_mean
            ) * 100

            print(f"\n📊 ZOOM 2 (Mixotrophy - L0=102):")
            print(f"   Biomass at t≈56.98h for C0=1/8:")
            print(f"      Replicate A: {N_C0_min_replicates[0]:.2e} cells/mL")
            print(f"      Replicate B: {N_C0_min_replicates[1]:.2e} cells/mL")
            print(f"      Replicate C: {N_C0_min_replicates[2]:.2e} cells/mL")
            print(f"      Mean: {N_C0_min_mean:.2e} cells/mL")
            print(f"   Biomass at t≈56.98h for C0=1:")
            print(f"      Replicate A: {N_C0_max_replicates[0]:.2e} cells/mL")
            print(f"      Replicate B: {N_C0_max_replicates[1]:.2e} cells/mL")
            print(f"      Replicate C: {N_C0_max_replicates[2]:.2e} cells/mL")
            print(f"      Mean: {N_C0_max_mean:.2e} cells/mL")
            print(f"   ➤ Percentage increase: {percent_increase_zoom2:.1f}%")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print(" COMBINED FIGURE: KAMBE DATA + CHLAMYDOMONAS DATA ")
    print("=" * 80)

    # Load Kambe data
    print("\nLoading Kambe data...")
    time_kambe, od_kambe = read_Kambe_data("data_Kambe")

    # Load Chlamydomonas data
    print("\nLoading Chlamydomonas data...")
    files_dict_chlamy = {
        1.000: "all_data/data_exp_Chlamy_07-07-25.csv",
        0.6: "all_data/data_exp_Chlamy_01-07-25.csv",
        0.3: "all_data/data_exp_Chlamy_17-02-25.csv",
        0.15: "all_data/data_exp_Chlamy_04-11-24.csv",
        0.07: "all_data/data_exp_Chlamy_16-09-24.csv",
    }
    all_data_chlamy = load_Chlamy_data(files_dict_chlamy, conv_OD_to_cell)

    # Create combined figure
    print("\nCreating combined figure...")
    fig = create_combined_figure(time_kambe, od_kambe, all_data_chlamy)

    # Save figure
    output_path = "microalgae_data.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"\n✓ Saved: {output_path}")

    print("\n" + "=" * 80)
    print(" FIGURE GENERATION COMPLETE ")
    print("=" * 80 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
