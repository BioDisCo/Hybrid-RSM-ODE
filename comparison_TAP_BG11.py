#!/usr/bin/env python3
"""
Comparative growth analysis of C. reinhardtii
in BG11 vs TAP medium at different dilutions and light intensities

Experimental conditions:
- Incubator at 1.5% CO2
- Two light intensities tested: 51 and 102 µmol photons/m²/s (from top)
- Temperature: 25°C
- Agitation: 100 rpm
- Working volume: 50 mL
- Initial OD: 0.004

Tested dilutions:
- C₀=1: undiluted medium
- C₀=1/2: 2x dilution with distilled water
- C₀=1/4: 4x dilution
- C₀=1/8: 8x dilution
- C₀=1/16: 16x dilution
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import tukey_hsd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]
mpl.rcParams.update(
    {
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 22,
    }
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths for 51 µmol/m²/s
PATH_BG11_51 = "all_data/data_exp_Chlamy_29-11-25.csv"
PATH_TAP_51 = "all_data/data_exp_Chlamy_17-02-25.csv"

# File paths for 102 µmol/m²/s
PATH_BG11_102 = "all_data/data_exp_Chlamy_19-01-26.csv"
PATH_TAP_102 = "all_data/data_exp_Chlamy_01-07-25.csv"

OUTPUT_DIR = "comparison_BG-11_TAP_chlamy/"

# Exponential phase parameters (in hours)
# 51 µmol/m²/s
EXP_PHASE_BG11_51 = {"start": 18, "end": 65}
EXP_PHASE_TAP_51 = {"start": 8, "end": 60}

# 102 µmol/m²/s (à ajuster selon tes données)
EXP_PHASE_BG11_102 = {"start": 22, "end": 50}
EXP_PHASE_TAP_102 = {"start": 5, "end": 60}

# Condition names
COND_NAMES = ["C0=1", "C0=1/2", "C0=1/4", "C0=1/8", "C0=1/16"]
CONDITIONS = {
    "0": ["0A", "0B", "0C"],
    "1": ["1A", "1B", "1C"],
    "2": ["2A", "2B", "2C"],
    "3": ["3A", "3B", "3C"],
    "4": ["4A", "4B", "4C"],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_and_prepare_data(filepath, medium_name):
    """
    Load and prepare data from an experiment CSV file.
    Uses standard OD columns (not NC columns).
    """
    df = pd.read_csv(filepath, sep=";")

    # Extract time in hours from Time (s)
    time_s = df["Time (s)"].dropna().values
    time_h = time_s / 3600

    # List of replicates
    replicates = [
        "0A",
        "0B",
        "0C",
        "1A",
        "1B",
        "1C",
        "2A",
        "2B",
        "2C",
        "3A",
        "3B",
        "3C",
        "4A",
        "4B",
        "4C",
    ]

    # Create data dictionary using standard OD columns
    data = {"time_h": time_h}
    for rep in replicates:
        col_name = f"OD {rep}"
        if col_name in df.columns:
            values = df[col_name].iloc[: len(time_h)].values
            data[rep] = values

    data_df = pd.DataFrame(data).dropna()
    print(
        f"[{medium_name}] Data loaded: {len(data_df)} time points, "
        f"from {data_df['time_h'].min():.1f}h to {data_df['time_h'].max():.1f}h"
    )

    return data_df


def calculate_mu_max(time, od, time_start, time_end):
    """
    Calculate maximum specific growth rate by linear regression on ln(OD).
    """
    mask = (time >= time_start) & (time <= time_end) & (od > 0)
    t_exp = time[mask]
    od_exp = od[mask]

    if len(t_exp) < 3:
        return np.nan, np.nan, np.nan

    ln_od = np.log(od_exp)
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_exp, ln_od)

    return slope, r_value**2, std_err


def calculate_all_mu(data_df, time_start, time_end):
    """
    Calculate μmax for all conditions and replicates.
    """
    mu_all = {}
    r2_all = {}

    for cond, reps in CONDITIONS.items():
        mu_all[cond] = []
        r2_all[cond] = []
        for rep in reps:
            if rep in data_df.columns:
                time = data_df["time_h"].values
                od = data_df[rep].values
                mu, r2, se = calculate_mu_max(time, od, time_start, time_end)
                mu_all[cond].append(mu)
                r2_all[cond].append(r2)

    return mu_all, r2_all


def calculate_carrying_capacity(data_df, n_last=3):
    """
    Calculate carrying capacity K (maximum OD) for each replicate.
    """
    k_all = {}

    for cond, reps in CONDITIONS.items():
        k_all[cond] = []
        for rep in reps:
            if rep in data_df.columns:
                od = data_df[rep].values
                k = np.mean(od[-n_last:])
                k_all[cond].append(k)

    return k_all


def compute_cld(mu_dict, alpha=0.05):
    """
    Perform one-way ANOVA, Kruskal-Wallis and Tukey HSD across concentrations.
    Returns ANOVA results and Compact Letter Display (CLD).

    Groups sharing a letter are NOT significantly different (Tukey HSD).
    """
    groups = [mu_dict[str(i)] for i in range(5)]
    n = len(groups)

    # One-way ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)

    # Kruskal-Wallis (non-parametric alternative)
    h_stat, p_kruskal = stats.kruskal(*groups)

    # Tukey HSD post-hoc
    tukey = tukey_hsd(*groups)

    # Compact Letter Display algorithm
    means = [np.nanmean(g) for g in groups]
    order = np.argsort(means)[::-1]  # descending mean order

    # Significance matrix in sorted order
    sig = [
        [tukey.pvalue[order[i]][order[j]] < alpha for j in range(n)] for i in range(n)
    ]

    # Greedy CLD: sweep sorted groups, assign/create letter groups
    letter_groups = []
    for i in range(n):
        added = False
        for group in letter_groups:
            if all(not sig[i][m] for m in group):
                group.add(i)
                added = True
        if not added:
            new_group = {i}
            # Absorb compatible previous elements
            for j in range(i):
                if all(not sig[j][m] for m in new_group):
                    new_group.add(j)
            letter_groups.append(new_group)

    # Assign letters
    letters_sorted = [set() for _ in range(n)]
    for idx, group in enumerate(letter_groups):
        letter = chr(ord("a") + idx)
        for member in group:
            letters_sorted[member].add(letter)

    # Map back to original order
    letters = [""] * n
    for i in range(n):
        letters[order[i]] = "".join(sorted(letters_sorted[i]))

    return {
        "f_stat": f_stat,
        "p_anova": p_anova,
        "h_stat": h_stat,
        "p_kruskal": p_kruskal,
        "letters": letters,
        "tukey": tukey,
    }


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    print_section("DATA LOADING")

    # Load data at 51 µmol/m²/s
    print("\n--- Light intensity: 51 µmol photons/m²/s ---")
    data_bg11_51 = load_and_prepare_data(PATH_BG11_51, "BG11 @ 51 µmol/m²/s")
    data_tap_51 = load_and_prepare_data(PATH_TAP_51, "TAP @ 51 µmol/m²/s")

    # Load data at 102 µmol/m²/s
    print("\n--- Light intensity: 102 µmol photons/m²/s ---")
    data_bg11_102 = load_and_prepare_data(PATH_BG11_102, "BG11 @ 102 µmol/m²/s")
    data_tap_102 = load_and_prepare_data(PATH_TAP_102, "TAP @ 102 µmol/m²/s")

    # -------------------------------------------------------------------------
    # GROWTH PARAMETER CALCULATION - 51 µmol/m²/s
    # -------------------------------------------------------------------------
    print_section("GROWTH RATE CALCULATION - 51 µmol/m²/s")

    # BG11 at 51
    mu_bg11_51, r2_bg11_51 = calculate_all_mu(
        data_bg11_51, EXP_PHASE_BG11_51["start"], EXP_PHASE_BG11_51["end"]
    )
    k_bg11_51 = calculate_carrying_capacity(data_bg11_51)

    # TAP at 51
    mu_tap_51, r2_tap_51 = calculate_all_mu(
        data_tap_51, EXP_PHASE_TAP_51["start"], EXP_PHASE_TAP_51["end"]
    )
    k_tap_51 = calculate_carrying_capacity(data_tap_51)

    # Display results for 51 µmol/m²/s
    print(
        f"\nExponential phase BG11: {EXP_PHASE_BG11_51['start']}-{EXP_PHASE_BG11_51['end']}h"
    )
    print(
        f"Exponential phase TAP: {EXP_PHASE_TAP_51['start']}-{EXP_PHASE_TAP_51['end']}h"
    )

    print("\n" + "-" * 70)
    print("BG11 medium (51 µmol/m²/s):")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        mean_mu = np.nanmean(mu_bg11_51[cond])
        std_mu = np.nanstd(mu_bg11_51[cond], ddof=1)
        mean_r2 = np.nanmean(r2_bg11_51[cond])
        td = np.log(2) / mean_mu if mean_mu > 0 else np.nan
        print(
            f"  {COND_NAMES[i]}: μmax = {mean_mu:.5f} ± {std_mu:.5f} h⁻¹ "
            f"(R² = {mean_r2:.3f}, Td = {td:.1f}h)"
        )

    print("\nTAP medium (51 µmol/m²/s):")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        mean_mu = np.nanmean(mu_tap_51[cond])
        std_mu = np.nanstd(mu_tap_51[cond], ddof=1)
        mean_r2 = np.nanmean(r2_tap_51[cond])
        td = np.log(2) / mean_mu if mean_mu > 0 else np.nan
        print(
            f"  {COND_NAMES[i]}: μmax = {mean_mu:.5f} ± {std_mu:.5f} h⁻¹ "
            f"(R² = {mean_r2:.3f}, Td = {td:.1f}h)"
        )

    # -------------------------------------------------------------------------
    # GROWTH PARAMETER CALCULATION - 102 µmol/m²/s
    # -------------------------------------------------------------------------
    print_section("GROWTH RATE CALCULATION - 102 µmol/m²/s")

    # BG11 at 102
    mu_bg11_102, r2_bg11_102 = calculate_all_mu(
        data_bg11_102, EXP_PHASE_BG11_102["start"], EXP_PHASE_BG11_102["end"]
    )
    k_bg11_102 = calculate_carrying_capacity(data_bg11_102)

    # TAP at 102
    mu_tap_102, r2_tap_102 = calculate_all_mu(
        data_tap_102, EXP_PHASE_TAP_102["start"], EXP_PHASE_TAP_102["end"]
    )
    k_tap_102 = calculate_carrying_capacity(data_tap_102)

    # Display results for 102 µmol/m²/s
    print(
        f"\nExponential phase BG11: {EXP_PHASE_BG11_102['start']}-{EXP_PHASE_BG11_102['end']}h"
    )
    print(
        f"Exponential phase TAP: {EXP_PHASE_TAP_102['start']}-{EXP_PHASE_TAP_102['end']}h"
    )

    print("\n" + "-" * 70)
    print("BG11 medium (102 µmol/m²/s):")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        mean_mu = np.nanmean(mu_bg11_102[cond])
        std_mu = np.nanstd(mu_bg11_102[cond], ddof=1)
        mean_r2 = np.nanmean(r2_bg11_102[cond])
        td = np.log(2) / mean_mu if mean_mu > 0 else np.nan
        print(
            f"  {COND_NAMES[i]}: μmax = {mean_mu:.5f} ± {std_mu:.5f} h⁻¹ "
            f"(R² = {mean_r2:.3f}, Td = {td:.1f}h)"
        )

    print("\nTAP medium (102 µmol/m²/s):")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        mean_mu = np.nanmean(mu_tap_102[cond])
        std_mu = np.nanstd(mu_tap_102[cond], ddof=1)
        mean_r2 = np.nanmean(r2_tap_102[cond])
        td = np.log(2) / mean_mu if mean_mu > 0 else np.nan
        print(
            f"  {COND_NAMES[i]}: μmax = {mean_mu:.5f} ± {std_mu:.5f} h⁻¹ "
            f"(R² = {mean_r2:.3f}, Td = {td:.1f}h)"
        )

    # -------------------------------------------------------------------------
    # STATISTICAL TESTS - 51 µmol/m²/s
    # -------------------------------------------------------------------------
    print_section("STATISTICAL TESTS - 51 µmol/m²/s")

    print("\nt-tests BG11 vs TAP (μmax) at 51 µmol/m²/s:")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        t_stat, p_val = stats.ttest_ind(mu_bg11_51[cond], mu_tap_51[cond])
        sig = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "ns"
        )
        print(f"  {COND_NAMES[i]}: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")

    # One-way ANOVA across concentrations within each medium - 51 µmol/m²/s
    cld_bg11_51 = compute_cld(mu_bg11_51)
    cld_tap_51 = compute_cld(mu_tap_51)

    print("\n--- One-way ANOVA across concentrations (51 µmol/m²/s) ---")
    sig_b = (
        "***"
        if cld_bg11_51["p_anova"] < 0.001
        else "**"
        if cld_bg11_51["p_anova"] < 0.01
        else "*"
        if cld_bg11_51["p_anova"] < 0.05
        else "ns"
    )
    sig_t = (
        "***"
        if cld_tap_51["p_anova"] < 0.001
        else "**"
        if cld_tap_51["p_anova"] < 0.01
        else "*"
        if cld_tap_51["p_anova"] < 0.05
        else "ns"
    )
    print(
        f"  BG11: F(4,10) = {cld_bg11_51['f_stat']:.3f}, p = {cld_bg11_51['p_anova']:.6f} ({sig_b})"
    )
    print(
        f"  TAP:  F(4,10) = {cld_tap_51['f_stat']:.3f}, p = {cld_tap_51['p_anova']:.6f} ({sig_t})"
    )

    print(f"\n  Kruskal-Wallis (non-parametric check):")
    sig_bk = (
        "***"
        if cld_bg11_51["p_kruskal"] < 0.001
        else "**"
        if cld_bg11_51["p_kruskal"] < 0.01
        else "*"
        if cld_bg11_51["p_kruskal"] < 0.05
        else "ns"
    )
    sig_tk = (
        "***"
        if cld_tap_51["p_kruskal"] < 0.001
        else "**"
        if cld_tap_51["p_kruskal"] < 0.01
        else "*"
        if cld_tap_51["p_kruskal"] < 0.05
        else "ns"
    )
    print(
        f"  BG11: H = {cld_bg11_51['h_stat']:.3f}, p = {cld_bg11_51['p_kruskal']:.6f} ({sig_bk})"
    )
    print(
        f"  TAP:  H = {cld_tap_51['h_stat']:.3f}, p = {cld_tap_51['p_kruskal']:.6f} ({sig_tk})"
    )

    print(f"\n  Tukey HSD Compact Letter Display (51 µmol/m²/s):")
    for i in range(5):
        print(
            f"    {COND_NAMES[i]}: BG11={cld_bg11_51['letters'][i]}  TAP={cld_tap_51['letters'][i]}"
        )

    # -------------------------------------------------------------------------
    # STATISTICAL TESTS - 102 µmol/m²/s
    # -------------------------------------------------------------------------
    print_section("STATISTICAL TESTS - 102 µmol/m²/s")

    print("\nt-tests BG11 vs TAP (μmax) at 102 µmol/m²/s:")
    for i, cond in enumerate(["0", "1", "2", "3", "4"]):
        t_stat, p_val = stats.ttest_ind(mu_bg11_102[cond], mu_tap_102[cond])
        sig = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "ns"
        )
        print(f"  {COND_NAMES[i]}: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")

    # One-way ANOVA across concentrations within each medium - 102 µmol/m²/s
    cld_bg11_102 = compute_cld(mu_bg11_102)
    cld_tap_102 = compute_cld(mu_tap_102)

    print("\n--- One-way ANOVA across concentrations (102 µmol/m²/s) ---")
    sig_b = (
        "***"
        if cld_bg11_102["p_anova"] < 0.001
        else "**"
        if cld_bg11_102["p_anova"] < 0.01
        else "*"
        if cld_bg11_102["p_anova"] < 0.05
        else "ns"
    )
    sig_t = (
        "***"
        if cld_tap_102["p_anova"] < 0.001
        else "**"
        if cld_tap_102["p_anova"] < 0.01
        else "*"
        if cld_tap_102["p_anova"] < 0.05
        else "ns"
    )
    print(
        f"  BG11: F(4,10) = {cld_bg11_102['f_stat']:.3f}, p = {cld_bg11_102['p_anova']:.6f} ({sig_b})"
    )
    print(
        f"  TAP:  F(4,10) = {cld_tap_102['f_stat']:.3f}, p = {cld_tap_102['p_anova']:.6f} ({sig_t})"
    )

    print(f"\n  Kruskal-Wallis (non-parametric check):")
    sig_bk = (
        "***"
        if cld_bg11_102["p_kruskal"] < 0.001
        else "**"
        if cld_bg11_102["p_kruskal"] < 0.01
        else "*"
        if cld_bg11_102["p_kruskal"] < 0.05
        else "ns"
    )
    sig_tk = (
        "***"
        if cld_tap_102["p_kruskal"] < 0.001
        else "**"
        if cld_tap_102["p_kruskal"] < 0.01
        else "*"
        if cld_tap_102["p_kruskal"] < 0.05
        else "ns"
    )
    print(
        f"  BG11: H = {cld_bg11_102['h_stat']:.3f}, p = {cld_bg11_102['p_kruskal']:.6f} ({sig_bk})"
    )
    print(
        f"  TAP:  H = {cld_tap_102['h_stat']:.3f}, p = {cld_tap_102['p_kruskal']:.6f} ({sig_tk})"
    )

    print(f"\n  Tukey HSD Compact Letter Display (102 µmol/m²/s):")
    for i in range(5):
        print(
            f"    {COND_NAMES[i]}: BG11={cld_bg11_102['letters'][i]}  TAP={cld_tap_102['letters'][i]}"
        )

    # -------------------------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # FIGURE: Statistical comparison (2x2 grid)
    # -------------------------------------------------------------------------
    print_section("GENERATING FIGURE")

    def format_sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.45, wspace=0.3)

    COLORS_BG11_FIG = ["#004d00", "#1a7a1a", "#2d8f2d", "#3da63d", "#5cb85c"]
    COLORS_TAP_FIG = ["#00008B", "#0000CD", "#4169E1", "#6495ED", "#87CEEB"]
    COND_LABELS_LATEX = [
        r"$C_0=1$",
        r"$C_0=1/2$",
        r"$C_0=1/4$",
        r"$C_0=1/8$",
        r"$C_0=1/16$",
    ]
    SCATTER_MARKERSIZE = 10

    # --- Top row: overlay growth curves ---
    OD_TO_CELLS = 4.77e6
    ax_row1 = None
    for col, (data_bg, data_tap, exp_bg, exp_tap, light_label) in enumerate(
        [
            (data_bg11_51, data_tap_51, EXP_PHASE_BG11_51, EXP_PHASE_TAP_51, "51"),
            (data_bg11_102, data_tap_102, EXP_PHASE_BG11_102, EXP_PHASE_TAP_102, "102"),
        ]
    ):
        ax = (
            fig.add_subplot(gs[0, col])
            if ax_row1 is None
            else fig.add_subplot(gs[0, col], sharey=ax_row1)
        )
        if ax_row1 is None:
            ax_row1 = ax
        for i, (cond, reps) in enumerate(CONDITIONS.items()):
            if all(rep in data_bg.columns for rep in reps):
                mean_bg = data_bg[reps].mean(axis=1) * OD_TO_CELLS
                std_bg = data_bg[reps].std(axis=1, ddof=1) * OD_TO_CELLS
                ax.errorbar(
                    data_bg["time_h"],
                    mean_bg,
                    yerr=std_bg,
                    fmt="o",
                    color=COLORS_BG11_FIG[i],
                    markersize=SCATTER_MARKERSIZE,
                    capsize=2,
                    capthick=0.8,
                    elinewidth=0.8,
                    alpha=0.8,
                    label=COND_LABELS_LATEX[i] + " BG-11",
                )
            if all(rep in data_tap.columns for rep in reps):
                mean_tap = data_tap[reps].mean(axis=1) * OD_TO_CELLS
                std_tap = data_tap[reps].std(axis=1, ddof=1) * OD_TO_CELLS
                ax.errorbar(
                    data_tap["time_h"],
                    mean_tap,
                    yerr=std_tap,
                    fmt="o",
                    color=COLORS_TAP_FIG[i],
                    markersize=SCATTER_MARKERSIZE,
                    markeredgewidth=1.5,
                    capsize=2,
                    capthick=0.8,
                    elinewidth=0.8,
                    label=COND_LABELS_LATEX[i] + " TAP",
                )
        ax.axvline(
            x=exp_bg["start"], color="mediumseagreen", linestyle="--", linewidth=2
        )
        ax.axvline(
            x=exp_bg["end"],
            color="mediumseagreen",
            linestyle="--",
            linewidth=2,
            label=f"BG-11 exponential phase: {exp_bg['start']}-{exp_bg['end']}h",
        )
        ax.axvline(x=exp_tap["start"], color="royalblue", linestyle="--", linewidth=2)
        ax.axvline(
            x=exp_tap["end"],
            color="royalblue",
            linestyle="--",
            linewidth=2,
            label=f"TAP exponential phase: {exp_tap['start']}-{exp_tap['end']}h",
        )
        ax.set_xlabel("Time (h)")
        if col == 0:
            ax.set_ylabel(r"Biomass (cell mL$^{-1}$)")
        light_title_ab = (
            r"$51\ \mu \mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$"
            if col == 0
            else r"$102\ \mu \mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$"
        )
        ax.set_title(f"{light_title_ab}", fontweight="bold", pad=18)
        ax.set_yscale("log")
        ax.set_xlim(right=107 if col == 0 else 155)
        # Reorder legend
        handles, labels = ax.get_legend_handles_labels()
        label_handle = dict(zip(labels, handles))
        ordered_labels = []
        for lbl in labels:
            if lbl.startswith("BG-11 exponential"):
                ordered_labels.append(lbl)
                break
        for lbl in labels:
            if lbl.startswith("TAP exponential"):
                ordered_labels.append(lbl)
                break
        for cl in COND_LABELS_LATEX:
            key = cl + " TAP"
            if key in label_handle:
                ordered_labels.append(key)
        for cl in COND_LABELS_LATEX:
            key = cl + " BG-11"
            if key in label_handle:
                ordered_labels.append(key)
        ordered_handles = [label_handle[l] for l in ordered_labels]
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            frameon=False,
            fontsize=12,
        )

    # Fix ylim for shared log-scale row 1
    y_bottom = np.inf
    y_top = 0.0
    for data_df in [data_bg11_51, data_tap_51, data_bg11_102, data_tap_102]:
        for cond, reps in CONDITIONS.items():
            existing = [r for r in reps if r in data_df.columns]
            if existing:
                vals = data_df[existing].mean(axis=1).values * OD_TO_CELLS
                pos = vals[np.isfinite(vals) & (vals > 0)]
                if len(pos):
                    y_bottom = min(y_bottom, pos.min())
                    y_top = max(y_top, pos.max())
    if np.isfinite(y_bottom):
        ax_row1.set_ylim(bottom=y_bottom * 0.5, top=y_top * 2)

    # --- Bottom row: µmax bar charts with stats ---
    x = np.arange(5)
    width = 0.35
    rev_order = [4, 3, 2, 1, 0]
    cond_labels_cd = [r"$1/16$", r"$1/8$", r"$1/4$", r"$1/2$", r"$1$"]

    ax_row2 = None
    for col, (mu_bg, mu_tap, cld_bg, cld_tap, light_label) in enumerate(
        [
            (mu_bg11_51, mu_tap_51, cld_bg11_51, cld_tap_51, "51"),
            (mu_bg11_102, mu_tap_102, cld_bg11_102, cld_tap_102, "102"),
        ]
    ):
        ax = (
            fig.add_subplot(gs[1, col])
            if ax_row2 is None
            else fig.add_subplot(gs[1, col], sharey=ax_row2)
        )
        if ax_row2 is None:
            ax_row2 = ax

        mean_bg = [np.nanmean(mu_bg[str(i)]) for i in rev_order]
        std_bg = [np.nanstd(mu_bg[str(i)], ddof=1) for i in rev_order]
        mean_tap = [np.nanmean(mu_tap[str(i)]) for i in rev_order]
        std_tap = [np.nanstd(mu_tap[str(i)], ddof=1) for i in rev_order]
        letters_bg = [cld_bg["letters"][i] for i in rev_order]
        letters_tap = [cld_tap["letters"][i] for i in rev_order]

        bg_face = mcolors.to_rgba("mediumseagreen", alpha=0.4)
        tap_face = mcolors.to_rgba("teal", alpha=0.4)

        ax.bar(
            x - width / 2,
            mean_bg,
            width,
            yerr=std_bg,
            label="BG-11",
            color=bg_face,
            capsize=5,
            edgecolor="mediumseagreen",
            linewidth=1.5,
            error_kw=dict(ecolor="mediumseagreen", capthick=1.5),
        )
        ax.bar(
            x + width / 2,
            mean_tap,
            width,
            yerr=std_tap,
            label="TAP",
            color=tap_face,
            capsize=5,
            edgecolor="teal",
            linewidth=1.5,
            error_kw=dict(ecolor="teal", capthick=1.5),
        )

        for i in range(5):
            y_b = mean_bg[i] + std_bg[i] + 0.002
            ax.text(
                x[i] - width / 2,
                y_b,
                letters_bg[i],
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="heavy",
                color="mediumseagreen",
            )
            y_t = mean_tap[i] + std_tap[i] + 0.002
            ax.text(
                x[i] + width / 2,
                y_t,
                letters_tap[i].upper(),
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="heavy",
                color="teal",
            )

        if col == 0:
            ax.set_ylabel(r"$\mu_{max}$ (h$^{-1}$)")
        ax.set_xlabel(r"$C_0$")
        light_title = (
            r"$51\ \mu \mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$"
            if col == 0
            else r"$102\ \mu \mathrm{mol}_{h\nu}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$"
        )
        ax.set_title(f"{light_title}", fontweight="bold", pad=18)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels_cd)
        ax.legend(loc="upper left", frameon=False)

        sig_bg_anova = format_sig(cld_bg["p_anova"])
        sig_tap_anova = format_sig(cld_tap["p_anova"])
        sig_bg_kw = format_sig(cld_bg["p_kruskal"])
        sig_tap_kw = format_sig(cld_tap["p_kruskal"])

        stats_text = (
            f"One-way ANOVA (across concentrations):  "
            f"BG-11: F(4,10)={cld_bg['f_stat']:.2f}, p={cld_bg['p_anova']:.4f} ({sig_bg_anova})  |  "
            f"TAP: F(4,10)={cld_tap['f_stat']:.2f}, p={cld_tap['p_anova']:.4f} ({sig_tap_anova})\n"
            f"Kruskal-Wallis:  "
            f"BG-11: H={cld_bg['h_stat']:.2f}, p={cld_bg['p_kruskal']:.4f} ({sig_bg_kw})  |  "
            f"TAP: H={cld_tap['h_stat']:.2f}, p={cld_tap['p_kruskal']:.4f} ({sig_tap_kw})\n"
            f"Tukey HSD CLD: lowercase (a-z) = BG-11, UPPERCASE (A-Z) = TAP  "
            f"(groups sharing a letter are not significantly different, α=0.05)"
        )
        ax.text(
            0.5,
            -0.25,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightyellow",
                edgecolor="gray",
                alpha=0.9,
            ),
        )

    plt.savefig(OUTPUT_DIR + "statistical_comparison.png", dpi=360, bbox_inches="tight")
    print(f"Figure saved: {OUTPUT_DIR}statistical_comparison.png")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE (CSV)
    # -------------------------------------------------------------------------
    print_section("SUMMARY TABLE")

    summary_data = []
    for light, mu_bg, mu_tap, k_bg, k_tap in [
        (51, mu_bg11_51, mu_tap_51, k_bg11_51, k_tap_51),
        (102, mu_bg11_102, mu_tap_102, k_bg11_102, k_tap_102),
    ]:
        for i, cond in enumerate(["0", "1", "2", "3", "4"]):
            mean_mu_b = np.nanmean(mu_bg[cond])
            std_mu_b = np.nanstd(mu_bg[cond], ddof=1)
            td_b = np.log(2) / mean_mu_b if mean_mu_b > 0 else np.nan
            mean_k_b = np.nanmean(k_bg[cond])
            std_k_b = np.nanstd(k_bg[cond], ddof=1)

            mean_mu_t = np.nanmean(mu_tap[cond])
            std_mu_t = np.nanstd(mu_tap[cond], ddof=1)
            td_t = np.log(2) / mean_mu_t if mean_mu_t > 0 else np.nan
            mean_k_t = np.nanmean(k_tap[cond])
            std_k_t = np.nanstd(k_tap[cond], ddof=1)

            ratio = mean_mu_t / mean_mu_b if mean_mu_b > 0 else np.nan

            summary_data.append(
                {
                    "Light_umol_m2_s": light,
                    "Condition": COND_NAMES[i],
                    "mumax_BG11_per_h": mean_mu_b,
                    "std_mumax_BG11": std_mu_b,
                    "Td_BG11_h": td_b,
                    "K_BG11": mean_k_b,
                    "std_K_BG11": std_k_b,
                    "mumax_TAP_per_h": mean_mu_t,
                    "std_mumax_TAP": std_mu_t,
                    "Td_TAP_h": td_t,
                    "K_TAP": mean_k_t,
                    "std_K_TAP": std_k_t,
                    "Ratio_mumax_TAP_BG11": ratio,
                    "Ratio_K_TAP_BG11": mean_k_t / mean_k_b,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR + "growth_results_combined.csv", index=False, sep=";")
    print(f"CSV saved: {OUTPUT_DIR}growth_results_combined.csv")

    # -------------------------------------------------------------------------
    # CONCLUSIONS
    # -------------------------------------------------------------------------
    print_section("CONCLUSIONS")

    for light, mu_bg, mu_tap in [
        (51, mu_bg11_51, mu_tap_51),
        (102, mu_bg11_102, mu_tap_102),
    ]:
        mu_ratio = []
        for cond in ["0", "1", "2", "3", "4"]:
            individual_ratios = [mu_tap[cond][j] / mu_bg[cond][j] for j in range(3)]
            mu_ratio.append(np.mean(individual_ratios))
        print(f"\nAt {light} µmol/m²/s:")
        print(
            f"  Mean ratio µmax(TAP/BG11) = {np.mean(mu_ratio):.2f} ± {np.std(mu_ratio, ddof=1):.2f}"
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
