import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# -------------------------
# Global font configuration
# -------------------------
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["figure.titlesize"] = 20


def saturation_model(x, a, b):
    return a * x / (b + x)


def plot_linearity_two_dates(
    file_dil_01,
    file_nodil_01,
    file_dil_07,
    file_nodil_07,
    num_points=8,
    fit_global=True,
):

    datasets = [(file_dil_01, file_nodil_01), (file_dil_07, file_nodil_07)]

    colors_by_group = {
        0: "mediumseagreen",
        1: "grey",
        2: "tomato",
        3: "teal",
        4: "orange",
    }

    markers_by_rep = {"A": "o", "B": "^", "C": "s"}
    linestyle_by_rep = {"A": ":", "B": "-", "C": "-."}

    n_groups = 5
    fig, axes = plt.subplots(nrows=n_groups, ncols=2, figsize=(16, n_groups * 4))

    axes = np.array(axes).reshape(n_groups, 2)

    for col_idx, (file_dil, file_nodil) in enumerate(datasets):
        df_dil = pd.read_csv(file_dil, sep=";")
        df_nodil = pd.read_csv(file_nodil, sep=";")

        all_x_global = {0: [], 1: []}
        all_y_global = {0: [], 1: []}

        for group_index in range(n_groups):
            ax = axes[group_index, col_idx]
            color = colors_by_group[group_index]

            for rep in ["A", "B", "C"]:
                label = f"{group_index}{rep}"

                try:
                    x_vals = df_dil[f"{label}_avg"].values
                    y_vals = df_nodil[f"{label}_avg"].values
                    xerr = df_dil[f"{label}_std"].values
                    yerr = df_nodil[f"{label}_std"].values
                except KeyError:
                    continue

                min_len = min(len(x_vals), len(y_vals), len(xerr), len(yerr))
                x_vals, y_vals = x_vals[:min_len], y_vals[:min_len]
                xerr, yerr = xerr[:min_len], yerr[:min_len]

                mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                x_all, y_all = x_vals[mask], y_vals[mask]
                xerr, yerr = xerr[mask], yerr[mask]

                ax.errorbar(
                    x_all,
                    y_all,
                    xerr=xerr,
                    yerr=yerr,
                    fmt=markers_by_rep[rep],
                    color=color,
                    linestyle="None",
                    capsize=3,
                    alpha=0.7,
                )

                if len(x_all) >= 2:
                    x_fit_pts = x_all[:num_points]
                    y_fit_pts = y_all[:num_points]

                    model = LinearRegression()
                    model.fit(x_fit_pts.reshape(-1, 1), y_fit_pts)
                    y_pred = model.predict(x_fit_pts.reshape(-1, 1))
                    r2 = r2_score(y_fit_pts, y_pred)

                    x_range = np.linspace(min(x_all), max(x_all), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))

                    ax.plot(
                        x_range, y_range, linestyle=linestyle_by_rep[rep], color=color
                    )

                    ax.plot(
                        [],
                        [],
                        color=color,
                        linestyle=linestyle_by_rep[rep],
                        marker=markers_by_rep[rep],
                        label=f"{label} (R²={r2:.2f})",
                    )

                if fit_global and group_index in [0, 1]:
                    all_x_global[group_index].extend(x_all)
                    all_y_global[group_index].extend(y_all)

            # ----- Titles & axes -----
            ax.set_title(f"$C_0 =$ {1 / (2**group_index):.3g}", fontsize=18)

            if group_index == n_groups - 1:
                ax.set_xlabel("OD (750 nm) - dilution", fontsize=16)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel("OD (750 nm) - raw", fontsize=16)
            else:
                ax.set_ylabel("")

            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.grid(False)
            # Place legend at upper left for first row to avoid curve overlap
            legend_loc = "upper left" if group_index == 0 else "lower right"
            ax.legend(loc=legend_loc, fontsize=12, frameon=False)

        # ----- Global saturation fit -----
        if fit_global:
            for g in [0, 1]:
                if all_x_global[g]:
                    xg = np.array(all_x_global[g])
                    yg = np.array(all_y_global[g])
                    popt, _ = curve_fit(saturation_model, xg, yg, bounds=(0, [10, 10]))
                    x_fit = np.linspace(min(xg), max(xg), 200)
                    y_fit = saturation_model(x_fit, *popt)
                    r2g = r2_score(yg, saturation_model(xg, *popt))

                    a_val, b_val = popt
                    legend_loc_global = "upper left" if g == 0 else "lower right"
                    axes[g, col_idx].plot(
                        x_fit,
                        y_fit,
                        "k-",
                        label=f"$y = \\frac{{{a_val:.3f} \\cdot x}}{{{b_val:.3f} + x}}$\nR²={r2g:.2f}",
                    )
                    axes[g, col_idx].legend(
                        loc=legend_loc_global, fontsize=12, frameon=False
                    )

    plt.tight_layout()
    plt.savefig("linearity_plate_reader.png", dpi=360)
    plt.show()


# ----- Call -----
plot_linearity_two_dates(
    "replicates_means_stds_01_07_2025.csv",
    "replicates_means_stds_no_dilution_01_07_2025.csv",
    "replicates_means_stds_07_07_2025.csv",
    "replicates_means_stds_no_dilution_07_07_2025.csv",
    num_points=8,
    fit_global=True,
)
