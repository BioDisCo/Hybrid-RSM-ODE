import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib as mpl

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


def plot_comparison_two_dates_custom_markers(
    means_stds_file1, raw_data_file1, means_stds_file2, raw_data_file2, num_points=8
):
    """
    Two columns of graphs for two different dates.
    C0 = 1 (top) → C0 = 0.0625 (bottom)
    Y-axis only on the first column and X-axis only on the last row.
    Custom colors and markers according to C0 and replicate.
    Line styles for regression: ':' for A, '-' for B, '-.' for C.
    """

    datasets = [(means_stds_file1, raw_data_file1), (means_stds_file2, raw_data_file2)]

    # Color definition by C0
    colors_by_group = {
        0: "mediumseagreen",  # C0 = 1
        1: "grey",  # C0 = 1/2
        2: "tomato",  # C0 = 1/4
        3: "teal",  # C0 = 1/8
        4: "orange",  # C0 = 1/16
    }

    # Marker definition by replicate
    markers_by_rep = {"A": "o", "B": "^", "C": "s"}

    # Line style definition for regression
    lines_by_rep = {"A": ":", "B": "-", "C": "-."}

    n_graphs = 5
    fig, axes = plt.subplots(nrows=n_graphs, ncols=2, figsize=(16, n_graphs * 4))
    axes = np.array(axes).reshape(n_graphs, 2)

    for col_index, (means_file, raw_file) in enumerate(datasets):
        df_means = pd.read_csv(means_file, sep=";")
        df_raw = pd.read_csv(raw_file, sep=";")
        df_raw = df_raw.iloc[1:].reset_index(drop=True)
        df_raw["Time (s)"] = pd.to_numeric(df_raw["Time (s)"], errors="coerce")

        for group_index in range(n_graphs):
            ax = axes[group_index, col_index]

            for rep in ["A", "B", "C"]:
                label = f"{group_index}{rep}"

                try:
                    x_vals = df_means[f"{label}_avg"].values
                    x_err = df_means[f"{label}_std"].values
                    y_vals = pd.to_numeric(
                        df_raw[f"OD {label}"], errors="coerce"
                    ).values
                    y_err = pd.to_numeric(
                        df_raw[f"Std {label}"], errors="coerce"
                    ).values
                except KeyError:
                    continue

                min_len = min(len(x_vals), len(y_vals), len(x_err), len(y_err))
                x_vals = x_vals[:min_len]
                y_vals = y_vals[:min_len]
                x_err = x_err[:min_len]
                y_err = y_err[:min_len]

                mask = (
                    ~np.isnan(x_vals)
                    & ~np.isnan(y_vals)
                    & ~np.isnan(x_err)
                    & ~np.isnan(y_err)
                )
                x_all = x_vals[mask]
                y_all = y_vals[mask]
                xerr_all = x_err[mask]
                yerr_all = y_err[mask]

                if len(x_all) == 0:
                    continue

                # Custom color and marker
                color = colors_by_group[group_index]
                marker = markers_by_rep[rep]
                line_style = lines_by_rep[rep]

                # Scatter plot
                ax.errorbar(
                    x_all,
                    y_all,
                    xerr=xerr_all,
                    yerr=yerr_all,
                    fmt=marker,
                    color=color,
                    capsize=3,
                    alpha=0.7,
                    linestyle="None",
                )

                # Regression on the first points
                x = x_all[:num_points]
                y = y_all[:num_points]

                if len(x) >= 2:
                    model = LinearRegression()
                    model.fit(x.reshape(-1, 1), y)
                    y_pred = model.predict(x.reshape(-1, 1))
                    r2 = r2_score(y, y_pred)

                    a = model.coef_[0]
                    b = model.intercept_

                    x_fit = np.linspace(min(x_all), max(x_all), 100).reshape(-1, 1)
                    y_fit = model.predict(x_fit)
                    ax.plot(x_fit, y_fit, linestyle=line_style, color=color)

                    label_eq = f"{label}: y = {a:.2f}x + {b:.2f} (R²={r2:.2f})"
                    ax.plot(
                        [],
                        [],
                        color=color,
                        linestyle=line_style,
                        marker=marker,
                        label=label_eq,
                    )

            # C0 title
            ax.set_title(f"$C_0 =$ {1 / (2**group_index):.3g}", fontsize=18)

            # X-axis only on the last row
            if group_index != n_graphs - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("OD (750 nm) - plate reader", fontsize=16)

            # Y-axis only on the first column
            if col_index == 0:
                ax.set_ylabel("OD (750 nm) - cuvette reader", fontsize=16)
            else:
                ax.set_ylabel("")

            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.grid(False)
            ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig("cuvette_plate_link.png", dpi=360)
    plt.show()


# Function call
plot_comparison_two_dates_custom_markers(
    "replicates_means_stds_01_07_2025.csv",
    "data_exp_Chlamy_01-07-25.csv",
    "replicates_means_stds_07_07_2025.csv",
    "data_exp_Chlamy_07-07-25.csv",
    num_points=9,
)
