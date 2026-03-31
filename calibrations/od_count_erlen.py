import pandas as pd
import matplotlib.pyplot as plt
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


df = pd.read_csv("link_od_count.csv", sep=";")

df["OD "] = pd.to_numeric(df["OD "], errors="coerce")
df["cell / mL"] = pd.to_numeric(df["cell / mL"], errors="coerce")

df = df.dropna(subset=["OD ", "cell / mL"])

# Extract group number from first column (e.g. "0A" -> 0)
df["group"] = df["C0 condition"].str[0].astype(int)

x = df["OD "].values
y = df["cell / mL"].values / 1e7

# Fit y = ax (through origin)
a = np.dot(x, y) / np.dot(x, x)

# R²
y_pred = a * x
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

x_line = np.linspace(0, x.max(), 200)

# Format coefficient in scientific notation for the legend
a_original = a * 1e7
exp = int(np.floor(np.log10(abs(a_original))))
mantissa = a_original / 10**exp

group_styles = {
    0: ("mediumseagreen", "$C_0 = 1$"),
    1: ("grey", "$C_0 = 0.5$"),
    2: ("tomato", "$C_0 = 0.25$"),
    3: ("teal", "$C_0 = 0.125$"),
    4: ("orange", "$C_0 = 0.0625$"),
}

fig, ax = plt.subplots()
for group, (color, label) in group_styles.items():
    mask = df["group"] == group
    ax.scatter(
        df.loc[mask, "OD "], df.loc[mask, "cell / mL"] / 1e7, color=color, label=label
    )

ax.plot(
    x_line,
    a * x_line,
    color="black",
    linestyle="--",
    label=f"$y = {mantissa:.2f} \\cdot 10^{{{exp}}} \\, x \\quad (R^2 = {r2:.3f})$",
)
ax.set_xlabel("OD (750 nm)")
ax.set_ylabel("Count $\\times 10^7$ (cell mL$^{-1}$)")

handles, labels = ax.get_legend_handles_labels()
legend_c0 = ax.legend(handles[:-1], labels[:-1], loc="upper left", frameon=False)
legend_fit = ax.legend(handles[-1:], labels[-1:], loc="lower right", frameon=False)
ax.add_artist(legend_c0)
plt.tight_layout()
plt.savefig("count_OD_link.png", dpi=360)
plt.show()
