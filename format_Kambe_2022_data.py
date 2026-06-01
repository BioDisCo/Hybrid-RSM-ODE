import pandas as pd
import os

# ── Read raw Excel ────────────────────────────────────────────────────────────
df_raw = pd.read_excel("OD_data.xlsx", header=None)

# Row 3 contains the header: col1=Ein, col2=C0, col3=NaN, col4..=time points
times = df_raw.iloc[3, 4:].values.astype(int)  # time points as integers

# Data rows start at index 4 (16 rows: 4 Ein × 4 C0)
data_rows = df_raw.iloc[4:].reset_index(drop=True)

# ── Build output directory ────────────────────────────────────────────────────
os.makedirs("data_Kambe", exist_ok=True)

# ── Generate one CSV per Ein value ────────────────────────────────────────────
ein_values = data_rows.iloc[:, 1].unique()  # [0.274, 0.521, 1.09, 2.92]

for ein in ein_values:
    group = data_rows[data_rows.iloc[:, 1] == ein].copy()

    # Sort by descending C0 so that:
    #   mean 0 = C0=1,  mean 1 = C0=0.5,  mean 2 = C0=0.25,  mean 3 = C0=0.125
    group = group.sort_values(by=group.columns[2], ascending=False)

    # Build output DataFrame
    out = pd.DataFrame({"Time (hour)": times})
    for i, (_, row) in enumerate(group.iterrows()):
        out[f"mean {i}"] = row.iloc[4:].values

    # Write CSV with semicolon separator, NaN as empty string
    ein_str = str(ein)
    filepath = os.path.join("data_Kambe", f"data-Ein-{ein_str}.csv")
    out.to_csv(filepath, sep=";", index=False, na_rep="")
    print(f"Written: {filepath}")

print("Done.")
