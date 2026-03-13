import pandas as pd
import os

# Folder
input_dir = "all_data/outliers_removed"
correction_dir = "all_data/linearity_correction"
output_dir = "all_data/final_corrected_data"
os.makedirs(output_dir, exist_ok=True)

## AVERAGE DATA
# Columns to correct in each file
corrections = {
    "mean_growth_curves_C_reinhardtii_plate_1_01-07-25.csv": [
        "OD C0 = 1",
        "OD C0 = 1/2",
    ],
    "mean_growth_curves_C_reinhardtii_plate_1_04-11-24.csv": ["OD C0 = 1"],
    "mean_growth_curves_C_reinhardtii_plate_1_07-07-25.csv": ["OD C0 = 1"],
    "mean_growth_curves_C_reinhardtii_plate_2_01-07-25.csv": [
        "OD C0 = 1",
        "OD C0 = 1/2",
    ],
    "mean_growth_curves_C_reinhardtii_plate_2_04-11-24.csv": ["OD C0 = 1"],
    "mean_growth_curves_C_reinhardtii_plate_2_07-07-25.csv": [
        "OD C0 = 1",
        "OD C0 = 1/2",
    ],
    "mean_growth_curves_C_reinhardtii_plate_21-10-24.csv": ["OD C0 = 1"],
    "mean_growth_curves_C_reinhardtii_plate_cleaned_16-09-24.csv": ["OD C0 = 1"],
    "mean_growth_curves_C_reinhardtii_plate_cleaned_17-02-25.csv": ["OD C0 = 1"],
}

# File to correct
for filename, cols_to_replace in corrections.items():
    original_path = os.path.join(input_dir, filename)
    corrected_path = os.path.join(
        correction_dir, filename.replace(".csv", "_nonlinearity_corrected.csv")
    )
    output_path = os.path.join(output_dir, filename)

    # Load all corrected file and original one without the outliers
    df_original = pd.read_csv(original_path, sep=";")
    df_corrected = pd.read_csv(corrected_path, sep=";")

    # Replace columns
    for col in cols_to_replace:
        if col in df_corrected.columns:
            df_original[col] = df_corrected[col]
        else:
            print(f"Warning ! Column '{col}' missing in {corrected_path}")

    # Save modified file
    df_original.to_csv(output_path, sep=";", index=False)
    print(f"Saved file : {output_path}")

## INTERMEDIATE REPLICATES DATA
# Mapping of columns to correct in each file
columns_to_replace = {
    "replicate_OD_intermediate_dilution_01_07_2025.csv": [
        f"C0 = {c} R{i}"
        for c in [
            "1.0",
            "0.95",
            "0.9",
            "0.85",
            "0.8",
            "0.75",
            "0.7",
            "0.65",
            "0.6",
            "0.55",
            "0.5",
            "0.45",
            "0.4",
        ]
        for i in range(1, 6)
    ],
    "replicate_OD_intermediate_dilution_04_11_2024.csv": [
        f"C0 = {c} R{i}" for c in ["1.0", "0.95", "0.9"] for i in range(1, 6)
    ]
    + ["C0 = 0.85 R1"]
    + [f"C0 = 0.45 R{i}" for i in range(1, 6)],
    "replicate_OD_intermediate_dilution_07_07_2025.csv": [
        f"C0 = {c} R{i}"
        for c in [
            "1.0",
            "0.95",
            "0.9",
            "0.85",
            "0.8",
            "0.75",
            "0.7",
            "0.65",
            "0.6",
            "0.55",
            "0.5",
            "0.45",
            "0.4",
            "0.35",
        ]
        for i in range(1, 6)
    ],
    "replicate_OD_intermediate_dilution_17_02_2025.csv": (
        [
            f"C0 = {c} R{i}"
            for c in ["1.0", "0.95", "0.9", "0.85", "0.8", "0.75", "0.7"]
            for i in range(1, 6)
        ]
        + ["C0 = 0.65 R1"]
        + [f"C0 = 0.6 R{i}" for i in range(1, 6)]
        + [f"C0 = 0.55 R{i}" for i in range(1, 6)]
        + ["C0 = 0.5 R1"]
        + [f"C0 = 0.45 R{i}" for i in range(1, 4)]
        + ["C0 = 0.15 R4"]
    ),
}

# Process file
for filename, columns in columns_to_replace.items():
    original_path = os.path.join(input_dir, filename)
    corrected_path = os.path.join(
        correction_dir, filename.replace(".csv", "_nonlinearity_corrected.csv")
    )
    output_path = os.path.join(output_dir, filename)

    df_original = pd.read_csv(original_path, sep=";")
    df_corrected = pd.read_csv(corrected_path, sep=";")

    for col in columns:
        if col in df_original.columns and col in df_corrected.columns:
            df_original[col] = df_corrected[col]
        else:
            print(f"Warning ! Missing columns : {col} in {filename}")

    df_original.to_csv(output_path, sep=";", index=False)

print("All corrected file have been created in all_data/final_corrected_data/")


## REPLICATES DATA FOR AVERAGE DATA

# Columns to replace
replacement_map = {
    "replicates_OD_16_09_2024_plate_1.csv": [
        *[
            f"C0 = 1 R{i}"
            for i in list(range(1, 9)) + list(range(10, 16)) + list(range(17, 25))
        ]
    ],
    "replicates_OD_17_02_2025_plate_2.csv": [*[f"C0 = 1 R{i}" for i in range(1, 25)]],
    "replicates_OD_04_11_2024_plate_2.csv": [
        *[f"C0 = 1 R{i}" for i in list(range(1, 10)) + list(range(13, 25))]
    ],
    "replicates_OD_01_07_2025_plate_2.csv": [
        *[f"C0 = 1 R{i}" for i in range(1, 25)],
        *[f"C0 = 1/2 R{i}" for i in range(1, 17)],
    ],
    "replicates_OD_07_07_2025_plate_1.csv": [*[f"C0 = 1 R{i}" for i in range(1, 25)]],
    "replicates_OD_01_07_2025_plate_1.csv": [
        *[f"C0 = 1 R{i}" for i in range(1, 25)],
        *[f"C0 = 1/2 R{i}" for i in range(1, 17)],
    ],
    "replicates_OD_07_07_2025_plate_2.csv": [
        *[f"C0 = 1 R{i}" for i in range(1, 25)],
        *[f"C0 = 1/2 R{i}" for i in [1, 4, 6] + list(range(10, 17))],
    ],
    "replicates_OD_17_02_2025_plate_1.csv": [
        *[f"C0 = 1 R{i}" for i in range(1, 25)],
        "C0 = 1/2 R10",
    ],
    "replicates_OD_04_11_2024_plate_1.csv": [
        *[f"C0 = 1 R{i}" for i in list(range(1, 10)) + list(range(17, 25))]
    ],
    # No modifications
    "replicates_OD_21_10_2024_plate_1.csv": None,
}

for filename, columns_to_replace in replacement_map.items():
    raw_path = os.path.join(input_dir, filename)
    raw_df = pd.read_csv(raw_path, sep=";")

    if columns_to_replace is not None:
        corrected_filename = filename.replace(".csv", "_nonlinearity_corrected.csv")
        corrected_path = os.path.join(correction_dir, corrected_filename)
        corrected_df = pd.read_csv(corrected_path, sep=";")

        for col in columns_to_replace:
            if col in raw_df.columns and col in corrected_df.columns:
                raw_df[col] = corrected_df[col]
            else:
                print(f"Warning ! Missing column : {col} in {filename}")

    # Sauvegarde du fichier modifié ou inchangé
    output_path = os.path.join(output_dir, filename)
    raw_df.to_csv(output_path, sep=";", index=False)
    print(f"Saved file : {output_path}")
