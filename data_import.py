"""Helpers to load experimental data."""

from enum import Enum
from typing import TypedDict
from pathlib import Path
import numpy as np
import pandas as pd

OdData = dict[str, list[float]]

def trimmed_mean(values: list | np.ndarray, fraction: float = 0.25) -> float:
    """Calculate mean by dropping min and max values (fault-tolerant).

    Args:
        values: List or array of values
        fraction: Fraction of values to drop from each end (default 0.25 = drop min and max)

    Returns:
        Mean of the remaining values
    """
    arr = np.array(values)
    if len(arr) <= 2:
        return np.nanmean(arr)

    # Sort and trim (drop min and max by default)
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    # Drop 1 from each end (min and max)
    trimmed = sorted_arr[1:-1]

    return np.nanmean(trimmed) if len(trimmed) > 0 else np.nanmean(arr)


class Container(Enum):
    ERLEN = 0
    PLATE = 1


class IndividualCurve(TypedDict):
    Time: list[float]
    Value: list[float]


class ExperimentalData(TypedDict):
    Time: list[float]
    Mean: list[float]
    Std: list[float]
    container: Container
    C0_factor: float
    L0_factor: float
    L_meters: float
    V_mL: float
    replicates: list[IndividualCurve]


Experiments = dict[str, ExperimentalData]

# FLASK DATA READING
def read_csv_data_erlen(
    filepath: str, conv_OD_to_cell: float | None = None
) -> Experiments:
    """Data reading from csv for erlen."""
    # Read CSV file, skipping the 2nd row (empty)
    df = pd.read_csv(filepath, sep=";", skiprows=[1])

    # Select useful columns
    colonnes_interessees = [
        col for col in df.columns if col.startswith(("OD ", "Std ", "mean ", "std "))
    ]

    # Find the last row where *all* columns of interest are valid (non-NaN)
    dernier_index_valide = df[colonnes_interessees].dropna(how="any").index[-1]

    # Truncate the DataFrame up to this row
    df = df.loc[:dernier_index_valide]

    # Convert time to hours
    time_h = (df["Time (s)"] / 3600).tolist()

    # Extract columns into a dictionary
    od_data: OdData = {col: df[col].tolist() for col in colonnes_interessees}

    # Mapping for L0:
    # 16/09/2024 - 0.07 * L0_REF = 11.9 µmol/m2/s
    # 21/10/2024 - 0.15 * L0_REF = 25.5 µmol/m2/s
    # 04/11/2024 - 0.15 * L0_REF = 25.5 µmol/m2/s
    # 17/02/2025 - 0.3 * L0_REF = 51 µmol/m2/s
    # 01/07/2025 - 0.6 * L0_REF = 102 µmol/m2/s
    # 07/07/2025 - 1.0 * L0_REF = 170 µmol/m2/s

    L0_factor: float = 1
    if "16-09-24" in filepath:
        L0_factor = 0.07
    elif "21-10-24" in filepath or "04-11-24" in filepath:
        L0_factor = 0.15
    elif "17-02-25" in filepath:
        L0_factor = 0.3
    elif "01-07-25" in filepath:
        L0_factor = 0.6
    elif "07-07-25" in filepath:
        L0_factor = 1
    else:
        msg = "Unknown experiment."
        raise NotImplementedError(msg)

    ret_dict: Experiments = {}
    for key in od_data:
        C0_factor: float = 1.0
        if key.startswith("mean"):
            c0 = key.strip().split(" ")[-1].strip()
            if c0 == "NC":
                continue
            values = (
                od_data[key]
                if conv_OD_to_cell is None
                else [x * conv_OD_to_cell for x in od_data[key]]
            )
            # C0 factor is encodes as powers of 1/2 for Erlen.
            C0_factor = 0.5 ** float(c0)
            C0_factor_corrected = 0.57 ** float(c0)  # just to check if this could have experimentally happened

            # get replicate data
            replicates: list[IndividualCurve] = []
            for r in ["A", "B", "C"]:
                rep_col = f"OD {c0}{r}"
                rep_values = (
                    od_data[rep_col]
                    if conv_OD_to_cell is None
                    else [x * conv_OD_to_cell for x in od_data[rep_col]]
                )
                replicates.append({"Time": time_h, "Value": rep_values})

            # use this as the new key
            exp_name = f"Erlen_C0x{C0_factor:.3f}_L0x{L0_factor:.3f}"

            # fill the dict
            ret_dict[exp_name] = {
                "Time": time_h,
                "Mean": values,
                "Std": [],
                "C0_factor": C0_factor_corrected,
                "L0_factor": L0_factor,
                "container": Container.ERLEN,
                "L_meters": 0.01,
                "V_mL": 0.050,
                "replicates": replicates,
            }
        elif key.startswith("std"):
            c0 = key.strip().split(" ")[-1].strip()
            if c0 == "NC":
                continue
            C0_factor = 1 / 2 ** float(c0)
            exp_name = f"Erlen_C0x{C0_factor:.3f}_L0x{L0_factor:.3f}"
            values = (
                od_data[key]
                if conv_OD_to_cell is None
                else [x * conv_OD_to_cell for x in od_data[key]]
            )
            ret_dict[exp_name]["Std"] = values

    for k, e in ret_dict.items():
        assert len(e["Time"]) == len(e["Mean"]), (
            f"Mean of {k} has not the length of Time: {filepath}. {len(e['Time'])} vs {len(e['Mean'])}"
        )
        assert len(e["Time"]) == len(e["Std"]), (
            f"Std of {k} has not the length of Time: {filepath}. {len(e['Time'])} vs {len(e['Std'])}"
        )

    return ret_dict

# PLATE RAW DATA READING
def read_csv_data_plate(
    filepath: str,
    conv_OD_plate_to_OD_erlen: float,
    conv_OD_to_cell: float | None = None,
) -> Experiments:
    """Data reading from csv for plate."""
    directory = Path(filepath).parent

    # Read CSV file, skipping the 2nd row (empty)
    df = pd.read_csv(filepath, sep=";", skiprows=[1])

    # Select useful columns
    colonnes_interessees = [
        col for col in df.columns if col.startswith(("OD C0 = ", "std C0 = "))
    ]

    # Find the last row where *all* columns of interest are valid (non-NaN)
    dernier_index_valide = df[colonnes_interessees].index[-1]  # better to deal with NaN value in corrected dataset instead of the line dernier_index_valide = df[colonnes_interessees].dropna(how="any").index[-1]

    # Truncate the DataFrame up to this row
    df = df.loc[:dernier_index_valide]

    # Convert time to hours
    time_h = df["Time (hour)"].tolist()

    # Extract columns into a dictionary
    od_data: OdData = {col: df[col].tolist() for col in colonnes_interessees}

    # Mapping date and L0 values:
    # plate_21-10-24: 0.15 * L0_REF = 25.5 µmol/m2/s
    # plate_1_04-11-24: 0.15 * L0_REF = 25.5 µmol/m2/s
    # plate_2_04-11-24: 0.15 * L0_REF = 25.5 µmol/m2/s
    # plate_cleaned_16-09-24: 0.07 * L0_REF = 11.9 µmol/m2/s
    # plate_cleaned_17-02-25: 0.3 * L0_REF = 51 µmol/m2/s
    # plate_1_01-07-2025: 0.6 * L0_REF = 102 µmol/m2/s
    # plate_2_01-07-2025: 0.6 * L0_REF = 102 µmol/m2/s
    # plate_1_07-07-2025: 1.0 * L0_REF = 170 µmol/m2/s
    # plate_2_07-07-2025: 1.0 * L0_REF = 170 µmol/m2/s

    L0_factor: float = 1
    date_str: str = ""
    if "21-10-24" in filepath or "04-11-24" in filepath:
        L0_factor = 0.15  
        date_str = "21_10_2024"
    elif "cleaned_16-09-24" in filepath:
        L0_factor = 0.07
        date_str = "16_09_2024"
    elif "cleaned_17-02-25" in filepath:
        L0_factor = 0.3
        date_str = "17_02_2025"
    elif "01-07-25" in filepath:
        L0_factor = 0.6
        date_str = "01_07_2025"
    elif "07-07-25" in filepath:
        L0_factor = 1
        date_str = "07_07_2025"
    else:
        msg = "Unknown experiment."
        raise NotImplementedError(msg)

    ret_dict: Experiments = {}
    for key in od_data:
        if key.startswith("OD"):
            c0 = key.strip().split("C0 = ")[-1].strip()
            values = (
                od_data[key]
                if conv_OD_to_cell is None
                else [
                    x * conv_OD_to_cell * conv_OD_plate_to_OD_erlen
                    for x in od_data[key]
                ]
            )
            # C0 factors are the keys: 1, 1/2, 1/4, ...
            C0_factor = float(eval(c0))
            L0_factor = L0_factor
            exp_name = f"Plate_C0x{C0_factor:.3f}_L0x{L0_factor:.3f}"

            # get replicates
            replicate_files = [
                directory / f"replicates_OD_{date_str}_plate_1.csv",
                directory / f"replicates_OD_{date_str}_plate_2.csv",
            ]

            # filter to those that exist
            replicate_files = [f for f in replicate_files if f.is_file()]
            assert len(replicate_files) > 0, filepath

            replicates: list[IndividualCurve] = []
            for f in replicate_files:
                df_rep = pd.read_csv(f, sep=";")
                time = df_rep["Time (hour)"].to_list()
                for r in range(1, 25): # 25 is the maximum number of replicates
                    col_name = f"C0 = {c0} R{r}"
                    if col_name in df_rep.keys():
                        value = (
                            df_rep[col_name].to_list()
                            if conv_OD_to_cell is None
                            else [
                                x * conv_OD_to_cell * conv_OD_plate_to_OD_erlen
                                for x in df_rep[col_name].to_list()
                            ]
                        )
                        replicates.append({"Time": time, "Value": value})

            ret_dict[exp_name] = {
                "Time": time_h,
                "Mean": values,
                "Std": [],
                "C0_factor": C0_factor,
                "L0_factor": L0_factor,
                "container": Container.PLATE,
                "L_meters": 0.005,
                "V_mL": 0.0002,
                "replicates": replicates,
            }
            # print(f"{replicates=}")

        elif key.startswith("std"):
            c0 = key.strip().split("C0 = ")[-1].strip()
            C0_factor = float(eval(c0))
            L0_factor = L0_factor
            exp_name = f"Plate_C0x{C0_factor:.3f}_L0x{L0_factor:.3f}"
            values = (
                od_data[key]
                if conv_OD_to_cell is None
                else [
                    x * conv_OD_to_cell * conv_OD_plate_to_OD_erlen
                    for x in od_data[key]
                ]
            )
            ret_dict[exp_name]["Std"] = values

    for k, e in ret_dict.items():
        assert len(e["Time"]) == len(e["Mean"]), (
            f"Mean of {k} has not the length of Time: {filepath}"
        )
        assert len(e["Time"]) == len(e["Std"]), (
            f"Std of {k} has not the length of Time: {filepath}"
        )
        assert "replicates" in e, filepath

    return ret_dict


# FLASK DATA READING (KAMBE ET AL. 2022 DATA)
def read_csv_data_erlen_Kambe(
    filepath: str, conv_OD_to_cell: float | None = None
) -> Experiments:
    """Data reading from csv for erlen."""
    # Read CSV file
    df = pd.read_csv(filepath, sep=";")

    # Select useful columns
    colonnes_interessees = [
        col for col in df.columns if col.startswith(("mean "))
    ]

    # Find the last row where *all* columns of interest are valid (non-NaN)
    dernier_index_valide = df[colonnes_interessees].dropna(how="any").index[-1]

    # Truncate the DataFrame up to this row
    df = df.loc[:dernier_index_valide]

    # Convert time to hours
    time_h = (df["Time (hour)"]).tolist()

    # Extract columns into a dictionary
    od_data: OdData = {col: df[col].tolist() for col in colonnes_interessees}

    # get factor for L0:
    # L0 = 0.274 µE/s
    # L0 = 0.521 µE/s
    # L0 = 1.09 µE/s
    # L0 = 2.92 µE/s

    L0_factor: float = 1
    if "0.274" in filepath:
        L0_factor = 0.093835616
    elif "0.521" in filepath:
        L0_factor = 0.178424658
    elif "1.09" in filepath:
        L0_factor = 0.373287671
    elif "2.92" in filepath:
        L0_factor = 1
    else:
        msg = "Unknown experiment."
        raise NotImplementedError(msg)

    ret_dict: Experiments = {}
    for key in od_data:
        C0_factor: float = 1.0
        if key.startswith("mean"):
            c0 = key.strip().split(" ")[-1].strip()
            values = (
                od_data[key]
                if conv_OD_to_cell is None
                else [x * conv_OD_to_cell for x in od_data[key]]
            )
            # C0 factor is encodes as powers of 1/2 for Erlen.
            C0_factor = 1 / 2 ** float(c0)

            # use this as the new key
            exp_name = f"Erlen_C0x{C0_factor:.3f}_L0x{L0_factor:.3f}"

            # fill the dict
            ret_dict[exp_name] = {
                "Time": time_h,
                "Mean": values,
                "Std": None,
                "C0_factor": C0_factor,
                "L0_factor": L0_factor,
                "container": Container.ERLEN,
                "L_meters": 0.011,
                "V_mL": 0.050,
                "replicates": None,
            }

    return ret_dict


def until_time(exp: Experiments, time: float) -> Experiments:
    """Cuts an experiment at time, and returns the 1st part."""
    ret_dict: Experiments = {}
    for key, e in exp.items():
        ret_dict[key] = {
            "Time": [],
            "Mean": [],
            "Std": [],
            "C0_factor": e["C0_factor"],
            "L0_factor": e["L0_factor"],
            "container": e["container"],
            "L_meters": e["L_meters"],
            "V_mL": e["V_mL"],
            "replicates": e["replicates"],
        }
        for i, t in enumerate(e["Time"]):
            if t <= time:
                ret_dict[key]["Time"] += [e["Time"][i]]
                ret_dict[key]["Mean"] += [e["Mean"][i]]
                ret_dict[key]["Std"] += [e["Std"][i]]
            else:
                break
    return ret_dict


def set_min(exp: Experiments, min_value: float = 0.01) -> Experiments:
    """Sets all measurments to a min of min_value."""
    ret_dict: Experiments = {}
    for key, e in exp.items():
        ret_dict[key] = {
            "Time": e["Time"],
            "Mean": [max(v, min_value) for v in e["Mean"]],
            "Std": e["Std"],
            "C0_factor": e["C0_factor"],
            "L0_factor": e["L0_factor"],
            "container": e["container"],
            "L_meters": e["L_meters"],
            "V_mL": e["V_mL"],
            "replicates": e["replicates"],
        }
    return ret_dict

# PLATE HAND-CLEANED DATA READING
def read_csv_data_plate_hand_cleaned(
    filepath: str,
    conv_OD_plate_to_OD_erlen: float,
    conv_OD_to_cell: float | None = None,
) -> Experiments:
    """Data reading from cleaned plate replicate CSV files.

    These files contain only individual replicate columns (C0 = X RY format)
    without mean/std columns.
    """
    df = pd.read_csv(filepath, sep=";")

    # Extract date from filename (format: replicates_OD_MM_DD_YYYY_plate_*.csv)
    filename = Path(filepath).stem
    parts = filename.split("_")

    # Find the date parts (should be after "OD")
    od_idx = parts.index("OD")
    month = parts[od_idx + 1]
    day = parts[od_idx + 2]
    year = parts[od_idx + 3]

    # Map to L0_factor based on date
    date_key = f"{month}_{day}_{year}"
    L0_factor: float = 1

    if date_key in ["21_10_2024", "04_11_2024"]:
        L0_factor = 0.15
    elif date_key == "16_09_2024":
        L0_factor = 0.07
    elif date_key == "17_02_2025":
        L0_factor = 0.3
    elif date_key == "01_07_2025":
        L0_factor = 0.6
    elif date_key == "07_07_2025":
        L0_factor = 1
    else:
        msg = f"Unknown experiment date: {date_key}"
        raise NotImplementedError(msg)

    # Get time column
    time_h = df["Time (hour)"].tolist()

    # Extract C0 factors and group replicates
    c0_replicates: dict[float, list[IndividualCurve]] = {}

    for col in df.columns:
        if col.startswith("C0 = "):
            c0_str = col.split("C0 = ")[1].split(" ")[0]
            c0_factor = float(eval(c0_str))

            if c0_factor not in c0_replicates:
                c0_replicates[c0_factor] = []

            values = (
                df[col].tolist()
                if conv_OD_to_cell is None
                else [
                    x * conv_OD_to_cell * conv_OD_plate_to_OD_erlen
                    for x in df[col].tolist()
                ]
            )
            c0_replicates[c0_factor].append({"Time": time_h, "Value": values})

    # Build experiments dict
    ret_dict: Experiments = {}
    for c0_factor, replicates in c0_replicates.items():
        # Calculate mean and std from replicates (using trimmed mean to drop outliers)
        mean_values = []
        std_values = []

        for i in range(len(time_h)):
            values_at_t = [rep["Value"][i] for rep in replicates]
            # Use trimmed mean (drop min and max values)
            mean_values.append(trimmed_mean(values_at_t))
            std_values.append(np.std(values_at_t))

        exp_name = f"Plate_C0x{c0_factor:.3f}_L0x{L0_factor:.3f}"
        ret_dict[exp_name] = {
            "Time": time_h,
            "Mean": mean_values,
            "Std": std_values,
            "C0_factor": c0_factor,
            "L0_factor": L0_factor,
            "container": Container.PLATE,
            "L_meters": 0.005,
            "V_mL": 0.0002,
            "replicates": replicates,
        }

    return ret_dict


def shift_by_time(exp: Experiments, by_time: dict[str, float]) -> Experiments:
    """Shifts the experimental data as specified by the dict by_time."""
    ret_dict: Experiments = {}
    for key, e in exp.items():
        shift = by_time.get(key, 0.0)
        ret_dict[key] = {
            "Time": [t + shift for t in e["Time"]],
            "Mean": e["Mean"],
            "Std": e["Std"],
            "C0_factor": e["C0_factor"],
            "L0_factor": e["L0_factor"],
            "container": e["container"],
            "L_meters": e["L_meters"],
            "V_mL": e["V_mL"],
            "replicates": e["replicates"],
        }
    return ret_dict
