import numpy as np
import pandas as pd
import torch

from datetime import datetime, timedelta

PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/"


def add_calendar_features(df):
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    df["hour"] = df["timestamp_utc"].dt.hour
    df["dayofweek"] = df["timestamp_utc"].dt.dayofweek
    df["dayofyear"] = df["timestamp_utc"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df.drop(columns=['hour', 'dayofweek', 'dayofyear'], inplace=True)

    return df

def create_sequences(demand, calendar, S=48, horizon=24):
    X_demand = []
    X_calendar = []

    T_total = len(demand)

    for t in range(T_total - (S + horizon)):
        # past demand
        d_seq = demand[t : t + S]                      # (S, Z)

        # calendar for past + future
        c_seq = calendar[t : t + S + horizon]          # (S+24, C_cal)

        X_demand.append(d_seq)
        X_calendar.append(c_seq)

    return np.array(X_demand), np.array(X_calendar)

def extract_year(s: str) -> str:
    return s[2:6]


def build_file_list(path, start="2019010100", end="2024123123"):
    """
    Generate filenames from X_2019010100.pt to X_2024123123.pt
    Format: X_YYYYMMDDХH.pt (year, month, day, hour)
    """
    # Parse start and end
    start_dt = datetime.strptime(start, "%Y%m%d%H")
    end_dt = datetime.strptime(end, "%Y%m%d%H")

    path += "2019/"
    assert path == PATH + "weather_data/2019/"
    
    filenames = []
    current = start_dt
    
    while current <= end_dt:
        filename = f"X_{current.strftime('%Y%m%d%H')}.pt"
        filename = path
        filenames.append(filename)
        current += timedelta(hours=1)
    
    return filenames

def assert_no_empty_values(df: pd.DataFrame) -> None:
    """
    Asserts that there are no empty values in a DataFrame.
    
    Empty values include:
    - NaN / None
    - Empty strings ""
    - Whitespace-only strings "   "

    Raises AssertionError with a detailed report if any are found.
    """
    issues = {}

    # Check for NaN / None
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        for col, count in nan_cols.items():
            issues.setdefault(col, []).append(f"{count} NaN/None value(s)")

    # Check for empty or whitespace-only strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        mask = df[col].astype(str).str.strip() == ""
        count = mask.sum()
        if count > 0:
            issues.setdefault(col, []).append(f"{count} empty/whitespace string(s)")

    if issues:
        report_lines = ["DataFrame contains empty values:"]
        for col, problems in issues.items():
            report_lines.append(f"  Column '{col}': {', '.join(problems)}")
        raise AssertionError("\n".join(report_lines))