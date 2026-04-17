import pandas as pd
import os
from pathlib import Path

def load_and_concat_csvs(directory: str):
    """
    Recursively find all CSV files in a directory, concatenate them,
    and sort by the 'timestamp_utc' column (parsed as datetime).
    """
    csv_files = list(Path(directory).rglob("*.csv"))

    print(csv_files)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{directory}'")

    dfs = [
        pd.read_csv(f, parse_dates=["timestamp_utc"])
        for f in csv_files
    ]

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    empty_counts = df.isnull().sum()
    cols_with_empties = empty_counts[empty_counts > 0]
    assert cols_with_empties.empty, (
        f"DataFrame contains empty values:\n{cols_with_empties.to_string()}"
    )

    df.to_csv('demand_raw.csv')

if __name__ == '__main__':
    load_and_concat_csvs('/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data')