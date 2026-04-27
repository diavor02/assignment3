"""
compute_spatial_means.py

For each hourly weather tensor, computes:
- Spatial mean temperature (or channel 0 if unlabeled)
- Spatial mean wind speed (or channel with highest temporal var)
- Spatial 95th-percentile temperature (captures heat extremes better than mean)
- Spatial 5th-percentile temperature (captures cold extremes)

Output: results/hourly_spatial_stats.csv with columns:
  timestamp, ch0_mean, ch0_std, ch0_p05, ch0_p95, ch1_mean, ...
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DATA_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data")
WEATHER_DIR = DATA_ROOT / "weather_data"

# UPDATED AFTER STEP 1:
TEMP_CHANNEL = 0
WIND_CHANNEL = 2


def get_all_weather_files():
    files = sorted(WEATHER_DIR.rglob("X_*.pt"))
    if not files:
        files = sorted(WEATHER_DIR.rglob("*.npy"))
    if not files:
        files = sorted(WEATHER_DIR.rglob("*.npz"))
    return files


def load_weather_array(f: Path) -> np.ndarray:
    if f.suffix == ".pt":
        try:
            obj = torch.load(f, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    elif f.suffix == ".npy":
        arr = np.load(f)
    elif f.suffix == ".npz":
        z = np.load(f)
        arr = z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported file type: {f}")

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim == 3 and arr.shape[0] == 7 and arr.shape[1] == 450 and arr.shape[2] == 449:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"Unexpected array shape {arr.shape} in {f}")

    return arr


def extract_timestamp_from_filename(f: Path):
    """
    Examples:
      X_2019010100.pt -> 2019-01-01 00:00
      2019010100.npy  -> 2019-01-01 00:00
    """
    stem = f.stem
    if stem.startswith("X_"):
        stem = stem[2:]

    try:
        return pd.Timestamp(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]} {stem[8:10]}:00")
    except Exception:
        pass

    try:
        return pd.Timestamp(stem)
    except Exception:
        return None


def compute_spatial_stats(arr, channel):
    """arr shape: (450, 449, c)"""
    ch_data = arr[:, :, channel].ravel()
    return {
        "mean": float(ch_data.mean()),
        "std": float(ch_data.std()),
        "p05": float(np.percentile(ch_data, 5)),
        "p95": float(np.percentile(ch_data, 95)),
        "max": float(ch_data.max()),
        "min": float(ch_data.min()),
    }


def process_one_file(path_str: str):
    f = Path(path_str)
    ts = extract_timestamp_from_filename(f)
    if ts is None:
        return None

    arr = load_weather_array(f)
    row = {"timestamp": ts}
    for ch in range(arr.shape[-1]):
        stats = compute_spatial_stats(arr, ch)
        for k, v in stats.items():
            row[f"ch{ch:02d}_{k}"] = v
    return row


def main():
    files = get_all_weather_files()
    print(f"Found {len(files)} weather files")
    print(f"Configured TEMP_CHANNEL={TEMP_CHANNEL}, WIND_CHANNEL={WIND_CHANNEL}")

    max_workers = int(os.environ.get("SPATIAL_STATS_WORKERS", max(1, min(8, os.cpu_count() or 1))))
    print(f"Using {max_workers} worker processes")

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        path_iter = (str(p) for p in files)
        for row in tqdm(executor.map(process_one_file, path_iter, chunksize=32), total=len(files), desc="Processing weather tensors"):
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv("results/hourly_spatial_stats.csv", index=False)
    print(f"Saved {len(df)} rows to results/hourly_spatial_stats.csv")
    print(df.head())


if __name__ == "__main__":
    main()
