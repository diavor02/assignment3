"""
collect_predictions.py

Runs evaluation-style inference and saves ALL predictions and targets to disk,
plus per-window metadata including start timestamp and weather bucket.

Outputs:
  results/all_preds.npy      shape: (N_windows, 24, Z)
  results/all_targets.npy    shape: (N_windows, 24, Z)
  results/window_meta.csv    columns: window_idx, start_timestamp, weather_bucket
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data")
WEATHER_DIR = DATA_ROOT / "weather_data"
ENERGY_DIR = DATA_ROOT / "energy_demand_data"

HISTORY_LEN = 168
FUTURE_LEN = 24
EVAL_YEARS = {2022, 2023}

MODEL_DIR = PROJECT_ROOT / "evaluation" / "me"
MODEL_PATH = MODEL_DIR / "model.py"

# Load event catalog
catalog = pd.read_csv(PROJECT_ROOT / "part3" / "results" / "extreme_event_catalog.csv", parse_dates=["timestamp"])
bucket_map = dict(zip(catalog["timestamp"], catalog["weather_bucket"]))


def get_bucket_for_window(start_ts: pd.Timestamp) -> str:
    """Return most severe bucket seen in the 24-hour prediction window."""
    window_hours = pd.date_range(start_ts, periods=24, freq="h")
    buckets = [bucket_map.get(ts, "normal") for ts in window_hours]
    severity = ["winter_storm", "extreme_heat", "extreme_cold", "high_wind", "normal"]
    for s in severity:
        if s in buckets:
            return s
    return "normal"


def load_weather(hour_int64: int, cache: dict) -> torch.Tensor:
    if hour_int64 in cache:
        return cache[hour_int64]
    dt = pd.Timestamp(int(hour_int64), unit="h")
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    try:
        t = torch.load(path, map_location="cpu", weights_only=True).float()
    except Exception:
        t = torch.load(path, map_location="cpu", weights_only=False)
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        t = t.float()
    cache[hour_int64] = t
    if len(cache) > 200:
        oldest = next(iter(cache))
        del cache[oldest]
    return t


def load_model(zone_cols):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))

    metadata = {
        "zone_names": zone_cols,
        "n_zones": len(zone_cols),
        "history_len": HISTORY_LEN,
        "future_len": FUTURE_LEN,
        "n_weather_vars": 7,
    }

    spec = importlib.util.spec_from_file_location("a3_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_model(metadata)
    model.eval()
    return model


def main():
    # Energy data
    dfs = []
    for csv_path in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(csv_path, parse_dates=["timestamp_utc"]))
    energy_df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)

    deltas = energy_df["timestamp_utc"].diff().dropna()
    assert (deltas == pd.Timedelta("1h")).all(), "Energy data has gaps"

    zone_cols = [c for c in energy_df.columns if c != "timestamp_utc"]
    energy_values = energy_df[zone_cols].values.astype(np.float32)
    all_hours = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)

    # Window starts at midnight during selected years and has full history/future in bounds.
    midnight_mask = (
        energy_df["timestamp_utc"].dt.hour == 0
    ) & (
        energy_df["timestamp_utc"].dt.year.isin(EVAL_YEARS)
    )
    test_dates = np.where(midnight_mask)[0]
    test_dates = test_dates[
        (test_dates >= HISTORY_LEN) &
        (test_dates + FUTURE_LEN <= len(energy_df))
    ]

    print(f"Selected windows: {len(test_dates)}")
    print(f"Years: {sorted(EVAL_YEARS)}")

    model = load_model(zone_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    all_preds = []
    all_targets = []
    all_meta = []

    weather_cache = {}

    with torch.no_grad():
        for window_idx, t_idx in enumerate(tqdm(test_dates, desc="Collecting predictions")):
            hist_slice = slice(t_idx - HISTORY_LEN, t_idx)
            future_slice = slice(t_idx, t_idx + FUTURE_LEN)

            hist_hours_arr = all_hours[hist_slice]
            future_hours_arr = all_hours[future_slice]

            hist_energy = energy_values[hist_slice]
            target = torch.from_numpy(energy_values[future_slice]).float()  # (24, Z)

            try:
                hist_weather = torch.stack([load_weather(int(h), weather_cache) for h in hist_hours_arr])
                fut_weather = torch.stack([load_weather(int(h), weather_cache) for h in future_hours_arr])
            except FileNotFoundError:
                continue

            fut_time = torch.tensor(future_hours_arr, dtype=torch.int64)

            raw_inputs = (
                hist_weather.unsqueeze(0).to(device),
                torch.from_numpy(hist_energy).unsqueeze(0).to(device),
                fut_weather.unsqueeze(0).to(device),
                fut_time.unsqueeze(0).to(device),
            )
            pred = model(*model.adapt_inputs(*raw_inputs)).squeeze(0).cpu().float()  # (24, Z)

            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())

            start_ts = energy_df["timestamp_utc"].iloc[t_idx]
            bucket = get_bucket_for_window(start_ts)
            all_meta.append({
                "window_idx": len(all_meta),
                "start_timestamp": start_ts,
                "weather_bucket": bucket,
            })

    if not all_preds:
        raise RuntimeError("No valid windows were collected.")

    preds_arr = np.stack(all_preds, axis=0)
    targets_arr = np.stack(all_targets, axis=0)
    meta_df = pd.DataFrame(all_meta)

    out_dir = PROJECT_ROOT / "part3" / "results"
    np.save(out_dir / "all_preds.npy", preds_arr)
    np.save(out_dir / "all_targets.npy", targets_arr)
    meta_df.to_csv(out_dir / "window_meta.csv", index=False)

    print(f"Saved windows: {len(preds_arr)}")
    print(f"Preds shape: {preds_arr.shape}")
    print("Bucket distribution:")
    print(meta_df["weather_bucket"].value_counts())


if __name__ == "__main__":
    main()
