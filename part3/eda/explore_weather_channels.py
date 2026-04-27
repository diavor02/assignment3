"""
explore_weather_channels.py

Loads a sample of weather tensors and reports:
1. Number of channels (c in shape 450x449xc)
2. Per-channel: mean, std, min, max, 1st/99th percentile
3. Spatial correlation structure (are channels spatially smooth or noisy?)
4. Temporal variance (which channels change the most hour-to-hour?)

Output: channel_stats.csv + printed summary
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data")
WEATHER_DIR = DATA_ROOT / "weather_data"


# --- Adjust this to however weather tensors are stored ---
# Option A: one .npy file per hour named by timestamp
# Option B: one large .npy array of shape (T, 450, 449, c)
# Option C: .npz files
# Option D: one .pt file per hour under year subdirectories
# We'll try to auto-detect:

def _load_tensor(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        arr = data[list(data.keys())[0]]
        return np.asarray(arr)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".pt":
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Assignment weather files are trusted course data and may contain pickled numpy arrays.
            tensor = torch.load(path, map_location="cpu", weights_only=False)
        arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        return arr
    raise ValueError(f"Unsupported file type: {path}")


def load_sample_tensors(n=200):
    """Load n evenly-spaced weather tensors. Returns list of (450,449,c) arrays."""
    files = sorted(WEATHER_DIR.rglob("X_*.pt"))
    if not files:
        files = sorted(WEATHER_DIR.rglob("*.npy"))
    if not files:
        files = sorted(WEATHER_DIR.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npy/.npz/.pt files found in {WEATHER_DIR}")

    n = min(n, len(files))
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    tensors = []
    selected_files = [files[i] for i in indices]

    for f in selected_files:
        arr = _load_tensor(f)
        if arr.ndim == 3 and arr.shape[0] == 7 and arr.shape[1] == 450 and arr.shape[2] == 449:
            # Convert CHW -> HWC if needed.
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"Unexpected tensor shape {arr.shape} in {f}")
        tensors.append(arr)

    return tensors, selected_files


def analyze_channels(tensors):
    stack = np.stack(tensors, axis=0)  # (N, 450, 449, c)
    c = stack.shape[-1]
    print(f"\nWeather tensor shape: 450 x 449 x {c} channels")
    print(f"Sample size: {len(tensors)} tensors\n")

    rows = []
    for ch in range(c):
        ch_data = stack[..., ch].ravel()
        row = {
            "channel": ch,
            "mean": ch_data.mean(),
            "std": ch_data.std(),
            "min": ch_data.min(),
            "max": ch_data.max(),
            "p01": np.percentile(ch_data, 1),
            "p99": np.percentile(ch_data, 99),
            # variance of spatial mean across time
            "temporal_var": stack[..., ch].mean(axis=(1, 2)).var(),
        }
        rows.append(row)
        print(
            f"  CH{ch:02d}: mean={row['mean']:8.3f}  std={row['std']:8.3f}  "
            f"range=[{row['min']:.2f}, {row['max']:.2f}]  "
            f"p1%={row['p01']:.2f}  p99%={row['p99']:.2f}  temporal_var={row['temporal_var']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/channel_stats.csv", index=False)
    print("\nSaved results/channel_stats.csv")
    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    tensors, files = load_sample_tensors(n=200)
    df = analyze_channels(tensors)
    print("\n--- INTERPRETATION GUIDE ---")
    print("Channel with highest temporal_var = most dynamic = likely temperature or wind speed")
    print("Channel with values ~273-310 range = likely temperature in Kelvin")
    print("Channel with values 0-30 range = likely wind speed (m/s)")
    print("Channel with values 900-1030 range = likely pressure (hPa)")
    print("Channel with values 0-1 range = likely humidity fraction or precipitation flag")
