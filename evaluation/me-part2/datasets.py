import os
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import pandas as pd

import torch.nn as nn

ENERGY_DATA_DIR = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data")
WEATHER_DATA_DIR = Path("/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data")

class WeatherLazyDataset(Dataset):
    def __init__(self, data_dir=WEATHER_DATA_DIR, S=168, horizon=24,
                 original_h=450, original_w=449, in_channels=7):
        # Save S and horizon so we can use them for slicing later
        self.S = S
        self.horizon = horizon
        self.seq_length = S + horizon
        self.original_h = original_h
        self.original_w = original_w
        self.in_channels = in_channels

        data_root = Path(data_dir)
        data_dirs = [
            data_root / "2020",
            data_root / "2021",
            data_root / "2022",
        ]
        self.file_paths = sorted([
            f for d in data_dirs
            for f in glob.glob(os.path.join(str(d), "**", "*.pt"), recursive=True)
        ])


        if len(self.file_paths) < self.seq_length:
            raise ValueError("Not enough files to create a single sequence.")

    def __len__(self):
        # print("self.file_paths:", len(self.file_paths))
        # print("self.seq_length weather:", self.seq_length)
        return len(self.file_paths) - self.seq_length + 1

    def __getitem__(self, idx):
        window_paths = self.file_paths[idx : idx + self.seq_length]

        tensors = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in window_paths
        ]
        
        # Shape: (S + horizon, H, W, C)
        sequence_tensor = torch.stack(tensors, dim=0)

        assert sequence_tensor.shape == (
            self.seq_length,
            self.original_h,
            self.original_w,
            self.in_channels
        ), f"Expected {(self.seq_length, self.original_h, self.original_w, self.in_channels)}, got {sequence_tensor.shape}"

        # ── Split the sequence into history and future ──
        history_weather = sequence_tensor[:self.S]     # Shape: (S, H, W, C)
        future_weather  = sequence_tensor[self.S:]     # Shape: (horizon, H, W, C)

        return history_weather, future_weather


class DemandTimeDataset(Dataset):
    """
    Loads raw tabular data once into memory and slices sliding windows on-the-fly.
    Perfectly tailored for `EnergyForecastModel.adapt_inputs`.
    """
    def __init__(self, csv_path=None, S=168, future_steps=24):
        self.S = S
        self.future_steps = future_steps
        self.seq_length = S + future_steps
        
        # 1. Load the cluster CSVs if no explicit file is provided.
        if csv_path is None:
            dfs = []
            for year in (2020, 2021, 2022):
                year_path = ENERGY_DATA_DIR / f"target_energy_zonal_{year}.csv"
                dfs.append(pd.read_csv(year_path, parse_dates=["timestamp_utc"]))
            df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
        else:
            df = pd.read_csv(csv_path)
        
        # 2. Extract RAW (un-normalized) energy data
        y_cols = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA_BOST']
        # Shape: (Total_Hours, n_zones)
        self.energy = torch.tensor(df[y_cols].values, dtype=torch.float32)
        
        # 3. Extract Unix Timestamps (Hours since epoch)
        # We assume your CSV has a 'datetime' column (e.g., '2023-01-01 00:00:00')
        # We convert it to integer hours to feed `future_time`
        if 'timestamp_utc' not in df.columns:
            raise ValueError("CSV must contain a 'timestamp_utc' column for the model to extract calendar features.")
            
        dt_index = pd.to_datetime(df['timestamp_utc'])
        # Convert nanoseconds to hours
        hours_since_epoch = dt_index.astype(np.int64) // 10**9 // 3600
        self.time_hours = torch.tensor(hours_since_epoch.values, dtype=torch.int64)

    def __len__(self):
        # Total valid sequences we can extract
        # print("self.energy:", len(self.energy))
        # print("self.seq_length:", self.seq_length)
        return len(self.energy) - self.seq_length + 1

    def __getitem__(self, idx):
        """
        Returns exactly what the training loop needs:
        1. history_energy (Raw MWh)
        2. future_time    (Int64 Hours)
        3. targets        (Raw MWh to calculate Loss)
        """
        # Historical Data (t-S : t)
        hist_energy = self.energy[idx : idx + self.S]
        
        # Future Time (t : t+24) -> The model needs this to compute calendar/history time
        future_time = self.time_hours[idx + self.S : idx + self.seq_length]
        
        # Future Target Energy (t : t+24) -> For your Loss Function
        targets = self.energy[idx + self.S : idx + self.seq_length]
        
        return hist_energy, future_time, targets

class JointEnergyWeatherDataset(Dataset):
    """Wraps both datasets to ensure perfect temporal synchronization."""
    def __init__(self, weather_ds: WeatherLazyDataset, tabular_ds: DemandTimeDataset):
        self.weather_ds = weather_ds
        self.tabular_ds = tabular_ds
        
        # CRITICAL: Both datasets must output the exact same number of sequences.
        if len(self.weather_ds) != len(self.tabular_ds):
            raise ValueError(
                f"Dataset length mismatch! "
                f"Weather has {len(self.weather_ds)} sequences, "
                f"Tabular has {len(self.tabular_ds)}."
            )

    def __len__(self):
        return len(self.weather_ds)

    def __getitem__(self, idx):
        # Fetch the identical timeframe from both sources
        hist_w, fut_w = self.weather_ds[idx]
        hist_e, fut_t, targets = self.tabular_ds[idx]
        
        return hist_w, hist_e, fut_w, fut_t, targets

def get_dataloader(
    batch_size: int = 2,
    is_train: bool = True,
    val_split: float = 0.2,
) -> DataLoader:
    # 1. Instantiate your raw datasets
    weather_ds = WeatherLazyDataset()
    tabular_ds = DemandTimeDataset()

    # 2. Bind them together
    joint_ds = JointEnergyWeatherDataset(weather_ds, tabular_ds)

    # 3. CHRONOLOGICAL SPLIT (No Data Leakage!)
    val_size   = int(len(joint_ds) * val_split)
    train_size = len(joint_ds) - val_size
    
    if is_train:
        # Train gets the first 80% of the timeline (e.g., 2019 - 2022)
        subset = Subset(joint_ds, range(0, train_size))
    else:
        # Val gets the last 20% of the timeline (e.g., 2023)
        subset = Subset(joint_ds, range(train_size, len(joint_ds)))

    # 4. Let PyTorch handle batching and multi-processing
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        # It is perfectly fine to shuffle the *training batches* # once the timeline has been safely split!
        shuffle=is_train, 
        num_workers=4,
        pin_memory=True,
        drop_last=is_train
    )

    return dataloader