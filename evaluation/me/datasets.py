import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pandas as pd

import torch.nn as nn

class WeatherLazyDataset(Dataset):
    def __init__(self, data_dir, S=168, horizon=24,
                 original_h=450, original_w=449, in_channels=7):
        # Save S and horizon so we can use them for slicing later
        self.S = S
        self.horizon = horizon
        self.seq_length = S + horizon
        self.original_h = original_h
        self.original_w = original_w
        self.in_channels = in_channels

        search_pattern = os.path.join(data_dir, "**", "*.pt")
        self.file_paths = sorted(glob.glob(search_pattern, recursive=True))

        if len(self.file_paths) < self.seq_length:
            raise ValueError("Not enough files to create a single sequence.")

    def __len__(self):
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
    def __init__(self, csv_path='demand_raw.csv', S=168, future_steps=24):
        self.S = S
        self.future_steps = future_steps
        self.seq_length = S + future_steps
        
        # 1. Load the CSV
        df = pd.read_csv(csv_path)
        
        # 2. Extract RAW (un-normalized) energy data
        y_cols = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA_BOST']
        # Shape: (Total_Hours, n_zones)
        self.energy = torch.tensor(df[y_cols].values, dtype=torch.float32)
        
        # 3. Extract Unix Timestamps (Hours since epoch)
        # We assume your CSV has a 'datetime' column (e.g., '2023-01-01 00:00:00')
        # We convert it to integer hours to feed `future_time`
        if 'datetime' not in df.columns:
            raise ValueError("CSV must contain a 'datetime' column for the model to extract calendar features.")
            
        dt_index = pd.to_datetime(df['datetime'])
        # Convert nanoseconds to hours
        hours_since_epoch = dt_index.astype(np.int64) // 10**9 // 3600
        self.time_hours = torch.tensor(hours_since_epoch.values, dtype=torch.int64)

    def __len__(self):
        # Total valid sequences we can extract
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