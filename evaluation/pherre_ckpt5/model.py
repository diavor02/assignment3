from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from base_model import get_model as build_base_model


MODEL_DIR = Path(__file__).parent
CHECKPOINT_PATH = MODEL_DIR / 'best_weights.pt'


def build_calendar_features(timestamps: pd.DatetimeIndex, calendar_dim: int = 8) -> torch.Tensor:
    if calendar_dim != 8:
        raise ValueError('build_calendar_features currently supports calendar_dim=8')

    hour = timestamps.hour.astype(np.float32)
    dow = timestamps.dayofweek.astype(np.float32)
    month = timestamps.month.astype(np.float32)

    hour_angle = 2 * np.pi * hour / 24.0
    dow_angle = 2 * np.pi * dow / 7.0
    month_angle = 2 * np.pi * (month - 1) / 12.0

    features = np.stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            np.sin(month_angle),
            np.cos(month_angle),
            (dow >= 5).astype(np.float32),
            np.zeros_like(hour, dtype=np.float32),
        ],
        axis=-1,
    )
    return torch.from_numpy(features.astype(np.float32))


class EvaluationWrapper(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        ckpt_meta = checkpoint.get('metadata', {})
        ckpt_args = ckpt_meta.get('args', {})
        scaler = ckpt_meta.get('demand_scaler', {})

        self.hist_len = int(ckpt_args.get('hist_len', metadata['history_len']))
        self.future_len = int(ckpt_args.get('future_len', metadata['future_len']))
        self.calendar_dim = int(ckpt_args.get('calendar_dim', 8))

        self.inner = build_base_model(
            weather_channels=int(ckpt_args.get('weather_channels', metadata['n_weather_vars'])),
            calendar_dim=self.calendar_dim,
            num_zones=metadata['n_zones'],
            hist_len=self.hist_len,
            future_len=self.future_len,
            d_model=int(ckpt_args.get('d_model', 256)),
            target_grid_size=int(ckpt_args.get('target_grid_size', 8)),
            nhead=int(ckpt_args.get('nhead', 8)),
            num_transformer_layers=int(ckpt_args.get('num_transformer_layers', 4)),
            dim_feedforward=int(ckpt_args.get('dim_feedforward', 512)),
            dropout=float(ckpt_args.get('dropout', 0.1)),
            use_learned_future_demand_mask=bool(ckpt_args.get('use_learned_future_demand_mask', False)),
        )
        self.inner.load_state_dict(checkpoint['model_state'])

        self.register_buffer('demand_mean', torch.tensor(scaler['mean'], dtype=torch.float32))
        std = torch.tensor(scaler['std'], dtype=torch.float32)
        std[std == 0] = 1.0
        self.register_buffer('demand_std', std)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_weather = history_weather[:, -self.hist_len :, ...]
        hist_energy = history_energy[:, -self.hist_len :, :]

        hist_time = future_time[:, :1] - torch.arange(
            self.hist_len, 0, -1, device=future_time.device, dtype=future_time.dtype
        ).view(1, -1)

        hist_timestamps = pd.DatetimeIndex(hist_time[0].cpu().numpy().astype('datetime64[h]'))
        fut_timestamps = pd.DatetimeIndex(future_time[0].cpu().numpy().astype('datetime64[h]'))
        hist_calendar = build_calendar_features(hist_timestamps, calendar_dim=self.calendar_dim).unsqueeze(0)
        fut_calendar = build_calendar_features(fut_timestamps, calendar_dim=self.calendar_dim).unsqueeze(0)

        hist_energy = (hist_energy.float() - self.demand_mean.view(1, 1, -1)) / self.demand_std.view(1, 1, -1)

        device = next(self.parameters()).device
        return (
            hist_weather.to(device),
            hist_energy.to(device),
            hist_calendar.to(device),
            future_weather.to(device),
            fut_calendar.to(device),
        )

    def forward(
        self,
        hist_weather: torch.Tensor,
        hist_energy: torch.Tensor,
        hist_calendar: torch.Tensor,
        fut_weather: torch.Tensor,
        fut_calendar: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.inner(
            hist_weather=hist_weather,
            hist_demand=hist_energy,
            hist_calendar=hist_calendar,
            fut_weather=fut_weather,
            fut_calendar=fut_calendar,
        )
        return pred * self.demand_std.view(1, 1, -1) + self.demand_mean.view(1, 1, -1)


def get_model(metadata: dict) -> EvaluationWrapper:
    return EvaluationWrapper(metadata)
