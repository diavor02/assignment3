"""
Part 1 — Baseline CNN-Transformer for day-ahead energy demand forecasting.

Architecture
------------
1. CNN  : downsample each (450, 449, 7) weather snapshot to P spatial tokens.
2. Historical tabular token : linear(normalised_energy + calendar features).
3. Future tabular token     : linear(learned_demand_mask + calendar features).
4. Sequence assembly        : (S + 24) × (P + 1) tokens with spatial + temporal
                              positional embeddings.
5. Transformer Encoder      : self-attention over the full sequence.
6. Prediction head          : slice future states → mean-pool → MLP → (B, 24, Z).

Energy values are z-score normalised internally; the model always outputs raw MWh.
Weather normalisation is handled by BatchNorm inside the CNN.

Evaluation interface (required by evaluate.py)
----------------------------------------------
    get_model(metadata) -> nn.Module

    model.adapt_inputs(history_weather, history_energy, future_weather, future_time)
        -> tuple   (same preprocessing as the training DataLoader)

    model.forward(*adapt_inputs(...)) -> (B, 24, n_zones)  raw MWh predictions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Path to the saved checkpoint (used by get_model at evaluation time)
_CKPT_PATH = Path(__file__).parent / "best_model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SpatialCNN(nn.Module):
    """
    Downsamples a (C, H, W) weather snapshot to P = grid_size² spatial tokens
    of dimension d_spatial using strided convolutions + adaptive average pooling.

    BatchNorm after each conv handles per-channel input normalisation without
    requiring precomputed statistics.
    """

    def __init__(self, in_channels: int = 7, d_spatial: int = 128, grid_size: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.P = grid_size * grid_size

        self.net = nn.Sequential(
            # 450×449 → ~150×150
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # ~150×150 → ~50×50
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # ~50×50 → ~25×25
            nn.Conv2d(64, d_spatial, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_spatial),
            nn.GELU(),
            # → grid_size × grid_size (exactly)
            nn.AdaptiveAvgPool2d(grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, H, W)

        Returns
        -------
        (N, P, d_spatial)
        """
        out = self.net(x)                            # (N, d_spatial, G, G)
        out = out.permute(0, 2, 3, 1)               # (N, G, G, d_spatial)
        return out.reshape(x.shape[0], self.P, -1)   # (N, P, d_spatial)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class EnergyForecastModel(nn.Module):

    def __init__(
        self,
        n_zones:        int   = 8,
        n_weather_vars: int   = 7,
        history_len:    int   = 48,   # S  — hours of history used by model
        future_len:     int   = 24,   # F
        grid_size:      int   = 5,    # CNN output grid (grid_size × grid_size)
        d_spatial:      int   = 128,  # CNN output channels
        d_model:        int   = 256,  # Transformer / embedding dim
        n_heads:        int   = 8,
        n_layers:       int   = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.history_len = history_len
        self.future_len  = future_len
        self.n_zones     = n_zones
        self.P           = grid_size * grid_size
        self.d_model     = d_model

        # ── Spatial pathway ───────────────────────────────────────────────
        self.cnn          = SpatialCNN(n_weather_vars, d_spatial, grid_size)
        self.spatial_proj = nn.Linear(d_spatial, d_model)

        # ── Tabular pathway ───────────────────────────────────────────────
        # Calendar features: sin/cos of {hour-of-day, day-of-week, month} = 6 dims
        n_cal = 6

        # Historical: normalised energy (n_zones) + calendar (n_cal)
        self.hist_tab_proj = nn.Linear(n_zones + n_cal, d_model)

        # Future: learned demand mask (n_zones) + calendar (n_cal)
        self.demand_mask   = nn.Parameter(torch.zeros(n_zones))
        self.fut_tab_proj  = nn.Linear(n_zones + n_cal, d_model)

        # ── Positional embeddings ─────────────────────────────────────────
        # Spatial: one embedding per patch position (shared across time)
        self.spatial_pos_emb  = nn.Embedding(self.P, d_model)
        # Temporal: one embedding per timestep slot (S + F slots total)
        self.temporal_pos_emb = nn.Embedding(history_len + future_len, d_model)

        # ── Transformer Encoder ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 2,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable gradient flow
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Prediction head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_zones),
        )

        # ── Energy normalisation stats (populated from training data) ─────
        # Shape (1, 1, n_zones) so they broadcast with (B, T, n_zones) tensors.
        self.register_buffer("energy_mean", torch.zeros(1, 1, n_zones))
        self.register_buffer("energy_std",  torch.ones(1, 1, n_zones))

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_calendar(time_hours: torch.Tensor) -> torch.Tensor:
        """
        Convert int64 hours-since-Unix-epoch to 6 circular calendar features.

        Parameters
        ----------
        time_hours : (B, T) int64

        Returns
        -------
        (B, T, 6) float32
            Columns: [sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month]
        """
        B, T  = time_hours.shape
        h_np  = time_hours.cpu().numpy().astype("datetime64[h]")
        feats = np.zeros((B, T, 6), dtype=np.float32)

        for b in range(B):
            dti   = pd.DatetimeIndex(h_np[b])
            hour  = dti.hour.values
            dow   = dti.dayofweek.values
            month = dti.month.values - 1          # 0-indexed

            feats[b, :, 0] = np.sin(2 * np.pi * hour  / 24)
            feats[b, :, 1] = np.cos(2 * np.pi * hour  / 24)
            feats[b, :, 2] = np.sin(2 * np.pi * dow   / 7)
            feats[b, :, 3] = np.cos(2 * np.pi * dow   / 7)
            feats[b, :, 4] = np.sin(2 * np.pi * month / 12)
            feats[b, :, 5] = np.cos(2 * np.pi * month / 12)

        return torch.from_numpy(feats).to(time_hours.device)

    def _encode_weather(self, weather: torch.Tensor) -> torch.Tensor:
        """
        Apply CNN + projection to a batch of weather snapshots.

        Parameters
        ----------
        weather : (B, T, H, W, C)

        Returns
        -------
        (B, T, P, d_model)
        """
        B, T, H, W, C = weather.shape
        x      = weather.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        tokens = self.cnn(x)                                             # (B*T, P, d_spatial)
        tokens = self.spatial_proj(tokens)                               # (B*T, P, d_model)
        return tokens.reshape(B, T, self.P, self.d_model)

    def _normalise(self, energy: torch.Tensor) -> torch.Tensor:
        return (energy - self.energy_mean) / (self.energy_std + 1e-8)

    def _denormalise(self, energy: torch.Tensor) -> torch.Tensor:
        return energy * (self.energy_std + 1e-8) + self.energy_mean

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation harness interface
    # ─────────────────────────────────────────────────────────────────────

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,  # (B, 168, 450, 449, 7)
        history_energy:  torch.Tensor,  # (B, 168, n_zones)
        future_weather:  torch.Tensor,  # (B, 24,  450, 449, 7)
        future_time:     torch.Tensor,  # (B, 24)  int64 hours-since-epoch
    ) -> tuple:
        """
        Pre-process raw inputs into lightweight tensors for forward().
        This method is also called during training (batch sizes may differ).

        Steps
        -----
        1. Sub-select the last `history_len` hours from the 168-hour window.
        2. Reconstruct historical timestamps from future_time.
        3. Extract circular calendar features for both windows.
        4. Apply CNN to weather snapshots → spatial patch tokens.
        5. Z-score normalise historical energy.
        """
        S      = self.history_len
        device = next(self.parameters()).device

        # Move to model device (no-op if already there)
        history_weather = history_weather.to(device)
        history_energy  = history_energy.to(device)
        future_weather  = future_weather.to(device)
        future_time     = future_time.to(device)

        # (1) sub-select last S hours
        hist_w = history_weather[:, -S:]   # (B, S, 450, 449, 7)
        hist_e = history_energy[:,  -S:]   # (B, S, n_zones)

        # (2) reconstruct historical timestamps
        # future_time[:, 0] == t+1  →  t == future_time[:, 0] - 1
        t_cur     = future_time[:, 0:1] - 1                                  # (B, 1)
        hist_time = t_cur + torch.arange(-S + 1, 1, device=device)           # (B, S)

        # (3) calendar features
        hist_cal = self.extract_calendar(hist_time)      # (B, S, 6)
        fut_cal  = self.extract_calendar(future_time)    # (B, 24, 6)

        # (4) CNN spatial encoding
        hist_sp = self._encode_weather(hist_w)           # (B, S,  P, d_model)
        fut_sp  = self._encode_weather(future_weather)   # (B, 24, P, d_model)

        # (5) normalise energy
        hist_e_norm = self._normalise(hist_e)            # (B, S, n_zones)

        return hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        hist_sp:  torch.Tensor,   # (B, S,  P, d_model)
        hist_e:   torch.Tensor,   # (B, S,  n_zones) — normalised
        hist_cal: torch.Tensor,   # (B, S,  6)
        fut_sp:   torch.Tensor,   # (B, F,  P, d_model)
        fut_cal:  torch.Tensor,   # (B, F,  6)
    ) -> torch.Tensor:
        """
        Returns raw (de-normalised) predictions of shape (B, F, n_zones).
        """
        B = hist_sp.shape[0]
        S = self.history_len
        F = self.future_len
        P = self.P
        D = self.d_model
        device = hist_sp.device

        # ── Spatial positional embeddings (shared across time) ────────────
        sp_emb  = self.spatial_pos_emb(torch.arange(P, device=device))  # (P, D)
        hist_sp = hist_sp + sp_emb    # broadcast: (B, S, P, D)
        fut_sp  = fut_sp  + sp_emb    # broadcast: (B, F, P, D)

        # ── Historical tabular tokens ─────────────────────────────────────
        hist_tab = self.hist_tab_proj(
            torch.cat([hist_e, hist_cal], dim=-1)          # (B, S, n_zones+6)
        )                                                   # (B, S, D)

        # ── Future tabular tokens (learned demand mask) ───────────────────
        mask    = self.demand_mask.view(1, 1, -1).expand(B, F, -1)  # (B, F, n_zones)
        fut_tab = self.fut_tab_proj(
            torch.cat([mask, fut_cal], dim=-1)             # (B, F, n_zones+6)
        )                                                   # (B, F, D)

        # ── Assemble per-timestep groups: [sp_0, …, sp_{P-1}, tab] ───────
        hist_grp = torch.cat([hist_sp, hist_tab.unsqueeze(2)], dim=2)  # (B, S, P+1, D)
        fut_grp  = torch.cat([fut_sp,  fut_tab.unsqueeze(2)],  dim=2)  # (B, F, P+1, D)
        all_grp  = torch.cat([hist_grp, fut_grp], dim=1)               # (B, S+F, P+1, D)

        # ── Temporal positional encoding (broadcast over P+1 tokens/step) ─
        t_emb   = self.temporal_pos_emb(torch.arange(S + F, device=device))   # (S+F, D)
        all_grp = all_grp + t_emb.unsqueeze(0).unsqueeze(2)                   # (B, S+F, P+1, D)

        # ── Transformer Encoder ───────────────────────────────────────────
        seq = all_grp.reshape(B, (S + F) * (P + 1), D)   # (B, L, D)
        out = self.transformer(seq)                        # (B, L, D)

        # ── Slice future states, aggregate, predict ───────────────────────
        # Future tokens start at position S*(P+1) in the sequence.
        fut_out  = out[:, S * (P + 1):].reshape(B, F, P + 1, D)  # (B, F, P+1, D)
        fut_repr = fut_out.mean(dim=2)                             # (B, F, D)
        pred_norm = self.head(fut_repr)                            # (B, F, n_zones)

        # De-normalise to raw MWh
        return self._denormalise(pred_norm)                        # (B, F, n_zones)


# ─────────────────────────────────────────────────────────────────────────────
# Factory (required by evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_model(metadata: dict) -> EnergyForecastModel:
    """
    Build the model and — if a checkpoint exists — load its weights.

    Parameters
    ----------
    metadata : dict
        Must contain: n_zones, n_weather_vars, future_len.
    """
    model = EnergyForecastModel(
        n_zones        = metadata["n_zones"],
        n_weather_vars = metadata["n_weather_vars"],
        history_len    = 24,
        future_len     = metadata["future_len"],
        grid_size      = 5,
        d_spatial      = 128,
        d_model        = 256,
        n_heads        = 8,
        n_layers       = 4,
        dropout        = 0.1,
    )

    if _CKPT_PATH.exists():
        state = torch.load(_CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {_CKPT_PATH}")
    else:
        print(f"[WARNING] No checkpoint found at {_CKPT_PATH} — using random weights.")

    return model
