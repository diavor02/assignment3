from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Converts a single (C, H, W) weather grid snapshot into P = grid_size²
    spatial patch tokens, each of dimension d_spatial.

    Architecture: three strided Conv2d layers progressively reduce spatial
    resolution (450×449 → ~150×150 → ~50×50 → ~25×25), followed by
    AdaptiveAvgPool2d that forces the output to exactly grid_size × grid_size
    regardless of rounding differences in intermediate sizes.

    BatchNorm2d after each convolution normalises activations per-channel,
    removing the need to precompute per-variable mean/std statistics over
    the full dataset before training.

    Output tokens are returned in sequence format (N, P, d_spatial) so they
    can be directly fed into a linear projection layer or a Transformer.
    """

    def __init__(self, in_channels: int = 7, d_spatial: int = 128, grid_size: int = 5):
        super().__init__()
        # grid_size² is the total number of spatial patch tokens per timestep.
        self.grid_size = grid_size
        self.P = grid_size * grid_size

        def depthwise_separable(in_c, out_c, k, s, p):
                    return nn.Sequential(
                        # Depthwise (spatial)
                        nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c),
                        nn.BatchNorm2d(in_c),
                        nn.GELU(),
                        # Pointwise (channel mixing)
                        nn.Conv2d(in_c, out_c, kernel_size=1),
                        nn.BatchNorm2d(out_c),
                        nn.GELU()
                    )

        self.net = nn.Sequential(
            # Stage 1: stride-3 ~150x150
            depthwise_separable(in_channels, 32, k=7, s=3, p=3),
            # Stage 2: stride-3 ~50x50
            depthwise_separable(32, 64, k=5, s=3, p=2),
            # Stage 3: stride-2 ~25x25
            depthwise_separable(64, d_spatial, k=3, s=2, p=1),
            
            nn.AdaptiveAvgPool2d(grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, H, W)
            Batch of N weather snapshots with C channels (weather variables),
            H rows and W columns of grid cells.

        Returns
        -------
        (N, P, d_spatial)
            Sequence of P = grid_size² spatial patch tokens per snapshot.
            Each token is a d_spatial-dimensional embedding summarising one
            spatial region of the weather grid.
        """
        out = self.net(x)                            # (N, d_spatial, G, G)
        # Move the channel dimension to the last axis so each (G, G) grid cell
        # is represented as a d_spatial-dimensional vector.
        out = out.permute(0, 2, 3, 1)               # (N, G, G, d_spatial)
        # Flatten the 2-D grid into a 1-D sequence of P patch tokens.
        return out.reshape(x.shape[0], self.P, -1)   # (N, P, d_spatial)




class EnergyForecastModel(nn.Module):

    def __init__(
        self,
        n_zones:        int   = 8,
        n_weather_vars: int   = 7,
        S:              int   = 168,   # S — number of past hourly timesteps seen by the model
        horizon:     int   = 24,    # F — number of future hourly timesteps to predict
        grid_size:      int   = 5,     # CNN output is grid_size × grid_size spatial patches
        d_spatial:      int   = 128,   # channel depth output by the CNN before projection
        d_model:        int   = 256,   # unified embedding dimension used throughout the Transformer
        n_heads:        int   = 8,
        n_layers:       int   = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.S = S
        self.horizon  = horizon
        self.n_zones     = n_zones
        self.P           = grid_size * grid_size  # total spatial patch tokens per timestep
        self.d_model     = d_model

        # ── Spatial pathway ───────────────────────────────────────────────
        # CNN reduces each (H, W, C) weather grid to P patch tokens of size d_spatial.
        # The linear projection then maps d_spatial → d_model so spatial tokens
        # live in the same embedding space as tabular tokens.
        self.cnn          = CNN(n_weather_vars, d_spatial, grid_size)
        self.spatial_proj = nn.Linear(d_spatial, d_model)

        # ── Tabular pathway ───────────────────────────────────────────────
        # Calendar features encode time cyclically as sin/cos pairs for:
        #   hour-of-day (period 24), day-of-week (period 7), month (period 12)
        # → 3 pairs × 2 = 6 dimensions total.
        n_cal = 6

        # Historical tabular token: concatenation of [normalised energy (n_zones), calendar (n_cal)]
        # projected to d_model so it can be mixed with spatial tokens in the Transformer.
        self.hist_tab_proj = nn.Linear(n_zones + n_cal, d_model)

        # Future tabular token: the model cannot see future energy, so a learned
        # per-zone "demand mask" vector acts as a trainable placeholder.
        # Concatenated with future calendar features and projected to d_model.
        self.demand_mask   = nn.Parameter(torch.zeros(n_zones))
        self.fut_tab_proj  = nn.Linear(n_zones + n_cal, d_model)

        # ── Positional embeddings ─────────────────────────────────────────
        # Spatial positional embedding: distinguishes the P patch positions within
        # each timestep. Shared across all timesteps (position 0 always means the
        # same grid region regardless of when the snapshot was taken).
        self.spatial_pos_emb  = nn.Embedding(self.P, d_model)

        # Temporal positional embedding: distinguishes timestep slots 0…(S+F-1).
        # Slots 0…S-1 are historical; slots S…S+F-1 are future. All P+1 tokens
        # at the same timestep receive the same temporal embedding.
        self.temporal_pos_emb = nn.Embedding(S + horizon, d_model)

        # ── Transformer Encoder ───────────────────────────────────────────
        # Pre-LayerNorm (norm_first=True) applies LayerNorm before each sub-layer
        # rather than after, which stabilises gradients in deep stacks and is
        # generally preferred for training from scratch.
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 2,  # FFN hidden size = 2× embedding dim
            dropout         = dropout,
            batch_first     = True,         # expects (B, seq_len, d_model) — not (seq_len, B, d_model)
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Prediction head ───────────────────────────────────────────────
        # Takes the mean-pooled future representation (B, F, d_model) and
        # produces per-zone normalised predictions (B, F, n_zones).
        # A second Linear + GELU allows the head to learn non-linear combinations
        # of the transformer output before the final regression layer.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_zones),  # output: one scalar per zone per future hour
        )

        # ── Energy normalisation statistics ───────────────────────────────
        # Populated with per-zone mean and std computed over the training set.
        # Stored as buffers (not parameters) so they are saved/loaded with the
        # checkpoint and moved to the correct device, but are never updated by
        # the optimiser.
        # Shape (1, 1, n_zones) broadcasts with (B, T, n_zones) energy tensors.
        self.register_buffer("energy_mean", torch.zeros(1, 1, n_zones))
        self.register_buffer("energy_std",  torch.ones(1, 1, n_zones))

    @staticmethod
    def extract_calendar(time_hours: torch.Tensor) -> torch.Tensor:
        """
        Convert integer hour timestamps into 6 continuous circular features
        that encode periodicity without discontinuities at period boundaries
        (e.g. hour 23 → hour 0, Sunday → Monday, December → January).

        Each time unit is encoded as (sin, cos) of its normalised angle so
        that the Euclidean distance between two feature vectors reflects
        the circular distance between the corresponding times.

        Parameters
        ----------
        time_hours : (B, T) int64
            Hours since the Unix epoch (1970-01-01 00:00 UTC).

        Returns
        -------
        (B, T, 6) float32
            Column order:
              0  sin_hour   — position within the 24-hour day
              1  cos_hour
              2  sin_dow    — position within the 7-day week
              3  cos_dow
              4  sin_month  — position within the 12-month year
              5  cos_month
        """
        B, T  = time_hours.shape
        # Convert int64 epoch-hours to numpy datetime64 for calendar decomposition.
        h_np  = time_hours.cpu().numpy().astype("datetime64[h]")
        feats = np.zeros((B, T, 6), dtype=np.float32)

        for b in range(B):
            # pandas DatetimeIndex provides .hour, .dayofweek, .month attributes.
            dti   = pd.DatetimeIndex(h_np[b])
            hour  = dti.hour.values           # 0–23
            dow   = dti.dayofweek.values      # 0 (Monday) – 6 (Sunday)
            month = dti.month.values - 1      # shift 1–12 to 0–11 for clean angle normalisation

            feats[b, :, 0] = np.sin(2 * np.pi * hour  / 24)
            feats[b, :, 1] = np.cos(2 * np.pi * hour  / 24)
            feats[b, :, 2] = np.sin(2 * np.pi * dow   / 7)
            feats[b, :, 3] = np.cos(2 * np.pi * dow   / 7)
            feats[b, :, 4] = np.sin(2 * np.pi * month / 12)
            feats[b, :, 5] = np.cos(2 * np.pi * month / 12)

        return torch.from_numpy(feats).to(time_hours.device)

    def _encode_weather(self, weather: torch.Tensor) -> torch.Tensor:
        """
        Apply the CNN and linear projection to a time series of weather grids,
        producing a sequence of spatial patch token embeddings in d_model space.

        The time axis is merged into the batch axis before passing through the
        CNN (which only accepts 4-D input), then restored afterwards.

        Parameters
        ----------
        weather : (B, T, H, W, C)
            B samples, T timesteps, H×W spatial grid, C weather variables.

        Returns
        -------
        (B, T, P, d_model)
            P = grid_size² patch tokens per timestep, each projected to d_model.
        """
        B, T, H, W, C = weather.shape
        # Merge batch and time so each snapshot is an independent CNN input.
        # Permute channels to front: (B*T, H, W, C) → (B*T, C, H, W).
        x      = weather.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        tokens = self.cnn(x)                                             # (B*T, P, d_spatial)
        tokens = self.spatial_proj(tokens)                               # (B*T, P, d_model)
        # Restore the time axis.
        return tokens.reshape(B, T, self.P, self.d_model)               # (B, T, P, d_model)

    def _normalise(self, energy: torch.Tensor) -> torch.Tensor:
        # Z-score normalisation using per-zone training statistics.
        # The 1e-8 epsilon prevents division by zero for zones with near-zero variance.
        return (energy - self.energy_mean) / (self.energy_std + 1e-8)

    def _denormalise(self, energy: torch.Tensor) -> torch.Tensor:
        # Inverse of _normalise: converts model outputs back to raw MWh values.
        return energy * (self.energy_std + 1e-8) + self.energy_mean


    def adapt_inputs(
        self,
        history_weather: torch.Tensor,  # (B, 168, 450, 449, 7) — full 168-hour weather history
        history_energy:  torch.Tensor,  # (B, 168, n_zones)      — corresponding energy readings
        future_weather:  torch.Tensor,  # (B, 24,  450, 449, 7) — NWP forecast for next 24 h
        future_time:     torch.Tensor,  # (B, 24)  int64 hours-since-epoch for each forecast hour
    ) -> tuple:
        """
        Pre-process raw evaluation inputs into the tensors expected by forward().
        Called both at inference time (by the evaluation harness) and during
        training (with potentially different batch sizes).

        Steps
        -----
        1. Trim the 168-hour history window to the last `S` hours,
           allowing the model to use a shorter context if configured that way.
        2. Reconstruct historical timestamps by stepping backwards from the first
           future timestamp — avoids passing timestamps for the history window as
           a separate input.
        3. Extract 6-dimensional circular calendar features for both windows.
        4. Run the CNN on all weather grids to produce spatial patch tokens.
        5. Z-score normalise historical energy using training-set statistics.
        """
        S      = self.S
        device = next(self.parameters()).device

        # Ensure all inputs are on the same device as the model weights.
        history_weather = history_weather.to(device)
        history_energy  = history_energy.to(device)
        future_weather  = future_weather.to(device)
        future_time     = future_time.to(device)

        # (1) Keep only the most recent S hours of the 168-hour history.
        hist_w = history_weather[:, -S:]   # (B, S, 450, 449, 7)
        hist_e = history_energy[:,  -S:]   # (B, S, n_zones)

        # (2) Reconstruct historical timestamps.
        # future_time[:, 0] is t+1 (the first forecast hour), so t = future_time[:, 0] - 1
        # is the last observed hour. Stepping back S-1 steps gives the full history range:
        # [t - S + 1, t - S + 2, …, t].
        t_cur     = future_time[:, 0:1] - 1                                  # (B, 1) — last observed hour
        hist_time = t_cur + torch.arange(-S + 1, 1, device=device)           # (B, S)

        # (3) Calendar features for history and forecast windows.
        hist_cal = self.extract_calendar(hist_time)      # (B, S, 6)
        fut_cal  = self.extract_calendar(future_time)    # (B, 24, 6)

        # (4) CNN spatial encoding: each weather grid → P patch token embeddings.
        hist_sp = self._encode_weather(hist_w)           # (B, S,  P, d_model)
        fut_sp  = self._encode_weather(future_weather)   # (B, 24, P, d_model)

        # (5) Normalise historical energy to zero mean / unit variance per zone.
        hist_e_norm = self._normalise(hist_e)            # (B, S, n_zones)

        return hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal

    def forward(
        self,
        hist_sp:  torch.Tensor,   # (B, S,  P, d_model) — historical spatial patch tokens
        hist_e:   torch.Tensor,   # (B, S,  n_zones)    — normalised historical energy
        hist_cal: torch.Tensor,   # (B, S,  6)          — historical calendar features
        fut_sp:   torch.Tensor,   # (B, F,  P, d_model) — future spatial patch tokens
        fut_cal:  torch.Tensor,   # (B, F,  6)          — future calendar features
    ) -> torch.Tensor:
        """
        Full forward pass: assemble a token sequence from spatial and tabular
        inputs, encode with the Transformer, then decode future tokens into
        zone-level energy predictions.

        Returns
        -------
        (B, F, n_zones)
            De-normalised energy forecasts in the original MWh scale,
            one value per zone per future hour.
        """
        B = hist_sp.shape[0]
        S = self.S
        F = self.horizon
        P = self.P
        D = self.d_model
        device = hist_sp.device

        # ── Spatial positional embeddings ─────────────────────────────────
        # One learned embedding per patch index (0…P-1), shared across timesteps.
        # Adding it to every (B, T, P, D) tensor lets the Transformer distinguish
        # "north-west corner patch" from "south-east corner patch" at any time.
        sp_emb  = self.spatial_pos_emb(torch.arange(P, device=device))  # (P, D)
        hist_sp = hist_sp + sp_emb    # broadcast over B and S: (B, S, P, D)
        fut_sp  = fut_sp  + sp_emb    # broadcast over B and F: (B, F, P, D)

        # ── Historical tabular tokens ─────────────────────────────────────
        # Combine normalised zone energy and calendar features into a single
        # d_model token per timestep that summarises the observed system state.
        hist_tab = self.hist_tab_proj(
            torch.cat([hist_e, hist_cal], dim=-1)          # (B, S, n_zones+6)
        )                                                   # (B, S, D)

        # ── Future tabular tokens ─────────────────────────────────────────
        # Since future energy is unknown, the learned demand_mask (one scalar
        # per zone) acts as a trainable "query" token that represents what the
        # model wants to know about each zone in the future.
        mask    = self.demand_mask.view(1, 1, -1).expand(B, F, -1)  # (B, F, n_zones)
        fut_tab = self.fut_tab_proj(
            torch.cat([mask, fut_cal], dim=-1)             # (B, F, n_zones+6)
        )                                                   # (B, F, D)

        # ── Assemble per-timestep token groups ────────────────────────────
        # Each timestep is represented by P+1 tokens:
        #   positions 0…P-1  → spatial patch tokens (weather)
        #   position  P      → tabular token (energy + calendar)
        # This groups related information together before feeding to the Transformer.
        hist_grp = torch.cat([hist_sp, hist_tab.unsqueeze(2)], dim=2)  # (B, S, P+1, D)
        fut_grp  = torch.cat([fut_sp,  fut_tab.unsqueeze(2)],  dim=2)  # (B, F, P+1, D)
        # Concatenate history and future along the time axis.
        all_grp  = torch.cat([hist_grp, fut_grp], dim=1)               # (B, S+F, P+1, D)

        # ── Temporal positional embeddings ────────────────────────────────
        # One embedding per timestep slot (0…S+F-1), broadcast identically to
        # all P+1 tokens within the same slot so attention can distinguish
        # "hour 5 of history" from "hour 2 of forecast".
        t_emb   = self.temporal_pos_emb(torch.arange(S + F, device=device))   # (S+F, D)
        all_grp = all_grp + t_emb.unsqueeze(0).unsqueeze(2)                   # (B, S+F, P+1, D)

        # ── Transformer Encoder ───────────────────────────────────────────
        # Flatten the (S+F) × (P+1) token grid into a single sequence of length
        # L = (S+F)*(P+1) before passing to the encoder. Full self-attention
        # allows every spatial and tabular token to attend to every other token
        # across both time and space.
        seq = all_grp.reshape(B, (S + F) * (P + 1), D)   # (B, L, D)
        out = self.transformer(seq)                        # (B, L, D)

        # ── Decode future predictions ─────────────────────────────────────
        # Future tokens occupy the last F*(P+1) positions in the sequence.
        # Restore the (F, P+1) structure, then average across the P+1 tokens
        # at each future timestep to get a single d_model representation per hour.
        fut_out  = out[:, S * (P + 1):].reshape(B, F, P + 1, D)  # (B, F, P+1, D)
        fut_repr = fut_out.mean(dim=2)                             # (B, F, D) — mean-pool over tokens
        pred_norm = self.head(fut_repr)                            # (B, F, n_zones) — normalised predictions

        # Convert normalised outputs back to the original MWh scale.
        return self._denormalise(pred_norm)                        # (B, F, n_zones)
