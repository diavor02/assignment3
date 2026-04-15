"""
model.py 
CS-137 Assignment 3
Part 1: Baseline CNN-Transformer Patch Architecture

NOTE: Weather tensors are pre-pooled to (grid_h, grid_w, 7) by the dataset/dataloader
before being passed to the model. The CNN here operates on the small grid,
not on the full 450×449 maps.

Evaluation harness interface:
    get_model(metadata: dict) -> CNNTransformerForecaster
    model.adapt_inputs(history_weather, history_energy, future_weather, future_time) -> tuple
    model.forward(*adapt_inputs(...)) -> (B, 24, n_zones)
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------
# spatial tokens
#-----------------
class WeatherCNNEncoder(nn.Module):

    """
    takes the already-pooled weather grids for every hour and runs it through 
    2 conv layers, then flattens to P tokens

    x: (B, T, grid_h, grid_w, C)  →  (B, T, P, D)   P = grid_h * grid_w
    """

    def __init__(self, in_channels: int, embed_dim: int, grid_h: int = 5, grid_w: int = 5):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32,       kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,        embed_dim,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, gh, gw, C = x.shape
        x = x.reshape(B * T, gh, gw, C).permute(0, 3, 1, 2)  # (B*T, C, gh, gw)
        x = self.cnn(x)                                         # (B*T, D, gh, gw)
        D = x.shape[1]
        x = x.permute(0, 2, 3, 1).reshape(B, T, gh * gw, D)   # (B, T, P, D)
        return x

#-----------------------
# positional encoding
#-----------------------
class SinusoidalTimestepEncoding(nn.Module):

    """
    add a per-timestep positional encoding to every token in that hour's group
    so that all data from the same timestep shares same temporal encoding before
    passing through full sequence 

    sine and cosine functions of different frequencies are used
    each dimension of the positional encoding corresponds to a sinusoid
    from "Attention is All You Need"
    """

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # 192 rows, one per timestep 
                                        # rows 0-167 = history, 168-191 = future


    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]

#-------------------
# calendar features
#-------------------
def hours_to_calendar(future_time: torch.Tensor) -> torch.Tensor:

    """
    converts hours into two numerical features using sine and cosine transformations

    future_time: (B, T) int64 hours since Unix epoch
    returns    : (B, T, 4) float32  [hour_sin, hour_cos, dow_sin, dow_cos]
    """

    h  = (future_time % 24).float()
    d  = ((future_time // 24) % 7).float()
    pi2 = 2.0 * math.pi
    return torch.stack([
        torch.sin(pi2 * h / 24), torch.cos(pi2 * h / 24),
        torch.sin(pi2 * d /  7), torch.cos(pi2 * d /  7),
    ], dim=-1)


def build_hist_calendar(history_len: int, future_time: torch.Tensor) -> torch.Tensor:
    
    """
    reconstruct historical calendar by counting back from future_time[:,0]
    """

    t0      = future_time[:, 0]                                            # (B,)
    offsets = torch.arange(-history_len, 0, device=future_time.device)    # (S,)
    return hours_to_calendar(t0.unsqueeze(1) + offsets)                   # (B, S, 4)


# -------------
# main model
# -------------
class CNNTransformerForecaster(nn.Module):

    def __init__(
        self,
        in_channels:    int   = 7,
        n_zones:        int   = 8,
        n_cal_features: int   = 4,
        embed_dim:      int   = 64,
        grid_h:         int   = 5,
        grid_w:         int   = 5,
        history_len:    int   = 168,
        n_heads:        int   = 4,
        n_layers:       int   = 2,
        dropout:        float = 0.1,
        mlp_hidden:     int   = 128,
        norm_mean:      np.ndarray | None = None,
        norm_std:       np.ndarray | None = None,
    ):
        super().__init__()
        self.history_len = history_len
        self.future_len  = 24
        self.n_zones     = n_zones
        self.embed_dim   = embed_dim
        self.n_spatial   = grid_h * grid_w   # P = 25

        total_steps = history_len + self.future_len   # 192

        # normalization stats — registered as buffers so they move with .to(device)
        if norm_mean is not None and norm_std is not None:
            self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
            self.register_buffer("norm_std",  torch.tensor(norm_std,  dtype=torch.float32))
        else:
            self.norm_mean = None
            self.norm_std  = None

        self.cnn             = WeatherCNNEncoder(in_channels, embed_dim, grid_h, grid_w)
        self.hist_tab_proj   = nn.Linear(n_zones + n_cal_features, embed_dim) 
        self.future_tab_proj = nn.Linear(n_zones + n_cal_features, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.randn(self.n_spatial, embed_dim) * 0.02)
        self.timestep_enc    = SinusoidalTimestepEncoding(total_steps, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_zones),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------
    # adapt_inputs — required by eval harness
    # -------------------------------------------

    def _pool_weather(self, w: torch.Tensor) -> torch.Tensor:

        """
        downsample full-res (B, T, H, W, C) → (B, T, grid_h, grid_w, C).
        """

        B, T, H, W, C = w.shape
        gh, gw = self.cnn.grid_h, self.cnn.grid_w
        if H == gh and W == gw:
            return w  # already at target resolution
        x = w.reshape(B * T, H, W, C).permute(0, 3, 1, 2)   # (B*T, C, H, W)
        x = F.adaptive_avg_pool2d(x, (gh, gw))                # (B*T, C, gh, gw)
        return x.permute(0, 2, 3, 1).reshape(B, T, gh, gw, C)

    def adapt_inputs(self, history_weather, history_energy, future_weather, future_time):

        """
        accepts full-res tensors from the evaluation harness:
          history_weather : (B, 168, 450, 449, 7)
          history_energy  : (B, 168, n_zones)
          future_weather  : (B, 24,  450, 449, 7)
          future_time     : (B, 24) 

        downsamples weather, normalizes energy inputs
        generates calendar features for future timesteps
        reconstructs calendar features for historical timesteps
        normalizes energy demand 
        """

        dev        = history_weather.device
        hist_w     = self._pool_weather(history_weather)
        fut_w      = self._pool_weather(future_weather)
        future_cal = hours_to_calendar(future_time).to(dev)
        hist_cal   = build_hist_calendar(self.history_len, future_time.to(dev))
        hist_e     = history_energy.to(dev)
        if self.norm_mean is not None:
            hist_e = (hist_e - self.norm_mean) / self.norm_std
        return hist_w, hist_e, hist_cal, fut_w, future_cal

    # -------------------
    # forward function
    # --------------------
    def forward(self, hist_weather, hist_demand, hist_calendar,
                future_weather, future_calendar) -> torch.Tensor:
        
        """
        returns (B, 24, Z)
        """

        B  = hist_weather.size(0)
        S  = self.history_len
        F  = self.future_len
        D  = self.embed_dim
        P  = self.n_spatial

        #-----------------
        # spatial tokens 
        all_weather    = torch.cat([hist_weather, future_weather], dim=1)  # (B, S+F, gh, gw, 7)
        spatial_tokens = self.cnn(all_weather)                              # (B, S+F, P, D)
        # add learnable spatial positional embedding to every timestep
        spatial_tokens = spatial_tokens + self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)

        #-----------------
        # tabular tokens 
        # historical: real demand + calendar
        hist_tab = self.hist_tab_proj(
            torch.cat([hist_demand, hist_calendar], dim=-1)
        ).unsqueeze(2)                                                       # (B, S, 1, D)

        # future
        # zeros in place of unknown demand + calendar
        zero_d      = torch.zeros(B, F, self.n_zones, device=hist_demand.device)
        future_tab  = self.future_tab_proj(
            torch.cat([zero_d, future_calendar], dim=-1)
        ).unsqueeze(2)                                                       # (B, F, 1, D)

        tab_tokens = torch.cat([hist_tab, future_tab], dim=1)               # (B, S+F, 1, D)

        #-------------------------------
        # assemble unified sequence 
        # for each timestep: [P spatial tokens | 1 tabular token]
        seq   = torch.cat([spatial_tokens, tab_tokens], dim=2)              # (B, S+F, P+1, D)

        # add per-timestep sinusoidal encoding so all tokens in the same hour share the same temporal signal
        t_enc = self.timestep_enc(S + F).unsqueeze(0).unsqueeze(2)          # (1, S+F, 1, D)
        seq   = (seq + t_enc).reshape(B, (S + F) * (P + 1), D)             # (B, L, D)
        
        #-------------------------------
        # transformer runs self-attention
        enc = self.transformer(seq)                                          # (B, L, D)
      
        #-------------------------------
        # slice future states & predict 
        future_enc  = enc[:, S * (P + 1):, :].reshape(B, F, P + 1, D)
        future_repr = future_enc.mean(dim=2)                                # (B, F, D)
        pred = self.pred_head(future_repr)    
                                      # (B, F, Z)
        # denormalize back to raw MWh for the evaluation harness
        if self.norm_mean is not None:
            pred = pred * self.norm_std + self.norm_mean
        return pred


# ----------------------
# get the trained model
# ------------------------

def get_model(metadata: dict = None) -> CNNTransformerForecaster:
    ckpt_dir = Path(__file__).parent / "checkpoints"

    # load normalization stats saved from training
    norm_mean = norm_std = None
    norm_path = ckpt_dir / "norm_stats.npz"
    if norm_path.exists():
        data      = np.load(norm_path)
        norm_mean = data["mean"]
        norm_std  = data["std"]
        print(f"Loaded norm stats from {norm_path}")
    else:
        print(f"WARNING: norm_stats.npz not found at {norm_path}")

    # load checkpoint — use its metadata if harness didn't provide any
    ckpt_path = ckpt_dir / "best_model.pt"
    ckpt = None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if metadata is None or len(metadata) == 0:
            metadata = ckpt.get("metadata", {})

    if metadata is None:
        metadata = {}

    model = CNNTransformerForecaster(
        in_channels    = metadata.get("n_weather_vars", 7),
        n_zones        = metadata.get("n_zones",        8),
        n_cal_features = 4,
        embed_dim      = metadata.get("embed_dim",      64),
        grid_h         = metadata.get("grid_h",         5),
        grid_w         = metadata.get("grid_w",         5),
        history_len    = metadata.get("history_len",    168),
        n_heads        = metadata.get("n_heads",        4),
        n_layers       = metadata.get("n_layers",       2),
        dropout        = metadata.get("dropout",        0.1),
        mlp_hidden     = metadata.get("mlp_hidden",     128),
        norm_mean      = norm_mean,
        norm_std       = norm_std,
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded weights from {ckpt_path}  (epoch {ckpt['epoch']+1}, best MAPE {ckpt['best_mape']:.3f}%)")
    else:
        print(f"WARNING: no checkpoint found at {ckpt_path} — using random weights!!!")

    return model

# ----------------
# sanity check
# ----------------
if __name__ == "__main__":
    meta  = {"n_zones": 8, "history_len": 168, "n_weather_vars": 7, "grid_h": 5, "grid_w": 5}
    model = get_model(meta)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}")
    # sequence length: (168+24) × (25+1) = 192 × 26 = 4,992 tokens
    print(f"Sequence length: {(168+24) * (5*5+1):,} tokens")

    B, S, F, gh, gw = 2, 168, 24, 5, 5
    hw = torch.randn(B, S, gh, gw, 7)
    he = torch.randn(B, S, 8)
    fw = torch.randn(B, F, gh, gw, 7)
    ft = torch.zeros(B, F, dtype=torch.int64)

    out = model(*model.adapt_inputs(hw, he, fw, ft))
    assert out.shape == (B, F, 8), f"Bad shape: {out.shape}"
    print(f"Output shape: {out.shape}  OKAY")