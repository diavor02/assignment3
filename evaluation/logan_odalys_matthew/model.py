'''
    model = BaselineHybridModel(
        in_channels=in_channels,
        num_zones=num_zones,
        dim_calendar=dim_calendar,
        seq_len_hist=config.get("seq_len_hist", 24),
        seq_len_fut=config.get("seq_len_fut", 24),
        grid_size=config.get("grid_size", (10, 10)),
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 3),
        mlp_hidden=config.get("mlp_hidden", 256)
    )
'''

import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, grid_size: tuple = (10, 10)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(grid_size)
        )

    def forward(self, x):
        return self.net(x)

class BaselineHybridModel(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_zones: int, 
                 dim_calendar: int = 4, # 4 for sin/cos of hour and day
                 seq_len_hist: int = 24,
                 seq_len_fut: int = 24, 
                 grid_size: tuple = (10, 10), # Make sure this matches what you trained with!
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 mlp_hidden: int = 256):
        super().__init__()
        
        self.S = seq_len_hist
        self.F = seq_len_fut
        self.P = grid_size[0] * grid_size[1]
        self.embed_dim = embed_dim
        self.num_zones = num_zones
        
        cnn_out_channels = 64
        self.cnn = CNNFeatureExtractor(in_channels, cnn_out_channels, grid_size)
        self.spatial_proj = nn.Linear(cnn_out_channels, embed_dim)
        
        self.hist_tab_proj = nn.Linear(num_zones + dim_calendar, embed_dim)
        self.fut_tab_proj = nn.Linear(num_zones + dim_calendar, embed_dim)
        
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 1, self.P, embed_dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.S + self.F, 1, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_zones)
        )
        
        # Placeholders for normalization stats (loaded via get_model)
        self.register_buffer("norm_mean", None)
        self.register_buffer("norm_std", None)

    def adapt_inputs(self, history_weather, history_energy, future_weather, future_time):
        """
        Called by evaluate.py FIRST. We use this to bridge the gap between 
        the evaluation inputs and our model's expected formatting.
        """
        B = history_weather.size(0)
        device = history_weather.device
        
        # 1. Slice history: evaluate.py provides 168 hours, but we only need self.S (e.g., 24)
        h_weather = history_weather[:, -self.S:] # (B, S, H, W, C)
        h_energy  = history_energy[:, -self.S:]  # (B, S, Z)
        
        # 2. Normalize historical energy using stats from training
        if self.norm_mean is not None and self.norm_std is not None:
            h_energy = (h_energy - self.norm_mean) / self.norm_std
            
        # 3. Permute weather to (B, T, C, H, W) and concatenate
        h_weather = h_weather.permute(0, 1, 4, 2, 3)
        f_weather = future_weather.permute(0, 1, 4, 2, 3)
        weather_seq = torch.cat([h_weather, f_weather], dim=1) # (B, S+24, C, H, W)
        
        # 4. Generate Calendar Features (Sine/Cosine) from Unix Hours
        # future_time is (B, 24). We need to reconstruct the historical hours back in time.
        start_times = future_time[:, 0] - self.S
        full_time_array = torch.stack([start_times + i for i in range(self.S + self.F)], dim=1) # (B, S+24)
        
        # Convert unix hours to pandas datetime to extract cycle information
        flat_hours = full_time_array.view(-1).cpu().numpy()
        dt = pd.to_datetime(flat_hours, unit='h')
        hours = dt.hour.values
        days = dt.dayofweek.values
        
        cal_features = np.column_stack([
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * days / 7.0),
            np.cos(2 * np.pi * days / 7.0)
        ])
        
        cal_tensor = torch.tensor(cal_features, dtype=torch.float32, device=device).view(B, self.S + self.F, 4)
        hist_cal = cal_tensor[:, :self.S]
        fut_cal = cal_tensor[:, self.S:]
        
        # Return exactly what your forward() function expects!
        return weather_seq, h_energy, hist_cal, fut_cal

    def forward(self, weather_seq, hist_demand, hist_cal, fut_cal):
        B = weather_seq.size(0)
        Total_T = self.S + self.F
        
        # --- Process Weather into Spatial Tokens ---
        w_flat = weather_seq.view(B * Total_T, weather_seq.size(2), weather_seq.size(3), weather_seq.size(4))
        spatial_feats = self.cnn(w_flat) 
        
        spatial_feats = spatial_feats.view(B, Total_T, spatial_feats.size(1), -1).permute(0, 1, 3, 2)
        spatial_tokens = self.spatial_proj(spatial_feats) + self.spatial_pos_embed
        
        # --- Process Tabular Data ---
        hist_tab_feat = torch.cat([hist_demand, hist_cal], dim=-1) 
        hist_tab_tokens = self.hist_tab_proj(hist_tab_feat) 
        
        fut_demand_mask = torch.zeros(B, self.F, self.num_zones, device=fut_cal.device)
        fut_tab_feat = torch.cat([fut_demand_mask, fut_cal], dim=-1) 
        fut_tab_tokens = self.fut_tab_proj(fut_tab_feat) 
        
        tab_tokens = torch.cat([hist_tab_tokens, fut_tab_tokens], dim=1).unsqueeze(2) 
        
        # --- Unified Sequence Assembly ---
        unified_tokens = torch.cat([spatial_tokens, tab_tokens], dim=2) + self.temporal_pos_embed
        
        seq_len = Total_T * (self.P + 1)
        transformer_in = unified_tokens.view(B, seq_len, self.embed_dim)
        
        # --- Transformer ---
        transformer_out = self.transformer(transformer_in) 
        
        # --- Slice & Predict ---
        transformer_out = transformer_out.view(B, Total_T, self.P + 1, self.embed_dim)
        fut_states = transformer_out[:, -self.F:, :, :] 
        fut_summary = fut_states[:, :, -1, :] 
        
        predictions = self.mlp(fut_summary) # These are normalized (z-scores)!
        
        # --- UN-NORMALIZE ---
        # evaluate.py needs raw MWh values to calculate the MAPE correctly
        if self.norm_mean is not None and self.norm_std is not None:
            predictions = (predictions * self.norm_std) + self.norm_mean
            
        return predictions

def get_model(metadata: dict):
    """
    Called by evaluate.py. Must load the model parameters AND the trained weights.
    """
    # 1. Initialize the architecture with the metadata from evaluate.py
    model = BaselineHybridModel(
        in_channels=metadata.get("n_weather_vars", 7),
        num_zones=metadata.get("n_zones", 8),
        dim_calendar=4, 
        seq_len_hist=24, # IMPORTANT: Match this to what you used in train.py!
        seq_len_fut=metadata.get("future_len", 24),
        grid_size=(10, 10) # IMPORTANT: Match this to the grid_size you trained with!
    )
    
    # 2. Load the trained weights and normalization stats
    # evaluate.py runs from the assignment root, but we can locate the model file dynamically
    checkpoint_path = Path(__file__).parent / "best_model.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load the normalization stats saved during training
        if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
            model.norm_mean = checkpoint['norm_mean'].to(torch.float32)
            model.norm_std = checkpoint['norm_std'].to(torch.float32)
            print(f"[*] Loaded normalization stats from checkpoint.")
            
        print(f"[*] Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"[WARNING] Could not find {checkpoint_path}. Evaluating an UNTRAINED model.")
        
    return model