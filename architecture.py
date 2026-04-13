import torch
import torch.nn as nn

import torch
import torch.nn as nn

# Input shape: (B, 7, 450, 449)
# Output shape: (B, d_model=128, ~110, ~110)
class WeatherCNN(nn.Module):
    def __init__(self, in_channels=7, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class TabularEmbedding(nn.Module):
    def __init__(self, demand_dim, cal_dim, d_model):
        super().__init__()

        self.demand_proj = nn.Linear(demand_dim, d_model)
        self.cal_proj = nn.Linear(cal_dim, d_model)

        self.fuse = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        demand, cal = x
        d_emb = self.demand_proj(demand)
        c_emb = self.cal_proj(cal)
        return self.fuse(d_emb + c_emb)

class ForecastModel(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4, Z=8):
        super().__init__()

        self.d_model = d_model
        self.Z = Z

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Final prediction head
        self.mlp = nn.Linear(d_model, Z)

    def forward(self, tokens, S=48):
        """
        tokens: (B, (S+24)*(P+1), d_model)
        S: number of historical timesteps
        """

        B, total_tokens, d_model = tokens.shape

        # --- Step 1: Transformer ---
        x = self.transformer(tokens)
        # still (B, (S+24)*(P+1), d_model)

        # --- Step 2: recover structure ---
        timesteps_total = S + 24
        tokens_per_timestep = total_tokens // timesteps_total  # = (P+1)

        x = x.view(B, timesteps_total, tokens_per_timestep, d_model)
        # (B, S+24, P+1, d_model)

        # --- Step 3: slice future ---
        x_future = x[:, S:, :, :]
        # (B, 24, P+1, d_model)

        # --- Step 4: reduce tokens per timestep ---
        # Option A: take LAST token (tabular token)
        x_reduced = x_future[:, :, -1, :]   # (B, 24, d_model)

        # Option B (alternative):
        # x_reduced = x_future.mean(dim=2)

        # --- Step 5: predict zones ---
        out = self.mlp(x_reduced)
        # (B, 24, Z)

        return out