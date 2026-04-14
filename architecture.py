import torch
import torch.nn as nn
import os

PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/"

# Input shape: (B, 7, 450, 449)
# Intended Output shape: (B, d_model=64, ~110, ~110)


import torch
import torch.nn as nn


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



# class WeatherDataset(torch.utils.data.Dataset):
#     def __init__(self, file_list, S=48, horizon=24, mean=None, std=None):
#         self.files = file_list
#         self.S = S
#         self.horizon = horizon
#         self.mean = mean
#         self.std = std

#     def __len__(self):
#         return len(self.files) - (self.S + self.horizon)

#     def __getitem__(self, idx):
#         seq_files = self.files[idx : idx + self.S + self.horizon]

#         weather_seq = []
#         for path in seq_files:
#             x = torch.load(path).float()  # (H, W, C)
#             weather_seq.append(x)

#         x = torch.stack(weather_seq)  # (T, H, W, C)

#         if self.mean is not None and self.std is not None:
#             x = (x - self.mean) / (self.std + 1e-6)

#         return x

# def compute_stats_from_files(file_list):
#     sum_ = 0
#     sum_sq = 0
#     count = 0

#     for path in file_list:
#         x = torch.load(path).float()  # (H, W, C)

#         sum_ += x.sum(dim=(0, 1))
#         sum_sq += (x ** 2).sum(dim=(0, 1))
#         count += x.shape[0] * x.shape[1]

#     mean = sum_ / count
#     result = sum_sq / count - mean ** 2
#     std = torch.sqrt(torch.as_tensor(result))

#     assert isinstance(mean, torch.Tensor)
#     assert isinstance(std, torch.Tensor)

#     return mean, std

# files = build_file_list(PATH + "weather_data/")

# print("Stuff")
# print(len(files))
# print(files[0])
# print(files[-1])
# print("------------------------------------")

# compute stats
# mean, std = compute_stats_from_files(files)
# mean = mean.view(1, 1, 1, -1)
# std = std.view(1, 1, 1, -1)

# # dataset
# dataset = WeatherDataset(files, mean=mean, std=std)