import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import TabularLazyDataset, WeatherLazyDataset
from helper import mape_loss

# ==========================================
# 1. SPATIAL COMPONENTS
# ==========================================


class WeatherCNN(nn.Module):
    """
    Stronger CNN backbone:
    - increases channels up to 128
    - aggressively downsamples
    - finishes with a fixed spatial grid via AdaptiveAvgPool2d
    """

    def __init__(self, in_channels=7, d_model=64, final_grid=4):
        super().__init__()
        self.features = nn.Sequential(
            # 450x449 -> ~225x225
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # ~225x225 -> ~113x113
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # ~113x113 -> ~57x57
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # ~57x57 -> ~29x29
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # ~29x29 -> ~15x15
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Force a tiny, fixed spatial grid
        self.pool = nn.AdaptiveAvgPool2d((final_grid, final_grid))

        # Project to model dimension
        self.proj = nn.Conv2d(128, d_model, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class SpatialTokenExtractor(nn.Module):
    def __init__(self, cnn_net, d_model=64, S=48, horizon=24, final_grid=4):
        super().__init__()
        self.cnn = cnn_net
        self.d_model = d_model
        self.T = S + horizon
        self.P = final_grid * final_grid

        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.P, d_model))
        self.timestep_embed = nn.Parameter(torch.randn(1, self.T, 1, d_model))

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)

        tokens = self.cnn(x)  # (B*T, d_model, g, g)

        _, _, g1, g2 = tokens.shape
        assert g1 * g2 == self.P, f"Expected {self.P} spatial tokens, got {g1 * g2}"

        tokens = tokens.reshape(B, T, self.d_model, self.P)
        tokens = tokens.permute(0, 1, 3, 2).contiguous()  # (B, T, P, d_model)

        assert tokens.shape == (B, self.T, self.P, self.d_model)

        tokens = tokens + self.spatial_embed + self.timestep_embed
        return tokens


def create_spatial_tokens(
    cnn, original_input_channels_size=7, original_h=450, original_w=449
):
    device = next(cnn.parameters()).device  # ✅ get model device

    dummy_input = torch.zeros(
        1, original_input_channels_size, original_h, original_w, device=device  # ✅ FIX
    )
    dummy_output = cnn(dummy_input)
    _, _, downsampled_h_size, downsampled_w_size = dummy_output.shape

    P = downsampled_h_size * downsampled_w_size
    print("downsampled_h_size:", downsampled_h_size)
    print("downsampled_w_size:", downsampled_w_size)
    print("P:", P)

    token_extractor = SpatialTokenExtractor(
        cnn_net=cnn, d_model=64, S=48, horizon=24, final_grid=downsampled_h_size
    )
    return token_extractor, P, downsampled_h_size, downsampled_w_size


# ==========================================
# 2. TABULAR / COMBINER COMPONENT
# ==========================================


class HybridTokenCombiner(nn.Module):
    def __init__(self, y_dim=8, c_dim=6, embed_dim=64):
        super().__init__()
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.tabular_embed = nn.Linear(y_dim + c_dim, embed_dim)

    def forward(
        self,
        y_hist,
        c_hist,
        c_future,
        spatial_tokens,
        downsampled_h_size,
        downsampled_w_size,
        d_model=64,
        S=48,
        fut=24,
    ):
        B = y_hist.shape[0]
        device = y_hist.device

        y_mask = torch.zeros((B, fut, self.y_dim), device=device)

        assert y_hist.shape == (B, S, self.y_dim)
        assert y_mask.shape == (B, fut, self.y_dim)
        assert c_hist.shape == (B, S, self.c_dim)
        assert c_future.shape == (B, fut, self.c_dim)

        P = downsampled_h_size * downsampled_w_size
        print("P:", P)
        assert spatial_tokens.shape == (B, S + fut, P, d_model)

        hist_input = torch.cat([y_hist, c_hist], dim=-1)
        future_input = torch.cat([y_mask, c_future], dim=-1)

        full_tabular_sequence = torch.cat([hist_input, future_input], dim=1)
        tabular_tokens = self.tabular_embed(full_tabular_sequence)  # (B, T, d_model)

        tabular_tokens = tabular_tokens.unsqueeze(2)  # (B, T, 1, d_model)

        unified_tokens = torch.cat([spatial_tokens, tabular_tokens], dim=2)
        assert unified_tokens.shape == (B, S + fut, P + 1, d_model)

        final_sequence = unified_tokens.reshape(B, -1, d_model)
        assert final_sequence.shape == (B, (S + fut) * (P + 1), d_model)

        return final_sequence


# ==========================================
# 3. PUTTING IT ALL TOGETHER
# ==========================================

if __name__ == "__main__":
    print("Starting...")

    # --- Config ---
    PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/"
    BATCH_SIZE = 4
    S = 48
    fut = 24
    d_model = 64
    original_h = 450
    original_w = 449
    in_channels = 7
    final_grid = 4  # 4x4 => P = 16, much smaller sequence

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # --- Initialize Models ---
    print("Initialize models")
    cnn_module = WeatherCNN(
        in_channels=in_channels, d_model=d_model, final_grid=final_grid
    ).to(DEVICE)
    token_extractor_module, P, d_h, d_w = create_spatial_tokens(
        cnn_module,
        original_input_channels_size=in_channels,
        original_h=original_h,
        original_w=original_w,
    )
    combiner_module = HybridTokenCombiner().to(DEVICE)
    token_extractor_module.to(DEVICE)

    transformer_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=8, batch_first=True
    ).to(DEVICE)

    # --- Tabular Data Setup ---
    tabular_dataset = TabularLazyDataset()

    tabular_loader = DataLoader(
        tabular_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,  # increase if cluster allows
        pin_memory=True,  # important for CUDA
    )

    # #  --- Weather Tabular Setup ---
    # weather_dataset = WeatherLazyDataset("/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data")

    # weather_loader = DataLoader(
    #     weather_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=1,      # increase if cluster allows
    #     pin_memory=True     # important for CUDA
    # )

    print("Mock data layer")
    mock_dataloader = [
        torch.randn(BATCH_SIZE, S + fut, original_h, original_w, in_channels)
        for _ in range(2)
    ]

    print("Initialize optimizer")
    prediction_head = nn.Linear(d_model, 8).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(cnn_module.parameters()) +
        list(token_extractor_module.parameters()) +
        list(combiner_module.parameters()) +
        list(transformer_layer.parameters()) +
        list(prediction_head.parameters()),
        lr=1e-4
    )

    # --- Training Loop ---
    for batch_idx, (weather_batch, tabular_batch) in enumerate(
        zip(mock_dataloader, tabular_loader)
    ):
        print(f"\n--- Processing Batch {batch_idx + 1} ---")

        # Move weather data
        weather_batch = weather_batch.to(DEVICE)

        # Unpack + move tabular data
        y_hist, y_future, c_hist, c_future = tabular_batch
        y_hist = y_hist.to(DEVICE, non_blocking=True)
        y_future = y_future.to(DEVICE, non_blocking=True)
        c_hist = c_hist.to(DEVICE, non_blocking=True)
        c_future = c_future.to(DEVICE, non_blocking=True)

        # 1. Spatial tokens
        spatial_tokens = token_extractor_module(weather_batch)

        print("y_hist shape:", y_hist.shape)
        print("y_future shape:", y_future.shape)
        print("c_hist shape:", c_hist.shape)
        print("c_future shape:", c_future.shape)
        print("spatial_tokens shape:", spatial_tokens.shape)

        # 2. Combine
        final_sequence = combiner_module(
            y_hist=y_hist,
            c_hist=c_hist,
            c_future=c_future,
            spatial_tokens=spatial_tokens,
            downsampled_h_size=d_h,
            downsampled_w_size=d_w,
            d_model=d_model,
            S=S,
            fut=fut,
        )

        print("final_sequence shape:", final_sequence.shape)

        # 3. Transformer
        transformer_out = transformer_layer(final_sequence)

        print(f"Success for batch idx {batch_idx}")

        # reshape back
        B = weather_batch.shape[0]
        transformer_out = transformer_out.view(B, S + fut, (d_h * d_w + 1), d_model)

        # extract tabular tokens
        tabular_tokens = transformer_out[:, :, -1, :]

        # future only
        future_tokens = tabular_tokens[:, S:, :]

        # prediction
        y_pred = prediction_head(future_tokens)

        # loss
        loss = mape_loss(y_pred, y_future)

        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
