from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # Using BatchNorm2d for stability
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, out_channels, H_out, W_out).
        """
        return self.act(self.norm(self.conv(x)))


class WeatherPatchEncoder(nn.Module):
    """Encode weather maps into a small grid of spatial patch tokens."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_dims: Sequence[int] = (32, 64, 128),
        target_grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.target_grid_size = target_grid_size
        self.embed_dim = embed_dim
        dims = [in_channels, *hidden_dims]
        blocks: list[nn.Module] = []

        # Build CNN encoder with downsampling
        for idx in range(len(dims) - 1):
            stride = 2 if idx < len(dims) - 1 else 1
            blocks.append(
                ConvBlock(
                    in_channels=dims[idx],
                    out_channels=dims[idx + 1],
                    stride=stride,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        # Projection layer to convert dimensions to embed_dim
        self.project = nn.Sequential(
            nn.Conv2d(dims[-1], embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Forces output to fixed spatial size
        self.pool = nn.AdaptiveAvgPool2d((target_grid_size, target_grid_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode weather maps into patch tokens.

        Args:
            x: Tensor of shape (B, T, Cw, H, W).

        Returns:
            Tensor of shape (B, T, P, D) where P = target_grid_size**2.
        """
        B, T, Cw, H, W = x.shape
        x = x.view(B * T, Cw, H, W)
        x = self.encoder(x)
        x = self.project(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        return x.view(B, T, -1, self.embed_dim)


class TabularTokenEncoder(nn.Module):
    """Encode tabular history or future features into a single token per timestep."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        use_future_demand_mask: bool = False,
        num_zones: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.num_zones = num_zones
        self.use_future_demand_mask = use_future_demand_mask

        # Simple MLP to project tabular features to embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Placeholder for future demand mask, which can be learned or fixed
        if use_future_demand_mask:
            self.future_demand_mask = nn.Parameter(torch.zeros(num_zones), requires_grad=True)
        else:
            self.register_buffer('future_demand_mask', torch.zeros(num_zones))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project tabular features to a single token.

        Args:
            x: Tensor of shape (B, T, input_dim).

        Returns:
            Tensor of shape (B, T, 1, D).
        """
        return self.mlp(x).unsqueeze(2)

    def pad_future_demand(self, batch_size: int, future_steps: int, device: torch.device) -> torch.Tensor:
        """Create padded demand input for future timesteps."""
        pad = self.future_demand_mask.view(1, 1, self.num_zones).expand(batch_size, future_steps, self.num_zones)
        return pad.to(device)


class ForecastHead(nn.Module):
    """Predict zonal demand from future tabular tokens."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        # Final predictor with standard transformer head architecture
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict demand for each future timestep.

        Args:
            x: Tensor of shape (B, future_len, D).

        Returns:
            Tensor of shape (B, future_len, Z).
        """
        return self.head(x)


class EnergyDemandForecastModel(nn.Module):
    """Baseline hybrid CNN–Transformer model for energy demand forecasting."""

    def __init__(
        self,
        weather_channels: int = 7,
        calendar_dim: int = 8,
        num_zones: int = 8,
        hist_len: int = 168,
        future_len: int = 24,
        d_model: int = 256,
        cnn_hidden_dims: Sequence[int] = (32, 64, 128),
        target_grid_size: int = 8,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_learned_future_demand_mask: bool = False,
        max_total_steps: int = 256,
    ) -> None:
        super().__init__()
        self.weather_channels = weather_channels
        self.calendar_dim = calendar_dim
        self.num_zones = num_zones
        self.hist_len = hist_len
        self.future_len = future_len
        self.d_model = d_model
        self.target_grid_size = target_grid_size
        self.patches_per_step = target_grid_size * target_grid_size


        ### ===== ENCODERS ===== ###
        self.weather_encoder = WeatherPatchEncoder(
            in_channels=weather_channels,
            embed_dim=d_model,
            hidden_dims=cnn_hidden_dims,
            target_grid_size=target_grid_size,
        )

        self.hist_tabular_encoder = TabularTokenEncoder(
            input_dim=num_zones + calendar_dim,
            embed_dim=d_model,
            hidden_dim=d_model,
            use_future_demand_mask=False,
            num_zones=num_zones,
        )

        self.fut_tabular_encoder = TabularTokenEncoder(
            input_dim=num_zones + calendar_dim,
            embed_dim=d_model,
            hidden_dim=d_model,
            use_future_demand_mask=use_learned_future_demand_mask,
            num_zones=num_zones,
        )


        ### ===== POSITIONAL EMBEDDINGS ===== ###
        self.spatial_pos_emb = nn.Parameter(torch.randn(self.patches_per_step, d_model))
        self.temporal_pos_emb = nn.Embedding(max_total_steps, d_model)


        # Standard transformer encoder to process combined weather + tabular tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.forecast_head = ForecastHead(input_dim=d_model, output_dim=num_zones, hidden_dim=d_model, dropout=dropout)

    @staticmethod
    def _normalize_weather_input(
        x: torch.Tensor,
        weather_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Normalize weather inputs to channel-first format.

        Accepts either (B, T, Cw, H, W) or (B, T, H, W, Cw).
        """
        if x.ndim != 5:
            raise ValueError(f'Weather input must be 5D, got shape {tuple(x.shape)}')
        if x.shape[2] == weather_channels and x.shape[3] == height and x.shape[4] == width:
            return x
        if x.shape[4] == weather_channels and x.shape[2] == height and x.shape[3] == width:
            return x.permute(0, 1, 4, 2, 3).contiguous()
        raise ValueError(
            f'Weather input shape {tuple(x.shape)} is not compatible with either '
            f'(B, T, Cw, H, W) or (B, T, H, W, Cw) for Cw={weather_channels}, H={height}, W={width}'
        )

    def forward(
        self,
        hist_weather: torch.Tensor,
        hist_demand: torch.Tensor,
        hist_calendar: torch.Tensor,
        fut_weather: torch.Tensor,
        fut_calendar: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for hybrid weather + tabular demand forecasting."""
        # Input validation and normalization
        B = hist_demand.shape[0]
        assert hist_demand.shape == (B, self.hist_len, self.num_zones), 'Incoming hist_demand shape mismatch'
        assert hist_calendar.shape == (B, self.hist_len, self.calendar_dim), 'Incoming hist_calendar shape mismatch'
        assert fut_calendar.shape == (B, self.future_len, self.calendar_dim), 'Incoming fut_calendar shape mismatch'

        hist_weather = self._normalize_weather_input(hist_weather, self.weather_channels, height=450, width=449)
        fut_weather = self._normalize_weather_input(fut_weather, self.weather_channels, height=450, width=449)

        # Encode wheater tokekens from both history and future
        hist_weather_tokens = self.weather_encoder(hist_weather)
        fut_weather_tokens = self.weather_encoder(fut_weather)
        # shapes: (B, hist_len, P, D) and (B, future_len, P, D)

        # Add shared spatial psitional embedings to weather tokens
        spatial_pos = self.spatial_pos_emb.view(1, 1, self.patches_per_step, self.d_model)
        hist_weather_tokens = hist_weather_tokens + spatial_pos
        fut_weather_tokens = fut_weather_tokens + spatial_pos

        # Encode tabular tokens from both history and future, applying future demand mask if enabled
        hist_tabular = torch.cat([hist_demand, hist_calendar], dim=-1)
        hist_tabular_tokens = self.hist_tabular_encoder(hist_tabular)
        # shape: (B, hist_len, 1, D)

        future_demand_pad = self.fut_tabular_encoder.pad_future_demand(
            batch_size=B,
            future_steps=self.future_len,
            device=fut_calendar.device,
        )
        fut_tabular = torch.cat([future_demand_pad, fut_calendar], dim=-1)
        fut_tabular_tokens = self.fut_tabular_encoder(fut_tabular)
        # shape: (B, future_len, 1, D)

        # Combine weather and tabular tokens
        hist_group = torch.cat([hist_weather_tokens, hist_tabular_tokens], dim=2)
        fut_group = torch.cat([fut_weather_tokens, fut_tabular_tokens], dim=2)
        # shapes: (B, hist_len, P+1, D) and (B, future_len, P+1, D)

        sequence = torch.cat([hist_group, fut_group], dim=1)
        total_steps = self.hist_len + self.future_len
        assert sequence.shape == (B, total_steps, self.patches_per_step + 1, self.d_model)

        time_indices = torch.arange(total_steps, device=sequence.device)

        # Add temporal positional embeddings
        time_emb = self.temporal_pos_emb(time_indices).unsqueeze(1)
        sequence = sequence + time_emb

        # Flatten for transformer input: (B, total_steps * (P+1), D)
        sequence = sequence.view(B, total_steps * (self.patches_per_step + 1), self.d_model)
        transformer_out = self.transformer(sequence)

        # Reshape back to (B, total_steps, P+1, D) and extract future tabular tokens for forecasting
        transformer_out = transformer_out.view(B, total_steps, self.patches_per_step + 1, self.d_model)

        future_tabular_out = transformer_out[:, self.hist_len :, self.patches_per_step, :]
        # shape: (B, future_len, D)

        return self.forecast_head(future_tabular_out)


def get_model(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> EnergyDemandForecastModel:
    """Create a model instance from config or kwargs."""
    params = {
        'weather_channels': 7,
        'calendar_dim': 8,
        'num_zones': 8,
        'hist_len': 168,
        'future_len': 24,
        'd_model': 256,
        'cnn_hidden_dims': (32, 64, 128),
        'target_grid_size': 8,
        'nhead': 8,
        'num_transformer_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'use_learned_future_demand_mask': False,
        'max_total_steps': 256,
    }
    if config is not None:
        params.update(config)
    params.update(kwargs)
    return EnergyDemandForecastModel(**params)


if __name__ == '__main__':
    model = get_model()
    B = 2
    hist_weather = torch.randn(B, model.hist_len, model.weather_channels, 450, 449)
    fut_weather = torch.randn(B, model.future_len, model.weather_channels, 450, 449)
    hist_demand = torch.randn(B, model.hist_len, model.num_zones)
    hist_calendar = torch.randn(B, model.hist_len, model.calendar_dim)
    fut_calendar = torch.randn(B, model.future_len, model.calendar_dim)
    out = model(
        hist_weather=hist_weather,
        hist_demand=hist_demand,
        hist_calendar=hist_calendar,
        fut_weather=fut_weather,
        fut_calendar=fut_calendar,
    )
    print('output shape', out.shape)
