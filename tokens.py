import torch

import torch
import torch.nn as nn
from torch import Tensor

import torch
import torch.nn as nn

class SpatialTokenExtractor(nn.Module):
    def __init__(self, cnn_net, d_model=64, S=48, horizon=24, downsampled_h=32, downsampled_w=32):
        super().__init__()
        self.cnn = cnn_net
        self.d_model = d_model
        
        # P = Number of Spatial Patches (Tokens) per timestep
        self.P = downsampled_h * downsampled_w
        self.T = S + horizon  # Total sequence length

        # Embeddings (Learnable Parameters) as shown in the diagram
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.P, d_model))
        self.timestep_embed = nn.Parameter(torch.randn(1, self.T, 1, d_model))

    def forward(self, x):
        """
        Expects input shape from DataLoader: (B, T, H, W, C)
        Returns unified sequence for Transformer: (B, T * P, d_model)
        """
        B, T, H, W, C = x.shape

        # 1. CNN Downsample
        # Permute to (B, T, C, H, W) and fold T into B for Conv2D
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * T, C, H, W)
        
        # Shape becomes (B*T, d_model, downsampled_h, downsampled_w)
        tokens = self.cnn(x) 

        # 2. Extract Spatial Tokens
        # Flatten spatial dimensions H and W into P patches
        tokens = tokens.view(B, T, self.d_model, self.P) # (B, T, d_model, P)
        tokens = tokens.permute(0, 1, 3, 2)              # (B, T, P, d_model)

        # 3. Inject Positional Encodings
        # Broadcasting naturally handles the (1, 1, P, d) and (1, T, 1, d) additions
        tokens = tokens + self.spatial_embed + self.timestep_embed

        # 4. Create Unified Sequence
        # Flatten Time (T) and Patches (P) into a single sequence length
        unified_seq = tokens.view(B, T * self.P, self.d_model) 

        return unified_seq

def build_tokens(
    weather: Tensor,
    demand: Tensor,
    calendar: Tensor,
    cnn: nn.Module,
    tabular_embed: nn.Module,
    S: int = 48,
) -> Tensor:
    """
    Tokenise weather, demand, and calendar inputs into a flat sequence of
    spatial + tabular tokens suitable for a transformer encoder.

    For each of the T = S + 24 timesteps, the function produces P + 1 tokens:
      - P spatial tokens extracted from the weather grid via ``cnn``
        (one token per spatial patch, where P = h * w after the CNN).
      - 1 tabular token formed by embedding the concatenation of the demand
        reading and the calendar features for that timestep.

    Demand is only observed for the first S timesteps.  For t >= S (the
    future horizon), the demand slice is replaced with zeros so the model
    cannot attend to ground-truth future load.

    Args:
        weather:        Raw weather grid features.
                        Shape: ``(B, S+24, H, W, C)``
                        ``B`` = batch, ``H``/``W`` = spatial dims,
                        ``C`` = feature channels.
        demand:         Zonal demand observations for the historical window.
                        Shape: ``(B, S, Z)``
                        ``Z`` = number of demand zones.
        calendar:       Calendar / time-of-day features for the full horizon.
                        Shape: ``(B, S+24, C_cal)``
                        ``C_cal`` = number of calendar features.
        cnn:            Convolutional feature extractor.
                        Input:  ``(B, C, H, W)``
                        Output: ``(B, d_model, h, w)``
        tabular_embed:  Linear (or MLP) that maps concatenated demand +
                        calendar features to the model dimension.
                        Input:  ``(B, Z + C_cal)``
                        Output: ``(B, d_model)``
        S:              Length of the historical (observed) window in
                        timesteps.  Defaults to 48.

    Returns:
        tokens: Flat token sequence over all timesteps.
                Shape: ``(B, (S+24) * (P+1), d_model)``
                where ``P = h * w`` is the number of spatial patches
                produced by ``cnn``.

    Raises:
        ValueError: If ``weather.shape[1] != S + 24`` or
                    ``demand.shape[1] != S``.
    """

    assert isinstance(weather, Tensor), f"Expected weather to be a Tensor, got {type(weather)}"
    assert isinstance(demand, Tensor), f"Expected demand to be a Tensor, got {type(demand)}"
    assert isinstance(calendar, Tensor), f"Expected calendar to be a Tensor, got {type(calendar)}"
    assert isinstance(cnn, nn.Module), f"Expected cnn to be an nn.Module, got {type(cnn)}"
    assert isinstance(tabular_embed, nn.Module), f"Expected tabular_embed to be an nn.Module, got {type(tabular_embed)}"
    assert isinstance(S, int), f"Expected S to be an int, got {type(S)}"


    if weather.shape[1] != S + 24:
        raise ValueError(
            f"Expected weather.shape[1] == S+24 ({S + 24}), "
            f"got {weather.shape[1]}"
        )
    if demand.shape[1] != S:
        raise ValueError(
            f"Expected demand.shape[1] == S ({S}), got {demand.shape[1]}"
        )

    B, T, H, W, C = weather.shape
    tokens: list[Tensor] = []

    for t in range(T):
        # --- Spatial tokens ---
        x: Tensor = weather[:, t].permute(0, 3, 1, 2)   # (B, C, H, W)
        feat: Tensor = cnn(x)                            # (B, d_model, h, w)
        feat = feat.flatten(2).transpose(1, 2)           # (B, P, d_model)

        # --- Tabular token ---
        # Future timesteps (t >= S) have no observed demand, so we mask with
        # zeros to prevent information leakage into the forecast horizon.
        if t < S:
            demand_t: Tensor = demand[:, t, :]           # (B, Z)
        else:
            demand_t = torch.zeros_like(demand[:, 0, :]) # (B, Z)

        tab_input: Tensor = torch.cat(
            [demand_t, calendar[:, t, :]], dim=-1
        )                                                # (B, Z + C_cal)
        tab: Tensor = tabular_embed(tab_input).unsqueeze(1)  # (B, 1, d_model)

        # --- Combine spatial and tabular tokens ---
        tokens.append(torch.cat([feat, tab], dim=1))    # (B, P+1, d_model)

    return torch.cat(tokens, dim=1)                     # (B, T*(P+1), d_model)