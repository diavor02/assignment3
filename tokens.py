import torch

import torch
import torch.nn as nn
from torch import Tensor


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