import torch

def build_tokens(weather, demand, calendar, cnn, tabular_embed, S=48):
    """
    weather: (B, S+24, H, W, C)
    demand: (B, S, Z)
    calendar: (B, S+24, C_cal)
    """

    B, T, H, W, C = weather.shape
    tokens = []

    #  T = S + 24
    for t in range(T):
        # --- spatial tokens ---
        x = weather[:, t].permute(0, 3, 1, 2)   # (B, C, H, W)
        feat = cnn(x)                           # (B, d_model, h, w)
        feat = feat.flatten(2).transpose(1, 2)  # (B, P, d_model), P = h * w

        # --- tabular token ---
        # demand.shape = (B, S, Z)
        # demand[:, t, :].shape = (B, Z), the expected output shape
        if t < S:
            tab_input = torch.cat([demand[:, t, :], calendar[:, t, :]], dim=-1)
        else:
            # mask future demand, basically select the first timestep nad make everything 0
            masked_y = torch.zeros_like(demand[:, 0, :])
            # calendar stays the same
            tab_input = torch.cat([masked_y, calendar[:, t, :]], dim=-1)

        tab = tabular_embed(tab_input).unsqueeze(1)  # (B, 1, d_model)

        # combine
        tokens.append(torch.cat([feat, tab], dim=1))  # (B, P+1, d_model)

    tokens = torch.cat(tokens, dim=1)
    # (B, (S+24)*(P+1), d_model)

    return tokens