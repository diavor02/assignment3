import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path

_CKPT_PATH = Path(__file__).parent / "rnn_energy_forecast_model.pt"


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




class RNNEnergyForecastModel(nn.Module):

    def __init__(
        self,
        n_zones:        int   = 8,
        n_weather_vars: int   = 7,
        S:              int   = 168,   
        horizon:        int   = 24,    
        grid_size:      int   = 5,     
        d_spatial:      int   = 128,   
        d_model:        int   = 256,   
        n_layers:       int   = 2,     # Generally, fewer layers are needed for RNNs than Transformers
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.S = S
        self.horizon  = horizon
        self.n_zones     = n_zones
        self.P           = grid_size * grid_size  
        self.d_model     = d_model

        # ── Spatial pathway (Unchanged) ───────────────────────────────────
        self.cnn          = CNN(n_weather_vars, d_spatial, grid_size)
        self.spatial_proj = nn.Linear(d_spatial, d_model)

        # ── Tabular pathway (Unchanged) ───────────────────────────────────
        n_cal = 6
        self.hist_tab_proj = nn.Linear(n_zones + n_cal, d_model)
        self.demand_mask   = nn.Parameter(torch.zeros(n_zones))
        self.fut_tab_proj  = nn.Linear(n_zones + n_cal, d_model)

        # ── Positional embeddings ─────────────────────────────────────────
        # We KEEP spatial embeddings so the model knows where patches are.
        self.spatial_pos_emb = nn.Embedding(self.P, d_model)
        
        # We REMOVE temporal_pos_emb. The LSTM handles time naturally.

        # ── Recurrent Encoder (Replacing Transformer) ─────────────────────
        # At each timestep, we have P spatial tokens + 1 tabular token.
        # We will flatten these (P + 1) tokens into a single vector per timestep.
        rnn_input_size = (self.P + 1) * d_model
        
        self.rnn = nn.LSTM(
            input_size  = rnn_input_size,
            hidden_size = d_model,         # The memory state size
            num_layers  = n_layers,
            batch_first = True,            # Expects (Batch, Timesteps, Features)
            dropout     = dropout if n_layers > 1 else 0.0
        )

        # ── Prediction head (Unchanged) ───────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_zones),  
        )

        # ── Normalisation statistics (Unchanged) ──────────────────────────
        self.register_buffer("energy_mean", torch.zeros(1, 1, n_zones))
        self.register_buffer("energy_std",  torch.ones(1, 1, n_zones))

    @staticmethod
    def extract_calendar(time_hours: torch.Tensor) -> torch.Tensor:
        # [Implementation remains exactly the same as your baseline]
        B, T  = time_hours.shape
        h_np  = time_hours.cpu().numpy().astype("datetime64[h]")
        feats = np.zeros((B, T, 6), dtype=np.float32)

        for b in range(B):
            dti   = pd.DatetimeIndex(h_np[b])
            hour  = dti.hour.values           
            dow   = dti.dayofweek.values      
            month = dti.month.values - 1      

            feats[b, :, 0] = np.sin(2 * np.pi * hour  / 24)
            feats[b, :, 1] = np.cos(2 * np.pi * hour  / 24)
            feats[b, :, 2] = np.sin(2 * np.pi * dow   / 7)
            feats[b, :, 3] = np.cos(2 * np.pi * dow   / 7)
            feats[b, :, 4] = np.sin(2 * np.pi * month / 12)
            feats[b, :, 5] = np.cos(2 * np.pi * month / 12)

        return torch.from_numpy(feats).to(time_hours.device)

    def _encode_weather(self, weather: torch.Tensor) -> torch.Tensor:
        # [Implementation remains exactly the same as your baseline]
        B, T, H, W, C = weather.shape
        x      = weather.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  
        tokens = self.cnn(x)                                             
        tokens = self.spatial_proj(tokens)                               
        return tokens.reshape(B, T, self.P, self.d_model)               

    def _normalise(self, energy: torch.Tensor) -> torch.Tensor:
        return (energy - self.energy_mean) / (self.energy_std + 1e-8)

    def _denormalise(self, energy: torch.Tensor) -> torch.Tensor:
        return energy * (self.energy_std + 1e-8) + self.energy_mean

    def adapt_inputs(self, history_weather, history_energy, future_weather, future_time) -> tuple:
        # [Implementation remains exactly the same as your baseline]
        S      = self.S
        device = next(self.parameters()).device

        history_weather = history_weather.to(device)
        history_energy  = history_energy.to(device)
        future_weather  = future_weather.to(device)
        future_time     = future_time.to(device)

        hist_w = history_weather[:, -S:]   
        hist_e = history_energy[:,  -S:]   

        t_cur     = future_time[:, 0:1] - 1                                  
        hist_time = t_cur + torch.arange(-S + 1, 1, device=device)           

        hist_cal = self.extract_calendar(hist_time)      
        fut_cal  = self.extract_calendar(future_time)    

        hist_sp = self._encode_weather(hist_w)           
        fut_sp  = self._encode_weather(future_weather)   

        hist_e_norm = self._normalise(hist_e)            

        return hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal

    def forward(
        self,
        hist_sp:  torch.Tensor,   
        hist_e:   torch.Tensor,   
        hist_cal: torch.Tensor,   
        fut_sp:   torch.Tensor,   
        fut_cal:  torch.Tensor,   
    ) -> torch.Tensor:
        
        B = hist_sp.shape[0]
        S = self.S
        F = self.horizon
        P = self.P
        D = self.d_model
        device = hist_sp.device

        # ── Spatial positional embeddings ─────────────────────────────────
        sp_emb  = self.spatial_pos_emb(torch.arange(P, device=device)) 
        hist_sp = hist_sp + sp_emb    
        fut_sp  = fut_sp  + sp_emb    

        # ── Tabular tokens ────────────────────────────────────────────────
        hist_tab = self.hist_tab_proj(torch.cat([hist_e, hist_cal], dim=-1)) 
        
        mask    = self.demand_mask.view(1, 1, -1).expand(B, F, -1)  
        fut_tab = self.fut_tab_proj(torch.cat([mask, fut_cal], dim=-1))      

        # ── Assemble per-timestep chunks ──────────────────────────────────
        hist_grp = torch.cat([hist_sp, hist_tab.unsqueeze(2)], dim=2)  # (B, S, P+1, D)
        fut_grp  = torch.cat([fut_sp,  fut_tab.unsqueeze(2)],  dim=2)  # (B, F, P+1, D)
        
        all_grp  = torch.cat([hist_grp, fut_grp], dim=1)               # (B, S+F, P+1, D)

        # ── Prepare sequence for RNN ──────────────────────────────────────
        # Flatten the (P+1) dimension into the feature dimension. 
        # The sequence length is just the timesteps (S+F).
        # Shape goes from (B, S+F, P+1, D) -> (B, S+F, (P+1)*D)
        seq = all_grp.reshape(B, S + F, (P + 1) * D)

        # ── Recurrent Processing ──────────────────────────────────────────
        # Process the sequence. We only need the output 'out', ignoring 
        # the final hidden states (hn, cn).
        out, _ = self.rnn(seq)  # out shape: (B, S+F, d_model)

        # ── Decode future predictions ─────────────────────────────────────
        # Extract only the representations for the future timesteps
        fut_repr = out[:, -F:, :]  # (B, F, d_model)
        
        pred_norm = self.head(fut_repr)     # (B, F, n_zones)

        return self._denormalise(pred_norm) # (B, F, n_zones)

def get_model(metadata: dict) -> RNNEnergyForecastModel:
    """
    Instantiate EnergyForecastModel from dataset metadata and optionally restore
    saved weights from disk.

    If a checkpoint exists at _CKPT_PATH, its state dict is loaded with
    map_location="cpu" to avoid device mismatches when loading a GPU-trained
    model on a CPU-only machine; the caller is responsible for moving the model
    to the target device afterwards.

    Parameters
    ----------
    metadata : dict
        Must contain:
          n_zones        — number of energy zones to forecast
          n_weather_vars — number of input weather channels
          future_len     — number of future hours to predict
    """
    model = RNNEnergyForecastModel(
        n_zones        = metadata["n_zones"],
        n_weather_vars = metadata["n_weather_vars"],
        S              = 168,    
        horizon        = metadata["future_len"],
        grid_size      = 5,     # 5×5 = 25 spatial patch tokens per timestep
        d_spatial      = 128,   # CNN output channels before d_model projection
        d_model        = 256,   # Transformer embedding dimension
        n_layers       = 4,
        dropout        = 0.1,
    )

    state = torch.load(_CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {_CKPT_PATH}")

    return model