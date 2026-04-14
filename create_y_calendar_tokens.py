import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZE = 128
DEVICE = 'cuda'

def create_sequences_from_csv(csv_path='demand_calendar_normalized.csv', S=48, future_steps=24):
    """Reads the CSV and creates rolling window tensors."""
    df = pd.read_csv(csv_path)
    
    # Define columns
    y_cols = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA_BOST']
    c_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
    
    # Convert to numpy arrays (assuming data is already scaled/normalized)
    Y_data = df[y_cols].values
    C_data = df[c_cols].values
    
    y_hist_list, c_hist_list, c_future_list = [], [], []
    
    # Create sliding windows
    total_window = S + future_steps
    for i in range(len(df) - total_window + 1):
        # Historical Data (t-S : t)
        y_hist_list.append(Y_data[i : i+S])
        c_hist_list.append(C_data[i : i+S])
        
        # Future Calendar Data (t+1 : t+24)
        c_future_list.append(C_data[i+S : i+total_window])
        
    # Convert to PyTorch Tensors: Shape [Batch, Sequence_Length, Features]
    y_hist = torch.tensor(np.array(y_hist_list), dtype=torch.float32)
    c_hist = torch.tensor(np.array(c_hist_list), dtype=torch.float32)
    c_future = torch.tensor(np.array(c_future_list), dtype=torch.float32)

    print("y_hist shape:", y_hist.shape)
    print("c_hist shape:", c_hist.shape)
    print("c_future shape:", c_future.shape)
    
    return y_hist, c_hist, c_future

def save_sequences_to_disk(csv_path='demand_calendar_normalized.csv', S=48, future_steps=24, output_dir='tensors'):
    """Creates rolling window tensors from CSV and saves them to disk."""
    import os
    
    y_hist, c_hist, c_future = create_sequences_from_csv(csv_path, S, future_steps)
    
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(y_hist,   os.path.join(output_dir, 'y_hist.pt'))
    torch.save(c_hist,   os.path.join(output_dir, 'c_hist.pt'))
    torch.save(c_future, os.path.join(output_dir, 'c_future.pt'))
    
    print(f"Tensors saved to '{output_dir}/'")
    print(f"  y_hist.pt   → {y_hist.shape}")
    print(f"  c_hist.pt   → {c_hist.shape}")
    print(f"  c_future.pt → {c_future.shape}")

class HybridTokenCombiner(nn.Module):
    def __init__(self, y_dim=8, c_dim=6, embed_dim=64):
        super().__init__()
        self.y_dim = y_dim
        self.c_dim = c_dim
        
        # The "Linear Embed" layer from the diagram
        # Projects the concatenated 14 features down to the 64-dim channel size
        self.tabular_embed = nn.Linear(y_dim + c_dim, embed_dim)

    def forward(self, y_hist, c_hist, c_future, spatial_tokens, downsampled_h_size=113, downsampled_w_size=113, d_model=64, S=48, fut=24):
        """
        y_hist: [Batch, 48, 8]
        c_hist: [Batch, 48, 6]
        c_future: [Batch, 24, 6]
        spatial_tokens: [Batch, 72, 12769, 64] (from CNN downsample)
        """
        y_mask = torch.zeros((BATCH_SIZE, fut, self.y_dim), device=DEVICE)

        assert y_hist.shape == (BATCH_SIZE, S, self.y_dim)
        assert y_mask.shape == (BATCH_SIZE, fut, self.y_dim)

        assert c_hist.shape == (BATCH_SIZE, S, self.c_dim)
        assert c_future.shape == (BATCH_SIZE, fut, self.c_dim)

        P = downsampled_h_size * downsampled_w_size
        assert spatial_tokens.shape == (BATCH_SIZE, S + fut, P, d_model)

        # 1. Process Historical Tabular Data
        # Concatenate Y and C: [Batch, 48, 14]
        hist_input = torch.cat([y_hist, c_hist], dim=-1)
        assert hist_input.shape == (BATCH_SIZE, S, self.y_dim + self.c_dim) 
        
        # Concatenate Masked Y and known future C: [Batch, 24, 14]
        future_input = torch.cat([y_mask, c_future], dim=-1) 
        assert future_input.shape == (BATCH_SIZE, fut, self.y_dim + self.c_dim)
        
        # 3. Create Full Tabular Sequence (S + 24)
        # Combine historical and future time steps: [Batch, 72, 14]
        full_tabular_sequence = torch.cat([hist_input, future_input], dim=1)
        assert full_tabular_sequence.shape == (BATCH_SIZE, S + fut, self.y_dim + self.c_dim)
        
        # 4. Linear Embed
        # Pass through linear layer: [Batch, 72, 64]
        tabular_tokens = self.tabular_embed(full_tabular_sequence)
        assert full_tabular_sequence.shape == (BATCH_SIZE, S + fut, d_model)
        
        # Reshape to [Batch, 72, 1, 64] so we can concatenate along the token dimension (P)
        tabular_tokens = tabular_tokens.unsqueeze(2)
        assert tabular_tokens.shape == (BATCH_SIZE, S + fut, 1, d_model)

        # 5. Build the Unified Sequence
        # spatial_tokens is [Batch, 72, 12769, 64]
        # Concatenate spatial (P) and tabular (1) tokens per timestep: [Batch, 72, 12770, 64]
        unified_tokens = torch.cat([spatial_tokens, tabular_tokens], dim=2)
        assert unified_tokens.shape == (BATCH_SIZE, S + fut, P + 1, d_model)
        
        # 6. Flatten to (S+24) * (P+1)
        # Final sequence shape for standard PyTorch Transformer: [Batch, 919440, 64]
        final_sequence = unified_tokens.view(BATCH_SIZE, -1, 64)
        assert final_sequence.shape == (BATCH_SIZE, (S+fut) * (P+1), d_model)
        
        return final_sequence