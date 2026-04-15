import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def create_sequences_from_csv(csv_path='demand_calendar_normalized.csv', S=48, future_steps=24):
    """Reads the CSV and creates rolling window tensors."""
    df = pd.read_csv(csv_path)
    
    # Define columns
    y_cols = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA_BOST']
    c_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
    
    # Convert to numpy arrays (assuming data is already scaled/normalized)
    Y_data = df[y_cols].values
    C_data = df[c_cols].values
    
    y_hist_list, y_future_list, c_hist_list, c_future_list = [], [], [], []
    
    # Create sliding windows
    total_window = S + future_steps
    for i in range(len(df) - total_window + 1):
        # Historical Data (t-S : t)
        y_hist_list.append(Y_data[i : i+S])
        c_hist_list.append(C_data[i : i+S])
        
        # Future Calendar Data (t+1 : t+24)
        y_future_list.append(Y_data[i+S : i+total_window])
        c_future_list.append(C_data[i+S : i+total_window])
        
    # Convert to PyTorch Tensors: Shape [Batch, Sequence_Length, Features]
    y_hist = torch.tensor(np.array(y_hist_list), dtype=torch.float32)
    y_future = torch.tensor(np.array(y_future_list), dtype=torch.float32)
    c_hist = torch.tensor(np.array(c_hist_list), dtype=torch.float32)
    c_future = torch.tensor(np.array(c_future_list), dtype=torch.float32)

    print("y_hist shape:", y_hist.shape)
    print("y_future shape", y_future.shape)
    print("c_hist shape:", c_hist.shape)
    print("c_future shape:", c_future.shape)
    
    return y_hist, y_future, c_hist, c_future

def save_sequences_to_disk(csv_path='demand_calendar_normalized.csv', S=48, future_steps=24, output_dir='tensors'):
    """Creates rolling window tensors from CSV and saves them to disk."""
    import os
    
    y_hist, y_future, c_hist, c_future = create_sequences_from_csv(csv_path, S, future_steps)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving tensors")
    torch.save(y_hist,   os.path.join(output_dir, 'y_hist.pt'))
    torch.save(y_future, os.path.join(output_dir, 'y_future.pt'))
    torch.save(c_hist,   os.path.join(output_dir, 'c_hist.pt'))
    torch.save(c_future, os.path.join(output_dir, 'c_future.pt'))
    
    print(f"Tensors saved to '{output_dir}/'")
    print(f"  y_hist.pt   → {y_hist.shape}")
    print(f"  y_future.pt → {y_future.shape}")
    print(f"  c_hist.pt   → {c_hist.shape}")
    print(f"  c_future.pt → {c_future.shape}")

if __name__ == "__main__":
    save_sequences_to_disk()