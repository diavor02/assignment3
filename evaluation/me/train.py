import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from model import EnergyForecastModel, SpatialCNN, get_model

# NOTE: Assuming your model classes (SpatialCNN, EnergyForecastModel) 
# and the get_model factory are imported here. For example:
# from your_model_file import EnergyForecastModel, SpatialCNN, get_model

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_CKPT_PATH = Path("energy_forecast_model.pt")

# Hyperparameters
BATCH_SIZE = 2      # Kept small due to large 450x449 weather inputs
EPOCHS = 5
LEARNING_RATE = 1e-4
N_ZONES = 10
N_WEATHER_VARS = 7
FUTURE_LEN = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dummy Data Generator
# ─────────────────────────────────────────────────────────────────────────────

def get_dummy_batch(batch_size: int, n_zones: int, device: torch.device):
    """
    Generates dummy tensors matching the expected inputs for adapt_inputs().
    """
    # historical window = 168 hours (1 week)
    history_weather = torch.randn(batch_size, 168, 450, 449, N_WEATHER_VARS, device=device)
    # Energy in raw MWh (e.g., ranging from 50 to 500)
    history_energy = torch.rand(batch_size, 168, n_zones, device=device) * 450 + 50 
    
    # future window = 24 hours
    future_weather = torch.randn(batch_size, FUTURE_LEN, 450, 449, N_WEATHER_VARS, device=device)
    
    # Target values we are trying to predict (raw MWh)
    future_energy_target = torch.rand(batch_size, FUTURE_LEN, n_zones, device=device) * 450 + 50
    
    # Time: hours since Unix epoch. Let's pick a random starting hour around year 2023.
    # 465000 hours ~ 53 years since 1970
    start_hour = 465000
    future_time = torch.arange(start_hour, start_hour + FUTURE_LEN, device=device)
    future_time = future_time.unsqueeze(0).expand(batch_size, -1) # (B, 24)

    return history_weather, history_energy, future_weather, future_time, future_energy_target

# ─────────────────────────────────────────────────────────────────────────────
# 2. Training Setup
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # Initialize the model
    metadata = {
        "n_zones": N_ZONES, 
        "n_weather_vars": N_WEATHER_VARS, 
        "future_len": FUTURE_LEN
    }
    
    # Use your factory function (or instantiate directly)
    # model = get_model(metadata) 
    model = EnergyForecastModel(
        n_zones=N_ZONES,
        n_weather_vars=N_WEATHER_VARS,
        history_len=24,
        future_len=FUTURE_LEN,
        grid_size=5,
        d_spatial=128,
        d_model=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(DEVICE)

    # Populate Normalization Stats
    # In reality, compute this across your entire training dataset before training!
    print("Populating dataset normalization statistics...")
    model.energy_mean.data = torch.full((1, 1, N_ZONES), 275.0, device=DEVICE) # Dummy mean
    model.energy_std.data  = torch.full((1, 1, N_ZONES), 125.0, device=DEVICE) # Dummy std
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Using Mean Squared Error for regression (predicting continuous MWh)
    criterion = nn.MSELoss()

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Training Loop
    # ─────────────────────────────────────────────────────────────────────────────
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        optimizer.zero_grad()
        
        # 1. Fetch data
        (hist_w, hist_e, fut_w, fut_t, targets) = get_dummy_batch(BATCH_SIZE, N_ZONES, DEVICE)
        
        # 2. Pre-process inputs through the model's adapter
        hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal = model.adapt_inputs(
            history_weather=hist_w,
            history_energy=hist_e,
            future_weather=fut_w,
            future_time=fut_t
        )
        
        # 3. Forward pass
        predictions = model(
            hist_sp=hist_sp,
            hist_e=hist_e_norm,
            hist_cal=hist_cal,
            fut_sp=fut_sp,
            fut_cal=fut_cal
        )
        
        # 4. Calculate Loss
        # predictions and targets are both in raw MWh shape: (B, F, n_zones)
        loss = criterion(predictions, targets)
        
        # 5. Backward pass & Optimize
        loss.backward()
        
        # Optional but highly recommended for Transformers: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Save Checkpoint
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\nTraining complete. Saving checkpoint...")
    
    # Save the state dictionary to the path expected by your get_model factory
    torch.save(model.state_dict(), _CKPT_PATH)
    print(f"Model saved successfully to: {_CKPT_PATH.resolve()}")

if __name__ == "__main__":
    # Suppress pandas DatetimeIndex warnings if necessary
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    train()