#!/usr/bin/env python3
"""
Training script for the Baseline CNN-Transformer day-ahead energy forecasting model.

This script demonstrates the optimization loop:
1. Loads data via a PyTorch DataLoader.
2. Performs forward passes to get predictions.
3. Calculates the loss (error) against true targets.
4. Uses backpropagation to update model weights.
5. Saves the best model checkpoint for evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import the model from your existing file
from model import EnergyForecastModel, _CKPT_PATH

# =============================================================================
# 1. Dataset Placeholder
# =============================================================================
class EnergyWeatherDataset(Dataset):
    """
    A placeholder PyTorch Dataset. 
    In your real code, this would handle the sliding window logic over your 
    target_energy_zonal_*.csv and X_*.pt weather tensors.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.n_samples = 1000 if is_train else 200 # Dummy lengths
        
        # Ensure dimensions match what the model expects:
        self.history_len = 168
        self.future_len = 24
        self.n_zones = 8      # Assuming 10 energy zones for this example
        self.weather_shape = (450, 449, 7)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # ---------------------------------------------------------------------
        # REPLACE THIS BLOCK with your actual data loading logic.
        # This just generates random tensors of the correct shape.
        # ---------------------------------------------------------------------
        hist_weather = torch.randn(self.history_len, *self.weather_shape)
        hist_energy  = torch.randn(self.history_len, self.n_zones)
        fut_weather  = torch.randn(self.future_len, *self.weather_shape)
        
        # future_time: int64 hours since Unix epoch (dummy sequential hours)
        base_hour = 400000 + idx
        fut_time = torch.arange(base_hour, base_hour + self.future_len, dtype=torch.int64)
        
        # The actual ground truth we want the model to predict
        target_energy = torch.randn(self.future_len, self.n_zones) 

        return hist_weather, hist_energy, fut_weather, fut_time, target_energy

# =============================================================================
# 2. Main Training Function
# =============================================================================
def train_model():
    # --- Configuration ---
    num_epochs = 20
    batch_size = 8
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Data Loaders ---
    train_dataset = EnergyWeatherDataset(is_train=True)
    val_dataset   = EnergyWeatherDataset(is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize Model ---
    model = EnergyForecastModel(
        n_zones=10,             # Match your dataset
        n_weather_vars=7,
        future_len=24
    ).to(device)

    # --- Optimizer and Loss Function ---
    # MSE is standard for regression, though MAE (L1Loss) is closer to the MAPE metric
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float('inf')

    # =========================================================================
    # 3. The Epoch Loop
    # =========================================================================
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # ---------------------------------------------------------------------
        # TRAINING PHASE
        # ---------------------------------------------------------------------
        model.train() # <--- CRITICAL: Turns on learning behaviors (Dropout, BatchNorm tracking)
        train_loss = 0.0

        for batch_idx, (h_w, h_e, f_w, f_t, targets) in enumerate(train_loader):
            # Move data to GPU
            h_w, h_e, f_w, f_t = h_w.to(device), h_e.to(device), f_w.to(device), f_t.to(device)
            targets = targets.to(device)

            # THE 5 STEPS OF MACHINE LEARNING OPTIMIZATION:
            # 1. Clear old gradients from the last batch
            optimizer.zero_grad() 

            # 2. Forward Pass: adapt inputs and make a prediction
            adapted_inputs = model.adapt_inputs(h_w, h_e, f_w, f_t)
            predictions = model(*adapted_inputs)

            # 3. Calculate Loss: How wrong was the prediction?
            loss = criterion(predictions, targets)

            # 4. Backward Pass: Calculate the gradients (the math maps for improvement)
            loss.backward()

            # 5. Optimize: Update the model weights based on the gradients
            optimizer.step()

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:03d} | Train Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ---------------------------------------------------------------------
        # VALIDATION PHASE
        # ---------------------------------------------------------------------
        model.eval() # <--- CRITICAL: Turns OFF learning behaviors (locks weights)
        val_loss = 0.0

        with torch.no_grad(): # <--- CRITICAL: Disables gradient tracking (saves memory)
            for h_w, h_e, f_w, f_t, targets in val_loader:
                h_w, h_e, f_w, f_t = h_w.to(device), h_e.to(device), f_w.to(device), f_t.to(device)
                targets = targets.to(device)

                adapted_inputs = model.adapt_inputs(h_w, h_e, f_w, f_t)
                predictions = model(*adapted_inputs)
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---------------------------------------------------------------------
        # CHECKPOINTING
        # ---------------------------------------------------------------------
        # If the model performed better on validation data than ever before, save it!
        # This is the file `evaluate.py` will look for.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), _CKPT_PATH)
            print(f"  --> Validation loss improved! Saved checkpoint to {_CKPT_PATH}")

if __name__ == "__main__":
    train_model()