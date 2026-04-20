import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from model import RNNEnergyForecastModel
from datasets import get_dataloader, DemandTimeDataset

# NOTE: Assuming your model classes (CNN, EnergyForecastModel) 
# and the get_model factory are imported here. For example:
# from your_model_file import EnergyForecastModel, CNN, get_model

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Hyperparameters
BATCH_SIZE = 20      # Kept small due to large 450x449 weather inputs
EPOCHS = 20
LEARNING_RATE = 1e-4
N_ZONES = 8
N_WEATHER_VARS = 7
horizon = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Training Setup
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print(f"--- Starting Training on {DEVICE} ---")
    
    tabular_ds = DemandTimeDataset(csv_path="filtered_demand_raw.csv", S=168, future_steps=24)
    
    model = RNNEnergyForecastModel(
        n_zones=N_ZONES,
        n_weather_vars=N_WEATHER_VARS,
        S=168,
        horizon=horizon,
        grid_size=5,
        d_spatial=128,
        d_model=256,
        n_layers=4,
        dropout=0.1,
    ).to(DEVICE)

    # 3. Populate REAL Normalization Stats from the tabular dataset
    print("Populating dataset normalization statistics...")
    
    # tabular_ds.energy has shape (Total_Hours, n_zones)
    # Calculate mean and std across the time dimension (dim=0)
    true_mean = tabular_ds.energy.mean(dim=0)  # Shape: (n_zones,)
    true_std  = tabular_ds.energy.std(dim=0)   # Shape: (n_zones,)
    
    # Reshape to (1, 1, n_zones) to match the buffer dimensions and move to GPU
    model.energy_mean.data = true_mean.view(1, 1, -1).to(DEVICE)
    model.energy_std.data  = true_std.view(1, 1, -1).to(DEVICE)

    print(f"Stats loaded. Example Zone 0 (ME) - Mean: {true_mean[0]:.2f}, Std: {true_std[0]:.2f}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Initialize the best loss tracker to infinity
    best_val_loss = float('inf')
    best_epoch = 0

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Training Loop
    # ─────────────────────────────────────────────────────────────────────────────
    train_loader = get_dataloader()
    val_loader   = get_dataloader(batch_size=2, is_train=False)
    
    # Set to strictly 2 epochs
    num_epochs = 2
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ─────────────────────────────────────────────────────────────────
        # --- TRAINING PHASE ---
        # ─────────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            hist_w, hist_e, fut_w, fut_t, targets = batch

            hist_w = hist_w.to(DEVICE)
            hist_e = hist_e.to(DEVICE)
            fut_w  = fut_w.to(DEVICE)
            fut_t  = fut_t.to(DEVICE)
            targets = targets.to(DEVICE)
            
            hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal = model.adapt_inputs(
                history_weather=hist_w, history_energy=hist_e,
                future_weather=fut_w, future_time=fut_t
            )

            predictions = model(hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal)
            loss = criterion(predictions, targets)

            loss.backward()
            
            # Track gradient norm BEFORE clipping — a sudden spike signals instability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item() * targets.size(0)

            # ── Intra-epoch progress (every 10 batches) ───────────────────
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(
                    f"  Epoch {epoch}/{num_epochs} "
                    f"[{batch_idx:>{len(str(len(train_loader)))}}/{len(train_loader)}] "
                    f"│ batch_loss={loss.item():.4f} "
                    f"│ grad_norm={grad_norm:.4f}",
                    end="\r"
                )

        print()  # newline after the \r progress line
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ─────────────────────────────────────────────────────────────────
        # --- VALIDATION PHASE ---
        # ─────────────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                hist_w, hist_e, fut_w, fut_t, targets = batch

                hist_w = hist_w.to(DEVICE)
                hist_e = hist_e.to(DEVICE)
                fut_w  = fut_w.to(DEVICE)
                fut_t  = fut_t.to(DEVICE)
                targets = targets.to(DEVICE)

                hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal = model.adapt_inputs(
                    history_weather=hist_w, history_energy=hist_e,
                    future_weather=fut_w, future_time=fut_t
                )

                predictions = model(hist_sp, hist_e_norm, hist_cal, fut_sp, fut_cal)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item() * targets.size(0)

        avg_val_loss   = total_val_loss / len(val_loader.dataset)
        epoch_duration = time.time() - epoch_start
        
        # ── Compute gap to detect overfitting ─────────────────────────────
        overfit_gap = avg_val_loss - avg_train_loss

        # ── Per-epoch summary ─────────────────────────────────────────────
        print(
            f"Epoch {epoch:>{len(str(num_epochs))}}/{num_epochs} "
            f"│ train={avg_train_loss:.4f} "
            f"│ val={avg_val_loss:.4f} "
            f"│ gap={overfit_gap:+.4f} "          # positive = overfitting, negative = underfitting
            f"│ lr={optimizer.param_groups[0]['lr']:.2e} "
            f"│ {epoch_duration:.1f}s"
        )

        # ── Track best loss ───────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        # ── Save model every epoch ─────────────────────────────────────────
        save_name = f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"  ✓ Model saved for epoch {epoch} to: {save_name}\n")

    print(f"Training complete. Best model performance was val_loss = {best_val_loss:.4f} at epoch {best_epoch}.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    train()