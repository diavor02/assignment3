import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from model import RNNEnergyForecastModel
from datasets import get_dataloader, DemandTimeDataset

from helper import (
    RegressionMetrics,
    best_checkpoint_path,
    epoch_checkpoint_path,
    format_metric_block,
    latest_checkpoint_path,
    load_checkpoint,
    register_active_run,
    resolve_run_dir,
    save_training_checkpoint,
)

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

    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", BATCH_SIZE))
    num_epochs = int(os.getenv("TRAIN_EPOCHS", EPOCHS))
    requested_run_dir = os.getenv("TRAIN_RUN_DIR")
    requested_resume_ckpt = os.getenv("TRAIN_RESUME_CKPT", "").strip()
    
    tabular_ds = DemandTimeDataset(S=168, future_steps=24)
    
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

    run_dir = resolve_run_dir("rnn", requested_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    register_active_run("rnn", run_dir)

    resume_path = None
    if requested_resume_ckpt:
        resume_path = Path(requested_resume_ckpt)
    else:
        candidate = latest_checkpoint_path(run_dir)
        if candidate.exists():
            resume_path = candidate

    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0

    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(resume_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        print(f"Resuming from checkpoint: {resume_path} (next epoch {start_epoch})")
    elif requested_resume_ckpt:
        print(f"Requested resume checkpoint not found: {resume_path}. Starting fresh.")
    else:
        print(f"No existing checkpoint found in {run_dir}. Starting fresh.")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Training Loop
    # ─────────────────────────────────────────────────────────────────────────────
    train_loader = get_dataloader(batch_size=batch_size)
    val_loader   = get_dataloader(batch_size=batch_size, is_train=False)
    
    if start_epoch > num_epochs:
        print(f"Requested {num_epochs} epochs, but checkpoint already reached epoch {start_epoch - 1}. Nothing to do.")
        return

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # ─────────────────────────────────────────────────────────────────
        # --- TRAINING PHASE ---
        # ─────────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        train_metrics = RegressionMetrics()

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
            train_metrics.update(predictions, targets)

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
        train_stats = train_metrics.compute()

        # ─────────────────────────────────────────────────────────────────
        # --- VALIDATION PHASE ---
        # ─────────────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        val_metrics = RegressionMetrics()

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
                val_metrics.update(predictions, targets)

        avg_val_loss   = total_val_loss / len(val_loader.dataset)
        val_stats = val_metrics.compute()
        epoch_duration = time.time() - epoch_start
        
        # ── Compute gap to detect overfitting ─────────────────────────────
        overfit_gap = avg_val_loss - avg_train_loss

        # ── Per-epoch summary ─────────────────────────────────────────────
        print(
            f"Epoch {epoch:>{len(str(num_epochs))}}/{num_epochs} "
            f"│ train_loss={avg_train_loss:.4f} "
            f"│ train {format_metric_block(train_stats)} "
            f"│ val_loss={avg_val_loss:.4f} "
            f"│ val {format_metric_block(val_stats)} "
            f"│ gap={overfit_gap:+.4f} "          # positive = overfitting, negative = underfitting
            f"│ lr={optimizer.param_groups[0]['lr']:.2e} "
            f"│ {epoch_duration:.1f}s"
        )

        checkpoint_config = {
            "model_kind": "rnn",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 1e-4,
            "n_zones": N_ZONES,
            "n_weather_vars": N_WEATHER_VARS,
            "history_length": 168,
            "horizon": horizon,
            "device": str(DEVICE),
            "run_dir": str(run_dir),
        }

        epoch_ckpt = epoch_checkpoint_path(run_dir, epoch)
        latest_ckpt = latest_checkpoint_path(run_dir)
        is_best = avg_val_loss < best_val_loss

        # ── Track best loss ───────────────────────────────────────────────
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            print(f"  ✓ New best checkpoint saved to: {best_checkpoint_path(run_dir)}")

        # ── Save model every epoch ─────────────────────────────────────────
        save_training_checkpoint(
            epoch_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            run_dir=run_dir,
            model_kind="rnn",
            config=checkpoint_config,
            train_metrics={**train_stats, "loss": avg_train_loss},
            val_metrics={**val_stats, "loss": avg_val_loss},
            is_best=is_best,
        )
        save_training_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            run_dir=run_dir,
            model_kind="rnn",
            config=checkpoint_config,
            train_metrics={**train_stats, "loss": avg_train_loss},
            val_metrics={**val_stats, "loss": avg_val_loss},
            is_best=False,
        )
        print(f"  ✓ Epoch checkpoint saved to: {epoch_ckpt}")
        print(f"  ✓ Latest checkpoint updated at: {latest_ckpt}\n")

    print(f"Training complete. Best model performance was val_loss = {best_val_loss:.4f} at epoch {best_epoch}.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    train()