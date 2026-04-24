import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/"
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = "v1"
CHECKPOINT_ROOT = PROJECT_ROOT / "evaluation" / "subhanga-additions" / "checkpoints" / CHECKPOINT_VERSION


def add_calendar_features(df):
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    df["hour"] = df["timestamp_utc"].dt.hour
    df["dayofweek"] = df["timestamp_utc"].dt.dayofweek
    df["dayofyear"] = df["timestamp_utc"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df.drop(columns=['hour', 'dayofweek', 'dayofyear'], inplace=True)

    return df

def create_sequences(demand, calendar, S=48, horizon=24):
    X_demand = []
    X_calendar = []

    T_total = len(demand)

    for t in range(T_total - (S + horizon)):
        # past demand
        d_seq = demand[t : t + S]                      # (S, Z)

        # calendar for past + future
        c_seq = calendar[t : t + S + horizon]          # (S+24, C_cal)

        X_demand.append(d_seq)
        X_calendar.append(c_seq)

    return np.array(X_demand), np.array(X_calendar)

def extract_year(s: str) -> str:
    return s[2:6]


def build_file_list(path, start="2019010100", end="2024123123"):
    """
    Generate filenames from X_2019010100.pt to X_2024123123.pt
    Format: X_YYYYMMDDХH.pt (year, month, day, hour)
    """
    # Parse start and end
    start_dt = datetime.strptime(start, "%Y%m%d%H")
    end_dt = datetime.strptime(end, "%Y%m%d%H")

    path += "2019/"
    assert path == PATH + "weather_data/2019/"
    
    filenames = []
    current = start_dt
    
    while current <= end_dt:
        filename = f"X_{current.strftime('%Y%m%d%H')}.pt"
        filename = path + filename
        filenames.append(filename)
        current += timedelta(hours=1)
    
    return filenames

def assert_no_empty_values(df: pd.DataFrame) -> None:
    """
    Asserts that there are no empty values in a DataFrame.
    
    Empty values include:
    - NaN / None
    - Empty strings ""
    - Whitespace-only strings "   "

    Raises AssertionError with a detailed report if any are found.
    """
    issues = {}

    # Check for NaN / None
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        for col, count in nan_cols.items():
            issues.setdefault(col, []).append(f"{count} NaN/None value(s)")

    # Check for empty or whitespace-only strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        mask = df[col].astype(str).str.strip() == ""
        count = mask.sum()
        if count > 0:
            issues.setdefault(col, []).append(f"{count} empty/whitespace string(s)")

    if issues:
        report_lines = ["DataFrame contains empty values:"]
        for col, problems in issues.items():
            report_lines.append(f"  Column '{col}': {', '.join(problems)}")
        raise AssertionError("\n".join(report_lines))

def mape_loss(y_pred, y_true, eps=1e-5):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps)))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_model_dir(model_kind: str) -> Path:
    return CHECKPOINT_ROOT / model_kind


def checkpoint_runs_dir(model_kind: str) -> Path:
    return checkpoint_model_dir(model_kind) / "runs"


def checkpoint_active_file(model_kind: str) -> Path:
    return checkpoint_model_dir(model_kind) / "active_run.txt"


def resolve_run_dir(model_kind: str, run_dir: str | None = None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()

    active_file = checkpoint_active_file(model_kind)
    if active_file.exists():
        candidate = Path(active_file.read_text().strip())
        if candidate.exists():
            return candidate

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_id = os.getenv("SLURM_JOB_ID")
    suffix = f"job-{job_id}" if job_id else f"run-{stamp}"
    return checkpoint_runs_dir(model_kind) / suffix


def register_active_run(model_kind: str, run_dir: Path) -> None:
    _ensure_dir(checkpoint_model_dir(model_kind))
    checkpoint_active_file(model_kind).write_text(str(run_dir.resolve()) + "\n")


def latest_checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "latest.pt"


def best_checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "best.pt"


def epoch_checkpoint_path(run_dir: str | Path, epoch: int) -> Path:
    return Path(run_dir) / f"epoch_{epoch:03d}.pt"


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path):
    return torch.load(Path(path), map_location="cpu")


def resolve_active_run_dir(model_kind: str) -> Path | None:
    active_file = checkpoint_active_file(model_kind)
    if not active_file.exists():
        return None

    candidate = Path(active_file.read_text().strip())
    if candidate.exists():
        return candidate
    return None


@dataclass
class RegressionMetrics:
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    target_sum: float = 0.0
    target_sum_squares: float = 0.0
    count: int = 0
    mape_sum: float = 0.0
    mape_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, include_mape: bool = False) -> None:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        diff = pred - target

        self.sum_squared_error += diff.square().sum().item()
        self.sum_absolute_error += diff.abs().sum().item()
        self.target_sum += target.sum().item()
        self.target_sum_squares += target.square().sum().item()
        self.count += target.numel()

        if include_mape:
            mask = target != 0
            if mask.any():
                self.mape_sum += (diff.abs()[mask] / target.abs()[mask]).sum().item()
                self.mape_count += int(mask.sum().item())

    def compute(self) -> dict:
        if self.count == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "rse": float("nan")}

        mse = self.sum_squared_error / self.count
        target_variance_sum = self.target_sum_squares - (self.target_sum ** 2) / self.count
        target_variance_sum = max(target_variance_sum, 1e-12)

        metrics = {
            "rmse": math.sqrt(mse),
            "mae": self.sum_absolute_error / self.count,
            "rse": math.sqrt(self.sum_squared_error / target_variance_sum),
        }

        if self.mape_count > 0:
            metrics["mape"] = (self.mape_sum / self.mape_count) * 100.0

        return metrics


def format_metric_block(metrics: dict, include_mape: bool = False) -> str:
    parts = [
        f"rmse={metrics['rmse']:.4f}",
        f"mae={metrics['mae']:.4f}",
        f"rse={metrics['rse']:.4f}",
    ]
    if include_mape and "mape" in metrics:
        parts.append(f"mape={metrics['mape']:.2f}%")
    return " │ ".join(parts)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    run_dir: str | Path,
    model_kind: str,
    config: dict,
    train_metrics: dict,
    val_metrics: dict,
    is_best: bool,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "run_dir": str(Path(run_dir).resolve()),
        "model_kind": model_kind,
        "config": config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "saved_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_checkpoint(path, payload)

    if is_best:
        save_checkpoint(best_checkpoint_path(run_dir), payload)