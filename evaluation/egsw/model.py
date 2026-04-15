"""
model.py — EvaluationWrapper for EnergyForecastModel

Entry point for evaluate.py:
    get_model(metadata: dict) -> EvaluationWrapper

The wrapper is fully constructable from two files that must sit alongside
this model.py in the evaluation model directory:
    config.json       — training config (model section is the key part)
    best_weights.pt   — state_dict produced by the training loop

The evaluation model directory is also added to sys.path by the evaluator,
so every sibling .py module (energy_forecast_model, spatial_encoder, etc.)
is importable directly.  If this file lives somewhere else, set the
optional "source_root" key at the top level of config.json to the directory
that contains those source files.

Interface
─────────
  adapt_inputs(history_weather, history_energy, future_weather, future_time)
      Converts the evaluator's 4-tuple into the 5-tuple expected by
      EnergyForecastModel.forward():
          history_weather : (B, 168, 450, 449, 7)  → sliced to (B, S, ...)
          history_energy  : (B, 168, n_zones)       → sliced to (B, S, ...)
          future_weather  : (B, 24, 450, 449, 7)    → passed through
          future_time     : (B, 24) int64            → used to build calendars
      Calendar features are computed from future_time (history timestamps are
      back-derived as future_time[b, 0] - S, ..., future_time[b, 0] - 1).

  forward(hist_weather, hist_energy, hist_calendar, fut_weather, fut_calendar)
      Calls the inner EnergyForecastModel and denormalizes its output from
      z-score normalized space back to raw MWh.

Cropping fix
────────────
Training used pre-cropped tight images, so the checkpoint's SpatialCropEncoder
buffers hold an identity crop (y: 0→288, x: 0→179) valid only for those
288×179 inputs.  At evaluation time the raw images are 450×449.

If config["model"]["spatial_cnn_args"]["real_crop_path"] is set, that JSON's
{y_min, y_max, x_min, x_max} values are written into the encoder's buffers
after loading the checkpoint, so the full-resolution images are cropped
correctly (to the same 288×179 region, but from the right place).
"""

import json
import sys
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

# ── sys.path: make sure sibling source modules are importable ─────────────────
# _DIR is checked first so a local copy of any file takes priority.
# The hardcoded project path is a fallback for environments where the source
# files have not been copied alongside model.py; if the path is inaccessible
# Python silently skips it and the local copy is used instead.
_DIR         = Path(__file__).parent
_PROJECT_DIR = "/cluster/tufts/c26sp1cs0137/swebbe01/CS137_DNN_Final_Project"

for _p in (str(_DIR), _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energy_forecast_model import EnergyForecastModel
from calendar_features import calendar_features


# ─────────────────────────────────────────────────────────────────────────────

class EvaluationWrapper(nn.Module):
    """
    Thin evaluation wrapper around EnergyForecastModel.

    Args:
        config:       Full training config dict.
        weights_path: Path to best_weights.pt state_dict.
    """

    def __init__(self, config: dict, weights_path: str):
        super().__init__()

        model_cfg = config["model"]

        # ── Build inner model ─────────────────────────────────────────────────
        # The paths in spatial_cnn_args / tabular_encoder_args are used only to
        # initialise buffers; they are immediately overwritten by load_state_dict.
        self.inner = EnergyForecastModel(**model_cfg)

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.inner.load_state_dict(state)

        # ── Cropping fix ──────────────────────────────────────────────────────
        # If the model was trained on pre-cropped images the checkpoint holds
        # an identity crop (e.g. y:0→288, x:0→179).  real_crop_path points to
        # the tight_crop.json whose indices address the full 450×449 images so
        # SpatialCropEncoder correctly extracts the right region at eval time.
        real_crop_path = config.get("training", {}).get("real_crop_path")
        if real_crop_path is not None:
            with open(real_crop_path) as f:
                crop = json.load(f)
            enc = self.inner.spatial_encoder
            enc.y_min.fill_(crop["y_min"])
            enc.y_max.fill_(crop["y_max"])
            enc.x_min.fill_(crop["x_min"])
            enc.x_max.fill_(crop["x_max"])

        self.S = model_cfg["S"]

        # Move everything (including the patched buffers) to the best device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _denorm(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Undo z-score normalization: pred_norm * (std + eps) + mean."""
        means = self.inner.tabular_encoder.means   # (n_zones,)
        stds  = self.inner.tabular_encoder.stds    # (n_zones,)
        return pred_norm * (stds + 1e-6) + means   # (B, 24, n_zones)

    # ── Evaluator interface ────────────────────────────────────────────────────

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,   # (B, 168, 450, 449, 7)
        history_energy:  torch.Tensor,   # (B, 168, n_zones)
        future_weather:  torch.Tensor,   # (B, 24, 450, 449, 7)
        future_time:     torch.Tensor,   # (B, 24) int64, hours since Unix epoch
    ):
        """
        Adapt the evaluator's 4-tuple into the 5-tuple expected by forward().

        Returns:
            (hist_weather, hist_energy, hist_calendar, fut_weather, fut_calendar)
            All tensors on the model's device.
        """
        B = history_weather.shape[0]
        S = self.S

        # ── Calendar features ─────────────────────────────────────────────────
        # future_time[b, 0] is the first prediction hour for sample b.
        # History runs [future_time[b,0] - S, ..., future_time[b,0] - 1].
        hist_cal_batches = []
        fut_cal_batches  = []

        for b in range(B):
            fut_hrs   = future_time[b].tolist()          # list of 24 ints
            hist_start = fut_hrs[0] - S
            hist_hrs   = range(hist_start, hist_start + S)

            hist_cal = torch.stack([
                calendar_features(pd.Timestamp(int(h), unit="h"))
                for h in hist_hrs
            ])                                           # (S, CAL_DIM)

            fut_cal = torch.stack([
                calendar_features(pd.Timestamp(int(h), unit="h"))
                for h in fut_hrs
            ])                                           # (24, CAL_DIM)

            hist_cal_batches.append(hist_cal)
            fut_cal_batches.append(fut_cal)

        hist_calendar = torch.stack(hist_cal_batches)    # (B, S, CAL_DIM)
        fut_calendar  = torch.stack(fut_cal_batches)     # (B, 24, CAL_DIM)

        # ── Slice history to model's lookback window ──────────────────────────
        # The evaluator always provides 168 h of history; the model may use S ≤ 168.
        hist_weather = history_weather[:, -S:, :, :, :]  # (B, S, 450, 449, 7)
        hist_energy  = history_energy[:,  -S:, :]         # (B, S, n_zones)

        # ── Move to model device ──────────────────────────────────────────────
        dev = self._device()
        return (
            hist_weather.to(dev),
            hist_energy.to(dev),
            hist_calendar.to(dev),
            future_weather.to(dev),
            fut_calendar.to(dev),
        )

    def forward(
        self,
        hist_weather:  torch.Tensor,   # (B, S, H, W, 7)
        hist_energy:   torch.Tensor,   # (B, S, n_zones)
        hist_calendar: torch.Tensor,   # (B, S, CAL_DIM)
        fut_weather:   torch.Tensor,   # (B, 24, H, W, 7)
        fut_calendar:  torch.Tensor,   # (B, 24, CAL_DIM)
    ) -> torch.Tensor:                 # (B, 24, n_zones)  raw MWh
        pred_norm = self.inner(
            hist_weather, hist_energy, hist_calendar,
            fut_weather,  fut_calendar,
        )
        return self._denorm(pred_norm)


# ── Entry point ───────────────────────────────────────────────────────────────

def get_model(
    metadata:     dict,
    config_path:  str = None,
    weights_path: str = None,
) -> EvaluationWrapper:
    """
    Called by evaluate.py.  Loads config.json and best_weights.pt from the
    same directory as this model.py file unless explicit paths are provided.

    Args:
        metadata:     Passed in by the evaluator (zone names, n_zones, etc.).
        config_path:  Path to config JSON.  Defaults to config.json alongside
                      this model.py.
        weights_path: Path to weights file.  Defaults to best_weights.pt
                      alongside this model.py.

    If config contains a top-level "source_root" key, that directory is
    prepended to sys.path so sibling source modules can be imported even when
    model.py lives in a separate evaluation sub-folder.
    """
    model_dir = Path(__file__).parent

    if config_path is None:
        config_path = model_dir / "config.json"
    if weights_path is None:
        weights_path = model_dir / "best_weights.pt"

    with open(config_path) as f:
        config = json.load(f)

    return EvaluationWrapper(config, str(weights_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Instantiate EvaluationWrapper from a config and weights file.")
    parser.add_argument("config_path",  help="Path to config.json")
    parser.add_argument("weights_path", help="Path to best_weights.pt")
    args = parser.parse_args()

    print(f"Loading config  : {args.config_path}")
    print(f"Loading weights : {args.weights_path}")

    model = get_model({}, config_path=args.config_path, weights_path=args.weights_path)
    model.eval()

    print(f"Model created successfully: {model.__class__.__name__}")
    print(f"  S (lookback)  : {model.S}")
    print(f"  Inner type    : {model.inner.__class__.__name__}")
    print(f"  Device        : {model._device()}")
    print(f"  Crop buffers  : y={model.inner.spatial_encoder.y_min.item()}:{model.inner.spatial_encoder.y_max.item()}"
          f"  x={model.inner.spatial_encoder.x_min.item()}:{model.inner.spatial_encoder.x_max.item()}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters    : {total:,}")
