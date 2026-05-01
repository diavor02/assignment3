# Energy Demand Forecasting with Weather Context

A multi-part deep learning project for spatiotemporal energy demand forecasting with focus on extreme weather event analysis.

## Overview

This repository implements an end-to-end forecasting pipeline across three parts:

- **Part 1–2**: Baseline model development using spatiotemporal transformers
- **Part 3**: Distribution shift analysis during extreme weather conditions

## Part 1–2: Baseline Model Development

### Problem Formulation

**Input:**
- Historical demand time series for 8 electricity zones (24 hours × 8 zones)
- Calendar features: hour-of-day, day-of-week, day-of-year (encoded as sine/cosine)
- High-resolution weather data: 450×449 spatial grid with 7 weather channels (hourly)
- Lookback window: 48 hours of historical data (S=48)

**Output:**
- 24-hour ahead forecast (horizon=24) for each of 8 zones

**Metric:** Mean Absolute Percentage Error (MAPE) with epsilon=1.0 for numerical stability.

### Data Processing

#### Calendar Features (`add_calendar_features`)
Transforms timestamp information into learnable temporal embeddings:
- Hourly sine/cosine encoding (24-hour period)
- Day-of-week sine/cosine encoding (7-day period)
- Day-of-year sine/cosine encoding (365-day period)

#### Sequence Construction
For each training/evaluation window:
- Stack S=48 consecutive historical demand observations: shape (48, 8)
- Stack S+horizon=72 consecutive calendar feature vectors: shape (72, C_cal)
- Maintain temporal alignment for encoder input

### Architecture

#### Spatial Feature Extraction: WeatherCNN

Processes raw weather tensors (450×449×7) through aggressive downsampling:

```
Input: (B, 7, 450, 449)
Conv2d(7 → 32, stride=2)  →  (B, 32, 225, 225)
Conv2d(32 → 64, stride=2) →  (B, 64, 113, 113)
Conv2d(64 → 128, stride=2) → (B, 128, 57, 57)
Conv2d(128 → 128, stride=2) → (B, 128, 29, 29)
Conv2d(128 → 128, stride=2) → (B, 128, 15, 15)
AdaptiveAvgPool2d(4×4)   →  (B, 128, 4, 4)
Conv2d(128 → d_model)    →  (B, 64, 4, 4)
Output: 64-dim tokens, spatial grid = 4×4 = 16 patches
```

#### Spatial Token Extraction: SpatialTokenExtractor

- **Input:** (B, T, H, W, C) weather sequences
- **Processing:** CNN extracts spatial patches at each timestep
- **Token embedding:** Add learnable spatial position embeddings and temporal embeddings
- **Output:** (B, T, P, d_model) = (B, 72, 16, 64) token sequence

#### Tabular Embedding Layer

Fuses demand and calendar information:
```
demand_proj: (B, 48, 8) → (B, 48, 64)
calendar_proj: (B, 72, C_cal) → (B, 72, 64)
Fusion: Concatenate + MLP projection
Output: (B, 72, 64) tabular tokens
```

#### Sequence Assembly

For each timestep t in the S+24 horizon:
- Concatenate P=16 spatial tokens + 1 tabular token = 17 tokens per timestep
- Total flattened sequence: (B, (72)×(17), 64) = (B, 1224, 64)

#### Transformer Encoder + Prediction Head

```
TransformerEncoder:
  - num_layers: 4
  - nhead: 8
  - d_model: 64
  - batch_first: True
  
Input: (B, 1224, 64) full token sequence
Processing: Self-attention over all spatial and tabular tokens
Output: (B, 1224, 64) contextual token representations

Prediction Head:
  - Extract future timesteps (forecast horizon, indices S:S+24)
  - Take final token per timestep (tabular token)
  - Linear projection: (B, 24, 64) → (B, 24, 8)
  - Output: 24-hour predictions for 8 zones
```

### Training

- **Optimizer:** AdamW
- **Loss:** MAPE with epsilon regularization
- **Batch size:** Standard (configurable via SLURM)
- **Checkpointing:** Best and per-epoch models saved to `evaluation/subhanga-additions/checkpoints/v2/`
- **Model variants:** Transformer (best) and RNN baseline both trained

**Best Checkpoint:**
- Path: `evaluation/subhanga-additions/checkpoints/v2/transformer/runs/job-483062/best.pt`
- Epoch: 10
- Validation Loss: 3663.27

---

## Part 3: Extreme Weather Event Analysis

### Motivation

Energy grid operations are particularly vulnerable during extreme weather. Understanding how model performance varies across weather regimes is critical for reliable forecasting. This analysis quantifies performance degradation and identifies zone-specific vulnerabilities.

### Part 3.1: Weather Channel EDA

**Script:** `part3/eda/explore_weather_channels.py`

Identifies physical meaning of the 7 weather channels through statistical analysis:

1. **Channel Identification:**
   - Load 200 evenly-spaced weather tensors (2019–2023)
   - Compute per-channel statistics: mean, std, min, max, 1st/99th percentile
   - Compute temporal variance (how much does each channel fluctuate hour-to-hour?)

2. **Channel Labeling:**
   - CH00: Temperature (mean=281.38 K, temporal_var=87.16) ✓
   - CH02: Wind speed (mean=7.81 m/s, range=0.02–38.16) ✓
   - CH01, CH03–CH06: Secondary variables (humidity, pressure, etc.)

3. **Output:** `results/channel_stats.csv` containing per-channel summary statistics

### Part 3.2: Spatial Aggregation

**Script:** `part3/eda/compute_spatial_means.py`

Process all 52,608 weather tensors (2019–2023) to extract hourly spatial statistics:

1. **Parallelization:** 8-worker ProcessPoolExecutor with 32-sample chunks
2. **Per-channel computation:** For each hour, compute:
   - Spatial mean, std, p05, p95, min, max
   - Result: Single scalar value per channel per hour
3. **Output:** `results/hourly_spatial_stats.csv` (52,608 rows × 42 columns)

**Execution time:** ~7.5 minutes for full dataset

### Part 3.3: Extreme Event Identification

**Script:** `part3/eda/identify_extreme_events.py`

Label each hour with weather regime and detect multi-hour events:

1. **Season-Aware Thresholds:**
   - Summer heat: temperature > 95th percentile (summer months)
   - Winter cold: temperature < 5th percentile (winter months)
   - High wind: wind speed > 95th percentile (year-round)

2. **Event Bucketing:**
   - Priority: winter_storm > extreme_heat > extreme_cold > high_wind > normal
   - Consecutive hours in same bucket grouped by event_id

3. **Evaluation Split (2022–2023):**
   - Normal: 607 windows
   - High wind: 68 windows
   - Extreme heat: 28 windows
   - Extreme cold: 22 windows
   - Winter storm: 5 windows (flagged as too sparse for reliable statistics)

4. **Outputs:**
   - `results/extreme_event_catalog.csv`: (52,608 rows) hourly bucket assignments
   - `results/thresholds.csv`: Temperature/wind thresholds used for bucketing

### Part 3.4: Prediction Collection

**Script:** `part3/eval/collect_predictions.py`

Run trained transformer on 730 evaluation windows (2022–2023):

1. **Window Selection:** Midnight-start 24-hour forecast windows
2. **Checkpoint Loading:** Best transformer checkpoint (v2, job-483062)
3. **Weather Caching:** Keep ~200 weather tensors in memory to avoid redundant disk I/O
4. **Inference Loop:** ~25–30 minutes on CPU (2.4–2.8 s/window)
5. **Bucket Assignment:** Map each 24-hour window to most severe weather bucket observed

6. **Outputs:**
   - `results/all_preds.npy`: (730, 24, 8) predictions
   - `results/all_targets.npy`: (730, 24, 8) ground truth
   - `results/window_meta.csv`: Window metadata and bucket assignments

### Part 3.5: Stratified MAPE Analysis

**Script:** `part3/analysis/zone_horizon_breakdown.py`

Compute MAPE stratified by weather bucket, zone, and forecast horizon:

1. **Pivot Tables:**
   - By bucket × zone: 4 buckets × 8 zones + bucket and zone aggregates
   - By bucket × horizon: 4 buckets × 24 horizons
   - By bucket × zone × horizon (full breakdown)

2. **Statistical Significance:**
   - Mann-Whitney U test: Compare each (bucket, zone) pair against normal regime
   - Identify zones with significant performance degradation under extremes

3. **Key Findings:**
   - Overall MAPE by bucket: normal=2.88%, high_wind=3.07%, extreme_cold=2.59%, extreme_heat=2.60%
   - Zone vulnerabilities: CT worst during high_wind/extreme_cold; ME worst during extreme_cold
   - Winter_storm bucket flagged as low-power (n=5) in statistical analysis

4. **Outputs:**
   - `results/stratified_mape_table.csv`: Full breakdown table (48 rows)
   - `results/significance_tests.csv`: Mann-Whitney U p-values and effect sizes

### Part 3.6: Visualization

**Script:** `part3/analysis/plot_results.py`

Generate 5 publication-ready figures:

1. **Figure 1: Channel Distributions**
   - Time series of spatial-mean temperature (2019–2023) colored by weather bucket
   - Seasonal histograms showing temperature distribution by bucket

2. **Figure 2: Extreme Event Timeline**
   - Scatter plot of identified extreme events across 5-year span
   - X-axis: timestamp, Y-axis: event type, color: duration

3. **Figure 3: Stratified MAPE Heatmap**
   - Zone × Bucket heatmap showing MAPE values per cell
   - Matplotlib-based (no seaborn dependency) with text annotations

4. **Figure 4: Horizon Curves**
   - Line plots: MAPE vs. forecast horizon (h=1..24) for each bucket
   - Shows if extreme-weather forecast errors grow differently over horizon

5. **Figure 5: Zone Vulnerability Ranking**
   - Horizontal bar chart: vulnerability ratio = max_extreme_mape / normal_mape per zone
   - Identifies which zones are most sensitive to weather extremes

**Outputs:** 5 PNG figures in `results/figures/`

### Part 3.7: LaTeX Writeup

**Output:** `results/content/part3_draft.tex`

Self-contained LaTeX section including:
- Methodology description
- Event bucket definitions and counts
- MAPE table with all stratifications
- Zone-specific vulnerability analysis
- Figure references and captions
- Discussion of grid operation implications
- Limitations and caveats (e.g., sparse winter_storm data)

Ready to paste into main paper.

---

## Repository Structure

```
.
├── README.md                          # This file
├── helper.py                          # Shared utilities: checkpoint mgmt, calendar features
├── architecture.py                    # Model definitions: WeatherCNN, Transformer
├── main.py                            # Training loop and dataset loading
├── evaluation/
│   ├── me/                            # Initial baseline checkpoint
│   ├── me-part2/                      # Refined baseline checkpoint
│   └── subhanga-additions/            # Final trained models (v2)
│       └── checkpoints/
│           └── v2/
│               ├── transformer/       # Best model (Part 1-2)
│               │   └── runs/job-483062/best.pt
│               └── rnn/               # RNN variant
├── part3/                             # Part 3 independent study
│   ├── eda/
│   │   ├── explore_weather_channels.py
│   │   ├── compute_spatial_means.py
│   │   └── identify_extreme_events.py
│   ├── eval/
│   │   └── collect_predictions.py
│   ├── analysis/
│   │   ├── zone_horizon_breakdown.py
│   │   └── plot_results.py
│   └── results/
│       ├── channel_stats.csv
│       ├── hourly_spatial_stats.csv
│       ├── extreme_event_catalog.csv
│       ├── thresholds.csv
│       ├── all_preds.npy
│       ├── all_targets.npy
│       ├── window_meta.csv
│       ├── stratified_mape_table.csv
│       ├── significance_tests.csv
│       ├── figures/
│       │   ├── fig1_channel_distributions.png
│       │   ├── fig2_extreme_event_map.png
│       │   ├── fig3_stratified_mape_heatmap.png
│       │   ├── fig4_horizon_curves.png
│       │   └── fig5_zone_vulnerability_ranking.png
│       └── content/
│           ├── part3_draft.tex
│           └── part3_writeup_notes.md
└── slurm/
    ├── train_baseline.sbatch          # Part 1-2 training
    └── part_three_runner.sbatch       # Part 3 reproducible execution
```

---

## Execution Guide

### Part 1–2: Training Models

**Environment Setup:**
```bash
source /cluster/home/supadh03/cs137/assignment3/.venv/bin/activate
module load class/default cs137/2026spring
```

**Submit Training Job:**
```bash
sbatch slurm/train_baseline.sbatch
```

**Monitor Training:**
```bash
squeue -u $(whoami)
tail -f slurm_logs/a3-train_*.out
```

### Part 3: Reproduce Full Analysis

**Run All Steps (Steps 1–6):**
```bash
sbatch slurm/part_three_runner.sbatch
```

**Run Individual Steps:**
```bash
cd part3

# Step 1: EDA
python eda/explore_weather_channels.py

# Step 2: Spatial aggregation
python eda/compute_spatial_means.py

# Step 3: Event identification
python eda/identify_extreme_events.py

# Step 4: Collect predictions
python eval/collect_predictions.py

# Step 5: Analysis
python analysis/zone_horizon_breakdown.py

# Step 6: Visualization
python analysis/plot_results.py
```

**Outputs:** Results and logs saved to `part3/results/`

---

## Key Dependencies

- **PyTorch** 2.3.0+ (CUDA support)
- **NumPy, Pandas, SciPy**
- **Matplotlib** (for visualization)
- Python 3.10+

**Note:** Seaborn dependency removed in favor of pure Matplotlib for cluster compatibility.

---

## Checkpointing & Reproducibility

### Model Checkpoints

All trained models stored in `evaluation/subhanga-additions/checkpoints/v2/`:

- **Transformer:** `job-483062/best.pt` (recommended for Part 3 inference)
- **RNN:** `job-483063/best.pt` (baseline comparison)

Active run metadata stored in `active_run.txt` for each model type.

### SLURM Logging

Batch jobs produce:
- Standard output/error: `slurm_logs/a3-train_<jobid>.{out,err}`
- Run-specific logs: `part3/results/logs/part3_runner_<jobid>.log`

---

## Performance Metrics (Part 1–2)

| Model | Checkpoint | Epoch | Val Loss |
|-------|-----------|-------|----------|
| Transformer | v2/job-483062 | 10 | 3663.27 |
| RNN | v2/job-483063 | 7 | 11125.99 |

---

## Part 3 Results Summary

**Weather Bucketing (2022–2023 Evaluation):**
- Normal: 607 windows (baseline regime)
- Extreme heat: 28 windows
- Extreme cold: 22 windows
- High wind: 68 windows
- Winter storm: 5 windows (sparse; use with caution)

**MAPE by Weather Regime:**
- Normal: 2.88%
- Extreme heat: 2.60% (slightly better)
- Extreme cold: 2.59% (slightly better)
- High wind: 3.07% (degraded performance)

**Zone Vulnerabilities:**
Significant performance degradation (Mann-Whitney U, α=0.05):
- **Connecticut (CT):** High wind, extreme cold
- **Maine (ME):** Extreme cold, extreme heat
- **NEMA/SEMA:** Cold sensitivity