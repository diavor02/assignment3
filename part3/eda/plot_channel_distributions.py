"""
plot_channel_distributions.py

Creates exploratory plots for weather channels from hourly spatial stats.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.read_csv("results/hourly_spatial_stats.csv", parse_dates=["timestamp"])
    cols = [c for c in stats.columns if c.endswith("_mean")]
    if not cols:
        print("No *_mean columns found in results/hourly_spatial_stats.csv")
        return

    n = min(4, len(cols))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols[:n]):
        ax.plot(stats["timestamp"], stats[col], linewidth=0.7)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    out_path = out_dir / "fig1_channel_distributions.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
