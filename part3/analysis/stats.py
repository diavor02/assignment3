"""
stats.py

Utility statistical helpers for Part 3 analysis.
"""

import numpy as np
from scipy import stats


def bootstrap_ci_mean(values, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(sample.mean())
    means = np.asarray(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def mann_whitney_greater(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan
    stat, p_value = stats.mannwhitneyu(x, y, alternative="greater")
    return float(stat), float(p_value)
