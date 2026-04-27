"""
zone_horizon_breakdown.py

Loads all_preds.npy, all_targets.npy, window_meta.csv and computes:
1. Overall MAPE per bucket
2. MAPE by bucket x zone
3. MAPE by bucket x forecast horizon h=1..24
4. MAPE by bucket x zone x horizon
5. Statistical significance tests (Mann-Whitney U vs normal)
"""

import numpy as np
import pandas as pd
from scipy import stats

ZONE_NAMES = ["CT", "ME", "NH", "RI", "VT", "WCMA", "NEMA", "SEMA"]


def mape(y_true, y_pred, eps=1.0):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100.0


def compute_mape_by_bucket_zone_horizon(preds, targets, meta):
    buckets = meta["weather_bucket"].values
    rows = []

    for bucket in meta["weather_bucket"].unique():
        mask = buckets == bucket
        p = preds[mask]
        t = targets[mask]
        n = int(mask.sum())

        if n < 10:
            print(f"  SKIP {bucket}: only {n} windows, too few for reliable stats")
            continue

        rows.append({
            "bucket": bucket,
            "zone": "ALL",
            "horizon": "ALL",
            "mape": mape(t, p),
            "n_windows": n,
        })

        z_count = p.shape[-1]
        for z in range(z_count):
            rows.append({
                "bucket": bucket,
                "zone": ZONE_NAMES[z] if z < len(ZONE_NAMES) else f"Z{z}",
                "horizon": "ALL",
                "mape": mape(t[:, :, z], p[:, :, z]),
                "n_windows": n,
            })

        for h in range(24):
            rows.append({
                "bucket": bucket,
                "zone": "ALL",
                "horizon": h + 1,
                "mape": mape(t[:, h, :], p[:, h, :]),
                "n_windows": n,
            })

        for z in range(z_count):
            for h in range(24):
                rows.append({
                    "bucket": bucket,
                    "zone": ZONE_NAMES[z] if z < len(ZONE_NAMES) else f"Z{z}",
                    "horizon": h + 1,
                    "mape": mape(t[:, h, z], p[:, h, z]),
                    "n_windows": n,
                })

    return pd.DataFrame(rows)


def run_significance_tests(preds, targets, meta):
    buckets = meta["weather_bucket"].values
    normal_mask = buckets == "normal"
    results = []

    for bucket in meta["weather_bucket"].unique():
        if bucket == "normal":
            continue
        bucket_mask = buckets == bucket

        for z in range(preds.shape[-1]):
            normal_mapes = np.abs(targets[normal_mask, :, z] - preds[normal_mask, :, z]).mean(axis=1)
            bucket_mapes = np.abs(targets[bucket_mask, :, z] - preds[bucket_mask, :, z]).mean(axis=1)

            if len(bucket_mapes) < 10:
                continue

            stat, p_value = stats.mannwhitneyu(bucket_mapes, normal_mapes, alternative="greater")
            results.append({
                "bucket": bucket,
                "zone": ZONE_NAMES[z] if z < len(ZONE_NAMES) else f"Z{z}",
                "median_normal_mape": float(np.median(normal_mapes)),
                "median_bucket_mape": float(np.median(bucket_mapes)),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "n_bucket": int(len(bucket_mapes)),
                "n_normal": int(len(normal_mapes)),
            })

    return pd.DataFrame(results)


def main():
    preds = np.load("results/all_preds.npy")
    targets = np.load("results/all_targets.npy")
    meta = pd.read_csv("results/window_meta.csv")

    print(f"Total windows: {len(preds)}")
    print(f"Bucket distribution:\n{meta['weather_bucket'].value_counts()}\n")

    results_df = compute_mape_by_bucket_zone_horizon(preds, targets, meta)
    results_df.to_csv("results/stratified_mape_table.csv", index=False)
    print("Saved results/stratified_mape_table.csv")

    sig_df = run_significance_tests(preds, targets, meta)
    sig_df.to_csv("results/significance_tests.csv", index=False)
    print("Saved results/significance_tests.csv")

    print("\n=== TABLE 1: Overall MAPE by Weather Bucket ===")
    overall = results_df[(results_df["zone"] == "ALL") & (results_df["horizon"] == "ALL")]
    print(overall[["bucket", "mape", "n_windows"]].to_string(index=False))

    print("\n=== TABLE 2: Zone MAPE during Extreme Events vs Normal ===")
    zone_table = results_df[(results_df["zone"] != "ALL") & (results_df["horizon"] == "ALL")]
    pivot = zone_table.pivot_table(values="mape", index="zone", columns="bucket")
    print(pivot.to_string())

    print("\n=== SIGNIFICANCE TESTS ===")
    if not sig_df.empty:
        print(sig_df[["bucket", "zone", "median_normal_mape", "median_bucket_mape", "p_value", "significant"]].to_string(index=False))


if __name__ == "__main__":
    main()
