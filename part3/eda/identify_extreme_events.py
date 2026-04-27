"""
identify_extreme_events.py

Reads hourly_spatial_stats.csv and labels each hour with:
- weather_bucket: normal, extreme_heat, extreme_cold, high_wind, winter_storm
- event_id: integer grouping consecutive extreme hours into events

Output: results/extreme_event_catalog.csv
"""

import pandas as pd

# UPDATED AFTER STEP 1
TEMP_MEAN_COL = "ch00_mean"
TEMP_P05_COL = "ch00_p05"
TEMP_P95_COL = "ch00_p95"
WIND_MEAN_COL = "ch02_mean"

EXTREME_PERCENTILE = 5


def label_season(ts):
    m = ts.month
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    return "fall"


def main():
    df = pd.read_csv("results/hourly_spatial_stats.csv", parse_dates=["timestamp"])
    df["season"] = df["timestamp"].apply(label_season)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    summer_mask = df["season"] == "summer"
    winter_mask = df["season"] == "winter"

    summer_p95 = df.loc[summer_mask, TEMP_MEAN_COL].quantile(0.95)
    winter_p05 = df.loc[winter_mask, TEMP_MEAN_COL].quantile(0.05)
    wind_p95 = df[WIND_MEAN_COL].quantile(0.95)

    print("Thresholds:")
    print(f"  Extreme heat (summer):   temp > {summer_p95:.2f}")
    print(f"  Extreme cold (winter):   temp < {winter_p05:.2f}")
    print(f"  High wind (all year):    wind > {wind_p95:.2f}")

    df["is_extreme_heat"] = (df["season"] == "summer") & (df[TEMP_MEAN_COL] > summer_p95)
    df["is_extreme_cold"] = (df["season"] == "winter") & (df[TEMP_MEAN_COL] < winter_p05)
    df["is_high_wind"] = df[WIND_MEAN_COL] > wind_p95
    df["is_winter_storm"] = df["is_extreme_cold"] & df["is_high_wind"]

    def assign_bucket(row):
        if row["is_winter_storm"]:
            return "winter_storm"
        if row["is_extreme_heat"]:
            return "extreme_heat"
        if row["is_extreme_cold"]:
            return "extreme_cold"
        if row["is_high_wind"]:
            return "high_wind"
        return "normal"

    df["weather_bucket"] = df.apply(assign_bucket, axis=1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["prev_bucket"] = df["weather_bucket"].shift(1)
    df["event_start"] = (df["weather_bucket"] != df["prev_bucket"]) & (df["weather_bucket"] != "normal")
    df["event_id"] = df["event_start"].cumsum()
    df.loc[df["weather_bucket"] == "normal", "event_id"] = -1

    print("\n--- BUCKET DISTRIBUTION ---")
    counts = df["weather_bucket"].value_counts()
    print(counts)
    print(f"\nTotal hours: {len(df)}")
    print(f"Non-normal hours: {(df['weather_bucket'] != 'normal').sum()}")
    print(f"Distinct extreme events: {df[df['event_id'] > 0]['event_id'].nunique()}")

    for bucket in ["extreme_heat", "extreme_cold", "high_wind", "winter_storm"]:
        n = int((df["weather_bucket"] == bucket).sum())
        windows_estimate = n // 24
        print(f"  {bucket}: {n} hours (~{windows_estimate} forecast windows)")
        if windows_estimate < 30:
            print(f"    WARNING: <30 windows for {bucket}. Results may be unreliable.")

    df.to_csv("results/extreme_event_catalog.csv", index=False)
    print("\nSaved results/extreme_event_catalog.csv")

    thresholds = {
        "summer_heat_threshold": summer_p95,
        "winter_cold_threshold": winter_p05,
        "wind_threshold": wind_p95,
        "temp_column": TEMP_MEAN_COL,
        "wind_column": WIND_MEAN_COL,
    }
    pd.Series(thresholds).to_csv("results/thresholds.csv")
    print("Saved results/thresholds.csv")


if __name__ == "__main__":
    main()
