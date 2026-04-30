"""
plot_results.py — Generate all 5 figures for the writeup
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

ZONE_NAMES = ["CT", "ME", "NH", "RI", "VT", "WCMA", "NEMA", "SEMA"]
BUCKET_COLORS = {
    "normal": "#4A90D9",
    "extreme_heat": "#E24B4A",
    "extreme_cold": "#5B9BF0",
    "high_wind": "#EF9F27",
    "winter_storm": "#7B2D8B",
}
BUCKET_LABELS = {
    "normal": "Normal",
    "extreme_heat": "Extreme Heat",
    "extreme_cold": "Extreme Cold",
    "high_wind": "High Wind",
    "winter_storm": "Winter Storm",
}


def fig1_channel_distributions():
    stats = pd.read_csv("results/hourly_spatial_stats.csv", parse_dates=["timestamp"])
    catalog = pd.read_csv("results/extreme_event_catalog.csv", parse_dates=["timestamp"])
    df = stats.merge(catalog[["timestamp", "weather_bucket"]], on="timestamp", how="left")
    df["weather_bucket"] = df["weather_bucket"].fillna("normal")

    temp_col = "ch00_mean"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    for bucket, color in BUCKET_COLORS.items():
        mask = df["weather_bucket"] == bucket
        ax.scatter(df.loc[mask, "timestamp"], df.loc[mask, temp_col], c=color, s=1, alpha=0.4, label=BUCKET_LABELS[bucket])
    ax.set_xlabel("Date")
    ax.set_ylabel("Spatial Mean Temperature (Channel 0)")
    ax.set_title("Temperature Time Series with Extreme Event Labels")
    ax.legend(markerscale=5, loc="upper right")

    ax = axes[1]
    df["season"] = df["timestamp"].dt.month.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    for season, color in zip(["Winter", "Spring", "Summer", "Fall"], ["#5B9BF0", "#63B363", "#E24B4A", "#EF9F27"]):
        mask = df["season"] == season
        ax.hist(df.loc[mask, temp_col], bins=60, alpha=0.5, color=color, label=season, density=True)
    ax.set_xlabel("Spatial Mean Temperature")
    ax.set_ylabel("Density")
    ax.set_title("Temperature Distribution by Season")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/fig1_channel_distributions.png", bbox_inches="tight")
    plt.close()
    print("Saved fig1")


def fig2_extreme_event_calendar():
    catalog = pd.read_csv("results/extreme_event_catalog.csv", parse_dates=["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 4))
    for bucket, color in BUCKET_COLORS.items():
        if bucket == "normal":
            continue
        mask = catalog["weather_bucket"] == bucket
        ax.scatter(catalog.loc[mask, "timestamp"], [BUCKET_LABELS[bucket]] * int(mask.sum()), c=color, s=8, alpha=0.6, label=BUCKET_LABELS[bucket])

    ax.set_xlabel("Date")
    ax.set_title("Extreme Weather Event Calendar (ISO-NE Domain, 2019–2023)")
    ax.legend(loc="upper right", markerscale=4)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig("results/figures/fig2_extreme_event_map.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2")


def fig3_mape_heatmap():
    results = pd.read_csv("results/stratified_mape_table.csv")
    zone_data = results[(results["zone"] != "ALL") & (results["horizon"] == "ALL")]

    pivot = zone_data.pivot_table(values="mape", index="zone", columns="bucket")
    col_order = [c for c in ["normal", "extreme_heat", "extreme_cold", "high_wind", "winter_storm"] if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.columns = [BUCKET_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("ISO-NE Load Zone")
    ax.set_title("Forecast Error by ISO-NE Load Zone and Weather Condition\n(MAPE % — lower is better)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAPE (%)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/figures/fig3_stratified_mape_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3")


def fig4_horizon_curves():
    results = pd.read_csv("results/stratified_mape_table.csv")
    horizon_data = results[(results["zone"] == "ALL") & (results["horizon"] != "ALL")].copy()
    horizon_data = horizon_data[horizon_data["horizon"].apply(lambda x: str(x).isdigit())]
    horizon_data["horizon"] = horizon_data["horizon"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    for bucket in ["normal", "extreme_heat", "extreme_cold", "high_wind", "winter_storm"]:
        sub = horizon_data[horizon_data["bucket"] == bucket].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub["horizon"], sub["mape"], color=BUCKET_COLORS.get(bucket, "gray"), label=BUCKET_LABELS.get(bucket, bucket), linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Forecast Horizon (hours ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast Error vs Forecast Horizon by Weather Condition")
    ax.legend()
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/fig4_horizon_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved fig4")


def fig5_zone_vulnerability():
    results = pd.read_csv("results/stratified_mape_table.csv")
    zone_data = results[(results["zone"] != "ALL") & (results["horizon"] == "ALL")]

    if "normal" not in zone_data["bucket"].values:
        print("No normal bucket data; skipping fig5")
        return

    normal_mape = zone_data[zone_data["bucket"] == "normal"].set_index("zone")["mape"]
    ratios = []

    zone_names = [z for z in ZONE_NAMES if z in zone_data["zone"].unique()]
    for zone in zone_names:
        zone_sub = zone_data[zone_data["zone"] == zone]
        extreme_sub = zone_sub[zone_sub["bucket"] != "normal"]
        if extreme_sub.empty:
            continue
        worst_mape = float(extreme_sub["mape"].max())
        normal = normal_mape.get(zone, None)
        if normal is None or normal == 0:
            continue
        ratios.append({
            "zone": zone,
            "normal_mape": float(normal),
            "worst_extreme_mape": worst_mape,
            "vulnerability_ratio": worst_mape / float(normal),
            "worst_bucket": extreme_sub.loc[extreme_sub["mape"].idxmax(), "bucket"],
        })

    ratio_df = pd.DataFrame(ratios).sort_values("vulnerability_ratio", ascending=True)
    if ratio_df.empty:
        print("No ratio data; skipping fig5")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(ratio_df["zone"], ratio_df["vulnerability_ratio"], color=[BUCKET_COLORS.get(b, "#888") for b in ratio_df["worst_bucket"]])
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="Baseline (ratio=1)")
    ax.set_xlabel("Vulnerability Ratio (Worst Extreme MAPE / Normal MAPE)")
    ax.set_title("Zone Vulnerability to Extreme Weather\n(Higher = More Error During Extremes)")
    ax.legend()

    for bar, row in zip(bars, ratio_df.itertuples()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{row.vulnerability_ratio:.2f}x ({BUCKET_LABELS.get(row.worst_bucket, row.worst_bucket)})", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/figures/fig5_zone_vulnerability_ranking.png", bbox_inches="tight")
    plt.close()
    print("Saved fig5")


def main():
    fig1_channel_distributions()
    fig2_extreme_event_calendar()
    fig3_mape_heatmap()
    fig4_horizon_curves()
    fig5_zone_vulnerability()
    print("\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
