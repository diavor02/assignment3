# Part 3 Writeup Notes

## 3.1 Motivation
- Grid forecasts matter most during extreme weather, when demand can shift fastest and model errors are most costly.
- Part 3 asks whether the transformer is less reliable during meteorological extremes and which zones are most exposed.

## 3.2 Data Analysis and Event Definition
- Weather channels identified from Step 1:
	- Channel 0: temperature-like, mean 281.38, range 235.50–312.16, highest temporal variance.
	- Channel 2: wind-like, mean 7.81, range 0.02–38.16.
- Thresholds from `results/thresholds.csv`:
	- Summer heat threshold: 297.24
	- Winter cold threshold: 262.79
	- High wind threshold: 12.56
- Event catalog from `results/extreme_event_catalog.csv`:
	- normal: 48,776 hours
	- high_wind: 2,518 hours
	- extreme_heat: 663 hours
	- extreme_cold: 538 hours
	- winter_storm: 113 hours
- Sparse categories are the main limitation: winter_storm is especially rare.

## 3.3 Stratified Evaluation Methodology
- Evaluation windows come from midnight starts in 2022–2023, yielding 730 forecast windows.
- Each window inherits the most severe weather bucket seen in its 24-hour forecast period.
- Metric: MAPE across all 24 forecast hours and all zones.
- Significance test: one-sided Mann-Whitney U against the normal bucket.

## 3.4 Results
- Overall MAPE by bucket:
	- normal: 2.8756
	- high_wind: 3.0673
	- extreme_cold: 2.5887
	- extreme_heat: 2.6015
- Interpretation:
	- High wind is slightly worse than normal overall.
	- Heat and cold do not raise aggregate MAPE in this split, but some zone-level effects are visible.
	- Sparse winter_storm cannot be trusted statistically.
- Zone-level highlights:
	- CT is significantly worse during high_wind and extreme_cold.
	- ME is significantly worse during extreme_cold and extreme_heat.
	- NEMA and SEMA show notable cold-weather sensitivity.
	- VT and WCMA show some heat sensitivity.
- Figures to reference:
	- fig1_channel_distributions.png
	- fig2_extreme_event_map.png
	- fig3_stratified_mape_heatmap.png
	- fig4_horizon_curves.png
	- fig5_zone_vulnerability_ranking.png

## 3.5 Discussion and Policy Implications
- Translate zone-specific extremes into actions for load balancing, weather sensing, and demand-response targeting.
- Use the zone vulnerability ranking to prioritize operational attention during cold snaps and high-wind events.

## 3.6 Limitations
- Climate rarity and the small number of winter_storm windows limit statistical power.
- The weather grid resolution may smooth out micro-climate effects.
- Evaluation is on a 2022–2023 proxy split, not the held-out 2024 test set.