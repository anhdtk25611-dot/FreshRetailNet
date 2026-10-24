# FreshRetailNet-50K experiment report

## Executive summary

This report was generated from the files and predictions produced by the current run. It compares one-day-ahead P10/P50/P90 forecasts of the dataset's normalized `sale_amount`. It does not estimate latent demand.

Lowest P50 MAE: lightgbm (0.367982; n=700).

Lowest mean pinball loss across P10/P50/P90: lightgbm (0.121486).

chronos: P50 MAE difference (stockout minus no stockout) = 0.0453152; interpret descriptively because status is observational and group sizes may differ.

lightgbm: P50 MAE difference (stockout minus no stockout) = 0.0499982; interpret descriptively because status is observational and group sizes may differ.

tft: P50 MAE difference (stockout minus no stockout) = 0.0202457; interpret descriptively because status is observational and group sizes may differ.

Coverage closest to nominal 80%: tft (79.1%). Coverage is an empirical calibration check, not a guarantee for future periods.

Comparisons are descriptive. They do not establish that stockouts cause the error difference, and the target remains normalized observed sales rather than latent demand.

## Evaluation protocol

- Official train dates: 2024-03-28 to 2024-06-25
- Official eval dates: 2024-06-26 to 2024-07-02
- Selected real store-product series: 100
- Train rows used: 9,000; eval rows scored: 700
- Evaluation is rolling one-day-ahead: previous observed days are available as history, including earlier eval days, but the target day's sale is excluded from its own features/context.
- Stockout status is attached after forecasts are generated and used only to stratify error summaries.

## Overall metrics

| model    |   n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:---------|----:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| lightgbm | 700 |   0.36798 |    0.64879 |    0.32213 |       0.08039 |       0.18399 |       0.10007 |        0.12149 |       0.67143 |         0.94616 |                    0.50714 |                          0.07938 |
| tft      | 700 |   0.43740 |    0.74172 |    0.38290 |       0.07967 |       0.21870 |       0.11489 |        0.13776 |       0.79143 |         1.21891 |                    0.62143 |                          0.25931 |
| chronos  | 700 |   0.37972 |    0.62668 |    0.33241 |       0.07777 |       0.18986 |       0.10719 |        0.12494 |       0.70714 |         0.95892 |                    0.52857 |                          0.10642 |

`coverage_80` is the empirical share of actuals between P10 and P90. The nominal target is 0.80. `mean_width_80` measures average interval width. Lower error metrics are better; coverage is best interpreted together with interval width.

## Results by target-day stockout label

| stockout_status      | model    |   n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:---------------------|:---------|----:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| No stockout recorded | lightgbm | 453 |   0.35034 |    0.63115 |    0.30489 |       0.07286 |       0.17517 |       0.09705 |        0.11503 |       0.70640 |         0.96001 |                    0.47461 |                          0.08498 |
| Stockout recorded    | lightgbm | 247 |   0.40034 |    0.67994 |    0.35429 |       0.09421 |       0.20017 |       0.10561 |        0.13333 |       0.60729 |         0.92076 |                    0.56680 |                          0.06909 |
| No stockout recorded | tft      | 453 |   0.43026 |    0.74565 |    0.37444 |       0.07783 |       0.21513 |       0.10683 |        0.13326 |       0.81898 |         1.25048 |                    0.59823 |                          0.27144 |
| Stockout recorded    | tft      | 247 |   0.45050 |    0.73444 |    0.39869 |       0.08306 |       0.22525 |       0.12968 |        0.14600 |       0.74089 |         1.16101 |                    0.66397 |                          0.23706 |
| No stockout recorded | chronos  | 453 |   0.36373 |    0.60511 |    0.31654 |       0.07089 |       0.18186 |       0.10561 |        0.11946 |       0.72627 |         0.97032 |                    0.50331 |                          0.11468 |
| Stockout recorded    | chronos  | 247 |   0.40904 |    0.66441 |    0.36200 |       0.09039 |       0.20452 |       0.11009 |        0.13500 |       0.67206 |         0.93802 |                    0.57490 |                          0.09128 |

The label `Stockout recorded` means `stock_hour6_22_cnt > 0`; it does not mean all lost demand is known. Subgroup differences are descriptive and may reflect product, store, day, or promotion mix.

## Metrics by evaluation date

| dt                  | model    |   n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:--------------------|:---------|----:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| 2024-06-26 00:00:00 | lightgbm | 100 |   0.29595 |    0.43221 |    0.27925 |       0.06090 |       0.14797 |       0.07944 |        0.09610 |       0.76000 |         0.88535 |                    0.52000 |                          0.02798 |
| 2024-06-27 00:00:00 | lightgbm | 100 |   0.30875 |    0.47133 |    0.28580 |       0.06167 |       0.15437 |       0.10041 |        0.10549 |       0.67000 |         0.87146 |                    0.45000 |                          0.05718 |
| 2024-06-28 00:00:00 | lightgbm | 100 |   0.28443 |    0.42491 |    0.29024 |       0.06213 |       0.14222 |       0.07149 |        0.09195 |       0.74000 |         0.90290 |                    0.44000 |                         -0.02550 |
| 2024-06-29 00:00:00 | lightgbm | 100 |   0.46602 |    0.88641 |    0.34128 |       0.09729 |       0.23301 |       0.12070 |        0.15033 |       0.66000 |         1.10244 |                    0.49000 |                          0.20138 |
| 2024-06-30 00:00:00 | lightgbm | 100 |   0.49134 |    0.93896 |    0.34819 |       0.10682 |       0.24567 |       0.14577 |        0.16609 |       0.59000 |         1.07661 |                    0.67000 |                          0.22233 |
| 2024-07-01 00:00:00 | lightgbm | 100 |   0.36223 |    0.56035 |    0.35066 |       0.08459 |       0.18112 |       0.09237 |        0.11936 |       0.64000 |         0.89691 |                    0.44000 |                          0.00604 |
| 2024-07-02 00:00:00 | lightgbm | 100 |   0.36716 |    0.61286 |    0.34424 |       0.08934 |       0.18358 |       0.09032 |        0.12108 |       0.64000 |         0.88745 |                    0.54000 |                          0.06623 |
| 2024-06-26 00:00:00 | tft      | 100 |   0.33424 |    0.49529 |    0.31538 |       0.06558 |       0.16712 |       0.08774 |        0.10681 |       0.85000 |         1.16712 |                    0.65000 |                          0.19976 |
| 2024-06-27 00:00:00 | tft      | 100 |   0.37848 |    0.56296 |    0.35035 |       0.06902 |       0.18924 |       0.10032 |        0.11952 |       0.75000 |         1.13849 |                    0.61000 |                          0.24073 |
| 2024-06-28 00:00:00 | tft      | 100 |   0.33372 |    0.49171 |    0.34053 |       0.06374 |       0.16686 |       0.08355 |        0.10472 |       0.87000 |         1.14866 |                    0.61000 |                          0.13308 |
| 2024-06-29 00:00:00 | tft      | 100 |   0.58825 |    1.07217 |    0.43080 |       0.09717 |       0.29413 |       0.16552 |        0.18560 |       0.73000 |         1.36178 |                    0.64000 |                          0.42235 |
| 2024-06-30 00:00:00 | tft      | 100 |   0.58553 |    0.92232 |    0.41494 |       0.10145 |       0.29276 |       0.14680 |        0.18034 |       0.77000 |         1.36777 |                    0.72000 |                          0.44279 |
| 2024-07-01 00:00:00 | tft      | 100 |   0.41054 |    0.68067 |    0.39743 |       0.07477 |       0.20527 |       0.12047 |        0.13350 |       0.76000 |         1.15108 |                    0.55000 |                          0.17380 |
| 2024-07-02 00:00:00 | tft      | 100 |   0.43105 |    0.76383 |    0.40413 |       0.08599 |       0.21552 |       0.09985 |        0.13379 |       0.81000 |         1.19746 |                    0.57000 |                          0.20263 |
| 2024-06-26 00:00:00 | chronos  | 100 |   0.28705 |    0.40414 |    0.27086 |       0.05749 |       0.14353 |       0.07255 |        0.09119 |       0.84000 |         0.94151 |                    0.54000 |                          0.05940 |
| 2024-06-27 00:00:00 | chronos  | 100 |   0.31117 |    0.41333 |    0.28804 |       0.05868 |       0.15559 |       0.07800 |        0.09742 |       0.76000 |         0.95041 |                    0.50000 |                          0.09464 |
| 2024-06-28 00:00:00 | chronos  | 100 |   0.28559 |    0.39282 |    0.29142 |       0.06072 |       0.14280 |       0.07372 |        0.09241 |       0.77000 |         0.92263 |                    0.39000 |                         -0.03527 |
| 2024-06-29 00:00:00 | chronos  | 100 |   0.50108 |    0.96132 |    0.36696 |       0.09004 |       0.25054 |       0.20786 |        0.18281 |       0.63000 |         0.91619 |                    0.63000 |                          0.35418 |
| 2024-06-30 00:00:00 | chronos  | 100 |   0.49799 |    0.74491 |    0.35291 |       0.10215 |       0.24899 |       0.12840 |        0.15985 |       0.59000 |         1.02722 |                    0.75000 |                          0.30710 |
| 2024-07-01 00:00:00 | chronos  | 100 |   0.37654 |    0.57226 |    0.36451 |       0.08470 |       0.18827 |       0.10160 |        0.12486 |       0.67000 |         1.01132 |                    0.38000 |                         -0.09675 |
| 2024-07-02 00:00:00 | chronos  | 100 |   0.39860 |    0.67384 |    0.37372 |       0.09063 |       0.19930 |       0.08820 |        0.12604 |       0.69000 |         0.94318 |                    0.51000 |                          0.06165 |

## LightGBM feature importance

| feature           |   importance |
|:------------------|-------------:|
| store_id          |     1133     |
| std_7             |      874.667 |
| day_of_month      |      810     |
| std_14            |      731     |
| discount          |      703.333 |
| lag_1             |      671.333 |
| std_28            |      659.333 |
| mean_7            |      628     |
| mean_14           |      597     |
| lag_28            |      586.333 |
| mean_28           |      583     |
| lag_3             |      559.333 |
| lag_7             |      545.667 |
| lag_2             |      540     |
| lag_14            |      530.667 |
| dow               |      523     |
| product_id        |      427     |
| week_of_year      |      396.333 |
| third_category_id |      164.667 |
| holiday_flag      |      133.333 |

Importance ranks model split use and is not a causal effect or a measure of feature importance for TFT/Chronos.

## Figures

See `figures/` for training trend, model error comparison, interval diagnostics, residuals by stockout label, actual-versus-P50 scatter, and one example series.

## Runtime and reproducibility

- Run timestamp (UTC): 2026-10-11T15:01:49.669430+00:00
- Seed: 17
- Max series setting: 100
- Model runtimes (seconds): `{"chronos": 93.5, "lightgbm": 2.5, "tft": 45.6}`
- Python: 3.12.14; pandas: 2.2.3; platform: Linux-6.18.44-x86_64-with-glibc2.39
- Full configuration: `run_manifest.json`; row-level forecasts: `predictions_*.csv`.

## Limitations

The series are short and the evaluation period is limited. `sale_amount` is globally normalized by the publisher, so it must not be labeled as raw physical units. Sales can be censored on stockout days. No causal claim is made about stockout status, and the forecasts do not recover unobserved demand. Results depend on the sampled series, data split, model settings, library versions, and available hardware.
