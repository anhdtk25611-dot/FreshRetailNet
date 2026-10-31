# FreshRetailNet-50K experiment report

## Executive summary

This report was generated from the files and predictions produced by the current run. It compares one-day-ahead P10/P50/P90 forecasts of the dataset's normalized `sale_amount`. It does not estimate latent demand.

Lowest P50 MAE: lightgbm (0.367982; n=700).

Lowest mean pinball loss across P10/P50/P90: lightgbm (0.121486).

chronos: P50 MAE difference (stockout minus no stockout) = 0.0395318; interpret descriptively because status is observational and group sizes may differ.

lightgbm: P50 MAE difference (stockout minus no stockout) = 0.0499982; interpret descriptively because status is observational and group sizes may differ.

tft: P50 MAE difference (stockout minus no stockout) = 0.0130523; interpret descriptively because status is observational and group sizes may differ.

Coverage closest to nominal 80%: chronos (70.3%). Coverage is an empirical calibration check, not a guarantee for future periods.

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
| tft      | 700 |   0.42751 |    0.72432 |    0.37425 |       0.07890 |       0.21376 |       0.13776 |        0.14347 |       0.68429 |         0.96045 |                    0.62429 |                          0.22404 |
| chronos  | 700 |   0.38110 |    0.63340 |    0.33361 |       0.07879 |       0.19055 |       0.11026 |        0.12653 |       0.70286 |         0.97094 |                    0.54143 |                          0.10479 |

`coverage_80` is the empirical share of actuals between P10 and P90. The nominal target is 0.80. `mean_width_80` measures average interval width. Lower error metrics are better; coverage is best interpreted together with interval width.

## Results by target-day stockout label

| stockout_status      | model    |   n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:---------------------|:---------|----:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| No stockout recorded | lightgbm | 453 |   0.35034 |    0.63115 |    0.30489 |       0.07286 |       0.17517 |       0.09705 |        0.11503 |       0.70640 |         0.96001 |                    0.47461 |                          0.08498 |
| Stockout recorded    | lightgbm | 247 |   0.40034 |    0.67994 |    0.35429 |       0.09421 |       0.20017 |       0.10561 |        0.13333 |       0.60729 |         0.92076 |                    0.56680 |                          0.06909 |
| No stockout recorded | tft      | 453 |   0.42291 |    0.74129 |    0.36804 |       0.07609 |       0.21145 |       0.14023 |        0.14259 |       0.70640 |         0.97342 |                    0.61810 |                          0.24565 |
| Stockout recorded    | tft      | 247 |   0.43596 |    0.69212 |    0.38582 |       0.08406 |       0.21798 |       0.13323 |        0.14509 |       0.64372 |         0.93666 |                    0.63563 |                          0.18441 |
| No stockout recorded | chronos  | 453 |   0.36715 |    0.62017 |    0.31952 |       0.07144 |       0.18357 |       0.10792 |        0.12098 |       0.71965 |         0.97976 |                    0.51435 |                          0.11502 |
| Stockout recorded    | chronos  | 247 |   0.40668 |    0.65695 |    0.35991 |       0.09226 |       0.20334 |       0.11455 |        0.13672 |       0.67206 |         0.95476 |                    0.59109 |                          0.08602 |

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
| 2024-06-26 00:00:00 | tft      | 100 |   0.32640 |    0.46770 |    0.30799 |       0.06155 |       0.16320 |       0.09328 |        0.10601 |       0.75000 |         0.93197 |                    0.66000 |                          0.16160 |
| 2024-06-27 00:00:00 | tft      | 100 |   0.38196 |    0.58850 |    0.35357 |       0.06817 |       0.19098 |       0.13892 |        0.13269 |       0.66000 |         0.86991 |                    0.65000 |                          0.27525 |
| 2024-06-28 00:00:00 | tft      | 100 |   0.35026 |    0.51694 |    0.35741 |       0.06572 |       0.17513 |       0.08769 |        0.10951 |       0.74000 |         0.90972 |                    0.61000 |                          0.12723 |
| 2024-06-29 00:00:00 | tft      | 100 |   0.57340 |    1.06009 |    0.41992 |       0.09554 |       0.28670 |       0.22807 |        0.20344 |       0.59000 |         1.00951 |                    0.63000 |                          0.39254 |
| 2024-06-30 00:00:00 | tft      | 100 |   0.52139 |    0.81573 |    0.36949 |       0.09394 |       0.26069 |       0.16081 |        0.17181 |       0.66000 |         1.09377 |                    0.69000 |                          0.30887 |
| 2024-07-01 00:00:00 | tft      | 100 |   0.42281 |    0.73735 |    0.40930 |       0.07823 |       0.21140 |       0.14234 |        0.14399 |       0.70000 |         0.93782 |                    0.56000 |                          0.15471 |
| 2024-07-02 00:00:00 | tft      | 100 |   0.41638 |    0.71223 |    0.39038 |       0.08919 |       0.20819 |       0.11321 |        0.13686 |       0.69000 |         0.97047 |                    0.57000 |                          0.14809 |
| 2024-06-26 00:00:00 | chronos  | 100 |   0.29510 |    0.43083 |    0.27845 |       0.06322 |       0.14755 |       0.07769 |        0.09615 |       0.76000 |         0.94163 |                    0.59000 |                          0.05349 |
| 2024-06-27 00:00:00 | chronos  | 100 |   0.30446 |    0.41016 |    0.28183 |       0.05681 |       0.15223 |       0.08483 |        0.09796 |       0.80000 |         0.95947 |                    0.48000 |                          0.07292 |
| 2024-06-28 00:00:00 | chronos  | 100 |   0.27649 |    0.37090 |    0.28213 |       0.05828 |       0.13824 |       0.07438 |        0.09030 |       0.81000 |         0.97152 |                    0.44000 |                         -0.04097 |
| 2024-06-29 00:00:00 | chronos  | 100 |   0.49525 |    0.96967 |    0.36269 |       0.08968 |       0.24762 |       0.19706 |        0.17812 |       0.61000 |         0.93583 |                    0.64000 |                          0.35783 |
| 2024-06-30 00:00:00 | chronos  | 100 |   0.51015 |    0.76879 |    0.36153 |       0.10470 |       0.25508 |       0.13973 |        0.16650 |       0.55000 |         1.02301 |                    0.74000 |                          0.32087 |
| 2024-07-01 00:00:00 | chronos  | 100 |   0.39467 |    0.60807 |    0.38206 |       0.08193 |       0.19733 |       0.10682 |        0.12870 |       0.70000 |         0.99616 |                    0.40000 |                         -0.07545 |
| 2024-07-02 00:00:00 | chronos  | 100 |   0.39156 |    0.64489 |    0.36711 |       0.09690 |       0.19578 |       0.09132 |        0.12800 |       0.69000 |         0.96897 |                    0.50000 |                          0.04482 |

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

- Run timestamp (UTC): 2026-10-31T04:08:08.129956+00:00
- Seed: 17
- Max series setting: 100
- Model runtimes (seconds): `{"chronos": 403.234, "lightgbm": 3.362, "tft": 100.63}`
- Python: 3.13.18; pandas: 2.2.3; platform: Linux-6.6.159+-x86_64-with-glibc2.39
- Full configuration: `run_manifest.json`; row-level forecasts: `predictions_*.csv`.

## Limitations

The series are short and the evaluation period is limited. `sale_amount` is globally normalized by the publisher, so it must not be labeled as raw physical units. Sales can be censored on stockout days. No causal claim is made about stockout status, and the forecasts do not recover unobserved demand. Results depend on the sampled series, data split, model settings, library versions, and available hardware.
