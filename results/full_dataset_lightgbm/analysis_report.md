# FreshRetailNet-50K experiment report

## Executive summary

This report was generated from the files and predictions produced by the current run. It compares one-day-ahead P10/P50/P90 forecasts of the dataset's normalized `sale_amount`. It does not estimate latent demand.

Lowest P50 MAE: lightgbm (0.375011; n=350,000).

Lowest mean pinball loss across P10/P50/P90: lightgbm (0.119122).

lightgbm: P50 MAE difference (stockout minus no stockout) = 0.0585642; interpret descriptively because status is observational and group sizes may differ.

Coverage closest to nominal 80%: lightgbm (76.3%). Coverage is an empirical calibration check, not a guarantee for future periods.

Comparisons are descriptive. They do not establish that stockouts cause the error difference, and the target remains normalized observed sales rather than latent demand.

## Evaluation protocol

- Official train dates: 2024-03-28 to 2024-06-25
- Official eval dates: 2024-06-26 to 2024-07-02
- Selected real store-product series: 50,000
- Train rows used: 4,500,000; eval rows scored: 350,000
- Evaluation is rolling one-day-ahead: previous observed days are available as history, including earlier eval days, but the target day's sale is excluded from its own features/context.
- Stockout status is attached after forecasts are generated and used only to stratify error summaries.

## Overall metrics

| model    |      n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:---------|-------:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| lightgbm | 350000 |   0.37501 |    0.65563 |    0.31433 |       0.08077 |       0.18751 |       0.08909 |        0.11912 |       0.76292 |         1.10686 |                    0.50453 |                          0.03160 |

`coverage_80` is the empirical share of actuals between P10 and P90. The nominal target is 0.80. `mean_width_80` measures average interval width. Lower error metrics are better; coverage is best interpreted together with interval width.

## Results by target-day stockout label

| stockout_status      | model    |      n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:---------------------|:---------|-------:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| No stockout recorded | lightgbm | 206562 |   0.35101 |    0.62002 |    0.28373 |       0.07267 |       0.17550 |       0.08693 |        0.11170 |       0.78901 |         1.11470 |                    0.47945 |                          0.05898 |
| Stockout recorded    | lightgbm | 143438 |   0.40957 |    0.70376 |    0.36258 |       0.09244 |       0.20479 |       0.09220 |        0.12981 |       0.72534 |         1.09558 |                    0.54064 |                         -0.00783 |

The label `Stockout recorded` means `stock_hour6_22_cnt > 0`; it does not mean all lost demand is known. Subgroup differences are descriptive and may reflect product, store, day, or promotion mix.

## Metrics by evaluation date

| dt                  | model    |     n |   mae_p50 |   rmse_p50 |   wape_p50 |   pinball_p10 |   pinball_p50 |   pinball_p90 |   mean_pinball |   coverage_80 |   mean_width_80 |   underprediction_rate_p50 |   bias_p50_actual_minus_forecast |
|:--------------------|:---------|------:|----------:|-----------:|-----------:|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|---------------------------:|---------------------------------:|
| 2024-06-26 00:00:00 | lightgbm | 50000 |   0.33285 |    0.54891 |    0.30272 |       0.07180 |       0.16643 |       0.07940 |        0.10587 |       0.78814 |         1.07425 |                    0.45322 |                         -0.03260 |
| 2024-06-27 00:00:00 | lightgbm | 50000 |   0.33708 |    0.58791 |    0.30415 |       0.07337 |       0.16854 |       0.07798 |        0.10663 |       0.77662 |         1.05774 |                    0.45580 |                         -0.01817 |
| 2024-06-28 00:00:00 | lightgbm | 50000 |   0.33569 |    0.52538 |    0.30263 |       0.07013 |       0.16784 |       0.07877 |        0.10558 |       0.76744 |         1.03961 |                    0.46562 |                         -0.01784 |
| 2024-06-29 00:00:00 | lightgbm | 50000 |   0.41434 |    0.69179 |    0.30479 |       0.08744 |       0.20717 |       0.09759 |        0.13073 |       0.75476 |         1.19994 |                    0.54584 |                          0.08995 |
| 2024-06-30 00:00:00 | lightgbm | 50000 |   0.48189 |    0.85379 |    0.31698 |       0.10139 |       0.24095 |       0.11971 |        0.15402 |       0.70376 |         1.23762 |                    0.63472 |                          0.20829 |
| 2024-07-01 00:00:00 | lightgbm | 50000 |   0.35505 |    0.62799 |    0.33719 |       0.07982 |       0.17753 |       0.08259 |        0.11331 |       0.77038 |         1.05265 |                    0.49314 |                         -0.01816 |
| 2024-07-02 00:00:00 | lightgbm | 50000 |   0.36818 |    0.69578 |    0.33419 |       0.08146 |       0.18409 |       0.08758 |        0.11771 |       0.77932 |         1.08622 |                    0.48334 |                          0.00972 |

## LightGBM feature importance

| feature            |   importance |
|:-------------------|-------------:|
| product_id         |    3932.33   |
| store_id           |    1634      |
| third_category_id  |    1231.67   |
| discount           |     961      |
| day_of_month       |     610      |
| dow                |     538      |
| lag_1              |     441.333  |
| mean_7             |     439.333  |
| week_of_year       |     365.333  |
| second_category_id |     233      |
| holiday_flag       |     227.667  |
| mean_28            |     170.333  |
| std_7              |     165.667  |
| lag_2              |     161.667  |
| mean_14            |     129      |
| std_14             |     113      |
| city_id            |     110      |
| lag_7              |     109.333  |
| lag_3              |      94      |
| lag_28             |      68.6667 |

Importance ranks model split use and is not a causal effect or a measure of feature importance for TFT/Chronos.

## Figures

See `figures/` for training trend, model error comparison, interval diagnostics, residuals by stockout label, actual-versus-P50 scatter, and one example series.

## Runtime and reproducibility

- Run timestamp (UTC): 2026-10-11T15:21:13.247908+00:00
- Seed: 17
- Max series setting: 0
- Model runtimes (seconds): `{"lightgbm": 115.969}`
- Python: 3.12.14; pandas: 2.2.3; platform: Linux-6.18.44-x86_64-with-glibc2.39
- Full configuration: `run_manifest.json`; row-level forecasts: `predictions_*.csv`.

## Limitations

The series are short and the evaluation period is limited. `sale_amount` is globally normalized by the publisher, so it must not be labeled as raw physical units. Sales can be censored on stockout days. No causal claim is made about stockout status, and the forecasts do not recover unobserved demand. Results depend on the sampled series, data split, model settings, library versions, and available hardware.
