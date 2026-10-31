"""Run a reproducible, rolling one-day-ahead comparison on FreshRetailNet-50K.

The official eval split is never used for fitting. For each eval date, only sales
from earlier dates are available as lags/context. The output target is the
dataset's globally normalized observed sale_amount, not latent demand.
"""
from __future__ import annotations

import argparse
import json
import random
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
QUANTILES = (0.1, 0.5, 0.9)
KEYS = ["store_id", "product_id"]
GROUP = [*KEYS]
ID_COLUMNS = ["city_id", "store_id", "management_group_id", "first_category_id",
              "second_category_id", "third_category_id", "product_id"]
BASE_FIELDS = [*ID_COLUMNS, "dt", "sale_amount", "stock_hour6_22_cnt",
               "holiday_flag", "activity_flag", "discount"]


def _require(columns: list[str], frame: pd.DataFrame, filename: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}. "
                         f"Found: {list(frame.columns)}")


def read_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two original Parquet files and validate their expected schema."""
    train_path, eval_path = DATA_DIR / "train.parquet", DATA_DIR / "eval.parquet"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "Original data files are missing. Run `python src/download_data.py` "
            "from the project root; do not substitute a modified dataset.")
    # The raw files also contain nested hourly sequences. This daily-target study
    # intentionally reads only daily model fields to keep Colab RAM use bounded.
    train = pd.read_parquet(train_path, columns=BASE_FIELDS)
    evaluation = pd.read_parquet(eval_path, columns=BASE_FIELDS)
    _require(BASE_FIELDS, train, "train.parquet")
    _require(BASE_FIELDS, evaluation, "eval.parquet")
    for frame in (train, evaluation):
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
        frame["sale_amount"] = pd.to_numeric(frame["sale_amount"], errors="coerce")
        frame["stock_hour6_22_cnt"] = pd.to_numeric(frame["stock_hour6_22_cnt"], errors="coerce")
    train = train.dropna(subset=[*KEYS, "dt", "sale_amount"]).copy()
    evaluation = evaluation.dropna(subset=[*KEYS, "dt", "sale_amount"]).copy()
    if train.duplicated([*KEYS, "dt"]).any() or evaluation.duplicated([*KEYS, "dt"]).any():
        raise ValueError("Duplicate store-product-date rows found; inspect source data before modeling.")
    if train.dt.max() >= evaluation.dt.min():
        raise ValueError("Expected official eval dates to follow train dates. "
                         "Inspect split dates; the script will not silently reorder the split.")
    return train, evaluation


def choose_panel(train: pd.DataFrame, evaluation: pd.DataFrame,
                 max_series: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the same real store-product IDs from both official splits."""
    train_keys = train[KEYS].drop_duplicates()
    eval_keys = evaluation[KEYS].drop_duplicates()
    keys = train_keys.merge(eval_keys, on=KEYS, how="inner").sort_values(KEYS)
    if max_series > 0 and len(keys) > max_series:
        keys = keys.sample(max_series, random_state=seed).sort_values(KEYS)
    if keys.empty:
        raise ValueError("No store-product series are shared by train and eval.")
    tr = train.merge(keys, on=KEYS, how="inner").sort_values([*KEYS, "dt"]).reset_index(drop=True)
    ev = evaluation.merge(keys, on=KEYS, how="inner").sort_values([*KEYS, "dt"]).reset_index(drop=True)
    return tr, ev


def make_features(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build lag/rolling features using only earlier observed sales."""
    both = pd.concat([train.assign(_split="train"), evaluation.assign(_split="eval")], ignore_index=True)
    both = both.sort_values([*KEYS, "dt"]).reset_index(drop=True)
    dt = both.dt
    both["dow"] = dt.dt.dayofweek.astype("int8")
    both["day_of_month"] = dt.dt.day.astype("int8")
    both["week_of_year"] = dt.dt.isocalendar().week.astype("int16")
    both["month"] = dt.dt.month.astype("int8")
    grouped = both.groupby(KEYS, sort=False, observed=True)["sale_amount"]
    for lag in (1, 2, 3, 7, 14, 28):
        both[f"lag_{lag}"] = grouped.shift(lag)
    prior = grouped.shift(1)
    for window in (7, 14, 28):
        roll = prior.groupby([both[k] for k in KEYS], sort=False, observed=True).rolling(window, min_periods=max(3, window // 2))
        both[f"mean_{window}"] = roll.mean().reset_index(level=[0, 1], drop=True)
        both[f"std_{window}"] = roll.std().reset_index(level=[0, 1], drop=True)
    features = ["dow", "day_of_month", "week_of_year", "month", "holiday_flag", "activity_flag",
                "discount", "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
                "mean_7", "std_7", "mean_14", "std_14", "mean_28", "std_28", *ID_COLUMNS]
    features = [c for c in features if c in both.columns]
    cats = [c for c in ID_COLUMNS if c in features]
    for col in cats:
        vals = both[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(vals.unique())
        both[col] = pd.Categorical(vals, categories=categories)
    tr = both[both._split.eq("train")].copy()
    ev = both[both._split.eq("eval")].copy()
    return tr, ev, features


def _base_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*KEYS, "dt", "sale_amount", "stock_hour6_22_cnt"]
    columns += [f"q{q:.1f}" for q in QUANTILES if f"q{q:.1f}" in frame.columns]
    out = frame[columns].copy()
    out["stockout_status"] = np.where(out.stock_hour6_22_cnt.fillna(0).gt(0),
                                       "Stockout recorded", "No stockout recorded")
    return out


def fit_lightgbm(train_f: pd.DataFrame, eval_f: pd.DataFrame, features: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRegressor

    usable = train_f.dropna(subset=["lag_28"]).copy()
    scoring = eval_f.dropna(subset=["lag_28"]).copy()
    if usable.empty or scoring.empty:
        raise ValueError("Too few rows with 28 days of history for LightGBM.")
    cats = [c for c in ID_COLUMNS if c in features]
    base = _base_prediction_frame(scoring)
    importances = []
    for q in QUANTILES:
        model = LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                              learning_rate=0.035, num_leaves=31, max_depth=-1,
                              subsample=0.85, colsample_bytree=0.85,
                              reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
        model.fit(usable[features], usable.sale_amount,
                  categorical_feature=cats if cats else "auto")
        base[f"q{q:.1f}"] = np.maximum(0, model.predict(scoring[features]))
        importances.append(pd.DataFrame({"feature": features, "importance": model.feature_importances_, "quantile": q}))
    importance = pd.concat(importances, ignore_index=True).groupby("feature", as_index=False).importance.mean()
    importance = importance.sort_values("importance", ascending=False)
    return base, importance


def fit_chronos(train: pd.DataFrame, evaluation: pd.DataFrame, batch_size: int = 64, seed: int = 17) -> pd.DataFrame:
    import torch
    from chronos import ChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Bound CPU activation memory; a larger batch is useful only when a GPU exists.
    if device == "cpu":
        batch_size = min(batch_size, 8)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        revision="da0eb423156e8761c8540ccdbad501f38bc46142",
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    hist = pd.concat([train, evaluation], ignore_index=True).sort_values([*KEYS, "dt"])
    eval_dates = sorted(evaluation.dt.unique())
    keys = evaluation[KEYS].drop_duplicates().sort_values(KEYS)
    histories = {key: group for key, group in hist.groupby(KEYS, sort=False, observed=True)}
    eval_lookup = evaluation.set_index([*KEYS, "dt"], drop=False)
    task_rows: list[tuple[tuple, pd.Timestamp, np.ndarray]] = []
    for key in keys.itertuples(index=False, name=None):
        sg = histories.get(key)
        if sg is None:
            continue
        for date in eval_dates:
            past = sg.loc[sg.dt < date, "sale_amount"].to_numpy(dtype=np.float32)
            try:
                target = eval_lookup.loc[(*key, pd.Timestamp(date))]
            except KeyError:
                continue
            if len(past) < 28:
                continue
            task_rows.append((key, pd.Timestamp(date), past[-90:]))
    predictions = []
    for start in range(0, len(task_rows), batch_size):
        batch = task_rows[start:start + batch_size]
        contexts = [torch.tensor(item[2], dtype=torch.float32) for item in batch]
        # Seed từng batch để việc lấy mẫu Chronos lặp lại với cùng cấu hình.
        torch.manual_seed(int(seed + start))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed + start))
        samples = pipeline.predict(contexts, prediction_length=1, num_samples=50)
        # Chronos-T5 returns [series, samples, horizon]. Take requested empirical quantiles.
        q_values = torch.quantile(samples.float(), torch.tensor(QUANTILES), dim=1).squeeze(-1).cpu().numpy()
        for j, (key, date, _) in enumerate(batch):
            target = eval_lookup.loc[(*key, date)]
            if isinstance(target, pd.DataFrame):
                target = target.iloc[0]
            row = {**dict(zip(KEYS, key)), "dt": date,
                   "sale_amount": float(target.sale_amount),
                   "stock_hour6_22_cnt": target.stock_hour6_22_cnt}
            for qi, q in enumerate(QUANTILES):
                row[f"q{q:.1f}"] = max(0.0, float(q_values[qi, j]))
            predictions.append(row)
    return _base_prediction_frame(pd.DataFrame(predictions).drop(columns="stockout_status", errors="ignore"))


def fit_tft(train: pd.DataFrame, evaluation: pd.DataFrame, seed: int, epochs: int, batch_size: int) -> pd.DataFrame:
    """Fit TFT Quantile on train; predict rolling one-step windows for official eval."""
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    seed_everything(seed, workers=True)
    data = pd.concat([train, evaluation], ignore_index=True).sort_values([*KEYS, "dt"]).copy()
    data["series_id"] = data.store_id.astype(str) + "__" + data.product_id.astype(str)
    data["time_idx"] = (data.dt - data.dt.min()).dt.days.astype("int32")
    data["dow_cat"] = data.dt.dt.dayofweek.astype(str).astype("category")
    data["store_cat"] = data.store_id.astype(str).astype("category")
    data["product_cat"] = data.product_id.astype(str).astype("category")
    train_end = int(train.time_idx.max()) if "time_idx" in train else int((train.dt.max() - data.dt.min()).days)
    max_encoder = min(28, max(14, int(train.groupby(KEYS).dt.nunique().min()) - 2))
    training = TimeSeriesDataSet(
        data[data.dt.le(train.dt.max())], time_idx="time_idx", target="sale_amount",
        group_ids=["series_id"], min_encoder_length=max_encoder,
        max_encoder_length=max_encoder, min_prediction_length=1, max_prediction_length=1,
        static_categoricals=["store_cat", "product_cat"],
        time_varying_known_reals=["time_idx", "holiday_flag", "activity_flag", "discount"],
        time_varying_known_categoricals=["dow_cat"],
        time_varying_unknown_reals=["sale_amount"],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    # All windows are generated in order. Eval target values are decoder labels only;
    # the target itself is registered as an unknown real and is not a future input.
    evaluation_set = TimeSeriesDataSet.from_dataset(training, data, predict=False,
                                                       stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    eval_loader = evaluation_set.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.01, hidden_size=16, attention_head_size=2,
        dropout=0.1, hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=list(QUANTILES)), log_interval=-1,
    )
    trainer = Trainer(max_epochs=epochs, accelerator="auto", devices=1,
                      gradient_clip_val=0.1, enable_checkpointing=False,
                      logger=False, enable_model_summary=False, enable_progress_bar=True,
                      deterministic=True)
    t0 = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader)
    train_seconds = time.perf_counter() - t0
    # `mode="prediction"` reduces quantile loss output to the median; raw mode
    # retains all three trained quantiles for scoring.
    result = model.predict(eval_loader, mode="raw", return_index=True,
                           trainer_kwargs={"enable_progress_bar": False})
    raw_output = result.output.prediction
    output = raw_output.detach().cpu().numpy()
    index = result.index.reset_index(drop=True)
    # The returned index identifies each prediction's series and target time index.
    if "series_id" not in index:
        raise RuntimeError(f"TFT prediction index has unexpected fields: {list(index.columns)}")
    time_col = "time_idx" if "time_idx" in index else next((c for c in index if c != "series_id"), None)
    if time_col is None:
        raise RuntimeError("Could not find decoder time in TFT prediction index.")
    eval_dates = set(pd.to_datetime(evaluation.dt).unique())
    date_from_idx = data.drop_duplicates("time_idx").set_index("time_idx").dt.to_dict()
    series_map = data[["series_id", *KEYS]].drop_duplicates().set_index("series_id")
    eval_lookup = evaluation.set_index([*KEYS, "dt"], drop=False)
    rows = []
    for i, rec in index.iterrows():
        time_idx = int(rec[time_col])
        date = pd.Timestamp(date_from_idx[time_idx])
        if date not in eval_dates:
            continue
        sid = str(rec.series_id)
        if sid not in series_map.index:
            continue
        store, product = series_map.loc[sid, KEYS].tolist()
        try:
            actual = eval_lookup.loc[(store, product, date)]
        except KeyError:
            continue
        if isinstance(actual, pd.DataFrame):
            actual = actual.iloc[0]
        vals = np.asarray(output[i]).reshape(-1, len(QUANTILES))[0]
        row = {"store_id": store, "product_id": product, "dt": date,
               "sale_amount": float(actual.sale_amount),
               "stock_hour6_22_cnt": actual.stock_hour6_22_cnt,
               "fit_seconds": train_seconds}
        for qi, q in enumerate(QUANTILES):
            row[f"q{q:.1f}"] = max(0.0, float(vals[qi]))
        rows.append(row)
    return _base_prediction_frame(pd.DataFrame(rows).drop(columns="stockout_status", errors="ignore"))


def metric_row(frame: pd.DataFrame, model_name: str) -> dict:
    if frame.empty:
        return {"model": model_name, "n": 0}
    y = frame.sale_amount.to_numpy(float)
    q10, q50, q90 = (frame[f"q{q:.1f}"].to_numpy(float) for q in QUANTILES)
    errors = y - q50
    denom = np.abs(y).sum()
    pinballs = {}
    for q in QUANTILES:
        pred = frame[f"q{q:.1f}"].to_numpy(float)
        err = y - pred
        pinballs[f"pinball_p{int(q*100)}"] = float(np.maximum(q * err, (q - 1) * err).mean())
    return {"model": model_name, "n": int(len(frame)),
            "mae_p50": float(np.abs(errors).mean()),
            "rmse_p50": float(np.sqrt(np.square(errors).mean())),
            "wape_p50": float(np.abs(errors).sum() / denom) if denom else None,
            **pinballs, "mean_pinball": float(np.mean(list(pinballs.values()))),
            "coverage_80": float(np.mean((y >= q10) & (y <= q90))),
            "mean_width_80": float(np.mean(q90 - q10)),
            "underprediction_rate_p50": float(np.mean(errors > 0)),
            "bias_p50_actual_minus_forecast": float(errors.mean())}


def summarize_predictions(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, by_status, by_date = [], [], []
    for name, frame in predictions.items():
        overall.append(metric_row(frame, name))
        for status, group in frame.groupby("stockout_status", dropna=False):
            by_status.append({"stockout_status": status, **metric_row(group, name)})
        for date, group in frame.groupby("dt"):
            by_date.append({"dt": date, **metric_row(group, name)})
    return pd.DataFrame(overall), pd.DataFrame(by_status), pd.DataFrame(by_date)


def _save_figures(predictions: dict[str, pd.DataFrame], overall: pd.DataFrame,
                  by_status: pd.DataFrame, train: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="colorblind")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    daily = train.groupby("dt", as_index=False).sale_amount.mean()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(daily.dt, daily.sale_amount, color="#455A64", linewidth=1.8)
    ax.set(title="Training panel: mean observed sales by date", xlabel="Date", ylabel="Mean normalized sale_amount")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "01_training_sales_trend.png", dpi=180); plt.close(fig)

    order = overall.sort_values("mae_p50").model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=overall, x="model", y="mae_p50", order=order, ax=axes[0])
    axes[0].set(title="Median forecast error", xlabel="Model", ylabel="MAE (P50)")
    sns.barplot(data=overall, x="model", y="mean_pinball", order=order, ax=axes[1])
    axes[1].set(title="Probabilistic forecast error", xlabel="Model", ylabel="Mean pinball loss")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "02_model_error_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=by_status, x="model", y="coverage_80", hue="stockout_status", ax=axes[0])
    axes[0].axhline(.8, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
    axes[0].set(title="Observed coverage by stockout label", xlabel="Model", ylabel="Coverage (P10–P90)", ylim=(0, 1))
    sns.barplot(data=by_status, x="model", y="mean_width_80", hue="stockout_status", ax=axes[1])
    axes[1].set(title="Prediction interval width by label", xlabel="Model", ylabel="Mean P90 − P10")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "03_interval_diagnostics.png", dpi=180); plt.close(fig)

    pieces = []
    for name, frame in predictions.items():
        tmp = frame.copy(); tmp["model"] = name; tmp["residual"] = tmp.sale_amount - tmp["q0.5"]
        pieces.append(tmp)
    allp = pd.concat(pieces, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.boxplot(data=allp, x="stockout_status", y="residual", hue="model", showfliers=False, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Median forecast residuals by recorded stockout status", xlabel="Target-day status", ylabel="Actual − P50")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "04_residuals_by_stockout.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.8), squeeze=False)
    for ax, (name, frame) in zip(axes.ravel(), predictions.items()):
        sample = frame.sample(min(len(frame), 2500), random_state=17) if len(frame) else frame
        sns.scatterplot(data=sample, x="sale_amount", y="q0.5", alpha=.28, s=16, ax=ax)
        lim = max(float(sample.sale_amount.max()), float(sample["q0.5"].max())) if len(sample) else 1
        ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
        ax.set(title=name, xlabel="Observed normalized sale_amount", ylabel="Predicted P50")
    fig.suptitle("Observed versus median forecast", y=1.02)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "05_actual_vs_median.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # One real, deterministic example series with quantile bands across eval dates.
    first_model = next(iter(predictions))
    frame = predictions[first_model].copy()
    if len(frame):
        key = frame.groupby(KEYS).size().sort_values(ascending=False).index[0]
        example = frame[(frame.store_id == key[0]) & (frame.product_id == key[1])].sort_values("dt")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(example.dt, example.sale_amount, marker="o", label="Observed")
        for name, pred in predictions.items():
            e = pred[(pred.store_id == key[0]) & (pred.product_id == key[1])].sort_values("dt")
            if len(e): ax.plot(e.dt, e["q0.5"], marker=".", label=f"{name} P50")
        ax.fill_between(example.dt, example["q0.1"], example["q0.9"], alpha=.16, label=f"{first_model} P10–P90")
        ax.set(title=f"Example series: store {key[0]}, product {key[1]}", xlabel="Eval date", ylabel="Normalized observed sales")
        ax.legend(ncol=2); fig.tight_layout(); fig.savefig(FIGURES_DIR / "06_example_forecast.png", dpi=180); plt.close(fig)


def _analysis_text(overall: pd.DataFrame, by_status: pd.DataFrame) -> str:
    if overall.empty:
        return "No model produced aligned predictions; inspect the error log and split manifest."
    best_mae = overall.loc[overall.mae_p50.idxmin()]
    best_pb = overall.loc[overall.mean_pinball.idxmin()]
    blocks = [f"Lowest P50 MAE: {best_mae.model} ({best_mae.mae_p50:.6g}; n={int(best_mae.n):,}).",
              f"Lowest mean pinball loss across P10/P50/P90: {best_pb.model} ({best_pb.mean_pinball:.6g})."]
    if len(by_status):
        mae = by_status.pivot(index="model", columns="stockout_status", values="mae_p50")
        if {"Stockout recorded", "No stockout recorded"}.issubset(mae.columns):
            for model, row in mae.iterrows():
                diff = row["Stockout recorded"] - row["No stockout recorded"]
                blocks.append(f"{model}: P50 MAE difference (stockout minus no stockout) = {diff:.6g}; interpret descriptively because status is observational and group sizes may differ.")
    coverage = overall.set_index("model").coverage_80
    closest = (coverage - .8).abs().idxmin()
    blocks.append(f"Coverage closest to nominal 80%: {closest} ({coverage[closest]:.1%}). Coverage is an empirical calibration check, not a guarantee for future periods.")
    blocks.append("Comparisons are descriptive. They do not establish that stockouts cause the error difference, and the target remains normalized observed sales rather than latent demand.")
    return "\n\n".join(blocks)


def _write_report(manifest: dict, overall: pd.DataFrame, by_status: pd.DataFrame,
                  by_date: pd.DataFrame, importance: pd.DataFrame, times: dict) -> None:
    table1 = overall.to_markdown(index=False, floatfmt=".5f")
    table2 = by_status.to_markdown(index=False, floatfmt=".5f") if len(by_status) else "No subgroup table available."
    table3 = by_date.to_markdown(index=False, floatfmt=".5f")
    top = importance.head(20).to_markdown(index=False) if len(importance) else "LightGBM importance not available."
    text = f'''# FreshRetailNet-50K experiment report

## Executive summary

This report was generated from the files and predictions produced by the current run. It compares one-day-ahead P10/P50/P90 forecasts of the dataset's normalized `sale_amount`. It does not estimate latent demand.

{_analysis_text(overall, by_status)}

## Evaluation protocol

- Official train dates: {manifest['train_min_date']} to {manifest['train_max_date']}
- Official eval dates: {manifest['eval_min_date']} to {manifest['eval_max_date']}
- Selected real store-product series: {manifest['series_count']:,}
- Train rows used: {manifest['train_rows']:,}; eval rows scored: {manifest['eval_rows']:,}
- Evaluation is rolling one-day-ahead: previous observed days are available as history, including earlier eval days, but the target day's sale is excluded from its own features/context.
- Stockout status is attached after forecasts are generated and used only to stratify error summaries.

## Overall metrics

{table1}

`coverage_80` is the empirical share of actuals between P10 and P90. The nominal target is 0.80. `mean_width_80` measures average interval width. Lower error metrics are better; coverage is best interpreted together with interval width.

## Results by target-day stockout label

{table2}

The label `Stockout recorded` means `stock_hour6_22_cnt > 0`; it does not mean all lost demand is known. Subgroup differences are descriptive and may reflect product, store, day, or promotion mix.

## Metrics by evaluation date

{table3}

## LightGBM feature importance

{top}

Importance ranks model split use and is not a causal effect or a measure of feature importance for TFT/Chronos.

## Figures

See `figures/` for training trend, model error comparison, interval diagnostics, residuals by stockout label, actual-versus-P50 scatter, and one example series.

## Runtime and reproducibility

- Run timestamp (UTC): {manifest['run_timestamp_utc']}
- Seed: {manifest['seed']}
- Max series setting: {manifest['max_series_setting']}
- Model runtimes (seconds): `{json.dumps(times, sort_keys=True)}`
- Python: {manifest['python_version']}; pandas: {manifest['pandas_version']}; platform: {manifest['platform']}
- Full configuration: `run_manifest.json`; row-level forecasts: `predictions_*.csv`.

## Limitations

The series are short and the evaluation period is limited. `sale_amount` is globally normalized by the publisher, so it must not be labeled as raw physical units. Sales can be censored on stockout days. No causal claim is made about stockout status, and the forecasts do not recover unobserved demand. Results depend on the sampled series, data split, model settings, library versions, and available hardware.
'''
    (RESULTS_DIR / "analysis_report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    global RESULTS_DIR, FIGURES_DIR
    if getattr(args, "results_dir", None) is not None:
        RESULTS_DIR = Path(args.results_dir).expanduser().resolve()
        FIGURES_DIR = RESULTS_DIR / "figures"
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, eval_raw = read_splits()
    train_raw_rows, eval_raw_rows = len(train_raw), len(eval_raw)
    train, evaluation = choose_panel(train_raw, eval_raw, args.max_series, args.seed)
    del train_raw, eval_raw
    import gc
    gc.collect()
    train_f, eval_f, features = make_features(train, evaluation)
    predictions: dict[str, pd.DataFrame] = {}
    times: dict[str, float] = {}
    importance = pd.DataFrame()
    model_names = args.models
    for name in model_names:
        print(f"\n{'='*72}\nFitting {name} on {len(train):,} train rows; scoring {len(evaluation):,} official eval rows\n{'='*72}")
        t0 = time.perf_counter()
        if name == "lightgbm":
            pred, importance = fit_lightgbm(train_f, eval_f, features, args.seed)
        elif name == "chronos":
            pred = fit_chronos(train, evaluation, args.inference_batch_size, args.seed)
        elif name == "tft":
            pred = fit_tft(train, evaluation, args.seed, args.tft_epochs, args.tft_batch_size)
        else:
            raise ValueError(f"Unknown model: {name}")
        times[name] = round(time.perf_counter() - t0, 3)
        # LightGBM's feature builder stores identifiers as pandas categoricals,
        # while the neural models return numeric identifiers. Normalize keys so
        # the cross-model intersection is stable across pandas merge semantics.
        for key_col in KEYS:
            pred[key_col] = pd.to_numeric(pred[key_col], errors="raise").astype("int64")
        pred["dt"] = pd.to_datetime(pred["dt"], errors="raise").dt.normalize()
        pred = pred.drop_duplicates([*KEYS, "dt"], keep="last")
        if pred.empty:
            warnings.warn(f"{name} returned no aligned predictions; skipping.")
            continue
        predictions[name] = pred
        pred.to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)
        print(f"{name}: {len(pred):,} predictions; {times[name]:,.1f}s")
    if not predictions:
        raise RuntimeError("No predictions were produced.")

    # Compare models only on the intersection of identical series-dates.
    common = None
    for pred in predictions.values():
        ids = pred[[*KEYS, "dt"]].drop_duplicates()
        common = ids if common is None else common.merge(ids, on=[*KEYS, "dt"], how="inner")
    if common is None or common.empty:
        raise RuntimeError("The models produced no common series-date rows for a fair comparison.")
    predictions = {name: pred.merge(common, on=[*KEYS, "dt"], how="inner") for name, pred in predictions.items()}
    for name, pred in predictions.items():
        pred.to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)

    overall, by_status, by_date = summarize_predictions(predictions)
    overall.to_csv(RESULTS_DIR / "metrics_overall.csv", index=False)
    by_status.to_csv(RESULTS_DIR / "metrics_by_stockout.csv", index=False)
    by_date.to_csv(RESULTS_DIR / "metrics_by_date.csv", index=False)
    if len(importance): importance.to_csv(RESULTS_DIR / "lightgbm_feature_importance.csv", index=False)
    _save_figures(predictions, overall, by_status, train)

    manifest = {
        "dataset": "Dingdong-Inc/FreshRetailNet-50K", "dataset_split": "official train/eval",
        "target": "sale_amount (publisher's globally normalized observed sales)",
        "forecasting_protocol": "rolling one-day-ahead; earlier eval observations may be used as prior history",
        "stockout_status_used_as_predictor": False,
        "train_rows_raw": int(train_raw_rows), "eval_rows_raw": int(eval_raw_rows),
        "train_rows": int(len(train)), "eval_rows": int(len(evaluation)),
        "series_count": int(train[KEYS].drop_duplicates().shape[0]),
        "train_min_date": str(train.dt.min().date()), "train_max_date": str(train.dt.max().date()),
        "eval_min_date": str(evaluation.dt.min().date()), "eval_max_date": str(evaluation.dt.max().date()),
        "common_scored_rows": int(len(common)), "models": list(predictions),
        "quantiles": list(QUANTILES), "max_series_setting": int(args.max_series),
        "seed": int(args.seed), "python_version": platform.python_version(),
        "pandas_version": pd.__version__, "platform": platform.platform(),
        "tft_epochs": int(args.tft_epochs), "tft_batch_size": int(args.tft_batch_size),
        "chronos_requested_batch_size": int(args.inference_batch_size),
        "chronos_samples_per_series": 50, "chronos_cpu_batch_cap": 8,
        "model_runtime_seconds": times,
        "total_runtime_seconds": round(time.perf_counter() - start, 3),
        "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest, overall, by_status, by_date, importance, times)
    print("\nOVERALL RESULTS")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nRESULTS BY STOCKOUT STATUS")
    print(by_status.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved metrics, row-level forecasts, analysis and plots under: {RESULTS_DIR}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-series", type=int, default=0,
                        help="number of real shared store-product series; 0 uses all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--models", nargs="+", choices=["lightgbm", "tft", "chronos"],
                        default=["lightgbm", "tft", "chronos"])
    parser.add_argument("--tft-epochs", type=int, default=8)
    parser.add_argument("--tft-batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="optional output directory; defaults to project results/")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
