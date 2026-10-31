"""Full-data EDA for the official FreshRetailNet-50K train/eval Parquet files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "full_dataset_eda"
FIG = OUT / "figures"
DAILY_COLUMNS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag",
    "activity_flag", "precpt", "avg_temperature", "avg_humidity",
    "avg_wind_level",
]


def read_full_data() -> pd.DataFrame:
    parts = []
    for split in ("train", "eval"):
        path = DATA / f"{split}.parquet"
        frame = pd.read_parquet(path, columns=DAILY_COLUMNS)
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
        frame["split"] = split
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data["stockout_recorded"] = data["stock_hour6_22_cnt"].fillna(0).gt(0)
    data["weekday"] = data["dt"].dt.dayofweek
    data["weekday_name"] = data["dt"].dt.day_name()
    data["month_day"] = data["dt"].dt.strftime("%m-%d")
    return data


def save_tables(df: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    numeric = ["sale_amount", "stock_hour6_22_cnt", "discount", "precpt",
               "avg_temperature", "avg_humidity", "avg_wind_level"]
    profile = {
        "rows_total": int(len(df)),
        "rows_train": int(df.split.eq("train").sum()),
        "rows_eval": int(df.split.eq("eval").sum()),
        "series_store_product": int(df[["store_id", "product_id"]].drop_duplicates().shape[0]),
        "unique_city": int(df.city_id.nunique()),
        "unique_store": int(df.store_id.nunique()),
        "unique_product": int(df.product_id.nunique()),
        "train_start": str(df.loc[df.split.eq("train"), "dt"].min().date()),
        "train_end": str(df.loc[df.split.eq("train"), "dt"].max().date()),
        "eval_start": str(df.loc[df.split.eq("eval"), "dt"].min().date()),
        "eval_end": str(df.loc[df.split.eq("eval"), "dt"].max().date()),
        "duplicate_series_dates": int(df.duplicated(["store_id", "product_id", "dt"]).sum()),
        "stockout_share": float(df.stockout_recorded.mean()),
        "zero_sale_share": float(df.sale_amount.fillna(0).eq(0).mean()),
        "missing_cells_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "sale_amount_quantiles": {str(q): float(df.sale_amount.quantile(q)) for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)},
        "sale_amount_mean": float(df.sale_amount.mean()),
        "sale_amount_std": float(df.sale_amount.std()),
    }
    (OUT / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    pd.DataFrame([{"field": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}
                  for k, v in profile.items()]).to_csv(OUT / "dataset_profile.csv", index=False)

    daily = df.groupby(["split", "dt"], as_index=False).agg(
        n=("sale_amount", "size"), mean_sale=("sale_amount", "mean"),
        median_sale=("sale_amount", "median"), std_sale=("sale_amount", "std"),
        stockout_rate=("stockout_recorded", "mean"), zero_sale_rate=("sale_amount", lambda x: x.eq(0).mean()),
        mean_discount=("discount", "mean"), mean_precpt=("precpt", "mean"),
    )
    daily.to_csv(OUT / "daily_summary_all_rows.csv", index=False)
    by_weekday = df.groupby(["split", "weekday", "weekday_name"], as_index=False).agg(
        n=("sale_amount", "size"), mean_sale=("sale_amount", "mean"), std_sale=("sale_amount", "std"),
        median_sale=("sale_amount", "median"), stockout_rate=("stockout_recorded", "mean"))
    by_weekday.to_csv(OUT / "weekday_summary_all_rows.csv", index=False)
    by_status = df.groupby(["split", "stockout_recorded"], as_index=False).agg(
        n=("sale_amount", "size"), mean_sale=("sale_amount", "mean"), median_sale=("sale_amount", "median"),
        sale_std=("sale_amount", "std"), zero_sale_rate=("sale_amount", lambda x: x.eq(0).mean()))
    by_status.to_csv(OUT / "stockout_summary_all_rows.csv", index=False)
    by_category = df.groupby("first_category_id", as_index=False).agg(
        n=("sale_amount", "size"), mean_sale=("sale_amount", "mean"), median_sale=("sale_amount", "median"),
        stockout_rate=("stockout_recorded", "mean"), mean_discount=("discount", "mean"))
    by_category.to_csv(OUT / "category_summary_all_rows.csv", index=False)
    df[numeric].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).T.to_csv(OUT / "numeric_descriptives_all_rows.csv")
    return profile


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#176B87", "eval": "#F28E2B"}

    # 1. Daily trend with variability band.
    daily = df.groupby(["split", "dt"]).sale_amount.agg(["mean", "median", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    for split, g in daily.groupby("split", sort=False):
        g=g.sort_values("dt"); ax.plot(g.dt,g["mean"],lw=2.2,color=colors[split],label=f"{split}: daily mean")
        ax.fill_between(g.dt,np.maximum(0,g["mean"]-g["std"]),g["mean"]+g["std"],color=colors[split],alpha=.12)
    ax.axvline(pd.Timestamp("2024-06-26"),color="#B23A48",ls="--",lw=1.8,label="train / eval boundary")
    ax.set(title="Full dataset: daily normalized sale_amount, mean ± 1 SD",xlabel="Date",ylabel="sale_amount (publisher-normalized)",ylim=(0,None)); ax.legend(ncol=3,fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/"01_full_daily_trend.png",dpi=180); plt.close(fig)

    # 2. Weekday effect for both splits.
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig,ax=plt.subplots(figsize=(13,6))
    sns.pointplot(data=df,x="weekday_name",y="sale_amount",hue="split",order=order,errorbar=("se",1.96),dodge=.25,markers=["o","s"],ax=ax)
    ax.set(title="Full dataset: normalized sales by weekday",xlabel="Weekday",ylabel="Mean sale_amount"); ax.tick_params(axis="x",rotation=15)
    fig.tight_layout(); fig.savefig(FIG/"02_weekday_effect.png",dpi=180); plt.close(fig)

    # 3. Long-tailed target distributions using deterministic display sample only.
    sample=df.groupby("split",group_keys=False).sample(n=min(150000,len(df)//2),random_state=17)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    sample["log1p_sale_amount"] = np.log1p(sample.sale_amount)
    sns.histplot(data=sample,x="log1p_sale_amount",hue="split",stat="density",common_norm=False,bins=70,element="step",fill=False,ax=axes[0])
    sns.boxplot(data=sample,x="split",y="sale_amount",hue="split",showfliers=False,ax=axes[1],legend=False)
    axes[0].set(title="Distribution (log1p scale; display sample)",xlabel="ln(1 + sale_amount)",ylabel="Density"); axes[1].set(title="Central spread (display truncated at plot scale)",ylabel="sale_amount")
    fig.tight_layout(); fig.savefig(FIG/"03_target_distribution.png",dpi=180); plt.close(fig)

    # 4. Heatmap: week-of-year × weekday mean.
    d=df.copy(); d["week"] = d.dt.dt.isocalendar().week.astype(int)
    h=d.groupby(["week","weekday_name"]).sale_amount.mean().unstack().reindex(columns=order)
    fig,ax=plt.subplots(figsize=(13,8)); sns.heatmap(h,cmap="YlGnBu",linewidths=.2,ax=ax,cbar_kws={"label":"Mean sale_amount"})
    ax.set(title="Calendar heatmap: mean sale_amount by ISO week and weekday",xlabel="Weekday",ylabel="ISO week")
    fig.tight_layout(); fig.savefig(FIG/"04_weekday_week_heatmap.png",dpi=180); plt.close(fig)

    # 5. Stockout group composition and observed sales, all rows.
    status=df.groupby(["split","stockout_recorded"],as_index=False).agg(n=("sale_amount","size"),mean_sale=("sale_amount","mean"),median_sale=("sale_amount","median"))
    status["label"]=np.where(status.stockout_recorded,"Stockout recorded","No stockout recorded")
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    sns.barplot(data=status,x="split",y="n",hue="label",ax=axes[0]); sns.barplot(data=status,x="split",y="mean_sale",hue="label",ax=axes[1])
    axes[0].set(title="Rows by recorded stockout",ylabel="Number of observations"); axes[1].set(title="Observed sale by stockout label",ylabel="Mean normalized sale_amount")
    fig.tight_layout(); fig.savefig(FIG/"05_stockout_composition.png",dpi=180); plt.close(fig)

    # 6. Category panel: mean sales and stockout incidence.
    cat=df.groupby("first_category_id",as_index=False).agg(n=("sale_amount","size"),mean_sale=("sale_amount","mean"),stockout_rate=("stockout_recorded","mean"))
    cat=cat.nlargest(15,"n").sort_values("mean_sale")
    cat["first_category_id"] = cat.first_category_id.astype(str)
    fig,axes=plt.subplots(1,2,figsize=(16,8),sharey=True)
    sns.barplot(data=cat,y="first_category_id",x="mean_sale",color="#176B87",ax=axes[0]); sns.barplot(data=cat,y="first_category_id",x="stockout_rate",color="#F28E2B",ax=axes[1])
    axes[0].set(title="Mean observed sales · 15 largest categories",xlabel="Mean sale_amount",ylabel="first_category_id"); axes[1].set(title="Recorded stockout rate",xlabel="Rate",ylabel="")
    fig.tight_layout(); fig.savefig(FIG/"06_category_comparison.png",dpi=180); plt.close(fig)

    # 7. Discount relationship using display sample; bins summarize the entire data.
    d=df.loc[df.discount.notna()].copy()
    d["discount_group"] = "= 1.00"
    low = d.discount < 1.0
    if low.any():
        bins = pd.qcut(d.loc[low,"discount"],q=5,duplicates="drop")
        codes = bins.cat.codes + 1
        d.loc[low,"discount_group"] = codes.map(lambda i: f"<1.00 · Q{i}").to_numpy()
    d.loc[d.discount > 1.0,"discount_group"] = "> 1.00"
    b=d.groupby("discount_group",observed=False).agg(n=("sale_amount","size"),mean_sale=("sale_amount","mean"),median_sale=("sale_amount","median"),stockout_rate=("stockout_recorded","mean")).reset_index()
    order = [f"<1.00 · Q{i}" for i in range(1,6)] + ["= 1.00", "> 1.00"]
    b["discount_group"] = pd.Categorical(b.discount_group,categories=order,ordered=True)
    b=b.sort_values("discount_group")
    b.to_csv(OUT/"discount_bin_summary_all_rows.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,6)); hb=axes[0].hexbin(d.discount,d.sale_amount,gridsize=55,bins="log",mincnt=1,cmap="viridis"); fig.colorbar(hb,ax=axes[0],label="log10(count)"); sns.barplot(data=b,y="discount_group",x="mean_sale",color="#4E9F3D",ax=axes[1])
    axes[0].set(title="Discount and observed sales (all rows)",xlabel="Discount field",ylabel="sale_amount"); axes[1].set(title="Full-data means by discount group",xlabel="Mean sale_amount",ylabel="Group (<1.0 groups split by quantiles)")
    fig.tight_layout(); fig.savefig(FIG/"07_discount_relationship.png",dpi=180); plt.close(fig)

    # 8. Numeric correlation matrix, full data.
    corr=df[["sale_amount","stock_hour6_22_cnt","discount","holiday_flag","activity_flag","precpt","avg_temperature","avg_humidity","avg_wind_level"]].corr(method="spearman")
    corr.to_csv(OUT/"spearman_correlation_all_rows.csv")
    fig,ax=plt.subplots(figsize=(11,9)); sns.heatmap(corr,annot=True,fmt=".2f",center=0,cmap="vlag",square=True,ax=ax,cbar_kws={"label":"Spearman correlation"})
    ax.set(title="Spearman correlation of numeric fields (all rows)"); fig.tight_layout(); fig.savefig(FIG/"08_numeric_correlation.png",dpi=180); plt.close(fig)

    # 9. Selected representative series, shown only for visual readability.
    keys=df.loc[df.split.eq("train"),["store_id","product_id"]].drop_duplicates().sample(n=4,random_state=17)
    plotdf=df.merge(keys,on=["store_id","product_id"],how="inner").sort_values("dt")
    fig,axes=plt.subplots(2,2,figsize=(15,8),sharex=True)
    for ax,((store,product),g) in zip(axes.flat,plotdf.groupby(["store_id","product_id"])):
        ax.plot(g.dt,g.sale_amount,color="#176B87",lw=1.4); ax.axvline(pd.Timestamp("2024-06-26"),color="#B23A48",ls="--",lw=1)
        ax.set(title=f"Store {store} · Product {product}",ylabel="sale_amount")
    fig.suptitle("Four real series selected from the full dataset (illustrative only)",y=1.02); fig.tight_layout(); fig.savefig(FIG/"09_example_series.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    df=read_full_data()
    profile=save_tables(df)
    make_figures(df)
    print(json.dumps({k:profile[k] for k in ("rows_total","rows_train","rows_eval","series_store_product","unique_city","unique_store","unique_product","train_start","train_end","eval_start","eval_end","stockout_share","zero_sale_share")},indent=2))
    print(f"Saved full-data EDA to {OUT}")


if __name__ == "__main__":
    main()
