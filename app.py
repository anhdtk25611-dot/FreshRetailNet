from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_NAMES = {"lightgbm": "LightGBM", "tft": "TFT", "chronos": "Chronos-T5-Small"}

st.set_page_config(page_title="FreshRetailNet-50K Dashboard", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def read_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def show_cards(row: pd.Series, note: str) -> None:
    cols = st.columns(4)
    cols[0].metric("Số dự báo", f"{int(row.get('n', 0)):,}")
    cols[1].metric("MAE P50", f"{row.get('mae_p50', float('nan')):.5f}")
    cols[2].metric("Coverage P10–P90", f"{row.get('coverage_80', float('nan')):.2%}")
    cols[3].metric("Pinball loss TB", f"{row.get('mean_pinball', float('nan')):.5f}")
    st.caption(note)


def show_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Chưa có bảng chỉ số trong gói kết quả.")
        return
    view = frame.copy()
    if "model" in view:
        view["model"] = view["model"].map(MODEL_NAMES).fillna(view["model"])
    if "stockout_status" in view:
        view["stockout_status"] = view["stockout_status"].replace({
            "No stockout recorded": "Không ghi nhận hết hàng",
            "Stockout recorded": "Có ghi nhận hết hàng",
        })
    view = view.rename(columns={
        "model": "Mô hình", "stockout_status": "Nhãn hết hàng",
        "dt": "Ngày eval", "n": "Số quan sát", "mae_p50": "MAE P50",
        "rmse_p50": "RMSE P50", "wape_p50": "WAPE P50",
        "mean_pinball": "Pinball loss TB", "coverage_80": "Coverage",
        "mean_width_80": "Độ rộng TB",
        "bias_p50_actual_minus_forecast": "Bias P50",
    })
    if "WAPE P50" in view:
        view["WAPE P50"] = view["WAPE P50"].map(lambda x: f"{x:.2%}")
    if "Coverage" in view:
        view["Coverage"] = view["Coverage"].map(lambda x: f"{x:.2%}")
    if "Ngày eval" in view:
        view["Ngày eval"] = pd.to_datetime(view["Ngày eval"]).dt.strftime("%d/%m/%Y")
    for col in ["MAE P50", "RMSE P50", "Pinball loss TB", "Độ rộng TB", "Bias P50"]:
        if col in view:
            view[col] = view[col].map(lambda x: f"{x:.5f}")
    st.dataframe(view, use_container_width=True, hide_index=True)


def show_image(relative_path: str, caption: str) -> None:
    path = ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"Chưa có hình: {caption}")


def load_panel_predictions() -> pd.DataFrame:
    frames = []
    for model in MODEL_NAMES:
        frame = read_csv(f"results/predictions_{model}.csv")
        required = {"store_id", "product_id", "dt", "sale_amount", "q0.1", "q0.5", "q0.9"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame[list(required)].copy()
        frame["dt"] = pd.to_datetime(frame["dt"])
        frame = frame.rename(columns={
            "sale_amount": f"actual_{model}", "q0.1": f"q10_{model}",
            "q0.5": f"q50_{model}", "q0.9": f"q90_{model}",
        })
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["store_id", "product_id", "dt"], how="inner")
    actual_cols = [f"actual_{m}" for m in MODEL_NAMES if f"actual_{m}" in merged]
    if actual_cols:
        merged["sale_amount"] = merged[actual_cols[0]]
    return merged.sort_values(["store_id", "product_id", "dt"])


def series_explorer() -> None:
    panel = load_panel_predictions()
    if panel.empty:
        st.info("Chưa có dự báo từng dòng của panel trong thư mục results.")
        return
    stores = sorted(panel["store_id"].dropna().unique().tolist())
    store = st.selectbox("Chọn store_id", stores)
    products = sorted(panel.loc[panel.store_id.eq(store), "product_id"].dropna().unique().tolist())
    product = st.selectbox("Chọn product_id", products)
    sample = panel[panel.store_id.eq(store) & panel.product_id.eq(product)].sort_values("dt")
    if sample.empty:
        st.warning("Không tìm thấy chuỗi dự báo.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(sample.dt, sample.sale_amount, marker="o", color="#222222",
            linewidth=2, label="Lượng bán quan sát")
    colors = {"lightgbm": "#277DA1", "tft": "#F8961E", "chronos": "#43AA8B"}
    for model, label in MODEL_NAMES.items():
        col = f"q50_{model}"
        if col in sample:
            ax.plot(sample.dt, sample[col], marker=".", linewidth=1.8,
                    color=colors[model], label=f"{label} P50")
    if {"q10_lightgbm", "q90_lightgbm"}.issubset(sample.columns):
        ax.fill_between(sample.dt, sample.q10_lightgbm, sample.q90_lightgbm,
                        color=colors["lightgbm"], alpha=.14, label="LightGBM P10–P90")
    ax.set(title=f"Chuỗi cửa hàng {store} – sản phẩm {product}",
           xlabel="Ngày eval", ylabel="sale_amount chuẩn hóa")
    ax.legend(ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.download_button(
        "Tải dự báo chuỗi này (CSV)", sample.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"forecast_store_{store}_product_{product}.csv", mime="text/csv",
    )


st.title("FreshRetailNet-50K")
st.subheader("Dashboard kết quả dự báo phân vị lượng bán")
st.write("Trang này hiển thị kết quả đã lưu; không huấn luyện lại các mô hình.")
st.info(
    "sale_amount là lượng bán quan sát đã chuẩn hóa toàn cục. Nhãn stockout chỉ dùng "
    "để phân tầng sai số; các khác biệt không chứng minh quan hệ nhân quả."
)

mode = st.sidebar.radio(
    "Chọn lượt thực nghiệm",
    ["LightGBM toàn bộ dữ liệu", "So sánh ba mô hình – panel 100 chuỗi"],
)
if mode == "LightGBM toàn bộ dữ liệu":
    base = "results/full_dataset_lightgbm"
    figs = f"{base}/figures"
    overall = read_csv(f"{base}/metrics_overall.csv")
    stockout = read_csv(f"{base}/metrics_by_stockout.csv")
    by_date = read_csv(f"{base}/metrics_by_date.csv")
    manifest = read_json(f"{base}/run_manifest.json")
    if not overall.empty:
        show_cards(overall.iloc[0], "50.000 chuỗi; 350.000 dòng eval.")
    st.caption(f"Toàn pipeline: {manifest.get('total_runtime_seconds', 'N/A')} giây.")
else:
    base = "results"
    figs = f"{base}/figures"
    overall = read_csv(f"{base}/metrics_overall.csv")
    stockout = read_csv(f"{base}/metrics_by_stockout.csv")
    by_date = read_csv(f"{base}/metrics_by_date.csv")
    manifest = read_json(f"{base}/run_manifest.json")
    lgb = overall.loc[overall.model.eq("lightgbm")] if not overall.empty else pd.DataFrame()
    if not lgb.empty:
        show_cards(lgb.iloc[0], "Panel 100 chuỗi; tối đa 700 quan sát chung.")
    runtimes = manifest.get("model_runtime_seconds", {})
    if runtimes:
        st.caption("Thời gian lượt panel: " + " · ".join(
            f"{MODEL_NAMES.get(k, k)} {v:.2f} giây" for k, v in runtimes.items()
        ))

tabs = st.tabs(["Tổng quan", "Theo stockout", "Biểu đồ", "Chuỗi dự báo", "EDA", "Tải kết quả"])
with tabs[0]:
    st.markdown("### Chỉ số tổng thể")
    show_table(overall)
    st.markdown("### Theo ngày eval")
    show_table(by_date)
with tabs[1]:
    st.markdown("### Sai số theo nhãn hết hàng ngày mục tiêu")
    st.caption("stockout được xác định bằng stock_hour6_22_cnt > 0; không đo lượng cầu bị mất.")
    show_table(stockout)
with tabs[2]:
    names = [
        ("02_model_error_comparison.png", "Sai số dự báo"),
        ("03_interval_diagnostics.png", "Coverage và độ rộng khoảng theo stockout"),
        ("04_residuals_by_stockout.png", "Phần dư theo nhãn stockout"),
        ("05_actual_vs_median.png", "Giá trị quan sát và dự báo P50"),
        ("06_example_forecast.png", "Chuỗi dự báo minh họa"),
    ]
    for filename, caption in names:
        show_image(f"{figs}/{filename}", caption)
with tabs[3]:
    if mode == "So sánh ba mô hình – panel 100 chuỗi":
        series_explorer()
    else:
        show_image(
            "results/full_dataset_lightgbm/figures/06_example_forecast.png",
            "Một chuỗi LightGBM trong bảy ngày eval",
        )
with tabs[4]:
    st.markdown("### Khám phá dữ liệu trên toàn bộ 4,85 triệu dòng")
    profile = read_csv("results/full_dataset_eda/dataset_profile.csv")
    if not profile.empty:
        st.dataframe(profile, use_container_width=True, hide_index=True)
    for filename, caption in [
        ("01_full_daily_trend.png", "Trung bình sale_amount theo ngày"),
        ("02_weekday_effect.png", "Sale_amount theo thứ trong tuần"),
        ("03_target_distribution.png", "Phân phối sale_amount sau log1p"),
        ("04_weekday_week_heatmap.png", "Heatmap tuần ISO và thứ"),
        ("05_stockout_composition.png", "Sale_amount theo nhãn stockout"),
        ("06_category_comparison.png", "So sánh nhóm category"),
        ("07_discount_relationship.png", "Discount và sale_amount"),
        ("08_numeric_correlation.png", "Tương quan Spearman"),
        ("09_example_series.png", "Bốn chuỗi minh họa"),
    ]:
        show_image(f"results/full_dataset_eda/figures/{filename}", caption)
with tabs[5]:
    st.markdown("### Tải bảng kết quả")
    for filename, label in [
        ("metrics_overall.csv", "Chỉ số tổng thể"),
        ("metrics_by_stockout.csv", "Chỉ số theo stockout"),
        ("metrics_by_date.csv", "Chỉ số theo ngày"),
    ]:
        path = ROOT / base / filename
        if path.is_file():
            st.download_button(
                f"Tải {label}", path.read_bytes(), file_name=filename,
                mime="text/csv", key=f"{base}_{filename}",
            )
    st.markdown("### Cấu hình lượt chạy")
    st.json(manifest)

st.divider()
st.caption(
    "Nguồn: FreshRetailNet-50K. Kết quả so sánh ba mô hình chỉ áp dụng cho panel "
    "100 chuỗi; xem README để biết thêm phạm vi và giới hạn."
)
