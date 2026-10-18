from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Coffee Shop Sales", page_icon="☕", layout="wide")

PALETTE = {
    "teal": "#087F8C",
    "teal_dark": "#075B63",
    "orange": "#E99A3C",
    "purple": "#7656B5",
    "blue": "#4D83B5",
    "rose": "#D66B75",
    "green": "#5A9B73",
    "ink": "#20343A",
    "muted": "#64777C",
    "grid": "#E8EEF0",
}
CHART_COLORS = [
    PALETTE["teal"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["blue"],
    PALETTE["rose"],
    PALETTE["green"],
]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = CHART_COLORS

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #F6F8F8; color: #20343A; }
      .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #20343A; letter-spacing: -0.025em; }
      h1 { font-weight: 750; }
      h2, h3 { margin-top: 0.5rem; }
      [data-testid="stSidebar"] { background: #EDF3F3; }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #20343A !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #20343A !important;
        border-color: #C9D6D8 !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #20343A !important;
      }
      [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #DCEDEC !important;
        color: #075B63 !important;
      }
      [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #075B63 !important;
      }
      [data-testid="stSidebar"] button {
        color: #20343A !important;
      }
      [data-testid="stRadio"] label,
      [data-testid="stRadio"] label p {
        color: #20343A !important;
      }
      [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E2EAEA; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 2px 8px rgba(32, 52, 58, 0.04);
      }
      [data-testid="stMetricLabel"] { color: #64777C; font-size: 0.88rem; }
      [data-testid="stMetricValue"] { color: #075B63; font-weight: 700; }
      div[data-testid="stVerticalBlock"] div[data-testid="stPlotlyChart"] {
        background: #FFFFFF; border: 1px solid #E7ECEC; border-radius: 14px;
        padding: 8px 8px 0 8px; box-shadow: 0 2px 8px rgba(32, 52, 58, 0.035);
      }
      [data-testid="stDataFrame"] { border: 1px solid #E2EAEA; border-radius: 12px; }
      [data-testid="stDownloadButton"] button {
        background: #087F8C !important; color: #FFFFFF !important;
        border: 1px solid #075B63 !important; border-radius: 9px;
      }
      [data-testid="stDownloadButton"] button p,
      [data-testid="stDownloadButton"] button span {
        color: #FFFFFF !important;
      }
      hr { border-color: #E0E8E8; }
    </style>
    """,
    unsafe_allow_html=True,
)

folder = Path(__file__).resolve().parent
file_data = folder / "data_prepared" / "coffee_sales_prepared.csv"


@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data["transaction_date"] = pd.to_datetime(data["transaction_date"])
    data["transaction_datetime"] = pd.to_datetime(data["transaction_datetime"])
    return data


def format_angka(value, decimals=2):
    if pd.isna(value):
        return "Belum tersedia"
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def summarize(data):
    total = data["sales"].sum()
    transactions = data["transaction_id"].nunique()
    average = total / transactions if transactions else None
    return total, transactions, average


def style_chart(fig, height=390):
    fig.update_layout(
        height=height,
        font={"family": "Arial, sans-serif", "size": 12, "color": PALETTE["ink"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 18, "r": 18, "t": 24, "b": 24},
        hoverlabel={
            "bgcolor": PALETTE["ink"],
            "font": {"color": "#FFFFFF", "size": 12},
            "bordercolor": PALETTE["ink"],
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(
        showline=False,
        zeroline=False,
        gridcolor=PALETTE["grid"],
        tickfont={"color": PALETTE["muted"]},
        title_font={"color": PALETTE["muted"]},
    )
    fig.update_yaxes(
        showline=False,
        zeroline=False,
        gridcolor=PALETTE["grid"],
        tickfont={"color": PALETTE["muted"]},
        title_font={"color": PALETTE["muted"]},
    )
    return fig


if not file_data.exists():
    st.error("File data belum ada. Jalankan audit_prepare_coffee.py terlebih dahulu.")
    st.stop()

df = load_data(str(file_data), file_data.stat().st_mtime_ns)
min_date = df["transaction_date"].min().date()
max_date = df["transaction_date"].max().date()

st.title("☕ Dashboard Coffee Shop Sales")
st.caption(
    f"Periode data: {min_date:%d %b %Y}–{max_date:%d %b %Y}. "
    "Mata uang tidak dinyatakan pada file sumber."
)

# Semua filter di sidebar dipakai oleh kartu, grafik, insight, dan unduhan.
st.sidebar.header("Filter data")
stores = sorted(df["store_location"].dropna().unique().tolist())
requested_stores = [
    value for value in st.query_params.get("stores", "").split(",") if value in stores
]
selected_stores = st.sidebar.multiselect(
    "Gerai", stores, default=requested_stores or stores, key="store_filter"
)
categories = sorted(df["product_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Kategori produk", categories, default=categories
)
date_range = st.sidebar.date_input(
    "Rentang tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
st.sidebar.caption("Semua kartu, grafik, dan tabel mengikuti filter ini.")

if not selected_stores or not selected_categories:
    st.info("Pilih minimal satu gerai dan satu kategori.")
    st.stop()
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir.")
    st.stop()

start_date, end_date = date_range
if start_date > end_date:
    st.error("Tanggal awal harus sebelum atau sama dengan tanggal akhir.")
    st.stop()

start_ts = pd.Timestamp(start_date)
end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
scope_mask = (
    df["store_location"].isin(selected_stores)
    & df["product_category"].isin(selected_categories)
)
filtered = df.loc[
    scope_mask
    & df["transaction_date"].ge(start_ts)
    & df["transaction_date"].lt(end_exclusive)
].copy()

if filtered.empty:
    st.warning("Tidak ada transaksi pada kombinasi filter tersebut.")
    st.stop()

# Bandingkan dengan rentang tepat sebelumnya yang jumlah harinya sama.
period_days = (end_date - start_date).days + 1
previous_start = start_ts - pd.Timedelta(days=period_days)
previous_end = start_ts
previous_data = df.loc[
    scope_mask
    & df["transaction_date"].ge(previous_start)
    & df["transaction_date"].lt(previous_end)
].copy()
current_total, current_transactions, current_average = summarize(filtered)
previous_total, previous_transactions, previous_average = summarize(previous_data)
monthly_kpi = (
    filtered.assign(month=filtered["transaction_date"].dt.strftime("%Y-%m"))
    .groupby("month")["sales"]
    .sum()
    .sort_index()
)
last_month = monthly_kpi.index[-1]
comparison_month = monthly_kpi.index[-2] if len(monthly_kpi) > 1 else None
monthly_growth = None
if comparison_month is not None and monthly_kpi.loc[comparison_month] != 0:
    monthly_growth = (
        (monthly_kpi.loc[last_month] - monthly_kpi.loc[comparison_month])
        / monthly_kpi.loc[comparison_month]
        * 100
    )

st.subheader("Ringkasan dan perbandingan periode")
st.caption(
    f"Periode aktif: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
    f"({period_days} hari). Pembanding: "
    f"{previous_start.date():%d %b %Y}–{(start_ts - pd.Timedelta(days=1)).date():%d %b %Y}."
)

delta_sales = (
    (current_total - previous_total) / previous_total * 100
    if previous_total
    else None
)
delta_transactions = (
    (current_transactions - previous_transactions) / previous_transactions * 100
    if previous_transactions
    else None
)
delta_average = (
    (current_average - previous_average) / previous_average * 100
    if previous_average
    else None
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Penjualan periode aktif",
    format_angka(current_total),
    f"{format_angka(delta_sales)}%" if delta_sales is not None else None,
)
c2.metric(
    "Transaksi tercatat",
    format_angka(current_transactions, 0),
    f"{format_angka(delta_transactions)}%" if delta_transactions is not None else None,
)
c3.metric(
    "Rata-rata per transaksi",
    format_angka(current_average),
    f"{format_angka(delta_average)}%" if delta_average is not None else None,
)
c4.metric(
    f"Growth {last_month} vs {comparison_month or 'bulan sebelumnya'}",
    f"{format_angka(monthly_growth)}%" if monthly_growth is not None else "Belum tersedia",
)
st.caption(f"Total unit terjual pada periode aktif: {format_angka(filtered['transaction_qty'].sum(), 0)} unit.")

if previous_data.empty:
    st.info(
        "Data periode sebelumnya tidak tersedia untuk filter yang dipilih; "
        "perubahan persentase belum dapat dihitung."
    )

st.divider()
st.subheader("Tren penjualan")
granularity = st.radio(
    "Kelompokkan tren berdasarkan",
    ["Harian", "Mingguan", "Bulanan"],
    horizontal=True,
    key="trend_granularity",
)
trend_data = filtered.copy()
if granularity == "Harian":
    trend_data["periode"] = trend_data["transaction_date"].dt.strftime("%Y-%m-%d")
elif granularity == "Mingguan":
    trend_data["periode"] = (
        trend_data["transaction_date"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    )
else:
    trend_data["periode"] = trend_data["transaction_date"].dt.strftime("%Y-%m")
trend = trend_data.groupby("periode", as_index=False).agg(
    total_sales=("sales", "sum"),
    transactions=("transaction_id", "nunique"),
)
fig_trend = px.line(
    trend,
    x="periode",
    y="total_sales",
    markers=True,
    hover_data={"transactions": True},
    labels={"periode": "Periode", "total_sales": "Penjualan", "transactions": "Transaksi"},
)
fig_trend.update_traces(
    line_color=PALETTE["teal"],
    line_width=3,
    marker={"size": 8, "color": PALETTE["teal"], "line": {"width": 2, "color": "#FFFFFF"}},
    connectgaps=False,
)
fig_trend.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
style_chart(fig_trend, height=390)
st.plotly_chart(fig_trend, use_container_width=True, key="trend")
peak_period = trend.loc[trend["total_sales"].idxmax()]
if len(trend) > 1:
    low_period = trend.loc[trend["total_sales"].idxmin()]
    st.markdown(
        f"**Kesimpulan:** Penjualan tertinggi terjadi pada {peak_period['periode']} "
        f"({format_angka(peak_period['total_sales'])}) dan terendah pada "
        f"{low_period['periode']} ({format_angka(low_period['total_sales'])})."
    )
else:
    st.markdown(
        f"**Kesimpulan:** Pada periode terpilih, penjualan tercatat "
        f"{format_angka(peak_period['total_sales'])}."
    )

with st.expander("Lihat perbandingan setiap bulan"):
    month_data = filtered.copy()
    month_data["Bulan"] = month_data["transaction_date"].dt.strftime("%Y-%m")
    monthly = month_data.groupby("Bulan", as_index=False).agg(
        Penjualan=("sales", "sum"),
        Transaksi=("transaction_id", "nunique"),
        Unit=("transaction_qty", "sum"),
    )
    monthly["Rata-rata transaksi"] = monthly["Penjualan"] / monthly["Transaksi"]
    monthly["Perubahan penjualan (%)"] = monthly["Penjualan"].pct_change() * 100
    st.dataframe(
        monthly.style.format(
            {
                "Penjualan": "{:,.2f}",
                "Transaksi": "{:,.0f}",
                "Unit": "{:,.0f}",
                "Rata-rata transaksi": "{:,.2f}",
                "Perubahan penjualan (%)": "{:,.2f}%",
            },
            na_rep="Belum tersedia",
            decimal=",",
            thousands=".",
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Perbandingan gerai dan kategori")
left, right = st.columns(2)
with left:
    store_compare = filtered.groupby("store_location", as_index=False).agg(
        Penjualan=("sales", "sum"),
        Transaksi=("transaction_id", "nunique"),
    ).sort_values("Penjualan", ascending=False)
    fig_store = px.bar(
        store_compare,
        x="store_location",
        y="Penjualan",
        color="store_location",
        text_auto=".2s",
        labels={"store_location": "Gerai", "Penjualan": "Penjualan"},
        color_discrete_sequence=CHART_COLORS,
        hover_data={"Transaksi": True},
    )
    fig_store.update_layout(showlegend=False)
    style_chart(fig_store, height=390)
    st.plotly_chart(fig_store, use_container_width=True, key="stores")
    top_store_chart = store_compare.iloc[0]
    st.markdown(
        f"**Kesimpulan:** {top_store_chart['store_location']} mencatat penjualan "
        f"tertinggi sebesar {format_angka(top_store_chart['Penjualan'])} "
        f"({format_angka(top_store_chart['Penjualan'] / current_total * 100)}% dari total)."
    )

with right:
    category_compare = filtered.groupby("product_category", as_index=False).agg(
        Penjualan=("sales", "sum"),
        Unit=("transaction_qty", "sum"),
    ).sort_values("Penjualan", ascending=False)
    fig_category = px.pie(
        category_compare,
        names="product_category",
        values="Penjualan",
        hole=0.55,
        color_discrete_sequence=CHART_COLORS,
        hover_data={"Unit": True},
    )
    fig_category.update_traces(
        textinfo="none",
        hovertemplate=(
            "%{label}<br>Penjualan: %{value:,.2f}<br>"
            "Kontribusi: %{percent}<extra></extra>"
        ),
    )
    fig_category.update_layout(legend_title_text="Kategori")
    fig_category.update_traces(
        marker={"line": {"color": "#FFFFFF", "width": 3}},
        insidetextfont={"color": "#FFFFFF"},
    )
    style_chart(fig_category, height=390)
    st.plotly_chart(fig_category, use_container_width=True, key="categories")
    top_category_chart = category_compare.iloc[0]
    st.markdown(
        f"**Kesimpulan:** Kategori {top_category_chart['product_category']} "
        f"memiliki kontribusi terbesar, yaitu "
        f"{format_angka(top_category_chart['Penjualan'] / current_total * 100)}% "
        "dari penjualan pada filter ini."
    )

st.divider()
st.subheader("Bandingkan dua gerai berdampingan")
st.caption("Perbandingan memakai rentang tanggal dan kategori aktif di sidebar.")
compare_left, compare_right = st.columns(2)
with compare_left:
    compare_store_a = st.selectbox("Gerai A", stores, index=0, key="compare_store_a")
with compare_right:
    default_b = 1 if len(stores) > 1 else 0
    compare_store_b = st.selectbox("Gerai B", stores, index=default_b, key="compare_store_b")

if compare_store_a == compare_store_b:
    st.info("Pilih dua gerai yang berbeda untuk melihat perbandingan.")
else:
    compare_data = df.loc[
        df["store_location"].isin([compare_store_a, compare_store_b])
        & df["product_category"].isin(selected_categories)
        & df["transaction_date"].ge(start_ts)
        & df["transaction_date"].lt(end_exclusive)
    ].copy()
    data_a = compare_data.loc[compare_data["store_location"] == compare_store_a]
    data_b = compare_data.loc[compare_data["store_location"] == compare_store_b]
    total_a, transactions_a, average_a = summarize(data_a)
    total_b, transactions_b, average_b = summarize(data_b)
    units_a = int(data_a["transaction_qty"].sum())
    units_b = int(data_b["transaction_qty"].sum())
    average_difference = (
        average_b - average_a
        if average_a is not None and average_b is not None
        else None
    )
    def signed_number(value, decimals):
        if value is None or pd.isna(value):
            return "Belum tersedia"
        sign = "+" if value > 0 else ""
        return f"{sign}{format_angka(value, decimals)}"

    comparison_table = pd.DataFrame(
        {
            "Metrik": ["Penjualan", "Transaksi tercatat", "Rata-rata per transaksi", "Unit terjual"],
            compare_store_a: [
                format_angka(total_a),
                format_angka(transactions_a, 0),
                format_angka(average_a),
                format_angka(units_a, 0),
            ],
            compare_store_b: [
                format_angka(total_b),
                format_angka(transactions_b, 0),
                format_angka(average_b),
                format_angka(units_b, 0),
            ],
            "Selisih (B - A)": [
                signed_number(total_b - total_a, 2),
                signed_number(transactions_b - transactions_a, 0),
                signed_number(average_difference, 2),
                signed_number(units_b - units_a, 0),
            ],
        }
    )
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    if total_a > 0:
        sales_change_pct = (total_b - total_a) / total_a * 100
        if total_b >= total_a:
            comparison_sentence = (
                f"{compare_store_b} mencatat penjualan "
                f"{format_angka(sales_change_pct)}% lebih tinggi daripada {compare_store_a}."
            )
        else:
            comparison_sentence = (
                f"{compare_store_b} mencatat penjualan "
                f"{format_angka(abs(sales_change_pct))}% lebih rendah daripada {compare_store_a}."
            )
    elif total_b > 0:
        comparison_sentence = (
            f"{compare_store_b} mencatat penjualan, sedangkan {compare_store_a} "
            "tidak memiliki penjualan pada filter ini."
        )
    else:
        comparison_sentence = "Kedua gerai belum memiliki penjualan pada filter ini."
    st.markdown(f"**Kesimpulan:** {comparison_sentence}")

    compare_data["Bulan"] = compare_data["transaction_date"].dt.strftime("%Y-%m")
    compare_monthly = compare_data.groupby(["Bulan", "store_location"], as_index=False).agg(
        Penjualan=("sales", "sum")
    )
    compare_chart = px.bar(
        compare_monthly,
        x="Bulan",
        y="Penjualan",
        color="store_location",
        barmode="group",
        category_orders={"store_location": [compare_store_a, compare_store_b]},
        labels={"Bulan": "Bulan", "Penjualan": "Penjualan", "store_location": "Gerai"},
        color_discrete_sequence=[PALETTE["teal"], PALETTE["orange"]],
    )
    compare_chart.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
    style_chart(compare_chart, height=380)
    st.plotly_chart(compare_chart, use_container_width=True, key="store_comparison")

st.divider()
st.subheader("Produk dan pola waktu")
left, right = st.columns(2)
with left:
    top_products = (
        filtered.groupby(["product_id", "product_detail"], as_index=False)
        .agg(Penjualan=("sales", "sum"), Unit=("transaction_qty", "sum"))
        .nlargest(10, "Penjualan")
        .sort_values("Penjualan", ascending=True)
    )
    top_products["Produk"] = (
        top_products["product_detail"] + " (" + top_products["product_id"].astype(str) + ")"
    )
    fig_products = px.bar(
        top_products,
        x="Penjualan",
        y="Produk",
        orientation="h",
        color_discrete_sequence=[PALETTE["purple"]],
        hover_data={"Unit": True},
        labels={"Penjualan": "Penjualan", "Produk": "Produk"},
    )
    fig_products.update_traces(marker_line_width=0, opacity=0.92)
    style_chart(fig_products, height=460)
    st.plotly_chart(fig_products, use_container_width=True, key="products")
    top_product_chart = top_products.iloc[-1]
    st.markdown(
        f"**Kesimpulan:** Produk {top_product_chart['product_detail']} "
        f"(ID {top_product_chart['product_id']}) menjadi produk teratas dengan "
        f"penjualan {format_angka(top_product_chart['Penjualan'])}."
    )

with right:
    by_hour = filtered.groupby("hour", as_index=False).agg(
        Transaksi=("transaction_id", "nunique")
    )
    by_hour["Jam"] = by_hour["hour"].astype(int).astype(str).str.zfill(2) + ":00"
    fig_hour = px.bar(
        by_hour,
        x="Jam",
        y="Transaksi",
        text_auto=True,
        color_discrete_sequence=[PALETTE["orange"]],
        labels={"Jam": "Jam transaksi", "Transaksi": "Transaksi tercatat"},
    )
    fig_hour.update_layout(xaxis_type="category")
    fig_hour.update_traces(marker_line_width=0, opacity=0.92)
    style_chart(fig_hour, height=460)
    st.plotly_chart(fig_hour, use_container_width=True, key="hours")
    peak_hour = by_hour.loc[by_hour["Transaksi"].idxmax()]
    st.markdown(
        f"**Kesimpulan:** Jam {peak_hour['Jam']} memiliki transaksi tercatat "
        f"terbanyak ({format_angka(peak_hour['Transaksi'], 0)}). Ini menunjukkan "
        "volume transaksi, bukan panjang antrean."
    )

weekday_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
weekday = filtered.groupby("day_name_id", as_index=False).agg(
    Penjualan=("sales", "sum"),
    Transaksi=("transaction_id", "nunique"),
)
weekday["day_name_id"] = pd.Categorical(
    weekday["day_name_id"], categories=weekday_order, ordered=True
)
weekday = weekday.sort_values("day_name_id")
fig_weekday = px.bar(
    weekday,
    x="day_name_id",
    y="Transaksi",
    text_auto=True,
    color_discrete_sequence=[PALETTE["teal"]],
    labels={"day_name_id": "Hari", "Transaksi": "Transaksi tercatat"},
    hover_data={"Penjualan": ":,.2f"},
)
fig_weekday.update_layout(xaxis_type="category")
fig_weekday.update_traces(marker_line_width=0, opacity=0.92)
style_chart(fig_weekday, height=350)
st.plotly_chart(fig_weekday, use_container_width=True, key="weekdays")
peak_weekday = weekday.loc[weekday["Transaksi"].idxmax()]
st.markdown(
    f"**Kesimpulan:** {peak_weekday['day_name_id']} memiliki transaksi tercatat "
    f"terbanyak ({format_angka(peak_weekday['Transaksi'], 0)}) pada periode terpilih."
)

st.divider()
st.subheader("Insight otomatis untuk filter saat ini")
insight_cols = st.columns(3)
best_month = trend.loc[trend["total_sales"].idxmax()]
best_store = store_compare.iloc[0]
best_category = category_compare.iloc[0]
best_hour = by_hour.loc[by_hour["Transaksi"].idxmax()]
best_weekday = weekday.loc[weekday["Transaksi"].idxmax()]
insight_cols[0].markdown(
    f"**Periode terkuat:** {best_month['periode']} dengan penjualan "
    f"{format_angka(best_month['total_sales'])}."
)
insight_cols[1].markdown(
    f"**Gerai teratas:** {best_store['store_location']} dengan penjualan "
    f"{format_angka(best_store['Penjualan'])}."
)
insight_cols[2].markdown(
    f"**Kategori terbesar:** {best_category['product_category']} menyumbang "
    f"{format_angka(best_category['Penjualan'] / current_total * 100)}% penjualan."
)
st.caption(
    f"Jam dengan transaksi terbanyak: {best_hour['Jam']} "
    f"({format_angka(best_hour['Transaksi'], 0)} transaksi). "
    f"Hari terbanyak: {best_weekday['day_name_id']} "
    f"({format_angka(best_weekday['Transaksi'], 0)} transaksi). "
    "Angka ini deskriptif dan tidak mengukur panjang antrean atau jumlah pelanggan unik."
)

st.divider()
st.subheader("Data transaksi")
download_df = filtered.sort_values("transaction_datetime", ascending=False)
csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Unduh transaksi sesuai filter (CSV)",
    data=csv_bytes,
    file_name=f"coffee_sales_{start_date}_{end_date}.csv",
    mime="text/csv",
)
st.caption(
    f"Menampilkan paling banyak 1.000 dari {len(download_df):,} baris terbaru. "
    "Unduhan memuat seluruh baris yang sesuai filter."
)
st.dataframe(
    download_df.head(1000)[
        [
            "transaction_datetime",
            "transaction_id",
            "store_location",
            "product_category",
            "product_detail",
            "transaction_qty",
            "unit_price",
            "sales",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Transaksi tercatat dihitung dari transaction_id unik. "
    "Rata-rata adalah nilai penjualan per transaction_id dalam dataset."
)
