from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Crypto OHLC Dashboard",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path("data/raw")

ASSETS = ["BTC", "ETH", "XRP", "ADA", "AVAX"]
TIMEFRAME_MAP = {
    "4h": "4h",
    "1day": "1d",
}


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_asset_data(asset: str, timeframe: str) -> pd.DataFrame:
    """
    Load parquet file for one asset and one timeframe.
    Expected file name: {asset_lower}_{timeframe}.parquet
    Example: btc_4h.parquet, btc_1d.parquet
    """
    file_path = DATA_DIR / f"{asset.lower()}_{timeframe}.parquet"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(file_path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def get_selected_assets() -> list[str]:
    selected = []
    st.sidebar.subheader("Assets")
    for asset in ASSETS:
        if st.sidebar.checkbox(asset, value=(asset == "BTC")):
            selected.append(asset)
    return selected


def build_ohlc_chart(df: pd.DataFrame, asset: str, timeframe: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=f"{asset}",
            )
        ]
    )
    fig.update_layout(
        title=f"{asset} - OHLC ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_close_chart(df: pd.DataFrame, asset: str, timeframe: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["close"],
            mode="lines",
            name=f"{asset} close",
        )
    )
    fig.update_layout(
        title=f"{asset} - Close price ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Close",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_normalized_chart(dfs: dict[str, pd.DataFrame], timeframe: str) -> go.Figure:
    fig = go.Figure()

    for asset, df in dfs.items():
        if df.empty:
            continue

        d = df.copy()
        d = d.sort_values("open_time").reset_index(drop=True)
        first_close = d["close"].iloc[0]

        if first_close == 0:
            continue

        d["close_pct"] = (d["close"] / first_close - 1.0) * 100.0

        fig.add_trace(
            go.Scatter(
                x=d["open_time"],
                y=d["close_pct"],
                mode="lines",
                name=asset,
            )
        )

    fig.update_layout(
        title=f"Normalized close variation ({timeframe})",
        xaxis_title="Date",
        yaxis_title="% variation vs first selected date",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    return fig


# ----------------------------
# UI
# ----------------------------
st.title("Crypto OHLC Dashboard")
st.write("Visualisation simple des données OHLC locales stockées en parquet.")

timeframe_label = st.sidebar.radio(
    "Timeframe",
    options=list(TIMEFRAME_MAP.keys()),
    index=0,
)
timeframe = TIMEFRAME_MAP[timeframe_label]

selected_assets = get_selected_assets()

if not selected_assets:
    st.info("Sélectionne au moins un asset dans la sidebar.")
    st.stop()

# Load all selected assets first to determine global date range
asset_data = {}
min_dates = []
max_dates = []

for asset in selected_assets:
    df = load_asset_data(asset, timeframe)
    asset_data[asset] = df
    if not df.empty:
        min_dates.append(df["open_time"].min())
        max_dates.append(df["open_time"].max())

if not min_dates or not max_dates:
    st.error("Aucune donnée trouvée pour la combinaison asset/timeframe sélectionnée.")
    st.stop()

global_min_date = min(min_dates).date()
global_max_date = max(max_dates).date()

st.sidebar.subheader("Date range")
date_range = st.sidebar.slider(
    "Select dates",
    min_value=global_min_date,
    max_value=global_max_date,
    value=(global_min_date, global_max_date),
)

start_date = pd.Timestamp(date_range[0], tz="UTC")
end_date = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

# Filter data
filtered_data = {}
for asset, df in asset_data.items():
    if df.empty:
        continue
    dff = df[(df["open_time"] >= start_date) & (df["open_time"] <= end_date)].copy()
    filtered_data[asset] = dff

# Remove empty after filtering
filtered_data = {asset: df for asset, df in filtered_data.items() if not df.empty}

if not filtered_data:
    st.warning("Aucune donnée dans la plage de dates sélectionnée.")
    st.stop()

# Main logic
if len(selected_assets) == 1:
    asset = selected_assets[0]
    df = filtered_data.get(asset, pd.DataFrame())

    if df.empty:
        st.warning("Pas de données pour cet asset sur cette période.")
        st.stop()

    st.caption(f"{asset} | {timeframe_label} | {len(df)} lignes visualisées")

    if len(df) < 365:
        fig = build_ohlc_chart(df, asset, timeframe_label)
    else:
        fig = build_close_chart(df, asset, timeframe_label)

    st.plotly_chart(fig, use_container_width=True)

else:
    # For multiple assets: normalized close variation from first selected date
    st.caption(f"{', '.join(selected_assets)} | {timeframe_label}")

    # Align on dates present for each asset (simple version: keep each asset as-is)
    # Normalization is done from the first point in the selected range for each asset.
    fig = build_normalized_chart(filtered_data, timeframe_label)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Affichage normalisé : variation du close en % par rapport à la première date de la période sélectionnée.")