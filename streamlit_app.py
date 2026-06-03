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
PRED_DIR = Path("data/predictions")

ASSETS = ["BTC", "ETH", "XRP", "ADA", "AVAX"]
TIMEFRAME_MAP = {
    "4h": "4h",
    "1day": "1d",
}


# ----------------------------
# Helpers
# ----------------------------
def load_asset_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_prediction_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "path_id" in df.columns:
        df["path_id"] = pd.to_numeric(df["path_id"], errors="coerce")
    df = df.sort_values(["timestamp", "path_id"]).reset_index(drop=True)
    return df


def get_selected_assets() -> list[str]:
    st.sidebar.subheader("Assets")
    selected = []
    for asset in ASSETS:
        if st.sidebar.checkbox(asset, value=(asset == "BTC")):
            selected.append(asset)
    return selected


def load_selected_data(asset: str, timeframe: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{asset.lower()}_{timeframe}.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    return load_asset_data(str(file_path), file_path.stat().st_mtime)


def load_selected_predictions(asset: str, timeframe: str) -> pd.DataFrame:
    file_path = PRED_DIR / f"{asset.lower()}_{timeframe}.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    return load_prediction_data(str(file_path), file_path.stat().st_mtime)


def build_ohlc_chart(df: pd.DataFrame, asset: str, timeframe_label: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=asset,
            )
        ]
    )
    fig.update_layout(
        title=f"{asset} - OHLC ({timeframe_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_close_chart(df: pd.DataFrame, asset: str, timeframe_label: str) -> go.Figure:
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
        title=f"{asset} - Close price ({timeframe_label})",
        xaxis_title="Date",
        yaxis_title="Close",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_normalized_chart(dfs: dict[str, pd.DataFrame], timeframe_label: str) -> go.Figure:
    fig = go.Figure()

    for asset, df in dfs.items():
        if df.empty:
            continue

        d = df.sort_values("open_time").reset_index(drop=True).copy()
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
        title=f"Normalized close variation ({timeframe_label})",
        xaxis_title="Date",
        yaxis_title="% variation vs first selected date",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    return fig


def summarize_prediction_paths(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in pred_df.columns if c.startswith("close_pred")]
    if "close_pred" in pred_df.columns:
        close_col = "close_pred"
    elif pred_cols:
        close_col = pred_cols[0]
    else:
        raise ValueError("Aucune colonne close_pred trouvée dans le fichier de prédictions.")

    summary = (
        pred_df.groupby("timestamp", as_index=False)[close_col]
        .agg(mean_close="mean", min_close="min", max_close="max")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return summary


def add_prediction_traces(fig: go.Figure, pred_summary: pd.DataFrame) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=pred_summary["timestamp"],
            y=pred_summary["mean_close"],
            mode="lines",
            name="Predicted close mean",
            line=dict(dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pd.concat([pred_summary["timestamp"], pred_summary["timestamp"][::-1]]),
            y=pd.concat([pred_summary["max_close"], pred_summary["min_close"][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prediction band",
            hoverinfo="skip",
            showlegend=True,
        )
    )
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

show_predictions = st.sidebar.checkbox("Afficher les prédictions", value=False)

selected_assets = get_selected_assets()

if not selected_assets:
    st.info("Sélectionne au moins un asset dans la sidebar.")
    st.stop()

# Load all selected assets to determine global date range
asset_data = {}
min_dates = []
max_dates = []

for asset in selected_assets:
    df = load_selected_data(asset, timeframe)
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

# Filter raw data
filtered_data = {}
for asset, df in asset_data.items():
    if df.empty:
        continue
    dff = df[(df["open_time"] >= start_date) & (df["open_time"] <= end_date)].copy()
    if not dff.empty:
        filtered_data[asset] = dff

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

    if show_predictions:
        pred_df = load_selected_predictions(asset, timeframe)

        if pred_df.empty:
            st.warning("Aucune prédiction trouvée pour cet asset/timeframe.")
        else:
            pred_df = pred_df[pred_df["timestamp"] >= df["open_time"].max()].copy()

            if pred_df.empty:
                st.warning("Les prédictions ne couvrent pas la période affichée.")
            else:
                pred_summary = summarize_prediction_paths(pred_df)
                fig = add_prediction_traces(fig, pred_summary)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.caption(f"{', '.join(selected_assets)} | {timeframe_label}")

    fig = build_normalized_chart(filtered_data, timeframe_label)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Affichage normalisé : variation du close en % par rapport à la première date de la période sélectionnée."
    )