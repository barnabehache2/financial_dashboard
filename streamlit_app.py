from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/raw")
PRED_DIR = Path("data/predictions")

ASSETS = ["BTC", "ETH", "XRP", "ADA", "AVAX"]

ASSET_COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "XRP": "#00AAE4",
    "ADA": "#0033AD",
    "AVAX": "#E84142",
}

TIMEFRAME_MAP = {
    "4h": "4h",
    "1day": "1d",
}

CANDLE_THRESHOLD = 365

# ----------------------------
# Custom CSS — dark financial theme
# ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] .stCheckbox label span {
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
    }
    [data-testid="stSidebar"] hr {
        border-color: #30363d;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e !important;
        padding: 0.6rem 0 0.3rem 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 0.5rem;
    }

    /* Main title */
    .dashboard-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #e6edf3;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .dashboard-subtitle {
        font-size: 0.8rem;
        color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.06em;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        flex: 1;
        min-width: 100px;
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .metric-value.positive { color: #3fb950; }
    .metric-value.negative { color: #f85149; }

    /* Chart container */
    .chart-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.3rem;
    }

    /* Prediction toggle */
    .pred-badge {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #58a6ff;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* Info / warning boxes */
    .stAlert {
        background-color: #161b22 !important;
        border-color: #30363d !important;
        color: #c9d1d9 !important;
    }

    /* Plotly chart borders */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Divider */
    .divider {
        height: 1px;
        background: #30363d;
        margin: 1rem 0;
    }

    /* ── Kronos metrics panel ─────────────────────────────── */
    .kronos-panel {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem 0.8rem 1.2rem;
        margin-top: 1.2rem;
    }
    .kronos-panel-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.4rem;
        margin-bottom: 0.9rem;
    }
    .kmetric-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.7rem;
    }
    /* Tooltip wrapper */
    .has-tip {
        position: relative;
        cursor: help;
    }
    .has-tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #1c2128;
        border: 1px solid #388bfd55;
        color: #c9d1d9;
        font-size: 0.72rem;
        font-family: 'IBM Plex Sans', sans-serif;
        line-height: 1.5;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        white-space: normal;
        width: 240px;
        z-index: 9999;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.18s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    .has-tip:hover::after { opacity: 1; }

    /* Metric cards inside kronos panel */
    .kmetric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 7px;
        padding: 0.6rem 0.9rem;
        flex: 1;
        min-width: 90px;
        transition: border-color 0.15s;
    }
    .kmetric-card:hover { border-color: #388bfd55; }
    .kmetric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.2rem;
    }
    .kmetric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .kmetric-value.positive { color: #3fb950; }
    .kmetric-value.negative { color: #f85149; }
    .kmetric-value.neutral  { color: #e3b341; }

    /* Signal badges */
    .signal-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.2rem;
    }
    .signal-card {
        border-radius: 7px;
        padding: 0.55rem 1rem;
        flex: 1;
        min-width: 140px;
    }
    .signal-card.long  { background: rgba(63,185,80,0.10);  border: 1px solid rgba(63,185,80,0.35); }
    .signal-card.short { background: rgba(248,81,73,0.10);  border: 1px solid rgba(248,81,73,0.35); }
    .signal-card.neutral{ background: rgba(227,179,65,0.10); border: 1px solid rgba(227,179,65,0.30); }
    .signal-card.stay  { background: rgba(63,185,80,0.07);  border: 1px solid rgba(63,185,80,0.20); }
    .signal-card.exit  { background: rgba(248,81,73,0.07);  border: 1px solid rgba(248,81,73,0.20); }
    .signal-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.2rem;
    }
    .signal-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .signal-value.long    { color: #3fb950; }
    .signal-value.short   { color: #f85149; }
    .signal-value.neutral { color: #e3b341; }
    .signal-value.stay    { color: #3fb950; }
    .signal-value.exit    { color: #f85149; }
    .signal-reason {
        font-size: 0.68rem;
        color: #8b949e;
        line-height: 1.4;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .kronos-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: #484f58;
        margin-top: 0.7rem;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Plotly dark theme base
# ----------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=11),
    xaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        zeroline=False,
        tickprefix="$",
    ),
    legend=dict(
        bgcolor="#161b22",
        bordercolor="#30363d",
        borderwidth=1,
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    height=560,
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#161b22",
        bordercolor="#30363d",
        font=dict(family="IBM Plex Mono, monospace", color="#e6edf3"),
    ),
)

# ----------------------------
# Cached data loaders
# The file_mtime param is used as a cache key:
# when the parquet file changes on disk, mtime changes
# → cache is invalidated → fresh data is loaded.
# ----------------------------
@st.cache_data(show_spinner=False)
def load_asset_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_prediction_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "path_id" in df.columns:
        df["path_id"] = pd.to_numeric(df["path_id"], errors="coerce")
    df = df.sort_values(["timestamp", "path_id"]).reset_index(drop=True)
    return df


# ----------------------------
# File helpers
# ----------------------------
def get_file_mtime(path: Path) -> float:
    """Returns mtime if file exists, else -1."""
    return path.stat().st_mtime if path.exists() else -1.0


def load_selected_data(asset: str, timeframe: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{asset.lower()}_{timeframe}.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    return load_asset_data(str(file_path), get_file_mtime(file_path))


def load_selected_predictions(asset: str, timeframe: str) -> pd.DataFrame:
    file_path = PRED_DIR / f"{asset.lower()}_{timeframe}.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    return load_prediction_data(str(file_path), get_file_mtime(file_path))


# ----------------------------
# Color helper
# ----------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------
# Chart builders
# ----------------------------
def build_ohlc_chart(df: pd.DataFrame, asset: str, timeframe_label: str) -> go.Figure:
    color = ASSET_COLORS.get(asset, "#58a6ff")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=asset,
                increasing_line_color="#3fb950",
                decreasing_line_color="#f85149",
                increasing_fillcolor="rgba(63,185,80,0.20)",
                decreasing_fillcolor="rgba(248,81,73,0.20)",
            )
        ]
    )
    layout = {**PLOTLY_LAYOUT}
    layout["title"] = dict(
        text=f"{asset} · OHLC · {timeframe_label}",
        font=dict(size=13, color="#8b949e"),
    )
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


def build_close_chart(df: pd.DataFrame, asset: str, timeframe_label: str) -> go.Figure:
    color = ASSET_COLORS.get(asset, "#58a6ff")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["close"],
            mode="lines",
            name=f"{asset}",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.09),
        )
    )
    layout = {**PLOTLY_LAYOUT}
    layout["title"] = dict(
        text=f"{asset} · Close · {timeframe_label}",
        font=dict(size=13, color="#8b949e"),
    )
    fig.update_layout(**layout)
    return fig


def build_normalized_chart(dfs: dict, timeframe_label: str) -> go.Figure:
    fig = go.Figure()

    for asset, df in dfs.items():
        if df.empty:
            continue
        d = df.sort_values("open_time").reset_index(drop=True).copy()
        first_close = d["close"].iloc[0]
        if first_close == 0:
            continue
        d["close_pct"] = (d["close"] / first_close - 1.0) * 100.0
        color = ASSET_COLORS.get(asset, "#58a6ff")
        fig.add_trace(
            go.Scatter(
                x=d["open_time"],
                y=d["close_pct"],
                mode="lines",
                name=asset,
                line=dict(color=color, width=1.8),
            )
        )

    layout = {**PLOTLY_LAYOUT}
    layout["title"] = dict(
        text=f"Normalized close variation · {timeframe_label}",
        font=dict(size=13, color="#8b949e"),
    )
    layout["yaxis"] = dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        zeroline=True,
        zerolinecolor="#30363d",
        zerolinewidth=1,
        ticksuffix="%",
        tickprefix="",
    )
    fig.update_layout(**layout)
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#30363d")
    return fig


def summarize_prediction_paths(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in pred_df.columns if c.startswith("close_pred")]
    if "close_pred" in pred_df.columns:
        close_col = "close_pred"
    elif pred_cols:
        close_col = pred_cols[0]
    else:
        raise ValueError("No close_pred column found in prediction file.")

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
            name="Forecast (mean)",
            line=dict(color="#58a6ff", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat(
                [pred_summary["timestamp"], pred_summary["timestamp"][::-1]]
            ),
            y=pd.concat(
                [pred_summary["max_close"], pred_summary["min_close"][::-1]]
            ),
            fill="toself",
            fillcolor="rgba(88, 166, 255, 0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Forecast range",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    return fig


# ----------------------------
# Metric helpers
# ----------------------------
def fmt_price(v: float) -> str:
    if v >= 1_000:
        return f"${v:,.0f}"
    return f"${v:,.4f}"


def render_metrics(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) >= 2 else last["close"]
    pct_change = (last["close"] / prev_close - 1) * 100 if prev_close else 0
    direction = "positive" if pct_change >= 0 else "negative"
    arrow = "▲" if pct_change >= 0 else "▼"

    metrics = [
        ("Open", fmt_price(last["open"])),
        ("High", fmt_price(last["high"])),
        ("Low", fmt_price(last["low"])),
        ("Close", fmt_price(last["close"])),
        ("Change", f'{arrow} {abs(pct_change):.2f}%', direction),
    ]

    if "volume" in df.columns:
        vol = last["volume"]
        vol_str = f"{vol/1e9:.2f}B" if vol >= 1e9 else f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol:,.0f}"
        metrics.append(("Volume", vol_str))

    cards_html = '<div class="metric-row">'
    for m in metrics:
        label = m[0]
        value = m[1]
        css_class = m[2] if len(m) == 3 else ""
        cards_html += (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {css_class}">{value}</div>'
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ----------------------------
# Kronos metrics helpers
# ----------------------------
import json as _json

METRIC_TOOLTIPS: dict[str, str] = {
    "sharpness": (
        "Coefficient of variation of final prices across all paths. "
        "Low sharpness means paths converge — the model is confident. "
        "High sharpness signals wide disagreement between scenarios."
    ),
    "direction_prob": (
        "Fraction of simulated paths whose final price exceeds the last observed close. "
        "Above 0.60 favours long; below 0.40 favours short; near 0.50 is ambiguous."
    ),
    "expected_return": (
        "Mean return across all paths at the forecast horizon, relative to the last close. "
        "This is the model's average expectation, not a guarantee."
    ),
    "signal_sharpe": (
        "Expected return divided by its cross-path standard deviation. "
        "Measures signal quality relative to uncertainty — higher is better."
    ),
    "upside": (
        "Average return of bullish paths only (those finishing above last close). "
        "Indicates the reward potential if the market follows the bullish scenario."
    ),
    "downside": (
        "Average return of bearish paths only (those finishing below last close). "
        "Indicates the loss potential if the market follows the bearish scenario."
    ),
    "risk_reward": (
        "Ratio of upside to absolute downside across paths. "
        "Above 1.0 means the reward potential outweighs the risk potential."
    ),
    "best_case_p95": (
        "95th percentile of final returns — what the top 5% of paths project. "
        "Represents the optimistic tail of the distribution."
    ),
    "worst_case_p5": (
        "5th percentile of final returns — what the bottom 5% of paths project. "
        "Represents the pessimistic tail; useful for stop-loss sizing."
    ),
    "conviction": (
        "Composite score [0–1] combining directional strength and path agreement. "
        "High conviction (>0.6) means the model has a clear, coherent view. "
        "Low conviction means mixed or uncertain signals."
    ),
    "entry": (
        "Whether conditions favour opening a new position right now. "
        "Based on direction probability, sharpness, expected return, and conviction thresholds."
    ),
    "stay_long": (
        "Whether an existing long position should be maintained. "
        "Triggers an exit recommendation if the forecast turns negative or paths diverge."
    ),
    "stay_short": (
        "Whether an existing short position should be maintained. "
        "Triggers an exit recommendation if the forecast turns positive or paths diverge."
    ),
}

SIGNAL_LABEL_MAP = {
    "long":        ("LONG",       "long"),
    "short":       ("SHORT",      "short"),
    "neutral":     ("NEUTRAL",    "neutral"),
    "stay_long":   ("STAY LONG",  "stay"),
    "stay_short":  ("STAY SHORT", "stay"),
    "exit_long":   ("EXIT LONG",  "exit"),
    "exit_short":  ("EXIT SHORT", "exit"),
}


@st.cache_data(show_spinner=False)
def load_metrics_file(file_path: str, file_mtime: float) -> dict | None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def load_asset_metrics(asset: str, timeframe: str) -> dict | None:
    file_path = PRED_DIR / f"{asset.lower()}_{timeframe}_metrics.json"
    if not file_path.exists():
        return None
    return load_metrics_file(str(file_path), get_file_mtime(file_path))


def _kcard(label: str, value: str, css_class: str = "", tooltip: str = "") -> str:
    tip_attr = 'data-tip="' + tooltip.replace('"', "'") + '"' if tooltip else ""
    wrapper_class = "kmetric-card has-tip" if tooltip else "kmetric-card"
    return (
        '<div class="' + wrapper_class + '" ' + tip_attr + ">"
        + '<div class="kmetric-label">' + label + "</div>"
        + '<div class="kmetric-value ' + css_class + '">' + value + "</div>"
        + "</div>"
    )


def _fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v*100:.2f}%"


def _fmt_score(v: float) -> str:
    return f"{v:.2f}"


def render_kronos_panel(asset: str, timeframe: str) -> None:
    data = load_asset_metrics(asset, timeframe)
    if data is None:
        st.caption(
            f"No Kronos metrics found for {asset} {timeframe} — "
            "run predict_kronos.py with --n-paths > 1."
        )
        return

    m = data.get("metrics", {})
    s = data.get("signals", {})
    computed_at  = data.get("computed_at", "")[:16].replace("T", " ")
    forecast_end = data.get("forecast_end", "")[:16].replace("T", " ")
    n_paths = m.get("n_paths", "?")

    color = ASSET_COLORS.get(asset, "#58a6ff")

    dir_prob   = m.get("direction_prob", 0)
    exp_ret    = m.get("expected_return", 0)
    sharpness  = m.get("sharpness", 0)
    conviction = m.get("conviction", 0)
    rr         = m.get("risk_reward", float("nan"))
    sharpe     = m.get("signal_sharpe", float("nan"))
    upside     = m.get("upside", 0)
    downside   = m.get("downside", 0)
    p95        = m.get("best_case_p95", 0)
    p5         = m.get("worst_case_p5", 0)

    dir_css   = "positive" if dir_prob >= 0.6 else ("negative" if dir_prob <= 0.4 else "neutral")
    ret_css   = "positive" if exp_ret > 0 else ("negative" if exp_ret < 0 else "")
    sharp_css = "positive" if sharpness < 0.03 else ("negative" if sharpness > 0.05 else "neutral")
    conv_css  = "positive" if conviction >= 0.6 else ("negative" if conviction < 0.35 else "neutral")
    rr_valid  = isinstance(rr, float) and rr == rr
    rr_css    = ("positive" if rr_valid and rr > 1.0 else ("negative" if rr_valid and rr < 0.8 else "neutral"))

    row1 = (
        _kcard("Direction Prob",  f"{dir_prob*100:.0f}%",                    dir_css,   METRIC_TOOLTIPS["direction_prob"])
        + _kcard("Expected Return", _fmt_pct(exp_ret),                        ret_css,   METRIC_TOOLTIPS["expected_return"])
        + _kcard("Sharpness",       _fmt_score(sharpness),                    sharp_css, METRIC_TOOLTIPS["sharpness"])
        + _kcard("Conviction",      _fmt_score(conviction),                   conv_css,  METRIC_TOOLTIPS["conviction"])
        + _kcard("Signal Sharpe",   _fmt_score(sharpe) if sharpe == sharpe else "n/a", "", METRIC_TOOLTIPS["signal_sharpe"])
    )

    row2 = (
        _kcard("Upside (avg)",  _fmt_pct(upside),                             "positive", METRIC_TOOLTIPS["upside"])
        + _kcard("Downside (avg)", _fmt_pct(downside),                        "negative", METRIC_TOOLTIPS["downside"])
        + _kcard("Risk / Reward",  _fmt_score(rr) if rr_valid else "n/a",     rr_css,    METRIC_TOOLTIPS["risk_reward"])
        + _kcard("Best (p95)",     _fmt_pct(p95),                             "positive", METRIC_TOOLTIPS["best_case_p95"])
        + _kcard("Worst (p5)",     _fmt_pct(p5),                              "negative", METRIC_TOOLTIPS["worst_case_p5"])
    )

    entry_val    = s.get("entry", "neutral")
    entry_reason = s.get("entry_reason", "")
    e_text, e_css = SIGNAL_LABEL_MAP.get(entry_val, (entry_val.upper(), "neutral"))

    stay_long_val    = s.get("stay_long", "stay_long")
    stay_long_reason = s.get("stay_long_reason", "")
    sl_text, sl_css  = SIGNAL_LABEL_MAP.get(stay_long_val, (stay_long_val.upper(), "neutral"))

    stay_short_val    = s.get("stay_short", "stay_short")
    stay_short_reason = s.get("stay_short_reason", "")
    ss_text, ss_css   = SIGNAL_LABEL_MAP.get(stay_short_val, (stay_short_val.upper(), "neutral"))

    def _sig(card_css, tip_key, label, val_text, val_css, reason):
        tip = METRIC_TOOLTIPS.get(tip_key, "").replace('"', "'")
        return (
            '<div class="signal-card ' + card_css + ' has-tip" data-tip="' + tip + '">'
            + '<div class="signal-label">' + label + "</div>"
            + '<div class="signal-value ' + val_css + '">' + val_text + "</div>"
            + '<div class="signal-reason">' + reason + "</div>"
            + "</div>"
        )

    signals_html = (
        _sig(e_css,  "entry",       "Entry Signal", e_text,  e_css,  entry_reason)
        + _sig(sl_css, "stay_long",  "If Long",      sl_text, sl_css, stay_long_reason)
        + _sig(ss_css, "stay_short", "If Short",     ss_text, ss_css, stay_short_reason)
    )

    html = (
        '<div class="kronos-panel">'
        + '<div class="kronos-panel-title" style="color:' + color + '88;">'
        + "&#9670; Kronos Analysis &middot; " + asset + " &middot; " + timeframe
        + "</div>"
        + '<div class="kmetric-row">' + row1 + "</div>"
        + '<div class="kmetric-row">' + row2 + "</div>"
        + '<div class="signal-row">'  + signals_html + "</div>"
        + '<div class="kronos-meta">'
        + str(n_paths) + " paths &middot; computed " + computed_at
        + " UTC &middot; forecast to " + forecast_end
        + "</div>"
        + "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)


# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">Timeframe</div>', unsafe_allow_html=True)
    timeframe_label = st.radio(
        "Timeframe",
        options=list(TIMEFRAME_MAP.keys()),
        index=0,
        label_visibility="collapsed",
        horizontal=True,
    )
    timeframe = TIMEFRAME_MAP[timeframe_label]

    st.markdown('<div class="sidebar-section">Assets</div>', unsafe_allow_html=True)
    selected_assets = []
    for asset in ASSETS:
        color = ASSET_COLORS[asset]
        checked = st.checkbox(
            f"● {asset}",
            value=(asset == "BTC"),
            key=f"cb_{asset}",
        )
        if checked:
            selected_assets.append(asset)

    st.markdown('<div class="sidebar-section">Predictions</div>', unsafe_allow_html=True)
    show_predictions = st.toggle("Show Kronos forecast", value=False)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.65rem; color:#484f58; font-family: IBM Plex Mono, monospace;">'
        "Data refreshes automatically<br>on file change · Parquet cache<br>invalidated via mtime</p>",
        unsafe_allow_html=True,
    )

# ----------------------------
# Guard: no asset selected
# ----------------------------
if not selected_assets:
    st.markdown('<div class="dashboard-title">Crypto Dashboard</div>', unsafe_allow_html=True)
    st.info("Select at least one asset in the sidebar.")
    st.stop()

# ----------------------------
# Load data for all selected assets
# ----------------------------
asset_data: dict[str, pd.DataFrame] = {}
min_dates, max_dates = [], []

for asset in selected_assets:
    df = load_selected_data(asset, timeframe)
    asset_data[asset] = df
    if not df.empty:
        min_dates.append(df["open_time"].min())
        max_dates.append(df["open_time"].max())

if not min_dates:
    st.error("No data found for the selected asset/timeframe combination.")
    st.stop()

global_min_dt = min(min_dates)
global_max_dt = max(max_dates)

# ----------------------------
# Date range — presets + date pickers
# ----------------------------
PRESETS = {
    "7D":  7,
    "1M":  30,
    "3M":  90,
    "6M":  180,
    "1Y":  365,
    "YTD": None,
    "Max": -1,
}

def _to_utc(dt) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def apply_preset(preset: str) -> None:
    end = global_max_dt.date()
    if preset == "Max":
        start = global_min_dt.date()
    elif preset == "YTD":
        start = global_max_dt.date().replace(month=1, day=1)
    else:
        start = (global_max_dt - pd.Timedelta(days=PRESETS[preset])).date()
    start = max(start, global_min_dt.date())
    st.session_state["dr_start"] = start
    st.session_state["dr_end"] = end

if "dr_start" not in st.session_state:
    apply_preset("Max")

with st.sidebar:
    st.markdown('<div class="sidebar-section">Period</div>', unsafe_allow_html=True)

    preset_keys = list(PRESETS.keys())
    cols1 = st.columns(4)
    cols2 = st.columns(3)
    for i, preset in enumerate(preset_keys):
        col = cols1[i] if i < 4 else cols2[i - 4]
        with col:
            if st.button(preset, use_container_width=True, key=f"preset_{preset}"):
                apply_preset(preset)

    st.markdown(
        '<p style="font-size:0.68rem; color:#8b949e; margin: 0.6rem 0 0.2rem 0;">From</p>',
        unsafe_allow_html=True,
    )
    picked_start = st.date_input(
        "From",
        value=st.session_state["dr_start"],
        min_value=global_min_dt.date(),
        max_value=global_max_dt.date(),
        key="dr_start",
        label_visibility="collapsed",
    )
    st.markdown(
        '<p style="font-size:0.68rem; color:#8b949e; margin: 0.4rem 0 0.2rem 0;">To</p>',
        unsafe_allow_html=True,
    )
    picked_end = st.date_input(
        "To",
        value=st.session_state["dr_end"],
        min_value=global_min_dt.date(),
        max_value=global_max_dt.date(),
        key="dr_end",
        label_visibility="collapsed",
    )

    if picked_end < picked_start:
        st.warning("End date must be after start date.")
        picked_end = picked_start

start_date = _to_utc(picked_start)
end_date = _to_utc(picked_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

# ----------------------------
# Filter data to selected date range
# ----------------------------
filtered_data: dict[str, pd.DataFrame] = {}
for asset, df in asset_data.items():
    if df.empty:
        continue
    dff = df[(df["open_time"] >= start_date) & (df["open_time"] <= end_date)].copy()
    if not dff.empty:
        filtered_data[asset] = dff

# ----------------------------
# Header
# ----------------------------
title_str = " · ".join(selected_assets)
pred_badge = '<span class="pred-badge">KRONOS ON</span>' if show_predictions else ""
st.markdown(
    f'<div class="dashboard-title">{title_str} {pred_badge}</div>'
    f'<div class="dashboard-subtitle">{timeframe_label} · {picked_start.strftime("%Y-%m-%d")} → {picked_end.strftime("%Y-%m-%d")}</div>',
    unsafe_allow_html=True,
)

if not filtered_data:
    st.warning("No data in the selected date range.")
    st.stop()

# ----------------------------
# Single asset view
# ----------------------------
if len(selected_assets) == 1:
    asset = selected_assets[0]
    df = filtered_data.get(asset, pd.DataFrame())

    if df.empty:
        st.warning("No data for this asset in this period.")
        st.stop()

    # Metrics row
    render_metrics(df)

    st.markdown(
        f'<div class="chart-header">'
        f'{"OHLC Candlestick" if len(df) < CANDLE_THRESHOLD else "Close Price"} · {len(df):,} bars'
        f"</div>",
        unsafe_allow_html=True,
    )

    if len(df) < CANDLE_THRESHOLD:
        fig = build_ohlc_chart(df, asset, timeframe_label)
    else:
        fig = build_close_chart(df, asset, timeframe_label)

    if show_predictions:
        pred_df = load_selected_predictions(asset, timeframe)
        if pred_df.empty:
            st.warning("No prediction file found for this asset/timeframe.")
        else:
            # Only keep predictions starting after the last visible candle
            cutoff = df["open_time"].max()
            pred_df = pred_df[pred_df["timestamp"] > cutoff].copy()
            if pred_df.empty:
                st.warning("Forecast does not extend beyond the visible period.")
            else:
                pred_summary = summarize_prediction_paths(pred_df)
                fig = add_prediction_traces(fig, pred_summary)

    st.plotly_chart(fig, use_container_width=True)

    # Kronos metrics panel (always shown when metrics file exists)
    render_kronos_panel(asset, timeframe)

# ----------------------------
# Multi-asset normalized view
# ----------------------------
else:
    n_assets = len(filtered_data)
    st.markdown(
        f'<div class="chart-header">Normalized % variation vs period start · {n_assets} assets</div>',
        unsafe_allow_html=True,
    )
    fig = build_normalized_chart(filtered_data, timeframe_label)
    st.plotly_chart(fig, use_container_width=True)

    # Small summary table
    rows = []
    for asset, df in filtered_data.items():
        first = df["close"].iloc[0]
        last = df["close"].iloc[-1]
        pct = (last / first - 1) * 100 if first else 0
        rows.append(
            {
                "Asset": asset,
                "Start": fmt_price(first),
                "Latest": fmt_price(last),
                "Change": f"{'+' if pct >= 0 else ''}{pct:.2f}%",
            }
        )

    summary_df = pd.DataFrame(rows).set_index("Asset")
    st.dataframe(
        summary_df,
        use_container_width=True,
        height=min(38 * (len(rows) + 1) + 20, 300),
    )

    # Kronos metrics panels — one per selected asset
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    cols = st.columns(len(selected_assets))
    for col, asset in zip(cols, selected_assets):
        with col:
            render_kronos_panel(asset, timeframe)