# Crypto OHLC Dashboard

An automated financial dashboard for real-time OHLC visualisation and probabilistic price forecasting across major cryptocurrency assets.

**Live application:** https://bh-financialdashboard.streamlit.app/

---

## Overview

This project provides an interactive dashboard for monitoring and analysing cryptocurrency price data across multiple timeframes. Beyond standard charting, it integrates a probabilistic forecasting engine that generates multi-path price scenarios and derives a structured set of quantitative metrics to support discretionary trade decision-making.

The entire stack is designed to operate at zero cost, relying exclusively on free-tier tooling: Streamlit Community Cloud for hosting, GitHub Actions for automated data collection and inference, and open-source model weights served locally within the CI environment.

Data collection and inference run on a scheduled GitHub Actions workflow. Both the raw market data and the prediction artefacts are committed directly to the repository, which Streamlit Community Cloud serves as a read-only filesystem. No database, no cloud storage, no paid API.

---

## Assets and Timeframes

Tracked assets: BTC, ETH, XRP, ADA, AVAX

Available timeframes: 4h, 1d

Data source : binance API

---

## Forecasting Engine

Forecasts are produced by **Kronos**, an open-source probabilistic time-series model for financial data.

> Kronos repository: https://github.com/shiyu-coder/Kronos

**Model configuration used in this project:**

| Parameter | Value |
|---|---|
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-2k` |
| Model | `NeoQuasar/Kronos-mini` |
| Lookback window | 360 bars |
| Forecast horizon | 24 bars |
| Number of simulated paths | 20 |

At each inference run, the model is conditioned on the 360 most recent bars and generates 20 independent price trajectories over the next 24 periods. The distribution of these paths is then used to compute the metrics described below.

---

## Quantitative Metrics

All metrics are computed from the cross-path distribution of final prices and are saved as a JSON file alongside the prediction parquet for each asset and timeframe.

| Metric | Description |
|---|---|
| **Direction Probability** | Fraction of paths finishing above the last observed close. Values above 0.60 indicate a bullish consensus; below 0.40, a bearish one. |
| **Expected Return** | Mean return across all paths at the forecast horizon, relative to the last close. Provides the model's average expectation net of path dispersion. |
| **Sharpness** | Coefficient of variation of final prices across paths. Low sharpness indicates that scenarios converge on a coherent outcome; high sharpness reflects fundamental uncertainty in the model. |
| **Signal Sharpe** | Expected return divided by the cross-path standard deviation of returns. Measures the quality of the directional signal relative to its own uncertainty, independently of time. |
| **Upside** | Mean return of bullish paths only. Quantifies the reward profile conditional on the market following the positive scenario. |
| **Downside** | Mean return of bearish paths only. Quantifies the loss profile conditional on the market following the negative scenario. |
| **Risk / Reward** | Ratio of upside to the absolute value of downside. Values above 1.0 indicate that the expected gain in bullish paths exceeds the expected loss in bearish paths. |
| **Best case (p95)** | 95th percentile of final returns across paths. Characterises the upper tail of the forecast distribution. |
| **Worst case (p5)** | 5th percentile of final returns across paths. Characterises the lower tail; relevant for stop-loss sizing. |
| **Conviction** | Composite score in [0, 1] combining directional strength (distance of direction probability from 0.50) and path agreement (inverse of normalised sharpness). High conviction indicates a model view that is both directionally clear and internally consistent. |

### Trade Signals

Three signals are derived from the metrics and displayed on the dashboard:

- **Entry Signal** (`long` / `short` / `neutral`): whether current conditions favour opening a new position, based on direction probability, sharpness, expected return, and conviction thresholds.
- **If Long**: whether an existing long position should be maintained or closed, based on whether the forecast remains supportive of upside.
- **If Short**: whether an existing short position should be maintained or closed, based on whether the forecast remains supportive of downside.

These signals are intended as decision-support tools, not as automated trading instructions.

---

## Dashboard Features

- OHLC candlestick chart (automatically switches to a close-price line chart beyond 365 bars to maintain rendering performance)
- Normalised percentage variation chart when multiple assets are selected simultaneously
- Overlay of Kronos forecast paths (mean trajectory and min/max band)
- Last-bar summary metrics: open, high, low, close, period change, volume
- Kronos analysis panel with all quantitative metrics and trade signals, including inline tooltips for each metric
- Period selector with preset buttons (7D, 1M, 3M, 6M, 1Y, YTD, Max) and precise date pickers

---


## Local Setup

```bash
git clone https://github.com/barnabe-hache/financial_dashboard
cd financial_dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To run inference locally:

```bash
pip install -r requirements_kronos.txt
python scripts/predict_kronos.py --input-dir data/raw --n-paths 20 --pred-len 24
```