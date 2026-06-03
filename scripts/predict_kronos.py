from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kronos_model.kronos import Kronos, KronosTokenizer, KronosPredictor


MODEL_NAME = "NeoQuasar/Kronos-mini"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-2k"
DEFAULT_LOOKBACK = 360
FREQ_MAP = {
    "4h": "4H",
    "1d": "1D",
}

# ----------------------------
# Thresholds for signal logic
# ----------------------------
# Entry: need at least this direction probability to consider long/short
ENTRY_DIR_PROB_THRESHOLD = 0.60
# Entry: sharpness must be below this (paths must agree enough)
ENTRY_SHARPNESS_THRESHOLD = 0.04
# Entry: minimum expected return (absolute) to bother entering
ENTRY_MIN_EXPECTED_RETURN = 0.005   # 0.5 %
# Entry: minimum conviction score
ENTRY_MIN_CONVICTION = 0.45
# Stay: exit if expected return flips strongly against position
STAY_EXIT_THRESHOLD = -0.003        # -0.3 %


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kronos forecasts on parquet files.")
    parser.add_argument("--input-file", type=str, default=None, help="Single parquet file to process.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Folder containing raw parquet files.")
    parser.add_argument("--output-dir", type=str, default="data/predictions", help="Folder for prediction parquet files.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Number of historical rows to use.")
    parser.add_argument("--pred-len", type=int, default=24, help="Forecast horizon in steps.")
    parser.add_argument("--n-paths", type=int, default=1, help="Number of simulated forecast paths.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    return parser.parse_args()


def load_predictor() -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return predictor


def detect_timeframe(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Impossible de déduire le timeframe depuis {path.name}")
    return parts[-1]


def detect_asset(path: Path) -> str:
    return path.stem.split("_")[0].upper()


def build_future_timestamps(last_timestamp: pd.Timestamp, timeframe: str, pred_len: int) -> pd.Series:
    if timeframe not in FREQ_MAP:
        raise ValueError(f"Timeframe non supporté: {timeframe}")
    freq = FREQ_MAP[timeframe]
    start = last_timestamp + pd.tseries.frequencies.to_offset(freq)
    return pd.Series(pd.date_range(start=start, periods=pred_len, freq=freq))


def prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    optional = [c for c in ["volume", "amount"] if c in df.columns]
    cols = required + optional
    return df[cols].copy()


# ----------------------------
# Metrics engine
# ----------------------------

def compute_metrics(out_df: pd.DataFrame, last_close: float) -> dict:
    """
    Compute quantitative metrics from all forecast paths.

    out_df must have columns: timestamp, path_id, close_pred (at minimum).
    last_close is the last observed close price before the forecast window.

    Returns a flat dict of metrics, all values are plain Python scalars
    so they serialise cleanly to JSON.
    """
    close_col = next(
        (c for c in out_df.columns if "close" in c and c != "path_id"),
        None,
    )
    if close_col is None:
        raise ValueError("No close_pred column found in prediction output.")

    # Final price of each path (last timestamp)
    final_prices: np.ndarray = (
        out_df.groupby("path_id")[close_col].last().values.astype(float)
    )
    n_paths = len(final_prices)

    # Per-path returns vs last close
    final_returns: np.ndarray = (final_prices - last_close) / last_close

    # ── Core distribution ────────────────────────────────────────────────────
    mean_final_price = float(np.mean(final_prices))
    std_final_price  = float(np.std(final_prices))

    # Sharpness: coefficient of variation of final prices.
    # Low  → paths agree on where price ends up.
    # High → wide dispersion, model is uncertain.
    sharpness = float(std_final_price / mean_final_price) if mean_final_price != 0 else float("nan")

    # ── Return metrics ───────────────────────────────────────────────────────
    expected_return  = float(np.mean(final_returns))
    std_return       = float(np.std(final_returns))

    # Sharpe of the signal (not annualised — cross-path, not cross-time)
    signal_sharpe = (
        float(expected_return / std_return) if std_return > 1e-10 else float("nan")
    )

    # ── Directional ──────────────────────────────────────────────────────────
    bullish_mask = final_returns > 0
    direction_prob = float(np.mean(bullish_mask))   # P(path ends above last close)

    upside   = float(np.mean(final_returns[bullish_mask]))  if bullish_mask.any()  else 0.0
    downside = float(np.mean(final_returns[~bullish_mask])) if (~bullish_mask).any() else 0.0
    risk_reward = (
        float(upside / abs(downside)) if abs(downside) > 1e-10 else float("nan")
    )

    # ── Tail risk ────────────────────────────────────────────────────────────
    best_case  = float(np.percentile(final_returns, 95))
    worst_case = float(np.percentile(final_returns, 5))

    # ── Conviction score  [0, 1] ─────────────────────────────────────────────
    # Combines directional strength and path agreement.
    # dir_strength: how far direction_prob is from 0.5 (scaled to [0,1])
    # agreement:    inverse of normalised sharpness, clipped to [0,1]
    dir_strength = abs(direction_prob - 0.5) * 2          # 0 = coin-flip, 1 = all paths agree on direction
    max_sharpness_ref = 0.10                               # sharpness above this → zero agreement score
    agreement = float(np.clip(1.0 - sharpness / max_sharpness_ref, 0.0, 1.0))
    conviction = float(dir_strength * agreement)

    return {
        "n_paths":          n_paths,
        "mean_final_price": round(mean_final_price, 6),
        "std_final_price":  round(std_final_price, 6),
        "sharpness":        round(sharpness, 6),
        "expected_return":  round(expected_return, 6),
        "std_return":       round(std_return, 6),
        "signal_sharpe":    round(signal_sharpe, 4),
        "direction_prob":   round(direction_prob, 4),
        "upside":           round(upside, 6),
        "downside":         round(downside, 6),
        "risk_reward":      round(risk_reward, 4),
        "best_case_p95":    round(best_case, 6),
        "worst_case_p5":    round(worst_case, 6),
        "conviction":       round(conviction, 4),
    }


def compute_signals(metrics: dict) -> dict:
    """
    Derive human-readable trade signals from computed metrics.

    Entry signal: should you open a position right now?
    Stay signal:  if you are already in a long or short, should you stay?

    Signals: "long" | "short" | "neutral"
    """
    dp        = metrics["direction_prob"]
    sharpness = metrics["sharpness"]
    exp_ret   = metrics["expected_return"]
    conviction= metrics["conviction"]

    # ── Entry signal ─────────────────────────────────────────────────────────
    reasons = []
    entry   = "neutral"

    paths_agree    = sharpness < ENTRY_SHARPNESS_THRESHOLD
    strong_bull    = dp >= ENTRY_DIR_PROB_THRESHOLD
    strong_bear    = dp <= (1 - ENTRY_DIR_PROB_THRESHOLD)
    enough_return  = abs(exp_ret) >= ENTRY_MIN_EXPECTED_RETURN
    enough_conv    = conviction >= ENTRY_MIN_CONVICTION

    if not paths_agree:
        reasons.append(f"high dispersion across paths (sharpness={sharpness:.3f})")

    if strong_bull and paths_agree and enough_return and enough_conv:
        entry = "long"
        reasons.append(
            f"{dp*100:.0f}% of paths are bullish, "
            f"expected return +{exp_ret*100:.2f}%, "
            f"conviction {conviction:.2f}"
        )
    elif strong_bear and paths_agree and enough_return and enough_conv:
        entry = "short"
        reasons.append(
            f"only {dp*100:.0f}% of paths are bullish, "
            f"expected return {exp_ret*100:.2f}%, "
            f"conviction {conviction:.2f}"
        )
    else:
        if not strong_bull and not strong_bear:
            reasons.append(f"directional probability too close to 50% ({dp*100:.0f}%)")
        if not enough_return:
            reasons.append(f"expected return too small ({exp_ret*100:.2f}%)")
        if not enough_conv:
            reasons.append(f"conviction too low ({conviction:.2f})")

    entry_reason = "; ".join(reasons) if reasons else "no clear edge"

    # ── Stay signal ───────────────────────────────────────────────────────────
    # For each side, assess whether the forecast still supports the position.
    def _stay_long() -> tuple[str, str]:
        if exp_ret < STAY_EXIT_THRESHOLD:
            return (
                "exit_long",
                f"model now expects negative return ({exp_ret*100:.2f}%) — close long",
            )
        if sharpness >= ENTRY_SHARPNESS_THRESHOLD and dp < 0.50:
            return (
                "exit_long",
                f"paths are dispersed and majority bearish ({dp*100:.0f}%) — close long",
            )
        return (
            "stay_long",
            f"forecast still supportive: expected return {exp_ret*100:.2f}%, "
            f"{dp*100:.0f}% bullish paths",
        )

    def _stay_short() -> tuple[str, str]:
        if exp_ret > abs(STAY_EXIT_THRESHOLD):
            return (
                "exit_short",
                f"model now expects positive return ({exp_ret*100:.2f}%) — close short",
            )
        if sharpness >= ENTRY_SHARPNESS_THRESHOLD and dp > 0.50:
            return (
                "exit_short",
                f"paths are dispersed and majority bullish ({dp*100:.0f}%) — close short",
            )
        return (
            "stay_short",
            f"forecast still supportive: expected return {exp_ret*100:.2f}%, "
            f"{(1-dp)*100:.0f}% bearish paths",
        )

    stay_long_signal,  stay_long_reason  = _stay_long()
    stay_short_signal, stay_short_reason = _stay_short()

    return {
        "entry":             entry,
        "entry_reason":      entry_reason,
        "stay_long":         stay_long_signal,
        "stay_long_reason":  stay_long_reason,
        "stay_short":        stay_short_signal,
        "stay_short_reason": stay_short_reason,
    }


def save_metrics(
    metrics_path: Path,
    asset: str,
    timeframe: str,
    last_close: float,
    pred_len: int,
    forecast_start: str,
    forecast_end: str,
    metrics: dict,
    signals: dict,
) -> None:
    payload = {
        "asset":          asset,
        "timeframe":      timeframe,
        "computed_at":    datetime.now(timezone.utc).isoformat(),
        "last_close":     last_close,
        "pred_len":       pred_len,
        "forecast_start": forecast_start,
        "forecast_end":   forecast_end,
        "metrics":        metrics,
        "signals":        signals,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics    -> {metrics_path}")


# ----------------------------
# Core runner
# ----------------------------

def run_one_file(input_path: Path, output_path: Path, predictor: KronosPredictor, args: argparse.Namespace) -> None:
    df = pd.read_parquet(input_path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    if len(df) < 2:
        print(f"Skip {input_path.name}: pas assez de lignes.")
        return

    timeframe  = detect_timeframe(input_path)
    asset      = detect_asset(input_path)
    last_close = float(df["close"].iloc[-1])

    lookback = min(len(df), args.lookback)
    hist     = df.tail(lookback).copy()

    x_df        = prepare_input_frame(hist)
    x_timestamp = hist["open_time"].reset_index(drop=True)
    y_timestamp = build_future_timestamps(hist["open_time"].iloc[-1], timeframe, args.pred_len)

    all_paths = []

    for path_id in range(args.n_paths):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=1,
            verbose=True,
        )

        pred_df = pred_df.reset_index(drop=True)
        pred_df.insert(0, "timestamp", y_timestamp.to_numpy())
        pred_df.insert(1, "path_id", path_id)
        pred_df = pred_df.rename(
            columns={c: f"{c}_pred" for c in pred_df.columns if c not in ["timestamp", "path_id"]}
        )

        all_paths.append(pred_df)

    out_df = pd.concat(all_paths, ignore_index=True)

    # ── Save predictions parquet (unchanged behaviour) ────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved predictions -> {output_path}")

    # ── Compute and save metrics ──────────────────────────────────────────
    if args.n_paths > 1:
        metrics = compute_metrics(out_df, last_close)
        signals = compute_signals(metrics)

        metrics_filename = f"{input_path.stem}_metrics.json"
        metrics_path     = output_path.parent / metrics_filename

        save_metrics(
            metrics_path    = metrics_path,
            asset           = asset,
            timeframe       = timeframe,
            last_close      = last_close,
            pred_len        = args.pred_len,
            forecast_start  = str(y_timestamp.iloc[0]),
            forecast_end    = str(y_timestamp.iloc[-1]),
            metrics         = metrics,
            signals         = signals,
        )
    else:
        print(
            f"Metrics skipped for {input_path.name}: "
            "use --n-paths > 1 to generate meaningful multi-path metrics."
        )


# ----------------------------
# Entry point
# ----------------------------

def main() -> None:
    args      = parse_args()
    predictor = load_predictor()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.input_file:
        input_path  = Path(args.input_file)
        output_path = output_dir / input_path.name
        run_one_file(input_path, output_path, predictor, args)
        return

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Aucun fichier parquet trouvé dans {input_dir}")
        return

    for input_path in parquet_files:
        output_path = output_dir / input_path.name
        run_one_file(input_path, output_path, predictor, args)


if __name__ == "__main__":
    main()