from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    # btc_4h.parquet -> 4h
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Impossible de déduire le timeframe depuis {path.name}")
    return parts[-1]


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


def run_one_file(input_path: Path, output_path: Path, predictor: KronosPredictor, args: argparse.Namespace) -> None:
    df = pd.read_parquet(input_path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    if len(df) < 2:
        print(f"Skip {input_path.name}: pas assez de lignes.")
        return

    timeframe = detect_timeframe(input_path)
    lookback = min(len(df), args.lookback)
    hist = df.tail(lookback).copy()

    x_df = prepare_input_frame(hist)
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
        pred_df = pred_df.rename(columns={c: f"{c}_pred" for c in pred_df.columns if c not in ["timestamp", "path_id"]})

        all_paths.append(pred_df)

    out_df = pd.concat(all_paths, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"Saved predictions -> {output_path}")


def main() -> None:
    args = parse_args()
    predictor = load_predictor()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.input_file:
        input_path = Path(args.input_file)
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