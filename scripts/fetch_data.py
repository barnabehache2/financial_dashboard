from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
TIMEFRAMES = ["1d", "4h"]
START_DATE = "2024-01-01"

BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )
    return session


def fetch_klines(session: requests.Session, symbol: str, interval: str) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000,
        }

        response = session.get(BASE_URL, params=params, timeout=(10, 60))
        response.raise_for_status()

        data = response.json()
        if not data:
            break

        rows.extend(data)
        start_ts = data[-1][0] + 1

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "number_of_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open_time", "open", "high", "low", "close", "volume", "number_of_trades"]].dropna()


def main():
    session = make_session()

    for symbol in SYMBOLS:
        asset = symbol.replace("USDT", "").lower()

        for timeframe in TIMEFRAMES:
            print(f"Fetching {symbol} {timeframe}")
            df = fetch_klines(session, symbol, timeframe)

            output_file = DATA_DIR / f"{asset}_{timeframe}.parquet"
            df.to_parquet(output_file, index=False)

            print(f"Saved -> {output_file} ({len(df)} rows)")


if __name__ == "__main__":
    main()