# tiingo_data/download_data.py
import os
import time
import pathlib
import pandas as pd
from src.tiingo_data.tiingo import get_daily_ohlcv
from src.tiingo_data.tiingo_tickers import TICKERS

DATA_DIR = "data/raw"

def download_and_cache_ohlcv(start_date: str, end_date: str, force: bool = False):
    root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        fp = out_dir / f"{ticker}.parquet"
        if fp.exists() and not force:
            print(f"[skip] {ticker} already cached")
            continue

        try:
            print(f"Downloading {ticker}...", end=" ")
            df = get_daily_ohlcv(ticker, start_date, end_date)
            if df.empty:
                print("Empty (skipped).")
                continue

            df = df.reset_index().rename(columns={"index": "date"})
            df.to_parquet(fp, index=False)
            print("Done.")
            time.sleep(0.6)  # rate limit
        except Exception as e:
            print(f"Failed: {e}")

def load_ohlcv_cached(ticker: str) -> pd.DataFrame:
    root = pathlib.Path(__file__).resolve().parents[1]
    fp = root / DATA_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")