# tiingo_data/tiingo.py
import os
import pandas as pd
from tiingo import TiingoClient

def _client() -> TiingoClient:
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise RuntimeError("Set TIINGO_API_KEY in your environment.")
    return TiingoClient({"session": True, "api_key": api_key})

def get_daily_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    client = _client()
    df = client.get_dataframe(
        ticker,
        startDate=start_date,
        endDate=end_date,
        frequency="daily",
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)

    wanted = ["open", "high", "low", "close", "adjClose", "volume"]
    cols = [c for c in wanted if c in df.columns]
    out = df[cols].copy()

    if "close" not in out.columns:
        return pd.DataFrame()

    return out