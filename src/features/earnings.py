# src/features/earnings.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _norm_ts(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s


def _cache_path(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "earnings_dates.parquet"


def load_earnings_cache(cache_dir: Path) -> pd.DataFrame:
    fp = _cache_path(cache_dir)
    if fp.exists():
        df = pd.read_parquet(fp)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
        return df
    return pd.DataFrame(columns=["item_id", "earnings_date"])


def save_earnings_cache(cache_dir: Path, df: pd.DataFrame) -> None:
    fp = _cache_path(cache_dir)
    df = df.drop_duplicates(["item_id", "earnings_date"]).sort_values(["item_id", "earnings_date"])
    df.to_parquet(fp, index=False)


def fetch_earnings_dates_yfinance(ticker: str, limit: int = 200) -> pd.Series:
    """
    Returns a Series of earnings datetimes (timezone-stripped).
    Uses yfinance (free, best-effort).
    """
    import yfinance as yf  # lazy import

    t = yf.Ticker(ticker)
    # get_earnings_dates returns a DataFrame indexed by datetime (depending on yfinance version)
    try:
        edf = t.get_earnings_dates(limit=limit)
    except Exception:
        return pd.Series([], dtype="datetime64[ns]")

    if edf is None or len(edf) == 0:
        return pd.Series([], dtype="datetime64[ns]")

    idx = pd.to_datetime(edf.index)
    idx = pd.Series(idx).dt.tz_localize(None)
    return idx.dropna().drop_duplicates().sort_values().reset_index(drop=True)


def ensure_ticker_in_cache(ticker: str, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    cache = load_earnings_cache(cache_dir)

    already = cache[cache["item_id"] == ticker]
    if (not refresh) and (len(already) > 0):
        return cache

    dates = fetch_earnings_dates_yfinance(ticker)
    if len(dates) == 0:
        # keep cache as-is
        return cache

    new_rows = pd.DataFrame({"item_id": ticker, "earnings_date": dates})
    cache = pd.concat([cache, new_rows], ignore_index=True)
    save_earnings_cache(cache_dir, cache)
    return cache


def earnings_flag(
    ticker: str,
    timestamps: Iterable[pd.Timestamp],
    cache_dir: Path,
    window_bdays: int = 2,
    refresh: bool = False,
) -> pd.Series:
    """
    Build a 0/1 flag for dates within +-window_bdays business days around an earnings date.
    """
    ts = pd.to_datetime(pd.Series(list(timestamps)))
    ts = _norm_ts(ts)

    cache = ensure_ticker_in_cache(ticker, cache_dir, refresh=refresh)
    ed = cache.loc[cache["item_id"] == ticker, "earnings_date"]
    ed = pd.to_datetime(ed).dt.tz_localize(None)

    if len(ed) == 0:
        return pd.Series(np.zeros(len(ts), dtype=np.int8), index=ts.index)

    # build a set of flagged business dates (small: ~4 per quarter)
    flagged = set()
    for d in ed:
        d = pd.Timestamp(d).normalize()
        rng = pd.bdate_range(d - pd.tseries.offsets.BDay(window_bdays),
                             d + pd.tseries.offsets.BDay(window_bdays),
                             freq="B")
        for x in rng:
            flagged.add(pd.Timestamp(x).normalize())

    out = ts.dt.normalize().isin(flagged).astype(np.int8)
    return out