# src/datasets/build_long_df.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.features.earnings import earnings_flag

DEFAULT_TECH_WINDOWS = (10, 20)


def _read_ohlcv_parquet(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)

    if "date" not in df.columns:
        raise ValueError(f"{fp.name}: missing 'date' column. Expected a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    cols = {c.lower(): c for c in df.columns}

    # Need at least one price column
    if ("adjclose" not in cols) and ("close" not in cols):
        raise ValueError(f"{fp.name}: missing close/adjClose column")

    out = pd.DataFrame({"timestamp": df["date"]})

    def _col(name: str):
        key = name.lower()
        return df[cols[key]] if key in cols else np.nan

    out["close"] = _col("close")
    out["adjClose"] = _col("adjClose")
    out["open"] = _col("open")
    out["high"] = _col("high")
    out["low"] = _col("low")
    out["volume"] = _col("volume")

    # Use adjClose if present else close
    out["price"] = out["adjClose"].where(out["adjClose"].notna(), out["close"]).astype(float)

    # Guard: price must be > 0 for log
    out.loc[out["price"] <= 0, "price"] = np.nan

    return out


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["dow"] = ts.dt.dayofweek.astype(np.int16)  # 0=Mon ... 6=Sun
    df["month"] = ts.dt.month.astype(np.int16)  # 1..12
    df["is_month_end"] = ts.dt.is_month_end.astype(np.int8)
    return df


def _add_technical_features_past_only(
    df: pd.DataFrame,
    windows: Iterable[int] = DEFAULT_TECH_WINDOWS,
) -> pd.DataFrame:
    """
    Past-only covariates.
    IMPORTANT: we SHIFT by 1 so that features at time t only use info up to t-1.
    """
    price = df["price"].astype(float)
    df["log_return"] = np.log(price).diff()

    for w in windows:
        df[f"ret_mean_{w}"] = df["log_return"].rolling(w).mean()
        df[f"ret_vol_{w}"] = df["log_return"].rolling(w).std(ddof=0)

    # Intraday range proxy (uses same-day high/low/close -> must be shifted!)
    if df["high"].notna().any() and df["low"].notna().any() and df["price"].notna().any():
        df["range_hl"] = (df["high"] - df["low"]) / df["price"]
    else:
        df["range_hl"] = np.nan

    # Volume z-score rolling
    if df["volume"].notna().any():
        for w in windows:
            roll = df["volume"].rolling(w)
            df[f"vol_z_{w}"] = (df["volume"] - roll.mean()) / roll.std(ddof=0)
    else:
        for w in windows:
            df[f"vol_z_{w}"] = np.nan

    tech_cols = [
        c for c in df.columns if c.startswith(("ret_mean_", "ret_vol_", "range_hl", "vol_z_"))
    ]
    df[tech_cols] = df[tech_cols].shift(1)

    return df


def _set_target(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """
    target_type:
      - returns: log-return
      - log_price: log(price)
      - price: raw price
    """
    price = df["price"].astype(float)

    if target_type == "returns":
        df["target"] = df["log_return"].astype(np.float32)

    elif target_type == "log_price":
        df["target"] = np.log(price).astype(np.float32)

    elif target_type == "price":
        df["target"] = price.astype(np.float32)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return df


def build_long_df(
    raw_dir: Path,
    out_path: Path,
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    windows: Iterable[int] = DEFAULT_TECH_WINDOWS,
    drop_na: bool = True,
    target_type: str = "returns",
    earnings_cache_dir: Path = Path("data/external"),
    earnings_window_bdays: int = 2,
    earnings_refresh: bool = False,
) -> pd.DataFrame:
    raw_dir = raw_dir.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    earnings_cache_dir = earnings_cache_dir.expanduser().resolve()
    earnings_cache_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        fps = sorted(raw_dir.glob("*.parquet"))
    else:
        fps = [raw_dir / f"{t}.parquet" for t in tickers]

    all_rows = []
    kept = 0
    skipped = 0

    for fp in fps:
        if not fp.exists():
            print(f"[skip] missing file: {fp}")
            skipped += 1
            continue

        ticker = fp.stem
        try:
            df = _read_ohlcv_parquet(fp)

            if start_date:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]

            df = _add_calendar_features(df)
            df = _add_technical_features_past_only(df, windows=windows)
            df = _set_target(df, target_type=target_type)

            # Earnings window flag (known-future covariate)
            df["is_earnings_window"] = earnings_flag(
                ticker=ticker,
                timestamps=df["timestamp"],
                cache_dir=earnings_cache_dir,
                window_bdays=earnings_window_bdays,
                refresh=earnings_refresh,
            ).astype(np.int8)

            tech_cols = [
                c for c in df.columns if c.startswith(("ret_mean_", "ret_vol_", "range_hl", "vol_z_"))
            ]
            tech_cols = [c for c in tech_cols if df[c].notna().any()]  # keep only usable

            keep_cols = [
                "timestamp",
                "target",
                "dow",
                "month",
                "is_month_end",
                "is_earnings_window",
            ] + tech_cols

            out = df[keep_cols].copy()
            out.insert(0, "item_id", ticker)

            if drop_na:
                # don't include earnings in dropna: should never be NaN
                out = out.dropna(subset=["target", "dow", "month", "is_month_end"] + tech_cols)

            if len(out) < 250:
                print(f"[skip] {ticker}: too few rows after cleaning ({len(out)})")
                skipped += 1
                continue

            all_rows.append(out)
            kept += 1
            print(f"[ok] {ticker}: {len(out)} rows")

        except Exception as e:
            print(f"[skip] {ticker}: {e}")
            skipped += 1

    if not all_rows:
        raise RuntimeError("No tickers produced a valid dataset. Check raw_dir and columns.")

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    long_df = long_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    long_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Tickers kept: {kept}, skipped: {skipped}")
    print(f"Total rows: {len(long_df)}")
    print("Columns:", list(long_df.columns))

    return long_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_path", type=str, default="data/processed/finance_long.parquet")
    ap.add_argument("--tickers", type=str, default="", help="Comma-separated tickers. Empty = all in raw_dir.")
    ap.add_argument("--start_date", type=str, default="")
    ap.add_argument("--end_date", type=str, default="")
    ap.add_argument("--drop_na", action="store_true", default=True)

    ap.add_argument(
        "--target_type",
        type=str,
        default="returns",
        choices=["returns", "log_price", "price"],
        help="Target to forecast: returns (log-return), log_price (log(adjClose/close)), or price (adjClose/close).",
    )

    ap.add_argument("--earnings_cache_dir", type=str, default="data/external")
    ap.add_argument("--earnings_window_bdays", type=int, default=2)
    ap.add_argument("--earnings_refresh", action="store_true", default=False)

    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    start_date = args.start_date or None
    end_date = args.end_date or None

    build_long_df(
        raw_dir=Path(args.raw_dir),
        out_path=Path(args.out_path),
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        drop_na=args.drop_na,
        target_type=args.target_type,
        earnings_cache_dir=Path(args.earnings_cache_dir),
        earnings_window_bdays=args.earnings_window_bdays,
        earnings_refresh=args.earnings_refresh,
    )


if __name__ == "__main__":
    main()