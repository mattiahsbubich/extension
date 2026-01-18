# src/backtest/backtest_covariates.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from pandas.tseries.offsets import BDay

import torch


BASE_KNOWN_FUTURE_COLS = ["dow", "month", "is_month_end"]
EARN_COL = "is_earnings_window"

TECH_PREFIXES = ("ret_mean_", "ret_vol_", "range_hl", "vol_z_")


def regularize_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex each item_id to a regular BusinessDay calendar so Chronos can infer freq.
    Missing days get NaN target; tech covariates are forward-filled (past-only).
    Calendar covariates are recomputed from timestamp.

    If df has is_earnings_window, it is kept (will be NaN on missing days, then filled with 0).
    """
    tech_cols = [c for c in df.columns if c.startswith(TECH_PREFIXES)]
    has_earn = EARN_COL in df.columns

    out = []
    for item, g in df.groupby("item_id", sort=False):
        g = g.sort_values("timestamp").copy()
        g = g.drop_duplicates(["timestamp"])

        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="B")
        g = g.set_index("timestamp").reindex(idx)
        g.index.name = "timestamp"

        g["item_id"] = item

        # recompute calendar covariates from the regular index
        ts = g.index.to_series()
        g["dow"] = ts.dt.dayofweek.astype(np.int16)
        g["month"] = ts.dt.month.astype(np.int16)
        g["is_month_end"] = ts.dt.is_month_end.astype(np.int8)

        # past-only technical covariates: forward fill is OK (they're shifted already)
        if tech_cols:
            g[tech_cols] = g[tech_cols].ffill()

        # earnings: if present, ensure it's 0/1 (fill missing with 0)
        if has_earn:
            g[EARN_COL] = g[EARN_COL].fillna(0).astype(np.int8)

        out.append(g.reset_index())

    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )


def load_long(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def get_tech_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(TECH_PREFIXES)]


def get_known_future_cols(df: pd.DataFrame) -> list[str]:
    cols = list(BASE_KNOWN_FUTURE_COLS)
    if EARN_COL in df.columns:
        cols.append(EARN_COL)
    return cols


def make_variant_dfs(df: pd.DataFrame, cutoff_ts: pd.Timestamp, horizon: int, variant: str):
    tech_cols = get_tech_cols(df)
    known_future_cols = get_known_future_cols(df)

    context = df[df["timestamp"] <= cutoff_ts].copy()

    # business-day future timestamps
    future_ts = pd.date_range(cutoff_ts + BDay(1), periods=horizon, freq="B")
    items = context["item_id"].unique()

    base_future = pd.DataFrame(
        {
            "item_id": np.repeat(items, horizon),
            "timestamp": np.tile(future_ts, len(items)),
        }
    )

    # calendar features are deterministic from timestamp
    base_future["dow"] = base_future["timestamp"].dt.dayofweek.astype(np.int16)
    base_future["month"] = base_future["timestamp"].dt.month.astype(np.int16)
    base_future["is_month_end"] = base_future["timestamp"].dt.is_month_end.astype(np.int8)

    # earnings flag: known-future covariate, but we already precomputed it in df.
    # Merge it for the future timestamps. If missing, fill with 0.
    if EARN_COL in df.columns:
        base_future = base_future.merge(
            df[["item_id", "timestamp", EARN_COL]],
            on=["item_id", "timestamp"],
            how="left",
        )
        base_future[EARN_COL] = base_future[EARN_COL].fillna(0).astype(np.int8)

    # Ground truth for those future timestamps (may include NaNs on holidays/weekends)
    y_true = df[df["timestamp"].isin(future_ts)][["item_id", "timestamp", "target"]].copy()

    if variant == "baseline":
        context_df = context[["item_id", "timestamp", "target"]]
        future_df = base_future[["item_id", "timestamp"]]

    elif variant == "past_only":
        context_df = context[["item_id", "timestamp", "target"] + tech_cols]
        future_df = base_future[["item_id", "timestamp"]]

    elif variant == "past_plus_known_future":
        context_df = context[["item_id", "timestamp", "target"] + tech_cols + known_future_cols]
        future_df = base_future[["item_id", "timestamp"] + known_future_cols]

    elif variant == "calendar_only":
        context_df = context[["item_id", "timestamp", "target"] + known_future_cols]
        future_df = base_future[["item_id", "timestamp"] + known_future_cols]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return context_df, future_df, y_true


def pinball_loss(y, qhat, q):
    e = y - qhat
    return np.mean(np.maximum(q * e, (q - 1) * e))


def run_backtest(
    data_path: str,
    model_id: str = "amazon/chronos-2",
    horizons=(1, 5, 20),
    context_length: int = 256,
    step: int = 20,
    n_cutoffs: int = 10,
    quantiles=(0.1, 0.5, 0.9),
    batch_size: int = 32,
):
    df = load_long(data_path)
    df = regularize_to_business_days(df)

    pipe = Chronos2Pipeline.from_pretrained(model_id, device_map="mps", dtype=torch.float16)

    # pick cutoffs
    all_ts = df["timestamp"].drop_duplicates().sort_values().to_list()
    if len(all_ts) < context_length + max(horizons) + 10:
        raise RuntimeError("Not enough history for chosen context_length/horizons.")

    last_possible = len(all_ts) - max(horizons) - 1
    first_possible = context_length
    candidates = list(range(first_possible, last_possible, step))
    if not candidates:
        raise RuntimeError("No valid cutoffs. Reduce context_length or horizon.")

    cut_idxs = candidates[-n_cutoffs:]
    cutoffs = [all_ts[i] for i in cut_idxs]

    variants = ["baseline", "past_only", "past_plus_known_future", "calendar_only"]
    rows = []

    for H in horizons:
        for cutoff_ts in cutoffs:
            for variant in variants:
                context_df, future_df, y_true = make_variant_dfs(df, cutoff_ts, H, variant)
                if future_df.empty:
                    continue

                context_df = (
                    context_df.sort_values(["item_id", "timestamp"])
                    .drop_duplicates(["item_id", "timestamp"])
                )
                future_df = (
                    future_df.sort_values(["item_id", "timestamp"])
                    .drop_duplicates(["item_id", "timestamp"])
                )

                pred_df = pipe.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=H,
                    quantile_levels=list(quantiles),
                    id_column="item_id",
                    timestamp_column="timestamp",
                    target="target",
                    validate_inputs=True,
                    context_length=context_length,
                    batch_size=batch_size,
                )

                qcols = [str(q) for q in quantiles if str(q) in pred_df.columns]
                if not qcols:
                    continue

                merged = (
                    y_true.merge(
                        pred_df[["item_id", "timestamp"] + qcols],
                        on=["item_id", "timestamp"],
                        how="inner",
                    )
                    .dropna(subset=["target"])
                )

                if len(merged) == 0:
                    continue
                if len(merged) < 50:
                    continue
                if "0.5" not in merged.columns:
                    continue

                mae = float(np.mean(np.abs(merged["target"].values - merged["0.5"].values)))

                pb = 0.0
                used = 0
                for q in quantiles:
                    qc = str(q)
                    if qc in merged.columns:
                        pb += pinball_loss(merged["target"].values, merged[qc].values, q)
                        used += 1
                if used == 0:
                    continue
                pinball = float(pb / used)

                if np.isnan(mae) or np.isnan(pinball):
                    continue

                rows.append(
                    {
                        "cutoff": cutoff_ts,
                        "horizon": H,
                        "variant": variant,
                        "mae": mae,
                        "pinball": pinball,
                        "n_points": len(merged),
                    }
                )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/processed/finance_long.parquet")
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--n_cutoffs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    res = run_backtest(
        data_path=args.data_path,
        context_length=args.context_length,
        step=args.step,
        n_cutoffs=args.n_cutoffs,
    )

    out = Path("data/processed/e1_backtest_results.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)

    print(res.groupby(["horizon", "variant"])["mae"].mean().sort_values())
    print("\n=== Mean Pinball ===")
    print(res.groupby(["horizon", "variant"])["pinball"].mean().sort_values())
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()