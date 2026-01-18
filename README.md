# Extension — E1: Covariate-Informed Forecasting with Chronos-2 (Finance)

This repository implements **Extension 1 (covariate-informed forecasting)** using **Amazon Chronos-2** on financial time series downloaded from **Tiingo** (daily OHLCV).  
We build a **long-format dataset** with technical + calendar covariates, run a **rolling-origin backtest** with multiple covariate variants, and analyze results in a set of notebooks.

---

## Project structure
- `bin/` – data download scripts (Tiingo)
- `src/datasets/` – build long-format dataset + covariates
- `src/backtest/` – rolling-origin backtest
- `notebooks/` – analysis notebooks
- `data/raw/` – cached OHLCV parquet
- `data/processed/` – processed datasets + backtest results

---

## What this extension does

### 1) Data ingestion (Tiingo → raw)
- Download daily OHLCV for a list of tickers.
- Save each ticker in `data/raw/{TICKER}.parquet`.

### 2) Dataset building (raw → long-format Chronos input)
We build a single **long-format** dataframe with schema:
- `item_id`: ticker symbol
- `timestamp`: date (daily)
- `target`: chosen forecasting target
- covariates: technical (past-only) and calendar (known-future)
- optional known-future event covariate: `is_earnings_window`

Outputs are saved in `data/processed/`:
- `finance_long.parquet` (returns target)
- `finance_long_logprice.parquet` (log price target)
- `finance_long_logprice_earn.parquet` (log price + earnings flag)

### 3) Backtest (rolling-origin)
We run a rolling-origin evaluation over multiple cutoffs and horizons:
- horizons: `H ∈ {1, 5, 20}`
- metrics: **MAE** and **Pinball loss** (for quantile forecasts)
- covariate variants:
  1. `baseline` (target only)
  2. `past_only` (target + technical covariates)
  3. `calendar_only` (target + known-future covariates)
  4. `past_plus_known_future` (target + technical + known-future)

Backtest outputs are stored as parquet:
- returns experiment:
  - `data/processed/e1_returns_ctx256_step10_cut15.parquet`
- log-price + earnings experiment:
  - `data/processed/e1_logprice_earn_ctx256_step10_cut15.parquet`

Additional summary files:
- `data/processed/e1_summary_mean_metrics.csv`
- `data/processed/e1_bootstrap_ci.csv`

---

## Setup

### Create and activate virtual environment (macOS / zsh)

python3 -m venv .venv

source .venv/bin/activate

python -m pip install -U pip


Install dependencies (minimal example):

pip install pandas pyarrow numpy matplotlib scipy

pip install chronos-forecasting

pip install tiingo

Notes:
- Chronos downloads the model weights automatically at first run.
- On Apple Silicon (M1/M2/M3), Chronos can run on MPS (GPU) via PyTorch.

---

## Notebooks (recommended order)

All notebooks assume you launch Jupyter from the project root (extension/), so relative paths like data/processed/... work.

00 - Data sanity checks

notebooks/00_data_sanity_checks.ipynb
  - Checks dataset schema and missingness
  - Confirms business-day irregularities and validates reindexing approach

01 — Covariates analysis (returns)

notebooks/01_e1_covariates_analysis.ipynb
  - Loads results and summarizes MAE + pinball by horizon/variant
  - Shows improvement distributions and bootstrap CI

02 — Log-price + earnings analysis

notebooks/02_e1_logprice_earnings_analysis.ipynb
  - Focus on log-price target with earnings known-future flag
	- Visual analysis + bootstrap CI and forest/scatter plots

03 — Returns vs Log-price comparison

notebooks/03_e1_returns_vs_logprice_comparison.ipynb
	- Direct comparison between the two experiments using the final parquet outputs:
	- data/processed/e1_returns_ctx256_step10_cut15.parquet
	- data/processed/e1_logprice_earn_ctx256_step10_cut15.parquet



