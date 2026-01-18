# Extension — Covariate-informed Forecasting (Chronos-2)

This project implements a covariate-informed forecasting pipeline on financial time series using Chronos-2.

## Project structure
- `bin/` – data download scripts (Tiingo)
- `src/datasets/` – build long-format dataset + covariates
- `src/backtest/` – rolling-origin backtest
- `notebooks/` – analysis notebooks
- `data/raw/` – cached OHLCV parquet
- `data/processed/` – processed datasets + backtest results

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TIINGO_API_KEY="YOUR_KEY"
PYTHONPATH=. python bin/download_ohlcv.py
PYTHONPATH=. python src/datasets/build_long_df.py --raw_dir data/raw --out_path data/processed/finance_long.parquet --target_type returns
PYTHONPATH=. python src/backtest/backtest_covariates.py --data_path data/processed/finance_long.parquet --context_length 256 --step 10 --n_cutoffs 15 --batch_size 32

