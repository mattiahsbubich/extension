# bin/download_ohlcv.py
from src.tiingo_data.download_data import download_and_cache_ohlcv

def main():
    download_and_cache_ohlcv(
        start_date="2015-01-01",
        end_date="2025-12-31",
        force=False,
    )

if __name__ == "__main__":
    main()