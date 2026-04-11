"""
benchmark_downloader.py
-----------------------
Utility to download Benchmark ETF (e.g., 510300.SS) from Yahoo Finance.
Used as a cleaner, lower-cost benchmark for backtesting.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def download_benchmark_etf(output_path="data_cn/benchmark_510300.csv", ticker="510300.SS", start_date="2012-01-01"):
    """
    Downloads Benchmark ETF adjusted close prices and saves to CSV.
    """
    print(f"Fetching {ticker} data from Yahoo Finance...")
    
    # Download data
    data = yf.download(ticker, start=start_date, auto_adjust=True)
    
    if data.empty:
        print(f"Error: No data found for {ticker}")
        return None
    
    # We only need the Close (Adjusted Close)
    # yfinance sometimes returns a MultiIndex if multiple tickers are passed, 
    # but for single ticker it's a simple index.
    df = data[['Close']].copy()
    
    # Use only digits for the column name to match local ticker format
    ticker_clean = "".join(filter(str.isdigit, ticker))
    df.columns = [ticker_clean]
    
    # Format index to YYYYMMDD to match other data files
    df.index = df.index.strftime('%Y%m%d')
    df.index.name = 'Date'
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved benchmark data to {output_path}")
    return df

if __name__ == "__main__":
    download_benchmark_etf()
