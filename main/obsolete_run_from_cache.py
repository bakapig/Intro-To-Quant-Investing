"""
run_from_cache.py
-----------------
Executes Backtrader backtests using cached HMM signals and VaR forecasts.
This avoids re-running the expensive HMM fitting and Hurst calculations.
"""

import os
import warnings
import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from data_loader import load_all_data
from backtest_engine import (
    generate_target_weights,
    run_backtrader_engine,
    setup_logger
)
from benchmark_downloader import download_benchmark_etf

# ===========================================================================
# Configuration (Must match your cached HMM parameters)
# ===========================================================================
DATA_DIR = "data_cn"
OUTPUT_DIR = "output_20260410"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

N_STATES = 2
HURST_WINDOW = 100
MOMENTUM_PERIODS = 5

def _run_task(task_kwargs):
    """Wrapper for parallel execution."""
    # run_backtrader_engine returns (cerebro, strat, metrics)
    _, _, metrics = run_backtrader_engine(**task_kwargs)
    return metrics

def main():
    # 1. Check for cache
    signal_cache_name = f"signal_df (STATES={N_STATES}, HURST WINDOW={HURST_WINDOW}, MOMENTUM PERIODS={MOMENTUM_PERIODS}).parquet"
    var_cache_name = f"var_df (STATES={N_STATES}, HURST WINDOW={HURST_WINDOW}, MOMENTUM PERIODS={MOMENTUM_PERIODS}).parquet"
    
    signal_path = os.path.join(CACHE_DIR, signal_cache_name)
    var_path = os.path.join(CACHE_DIR, var_cache_name)

    if not os.path.exists(signal_path) or not os.path.exists(var_path):
        print(f"ERROR: Cache not found at {CACHE_DIR}")
        print(f"Looked for: {signal_cache_name}")
        return

    # 2. Load cached signals and market data
    print(f"Loading cached signals from {CACHE_DIR}...")
    signal_df = pd.read_parquet(signal_path)
    var_df = pd.read_parquet(var_path)

    print("Loading market data...")
    data = load_all_data(DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    # 3. Dynamic 2012 Alignment (Matching CSI 300 launch)
    BENCHMARK_TICKER = "510300"
    BENCHMARK_PATH = os.path.join(DATA_DIR, f"benchmark_{BENCHMARK_TICKER}.csv")
    if not os.path.exists(BENCHMARK_PATH):
        download_benchmark_etf(BENCHMARK_PATH, ticker=f"{BENCHMARK_TICKER}.SS")
    
    df_benchmark_src = pd.read_csv(BENCHMARK_PATH)
    df_benchmark_src["Date"] = pd.to_datetime(df_benchmark_src["Date"].astype(str), format="%Y%m%d")
    df_benchmark_src.set_index("Date", inplace=True)
    
    GLOBAL_START_DATE = df_benchmark_src.index.min()
    print(f"Aligning backtest to benchmark launch: {GLOBAL_START_DATE.date()}")

    df_prices = df_prices[df_prices.index >= GLOBAL_START_DATE]
    df_mcap = df_mcap[df_mcap.index >= GLOBAL_START_DATE]

    # 4. Generate target weights
    # Benchmark weights for CSI 300 ETF
    etf_prices = df_benchmark_src.reindex(df_prices.index).ffill()
    benchmark_weights = pd.DataFrame(index=etf_prices.index, columns=[BENCHMARK_TICKER], data=1.0)
    
    # Active strategies
    strat_weights = generate_target_weights(signal_df, df_mcap, is_buy_and_hold=False)
    rp_weights = generate_target_weights(signal_df, df_mcap, df_var=var_df, target_risk=0.01)

    common_cols = list(col for col in df_prices.columns if col in strat_weights.columns)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 5. Define Backtest Tasks
    tasks = [
        {
            "df_prices": etf_prices[etf_prices.notna().any(axis=1)],
            "target_weights_df": benchmark_weights,
            "test_name": f"CSI 300 ETF Buy & Hold (Benchmark) {current_time}",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": strat_weights[common_cols],
            "test_name": f"HMM & Hurst Strategy {current_time}",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": strat_weights[common_cols],
            "test_name": f"HMM & Hurst Strategy (Elder Rules) {current_time}",
            "use_elder_rules": True,
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": rp_weights[common_cols],
            "test_name": f"HMM & Hurst Strategy (GARCH VaR Parity) {current_time}",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        }
    ]

    print(f"\nDispatching {len(tasks)} parallel backtests...")
    results = Parallel(n_jobs=-1)(delayed(_run_task)(t) for t in tasks)

    # 6. Display Summary Table
    print("\n" + "=" * 110)
    print(f"{'Strategy Name':<50} | {'Final Value':>15} | {'Trades':>8} | {'Win%':>8} | {'MaxDD':>8} | {'Sharpe':>7}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['Strategy']:<50} | ${r['Final Value']:>14,.2f} | "
            f"{r['Trades']:>8} | {r['Win Rate (%)']:>7.1f}% | "
            f"{r['Max DD (%)']:>7.2f}% | {r['Sharpe']:>7.2f}"
        )
    print("=" * 110)

if __name__ == "__main__":
    main()
