"""
run_from_cache.py
-----------------
Executes Backtrader backtests using cached HMM signals and VaR forecasts.
This avoids re-running the expensive HMM fitting and Hurst calculations.

Usage:
    python run_from_cache.py
"""

import os
import warnings
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from data_loader import load_all_data
from backtest_engine import (
    generate_target_weights,
    run_backtrader_engine
)

DATA_DIR = "data_cn"
OUTPUT_DIR = "output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

def _run_task(task_kwargs):
    """Wrapper for parallel execution."""
    _, _, metrics = run_backtrader_engine(**task_kwargs)
    return metrics

def main():
    # 1. Check for cache
    signal_path = os.path.join(CACHE_DIR, "signal_df.pkl")
    var_path = os.path.join(CACHE_DIR, "var_df.pkl")

    if not os.path.exists(signal_path) or not os.path.exists(var_path):
        print(f"ERROR: Cache not found at {CACHE_DIR}")
        print("Please run 'python run_strategy.py' first to generate signals.")
        return

    # 2. Load cached signals and data
    print(f"Loading cached signals from {CACHE_DIR}...")
    signal_df = pd.read_pickle(signal_path)
    var_df = pd.read_pickle(var_path)

    print("Loading market data...")
    data = load_all_data(DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    # 3. Generate weights for all strategies
    # Buy & Hold (Benchmark)
    bh_weights = generate_target_weights(signal_df, df_mcap, is_buy_and_hold=True)
    
    # Standard Strategy
    strat_weights = generate_target_weights(signal_df, df_mcap, is_buy_and_hold=False)
    
    # Risk Parity (GARCH VaR)
    rp_weights = generate_target_weights(
        signal_df, 
        df_mcap, 
        df_var=var_df, 
        target_risk=0.01
    )

    common_cols = list(set(df_prices.columns).intersection(signal_df.columns))
    
    # 4. Define Backtest Tasks
    tasks = [
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": bh_weights[common_cols],
            "test_name": "Buy & Hold (Market Cap Weighted)",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": strat_weights[common_cols],
            "test_name": "HMM & Hurst Strategy",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": strat_weights[common_cols],
            "test_name": "HMM & Hurst Strategy (Elder Rules)",
            "use_elder_rules": True,
            "output_dir": OUTPUT_DIR,
            "console_out": False
        },
        {
            "df_prices": df_prices[common_cols],
            "target_weights_df": rp_weights[common_cols],
            "test_name": "HMM & Hurst Strategy (GARCH VaR Parity)",
            "output_dir": OUTPUT_DIR,
            "console_out": False
        }
    ]

    print(f"\nRunning {len(tasks)} backtests in parallel...")
    results = Parallel(n_jobs=-1)(delayed(_run_task)(t) for t in tasks)

    # 5. Display Summary
    print("\n" + "=" * 110)
    print(f"{'Strategy Name':<45} | {'Final Value':>15} | {'Trades':>8} | {'Win%':>8} | {'MaxDD':>8} | {'Sharpe':>7}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['Strategy']:<45} | ${r['Final Value']:>14,.2f} | "
            f"{r['Trades']:>8} | {r['Win Rate (%)']:>7.1f}% | "
            f"{r['Max DD (%)']:>7.2f}% | {r['Sharpe']:>7.2f}"
        )
    print("=" * 110)

if __name__ == "__main__":
    main()
