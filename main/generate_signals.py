"""
generate_signals.py
-------------------
STAGE 1: Research & Signal Generation.
This script performs the heavy lifting:
  1. Loads universe data (Prices/MCap).
  2. Fits N-state HMM market regimes.
  3. Calculates Hurst exponents and Momentum signals.
  4. Performs BIC Analysis.
  5. Evaluates theoretical portfolio performance (Step 4).
  6. Saves signals to cache for Stage 2.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from data_loader import load_all_data
from hmm_strategy import (
    process_universe,
    evaluate_backtest,
)
# removed: from benchmark_downloader import download_benchmark_etf

def main(hurst_window):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  STAGE 1: Signal Generation (Hurst Window={hurst_window})")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading universe data...")
    data = load_all_data(config.DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    # Dynamic Start Date: config.START_YEAR
    GLOBAL_START_DATE = pd.Timestamp(year=config.START_YEAR, month=1, day=1)
    print(f"  Aligning to effective start date: {GLOBAL_START_DATE.date()}")

    df_prices = df_prices[df_prices.index >= GLOBAL_START_DATE]
    df_mcap = df_mcap[df_mcap.index >= GLOBAL_START_DATE]
    
    # Calculate daily log-returns for the universe for benchmark purposes
    df_univ_log_ret = np.log(df_prices / df_prices.shift(1)).dropna(how='all')

    # 2. HMM & Hurst Processing
    print(f"\n[2/4] Processing universe with {config.N_STATES}-state HMM...")
    tickers_csv_path = os.path.join(config.DATA_DIR, "tickers.csv")
    
    # Grid Search over Momentum Periods
    for mp in config.MOMENTUM_PERIODS:
        strat_ret, bh_ret, signal_df, bic_all, var_df = process_universe(
            df_prices, 
            n_states=config.N_STATES, 
            tickers_csv_path=tickers_csv_path, 
            hurst_window=hurst_window, 
            momentum_periods=mp,
            hurst_upper=config.HURST_UPPER,
            hurst_lower=config.HURST_LOWER,
            holding_days=config.HOLDING_DAYS
        )

        # Cache results with unique names for this combination
        combo_suffix = f"H{hurst_window}_M{mp}"
        signal_cache_name = f"signal_df_{combo_suffix}.parquet"
        var_cache_name = f"var_df_{combo_suffix}.parquet"
        
        signal_df.to_parquet(os.path.join(config.CACHE_DIR, signal_cache_name))
        var_df.to_parquet(os.path.join(config.CACHE_DIR, var_cache_name))
        print(f"  Signals (M={mp}) cached to {config.CACHE_DIR}")

        # --- Performance Reporting (Now Inside Loop) ---
        
        # 3. BIC Analysis
        print(f"\n[3/4] Running BIC analysis for M={mp}...")
        bic_summary = bic_all.groupby("n_states")["bic"].agg(["mean", "std", "count"])
        print(bic_summary)
        bic_summary.to_csv(os.path.join(config.OUTPUT_DIR, f"bic_summary_{combo_suffix}.csv"))

        # 4. Theoretical Portfolio Evaluation
        print(f"\n[4/4] Evaluating theoretical performance for M={mp}...")
        
        # Calculate a synthetic Benchmark Return from the universe (Market-Cap Weighted)
        mcap_aligned = df_mcap.reindex(index=strat_ret.index, columns=strat_ret.columns).shift(1)
        weights = mcap_aligned.div(mcap_aligned.sum(axis=1), axis=0).fillna(0)
        
        # Strat returns
        port_simple_strat = ((np.exp(strat_ret) - 1) * weights).sum(axis=1)
        port_log_strat = np.log(1 + port_simple_strat)
        
        # Universe (BH) returns
        port_simple_bh = ((np.exp(bh_ret) - 1) * weights).sum(axis=1)
        port_log_bh = np.log(1 + port_simple_bh)

        portfolio_eval_df = pd.DataFrame({
            "Log_Return": port_log_bh,
            "Strategy_Return": port_log_strat
        }).replace(0, np.nan).dropna()

        performance_metrics = evaluate_backtest(portfolio_eval_df)
        print(f"\n--- Theoretical Performance (Market Cap Weighted, M={mp}) ---")
        print(performance_metrics)
        performance_metrics.to_csv(os.path.join(config.OUTPUT_DIR, f"theoretical_performance_{combo_suffix}.csv"))

    print(f"\nSTAGE 1 COMPLETE for Hurst Window={hurst_window}.")

if __name__ == "__main__":
    for hw in config.HURST_WINDOWS:
        main(hurst_window=hw)
