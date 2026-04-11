"""
run_strategy.py
---------------
Main entry point: replicates JC's full HMM+Hurst strategy pipeline.

Steps:
  1. Load price and market cap data
  2. Run HMM regime detection + Hurst + Momentum signal generation (sector-neutral)
  3. BIC analysis (2-state vs 3-state)
  4. Evaluate portfolio-level returns (MCap weighted)
  5. Run Backtrader engine for Buy & Hold and HMM & Hurst strategies

Usage:
    cd main
    python run_strategy.py
"""

import datetime
import os
import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from data_loader import load_all_data
from hmm_strategy import (
    process_universe,
    calculate_mcap_weighted_returns,
    evaluate_backtest,
)
from backtest_engine import (
    generate_target_weights,
    run_backtrader_engine,
    setup_logger,
)
from benchmark_downloader import download_benchmark_etf

# ===========================================================================
# Configuration
# ===========================================================================

DATA_DIR = "data_cn"
OUTPUT_DIR = "output_20260410"
USE_CACHE = True  # Set to True to skip HMM/BIC and jump straight to Backtrader

def _run_backtest_task(task_kwargs):
    """Wrapper to run backtest and prevent pickling Cerebro objects back to main process."""
    run_backtrader_engine(**task_kwargs)
    return True


def main(hurst_window, n_states):
    momentum_periods = 5

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  HMM + Hurst Regime-Switching Strategy (JC Pipeline)")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    data = load_all_data(DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    # Ensure Benchmark ETF Data (CSI 300)
    BENCHMARK_TICKER = "510300"
    BENCHMARK_PATH = os.path.join(DATA_DIR, f"benchmark_{BENCHMARK_TICKER}.csv")
    if not os.path.exists(BENCHMARK_PATH):
        download_benchmark_etf(BENCHMARK_PATH, ticker=f"{BENCHMARK_TICKER}.SS")
    
    df_benchmark_src = pd.read_csv(BENCHMARK_PATH)
    df_benchmark_src["Date"] = pd.to_datetime(df_benchmark_src["Date"].astype(str), format="%Y%m%d")
    df_benchmark_src.set_index("Date", inplace=True)
    
    # Dynamic Start Date: Start exactly when we have benchmark data
    GLOBAL_START_DATE = df_benchmark_src.index.min()
    print(f"  Dynamically aligning backtest to start on benchmark launch: {GLOBAL_START_DATE.date()}")

    # Align Universe to Benchmark Start
    df_prices = df_prices[df_prices.index >= GLOBAL_START_DATE]
    df_mcap = df_mcap[df_mcap.index >= GLOBAL_START_DATE]

    benchmark_log_ret = np.log(df_benchmark_src / df_benchmark_src.shift(1)).dropna()

    print(f"  Adjusted prices shape: {df_prices.shape}")
    print(f"  Market cap shape:      {df_mcap.shape}")

    tickers_csv_path = os.path.join(DATA_DIR, "tickers.csv")

    # ------------------------------------------------------------------
    # Step 2-4 Cache Logic
    # ------------------------------------------------------------------
    cache_dir = os.path.join(OUTPUT_DIR, "cache")
    signal_cache_name = f"signal_df (STATES={n_states}, HURST WINDOW={hurst_window}, MOMENTUM PERIODS={momentum_periods}).parquet"
    var_cache_name = f"var_df (STATES={n_states}, HURST WINDOW={hurst_window}, MOMENTUM PERIODS={momentum_periods}).parquet"
    
    signal_cache_path = os.path.join(cache_dir, signal_cache_name)
    var_cache_path = os.path.join(cache_dir, var_cache_name)

    if USE_CACHE and os.path.exists(signal_cache_path) and os.path.exists(var_cache_path):
        print("\n[USE_CACHE=True] Detected valid cache. Skipping Steps 2, 3, and 4.")
        print(f"  Loading signals from: {signal_cache_name}")
    else:
        # ------------------------------------------------------------------
        # Step 2: Run the Main Engine (Extract Returns and Hurst)
        # ------------------------------------------------------------------
        print("\n[2/5] Processing universe with HMM regimes...")

        # Run with the specified number of states
        print(f"\n  --- {n_states}-State HMM ---")
        strat_ret, bh_ret, signal_df, bic_all, var_df = process_universe(
            df_prices, n_states=n_states, tickers_csv_path=tickers_csv_path, hurst_window=hurst_window, momentum_periods=momentum_periods
        )

        # Cache signals and VaR so extra backtests can skip HMM+Hurst
        os.makedirs(cache_dir, exist_ok=True)
        signal_df.to_parquet(signal_cache_path)
        var_df.to_parquet(var_cache_path)
        print("  Cached signals and VaR to output/cache/")

        # ------------------------------------------------------------------
        # Step 3: BIC Analysis
        # ------------------------------------------------------------------
        print("\n[3/5] Running BIC analysis...")
        bic_summary = bic_all.groupby("n_states")["bic"].agg(["mean", "std", "count"])

        print("\nBIC Summary:")
        print(bic_summary)

        bic_summary.to_csv(os.path.join(OUTPUT_DIR, "bic_summary.csv"))

        # ------------------------------------------------------------------
        # Step 4: Evaluate Portfolio Returns (MCap Weighted)
        # ------------------------------------------------------------------
        print(f"\n[4/5] Evaluating portfolio performance ({n_states}-state model)...")

        # Align benchmark returns with strategy index
        benchmark_log_ret_aligned = benchmark_log_ret.reindex(strat_ret.index).fillna(0)
        benchmark_series = benchmark_log_ret_aligned.iloc[:, 0] if isinstance(benchmark_log_ret_aligned, pd.DataFrame) else benchmark_log_ret_aligned

        # Calculate Portfolio Strategy Returns (weighted sum of universe strategy returns)
        mcap_aligned = df_mcap.reindex(index=strat_ret.index, columns=strat_ret.columns).shift(1)
        weights = mcap_aligned.div(mcap_aligned.sum(axis=1), axis=0).fillna(0)
        
        port_simple_strat = ((np.exp(strat_ret) - 1) * weights).sum(axis=1)
        port_log_strat = np.log(1 + port_simple_strat)

        portfolio_eval_df = pd.DataFrame({
            "Log_Return": benchmark_series,
            "Strategy_Return": port_log_strat
        }).replace(0, np.nan).dropna()

        performance_metrics = evaluate_backtest(portfolio_eval_df)

        print("\n--- Strategy vs Benchmark Performance ---")
        print(performance_metrics)
        print("-" * 50)

        performance_metrics.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    # ------------------------------------------------------------------
    # Step 5: Backtrader Execution
    # ------------------------------------------------------------------
    print("\n[5/5] Running Backtrader backtests...")
    signal_df = pd.read_parquet(signal_cache_path)
    var_df = pd.read_parquet(var_cache_path)

    target_weights = generate_target_weights(signal_df, df_mcap)

    common_columns = list(
        col for col in df_prices.columns if col in target_weights.columns
    )

    # Target Weights for CSI 300 ETF (Benchmark Task)
    etf_prices = df_benchmark_src.reindex(df_prices.index).ffill()
    benchmark_weights = pd.DataFrame(index=etf_prices.index, columns=[BENCHMARK_TICKER], data=1.0)

    bh_weights = generate_target_weights(
        signals_df=signal_df, df_mcap=df_mcap, is_buy_and_hold=True
    )

    strat_weights = generate_target_weights(
        signals_df=signal_df, df_mcap=df_mcap, is_buy_and_hold=False
    )

    rp_weights = generate_target_weights(
        signals_df=signal_df,
        df_mcap=df_mcap,
        df_var=var_df,
        target_risk=0.01,  # 1% VaR risk budget per asset
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Buy & Hold Weights:")
    print((bh_weights[common_columns].iloc[0:200, :] != 0).sum(axis=1))
    bh_weights[common_columns].to_csv(f"{OUTPUT_DIR}/bh_weights_hurst_window_{current_time}_(States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}).csv")

    print("\nHMM & Hurst Strategy Weights:")
    print((strat_weights[common_columns].iloc[0:200, :] != 0).sum(axis=1).head(300))
    strat_weights[common_columns].to_csv(f"{OUTPUT_DIR}/strat_weights_hurst_window_{current_time}_(States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}).csv")

    print("\nHMM & Hurst Strategy with GARCH VaR Parity Weights:")
    print((rp_weights[common_columns].iloc[0:200, :] != 0).sum(axis=1).head(300))
    rp_weights[common_columns].to_csv(f"{OUTPUT_DIR}/rp_weights_hurst_window_{current_time}_(States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}).csv")


    print("\n  Dispatching parallel backtests to CPU worker pool...")
    backtest_tasks = [
        {
            "df_prices": etf_prices[etf_prices.notna().any(axis=1)],
            "target_weights_df": benchmark_weights,
            "test_name": f"CSI 300 ETF Buy & Hold (Benchmark) {current_time}",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": False,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": strat_weights[common_columns],
            "test_name": f"HMM & Hurst Strategy (States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}) {current_time}",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": False,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": strat_weights[common_columns],
            "test_name": f"HMM & Hurst Strategy (Elder Rules) (States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}) {current_time}",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": True,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": rp_weights[common_columns],
            "test_name": f"HMM & Hurst Strategy (GARCH VaR Parity) (States = {n_states}, Hurst Window={hurst_window}, Momentum Periods={momentum_periods}) {current_time}",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": False,
        },
    ]

    Parallel(n_jobs=-1)(delayed(_run_backtest_task)(task) for task in backtest_tasks)

    print("\n" + "=" * 70)
    print("  Strategy pipeline complete! Check output/ for results.")
    print("=" * 70)


if __name__ == "__main__":
    n_states = 2

    for hurst_window in [100]:
        print("\n" + "=" * 70)
        print(f"Running strategy with States={n_states}, Hurst Window={hurst_window}")
        print("=" * 70)
        main(hurst_window, n_states)

    # main()
