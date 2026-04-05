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

# ===========================================================================
# Configuration
# ===========================================================================

DATA_DIR = "data_cn"
OUTPUT_DIR = "output"
N_STATES = 3  # Default number of HMM states


def _run_backtest_task(task_kwargs):
    """Wrapper to run backtest and prevent pickling Cerebro objects back to main process."""
    run_backtrader_engine(**task_kwargs)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  HMM + Hurst Regime-Switching Strategy (JC Pipeline)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load Price Data and Market Cap Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    data = load_all_data(DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    print(f"  Adjusted prices shape: {df_prices.shape}")
    print(f"  Market cap shape:      {df_mcap.shape}")

    tickers_csv_path = os.path.join(DATA_DIR, "tickers.csv")

    # ------------------------------------------------------------------
    # Step 2: Run the Main Engine (Extract Returns and Hurst)
    # ------------------------------------------------------------------
    print("\n[2/5] Processing universe with HMM regimes...")

    # Run with 2 states
    print("\n  --- 2-State HMM ---")
    strat_ret_2, bh_ret_2, signal_df_2, bic_all_2, var_df_2 = process_universe(
        df_prices, n_states=2, tickers_csv_path=tickers_csv_path
    )

    # # Run with 3 states (commented out for faster computation)
    # print("\n  --- 3-State HMM ---")
    # strat_ret_3, bh_ret_3, signal_df_3, bic_all_3, var_df_3 = process_universe(
    #     df_prices, n_states=3, tickers_csv_path=tickers_csv_path
    # )

    # Use 2-state results for everything downstream
    strat_ret_3 = strat_ret_2
    bh_ret_3 = bh_ret_2
    signal_df_3 = signal_df_2
    bic_all_3 = bic_all_2
    var_df_3 = var_df_2

    # Cache signals and VaR so extra backtests can skip HMM+Hurst
    cache_dir = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    signal_df_3.to_parquet(os.path.join(cache_dir, "signal_df.parquet"))
    var_df_3.to_parquet(os.path.join(cache_dir, "var_df.parquet"))
    print("  Cached signals and VaR to output/cache/")

    # ------------------------------------------------------------------
    # Step 3: BIC Analysis
    # ------------------------------------------------------------------
    print("\n[3/5] Running BIC analysis...")
    bic_all = bic_all_3  # pd.concat([bic_all_2, bic_all_3], ignore_index=True)
    bic_summary = bic_all.groupby("n_states")["bic"].agg(["mean", "std", "count"])

    # best_model = bic_all.loc[bic_all.groupby("ticker")["bic"].idxmin()]
    # selection_freq = best_model["n_states"].value_counts(normalize=True)

    print("\nBIC Summary:")
    print(bic_summary)
    # print("\nModel Selection Frequency:")
    # print(selection_freq)

    bic_summary.to_csv(os.path.join(OUTPUT_DIR, "bic_summary.csv"))

    # ------------------------------------------------------------------
    # Step 4: Evaluate Portfolio Returns (MCap Weighted)
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating portfolio performance (3-state model)...")

    portfolio_eval_df = calculate_mcap_weighted_returns(
        strat_log_returns=strat_ret_3, bh_log_returns=bh_ret_3, mcap_df=df_mcap
    )

    performance_metrics = evaluate_backtest(portfolio_eval_df)

    print("\n--- Strategy vs Benchmark Performance ---")
    print(performance_metrics)
    print("-" * 50)

    performance_metrics.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    # ------------------------------------------------------------------
    # Step 5: Backtrader Execution
    # ------------------------------------------------------------------
    print("\n[5/5] Running Backtrader backtests...")

    target_weights = generate_target_weights(signal_df_3, df_mcap)
    common_columns = list(
        set(df_prices.columns).intersection(set(target_weights.columns))
    )

    bh_weights = generate_target_weights(
        signals_df=signal_df_3, df_mcap=df_mcap, is_buy_and_hold=True
    )

    strat_weights = generate_target_weights(
        signals_df=signal_df_3, df_mcap=df_mcap, is_buy_and_hold=False
    )

    rp_weights = generate_target_weights(
        signals_df=signal_df_3,
        df_mcap=df_mcap,
        df_var=var_df_3,
        target_risk=0.01,  # 1% VaR risk budget per asset
    )

    print("\n  Dispatching parallel backtests to CPU worker pool...")
    backtest_tasks = [
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": bh_weights[common_columns],
            "test_name": "Buy & Hold (Market Cap Weighted)",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": False,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": strat_weights[common_columns],
            "test_name": "HMM & Hurst Strategy",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": False,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": strat_weights[common_columns],
            "test_name": "HMM & Hurst Strategy (Elder Rules)",
            "print_logs": False,
            "output_dir": OUTPUT_DIR,
            "use_elder_rules": True,
        },
        {
            "df_prices": df_prices[common_columns],
            "target_weights_df": rp_weights[common_columns],
            "test_name": "HMM & Hurst Strategy (GARCH VaR Parity)",
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
    main()
