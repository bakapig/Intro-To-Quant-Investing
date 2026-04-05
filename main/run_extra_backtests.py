"""
run_extra_backtests.py
----------------------
Run Elder Rules and GARCH VaR Parity backtests WITHOUT re-running HMM+Hurst.
Loads saved signals/VaR from output/ (cached by run_strategy.py).

Usage:
    cd main
    python run_extra_backtests.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from data_loader import load_all_data
from backtest_engine import (
    generate_target_weights,
    run_backtrader_engine,
    setup_logger,
)

DATA_DIR = "data_cn"
OUTPUT_DIR = "output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")


def _run_backtest_task(task_kwargs):
    run_backtrader_engine(**task_kwargs)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cached signals & VaR (saved by run_strategy.py)
    # ------------------------------------------------------------------
    print("Loading cached signals and VaR from previous run...")

    signal_path = os.path.join(CACHE_DIR, "signal_df.parquet")
    var_path = os.path.join(CACHE_DIR, "var_df.parquet")

    if not os.path.exists(signal_path) or not os.path.exists(var_path):
        print(
            "ERROR: Cached files not found. Run 'python run_strategy.py' first "
            "to generate signals and VaR forecasts."
        )
        return

    signal_df = pd.read_parquet(signal_path)
    var_df = pd.read_parquet(var_path)

    print(f"  Signals shape: {signal_df.shape}")
    print(f"  VaR shape:     {var_df.shape}")

    # ------------------------------------------------------------------
    # 2. Load price and market cap data
    # ------------------------------------------------------------------
    print("Loading price and market cap data...")
    data = load_all_data(DATA_DIR)
    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    # ------------------------------------------------------------------
    # 3. Generate weights
    # ------------------------------------------------------------------
    strat_weights = generate_target_weights(
        signals_df=signal_df, df_mcap=df_mcap, is_buy_and_hold=False
    )
    common_columns = list(
        set(df_prices.columns).intersection(set(strat_weights.columns))
    )

    rp_weights = generate_target_weights(
        signals_df=signal_df,
        df_mcap=df_mcap,
        df_var=var_df,
        target_risk=0.01,
    )

    # ------------------------------------------------------------------
    # 4. Run backtests
    # ------------------------------------------------------------------
    print("\nDispatching Elder Rules + GARCH VaR Parity backtests...")

    backtest_tasks = [
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
    print("  Extra backtests complete! Check output/ for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
