"""
run_backtest.py
---------------
STAGE 2: Fast Backtesting & Execution.
This script uses cached signals to simulate trading:
  1. Loads universe data and integrates univ_h.csv for constituent tracking.
  2. Loads cached HMM/Hurst signals from config.
  3. Aligns dates to max(config.START_YEAR, Benchmark_Launch).
  4. Generates target weights with univ_h.csv filtering.
  5. Executes Parallel Backtrader tests (Benchmark replication, Standard, Elder, Risk Parity).
  6. Prints final KPI comparison tables.
"""

import os
import warnings
import datetime
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

try:
    import config_backtest as cfg
except ImportError:
    import config as cfg

from data_loader import load_all_data
from backtest_engine import (
    generate_target_weights,
    run_backtrader_engine,
)

def _run_backtest_task(task_kwargs):
    """Wrapper for parallel execution."""
    _, _, metrics = run_backtrader_engine(**task_kwargs)
    return metrics

def main(hurst_window, start_year=None, output_subdir=None):
    print("\n" + "=" * 70)
    print(f"  STAGE 2: Backtesting (Hurst Window={hurst_window})")
    print("=" * 70)
    
    # Create window-specific output directory
    base_out = os.path.join(cfg.OUTPUT_DIR, output_subdir) if output_subdir else cfg.OUTPUT_DIR
    window_out_dir = os.path.join(base_out, f"hurst_{hurst_window}")
    os.makedirs(window_out_dir, exist_ok=True)

    # 1. Load Data
    print("\n[1/4] Loading universe and benchmark data...")
    data = load_all_data(cfg.DATA_DIR)
    df_close = data["adjusted"]
    df_open = data["open"]
    df_mcap = data["mktcap"]

    # Load univ_h.csv (Constituent Matrix)
    univ_h_path = os.path.join(cfg.DATA_DIR, "univ_h.csv")
    if os.path.exists(univ_h_path):
        univ_h = pd.read_csv(univ_h_path)
        univ_h.set_index("year", inplace=True)
        # Create a Daily Mask
        constituent_mask = pd.DataFrame(index=df_close.index, columns=df_close.columns, data=0)
        for yr in univ_h.index:
            mask_row = univ_h.loc[yr]
            valid_tickers = mask_row[mask_row == 1].index.tolist()
            avail_tickers = [t for t in valid_tickers if t in df_close.columns]
            start_yr = pd.Timestamp(year=int(yr), month=1, day=1)
            end_yr = pd.Timestamp(year=int(yr), month=12, day=31)
            constituent_mask.loc[start_yr:end_yr, avail_tickers] = 1
        print(f"  Integrated 'univ_h.csv' constituent matrix.")
    else:
        constituent_mask = None

    # Align to Effective Start Year
    effective_year = start_year if start_year is not None else cfg.START_YEAR
    GLOBAL_START_DATE = pd.Timestamp(year=effective_year, month=1, day=1)
    print(f"  Aligning all data to start date: {GLOBAL_START_DATE.date()} (Parameter: {start_year}, Config: {cfg.START_YEAR})")

    df_close = df_close[df_close.index >= GLOBAL_START_DATE]
    df_open = df_open[df_open.index >= GLOBAL_START_DATE]
    df_mcap = df_mcap[df_mcap.index >= GLOBAL_START_DATE]

    # Combine Open and Close into MultiIndex (Ticker, Field)
    common_tickers = df_close.columns.intersection(df_open.columns)
    df_prices = pd.concat(
        {"Close": df_close[common_tickers], "Open": df_open[common_tickers]}, axis=1
    ).swaplevel(0, 1, axis=1).sort_index(axis=1)


    all_kpis = []
    master_task_list = [] # Create a global list for ALL tasks

    # 2. Grid Search over Momentum Periods
    for mp in cfg.MOMENTUM_PERIODS:
        print(f"\n--- Testing Momentum Period: {mp} ({cfg.N_STATES}-State HMM) ---")
        
        combo_suffix = f"H{hurst_window}_M{mp}"
        signal_cache_name = f"signal_df_{combo_suffix}.parquet"
        var_cache_name = f"var_df_{combo_suffix}.parquet"
        
        signal_path = os.path.join(cfg.CACHE_DIR, signal_cache_name)
        var_path = os.path.join(cfg.CACHE_DIR, var_cache_name)

        if not os.path.exists(signal_path):
            print(f"  SKIP: No cache found for {combo_suffix}. Run Stage 1 first.")
            continue

        signal_df = pd.read_parquet(signal_path)
        var_df = pd.read_parquet(var_path)

        # 3. Weight Generation
        strat_weights = generate_target_weights(signal_df, df_mcap)
        rp_weights = generate_target_weights(signal_df, df_mcap, df_var=var_df, target_risk=0.01)
        bench_weights = generate_target_weights(
            signal_df, df_mcap, is_buy_and_hold=True, constituent_df=constituent_mask
        )

        available_tickers = df_prices.columns.get_level_values(0).unique()
        common_cols = [
            col for col in df_close.columns 
            if col in strat_weights.columns 
            and col in bench_weights.columns
            and col in rp_weights.columns
            and col in common_tickers 
            and col in available_tickers
        ]

        # OPTIMIZATION: Slice the massive MultiIndex DataFrame exactly ONCE per loop
        sliced_prices = df_prices.loc[:, common_cols]

        # 4. Append tasks to the MASTER list (Do not execute yet)
        master_task_list.extend([
            {
                "df_prices": sliced_prices,
                "target_weights_df": bench_weights[common_cols],
                "test_name": f"Univ_H_Benchmark_{combo_suffix}",
                "rebalance_freq": getattr(cfg, "REBALANCE_FREQ_BENCH", "Q"),
                "output_dir": window_out_dir,
                "console_out": False
            },
            {
                "df_prices": sliced_prices,
                "target_weights_df": strat_weights[common_cols],
                "test_name": f"HMM_Standard_{combo_suffix}",
                "rebalance_freq": getattr(cfg, "REBALANCE_FREQ_STRAT", "W"),
                "output_dir": window_out_dir,
                "console_out": False
            },
            {
                "df_prices": sliced_prices,
                "target_weights_df": strat_weights[common_cols],
                "test_name": f"HMM_Elder_{combo_suffix}",
                "use_elder_rules": True,
                "elder_max_pos": getattr(cfg, "ELDER_MAX_POS", 0.02),
                "elder_max_dd": getattr(cfg, "ELDER_MAX_DD", 0.06),
                "rebalance_freq": getattr(cfg, "REBALANCE_FREQ_STRAT", "W"),
                "output_dir": window_out_dir,
                "console_out": False
            },
            {
                "df_prices": sliced_prices,
                "target_weights_df": rp_weights[common_cols],
                "test_name": f"HMM_VaR_Parity_{combo_suffix}",
                "rebalance_freq": getattr(cfg, "REBALANCE_FREQ_STRAT", "W"),
                "output_dir": window_out_dir,
                "console_out": False
            }
        ])

    # 5. Global Parallel Execution
    print(f"\n[4/4] Executing all {len(master_task_list)} Backtrader tasks across all CPU cores...")
    all_kpis = Parallel(n_jobs=-1)(delayed(_run_backtest_task)(t) for t in master_task_list)

    # 6. Display Final KPI Table
    print("\n" + "=" * 130)
    print(f"{'Execution Strategy (H=' + str(hurst_window) + ', States=' + str(cfg.N_STATES) + ')':<65} | {'Final Value':>15} | {'Trades':>8} | {'Win%':>8} | {'MaxDD':>8} | {'Sharpe':>7}")
    print("-" * 130)
    for r in all_kpis:
        print(
            f"{r['Strategy']:<65} | ${r['Final Value']:>14,.2f} | "
            f"{r['Trades']:>8} | {r['Win Rate (%)']:>7.1f}% | "
            f"{r['Max DD (%)']:>7.2f}% | {r['Sharpe']:>7.2f}"
        )
    print("-" * 130)

    return all_kpis

if __name__ == "__main__":
    for hw in cfg.HURST_WINDOWS:
        main(hurst_window=hw)