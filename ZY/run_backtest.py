"""
run_backtest.py
---------------
Main entry point: loads data, computes Hurst/HMM regimes, runs backtrader backtest.

Usage:
    python run_backtest.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import backtrader as bt

from data_loader import load_all_data, filter_stocks, prepare_prices, build_index
from hurst_dfa import rolling_hurst
from regime_hmm import compute_features, fit_hmm, predict_regime, label_regimes, walk_forward_regimes
from bt_strategy import RegimeSwitchingStrategy, PandasDataFeed

warnings.filterwarnings("ignore")

# ===========================================================================
# Configuration
# ===========================================================================

DATA_DIR = "data_cn"
TOP_N_STOCKS = 30         # number of stocks to select
HURST_LOOKBACK = 252      # 1 year for rolling Hurst
TRAIN_END = "2018-12-31"  # train/test split
TEST_START = "2019-01-01"
INITIAL_CASH = 1_000_000  # 1M CNY
COMMISSION = 0.001        # 0.1% per trade (typical for China A-shares)


def main():
    print("=" * 70)
    print("  Hurst Exponent + HMM Regime-Switching Trading Strategy")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load & clean data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    data = load_all_data(DATA_DIR)
    print(f"  Adjusted prices: {data['adjusted'].shape}")

    # ------------------------------------------------------------------
    # Step 2: Select stock universe
    # ------------------------------------------------------------------
    print("\n[2/6] Filtering stocks...")
    tickers = filter_stocks(data, top_n=TOP_N_STOCKS)
    print(f"  Selected {len(tickers)} stocks: {tickers[:10]}...")
    prices = prepare_prices(data, tickers)

    # ------------------------------------------------------------------
    # Step 3: Build equal-weight index & compute features
    # ------------------------------------------------------------------
    print("\n[3/6] Building index & computing features...")
    index_prices = build_index(prices["adjusted"])
    features = compute_features(
        index_prices,
        hurst_lookback=HURST_LOOKBACK,
        hurst_step=5,
    )
    print(f"  Feature matrix: {features.shape}")
    print(f"  Valid (non-NaN) rows: {features.dropna().shape[0]}")

    # ------------------------------------------------------------------
    # Step 4: Walk-forward HMM regime detection
    # ------------------------------------------------------------------
    print("\n[4/6] Running walk-forward HMM regime detection...")
    regimes = walk_forward_regimes(
        features,
        train_end=TRAIN_END,
        refit_every=63,  # quarterly
        n_states=3,
    )
    print(f"  Regime distribution (full period):")
    print(f"    {regimes.value_counts().to_dict()}")

    # Save regimes for inspection
    regimes.to_csv("regimes_output.csv", header=True)
    print("  -> Saved to regimes_output.csv")

    # ------------------------------------------------------------------
    # Step 5: Run backtest for each stock
    # ------------------------------------------------------------------
    print("\n[5/6] Running backtest...")

    # We'll test on the out-of-sample period for a few representative stocks
    test_tickers = tickers[:5]  # top 5 by liquidity
    all_results = {}

    for ticker in test_tickers:
        print(f"\n  --- Backtesting {ticker} ---")
        result = run_single_backtest(
            ticker=ticker,
            prices=prices,
            regimes=regimes,
            test_start=TEST_START,
        )
        if result is not None:
            all_results[ticker] = result

    # ------------------------------------------------------------------
    # Step 6: Summary & plots
    # ------------------------------------------------------------------
    print("\n[6/6] Generating summary...")
    print_summary(all_results)
    plot_results(all_results, regimes, index_prices)

    print("\nDone! Check the generated plots.")


def run_single_backtest(
    ticker: str,
    prices: dict,
    regimes: pd.Series,
    test_start: str,
) -> dict:
    """Run a single-stock backtest and return performance metrics."""

    adj = prices["adjusted"][ticker].dropna()
    opn = prices["open"][ticker].reindex(adj.index).ffill()
    high = prices["high"][ticker].reindex(adj.index).ffill()
    low = prices["low"][ticker].reindex(adj.index).ffill()
    dv = prices["dv"][ticker].reindex(adj.index).ffill().fillna(0)

    # Build OHLCV DataFrame for backtrader
    ohlcv = pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": adj,
        "volume": dv,
    }).dropna()

    # Filter to test period (but keep some warmup before test_start)
    warmup_start = pd.Timestamp(test_start) - pd.DateOffset(days=365)
    ohlcv = ohlcv[ohlcv.index >= warmup_start]

    if len(ohlcv) < 100:
        print(f"    WARNING: Skipping {ticker}: not enough data ({len(ohlcv)} rows)")
        return None

    # Create Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Add data feed
    data_feed = PandasDataFeed(dataname=ohlcv)
    cerebro.adddata(data_feed)

    # Add strategy with regime series
    cerebro.addstrategy(
        RegimeSwitchingStrategy,
        regime_series=regimes,
        fast_period=20,
        slow_period=60,
        mr_lookback=20,
        mr_entry_z=1.5,
        mr_exit_z=0.5,
    )

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    strat = results[0]

    # Extract metrics
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    ret = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    sharpe_ratio = sharpe.get("sharperatio", None)
    max_dd = dd.get("max", {}).get("drawdown", 0)
    total_return = (end_value / start_value - 1) * 100

    # Calculate number of years for CAGR
    n_years = len(ohlcv) / 252
    cagr = ((end_value / start_value) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    total_trades = trades.get("total", {}).get("total", 0)
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0

    print(f"    Start: {start_value:,.0f} CNY -> End: {end_value:,.0f} CNY")
    print(f"    Return: {total_return:.2f}%  |  CAGR: {cagr:.2f}%")
    print(f"    Sharpe: {sharpe_ratio}  |  Max DD: {max_dd:.2f}%")
    print(f"    Trades: {total_trades}  |  Win rate: {win_rate:.1f}%")

    return {
        "ticker": ticker,
        "start_value": start_value,
        "end_value": end_value,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe_ratio,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
    }


def print_summary(results: dict):
    """Print a summary table of all backtest results."""
    if not results:
        print("  No results to summarise.")
        return

    print("\n" + "=" * 70)
    print(f"  {'Ticker':<10} {'Return%':>10} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8} {'WinRate%':>10}")
    print("-" * 70)
    for ticker, r in results.items():
        sharpe_str = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else "N/A"
        print(
            f"  {r['ticker']:<10} {r['total_return']:>10.2f} {r['cagr']:>8.2f} "
            f"{sharpe_str:>8} {r['max_dd']:>8.2f} {r['total_trades']:>8} {r['win_rate']:>10.1f}"
        )
    print("=" * 70)


def plot_results(results: dict, regimes: pd.Series, index_prices: pd.Series):
    """Generate summary plots."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Regime timeline ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 1, 1]})

    # Index price with regime coloring
    ax = axes[0]
    ax.plot(index_prices.index, index_prices.values, color="black", linewidth=0.8, alpha=0.9)
    ax.set_title("Equal-Weight Index with HMM Regime Coloring", fontsize=14, fontweight="bold")
    ax.set_ylabel("Index Level")

    # Color background by regime
    regime_colors = {
        "trending": "#2ecc71",       # green
        "mean_reverting": "#e74c3c", # red
        "random_walk": "#95a5a6",    # grey
    }
    prev_date = None
    prev_regime = None
    for date, regime in regimes.items():
        if prev_date is not None and prev_regime is not None:
            color = regime_colors.get(prev_regime, "#cccccc")
            ax.axvspan(prev_date, date, alpha=0.15, color=color, linewidth=0)
        prev_date = date
        prev_regime = regime

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#2ecc71", alpha=0.3, label="Trending"),
        Patch(facecolor="#e74c3c", alpha=0.3, label="Mean Reverting"),
        Patch(facecolor="#95a5a6", alpha=0.3, label="Random Walk"),
    ]
    ax.legend(handles=legend_patches, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Regime distribution over time (stacked bar by year)
    ax = axes[1]
    regimes_yearly = regimes.groupby(regimes.index.year).value_counts(normalize=True).unstack(fill_value=0)
    for col in ["trending", "mean_reverting", "random_walk"]:
        if col not in regimes_yearly.columns:
            regimes_yearly[col] = 0
    regimes_yearly[["trending", "mean_reverting", "random_walk"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#2ecc71", "#e74c3c", "#95a5a6"], alpha=0.8,
    )
    ax.set_title("Regime Distribution by Year", fontsize=12)
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Hurst exponent over time
    ax = axes[2]
    from hurst_dfa import rolling_hurst as rh
    hurst_series = rh(index_prices, lookback=252, step=5)
    ax.plot(hurst_series.index, hurst_series.values, color="#3498db", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="H=0.5 (random walk)")
    ax.set_title("Rolling Hurst Exponent (252-day window)", fontsize=12)
    ax.set_ylabel("Hurst Exponent")
    ax.set_ylim(0.2, 0.8)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {output_dir}/regime_analysis.png")

    # --- Plot 2: Performance summary bar chart ---
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        tickers_list = list(results.keys())
        returns = [results[t]["total_return"] for t in tickers_list]
        sharpes = [results[t]["sharpe"] if results[t]["sharpe"] is not None else 0 for t in tickers_list]
        max_dds = [results[t]["max_dd"] for t in tickers_list]

        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in returns]

        axes[0].bar(tickers_list, returns, color=colors, alpha=0.8)
        axes[0].set_title("Total Return (%)", fontweight="bold")
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(tickers_list, sharpes, color="#3498db", alpha=0.8)
        axes[1].set_title("Sharpe Ratio", fontweight="bold")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(tickers_list, max_dds, color="#e74c3c", alpha=0.8)
        axes[2].set_title("Max Drawdown (%)", fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved {output_dir}/performance_summary.png")


if __name__ == "__main__":
    main()
