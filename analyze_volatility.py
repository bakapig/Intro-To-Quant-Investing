"""
analyze_volatility.py
---------------------
Creates a market-cap weighted index and calculates/plots annualized volatility,
similar to the user-provided reference image.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import load_all_data, filter_stocks, prepare_prices

# ===========================================================================
# Configuration
# ===========================================================================
DATA_DIR = "data_cn"
TOP_N_STOCKS = 300  # Wider universe for a more representative index
ROLLING_WINDOW = 20
ANNUAL_FACTOR = 252


def build_universe_mask(
    dates: pd.DatetimeIndex, tickers: list, univ_h: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a daily (date × ticker) boolean mask indicating whether each
    stock was in the investable universe on that day.

    univ_h has one row per year with columns = ticker codes and values 0/1.
    Each trading day inherits the membership of its calendar year.
    """
    years = dates.year
    ticker_set = set(tickers)
    univ_tickers = [c for c in univ_h.columns if c != "year"]
    # Build a year -> set-of-tickers lookup
    year_members = {}
    for _, row in univ_h.iterrows():
        yr = int(row["year"])
        year_members[yr] = set(
            t for t in univ_tickers if t in ticker_set and row[t] == 1
        )

    # Expand to daily mask
    mask_data = np.zeros((len(dates), len(tickers)), dtype=bool)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    for i, yr in enumerate(years):
        members = year_members.get(yr, set())
        for t in members:
            mask_data[i, ticker_idx[t]] = True

    return pd.DataFrame(mask_data, index=dates, columns=tickers)


def build_mcw_index(
    adj: pd.DataFrame, mktcap: pd.DataFrame, univ_mask: pd.DataFrame
) -> pd.Series:
    """
    Build a Market-Cap Weighted index using only stocks that are
    in the investable universe on each day (survivorship-bias free).

    Daily Return = sum(Return_i * MktCap_{i, t-1}) / sum(MktCap_{i, t-1})
    where i ∈ universe(year(t)).
    """
    # Align dates and tickers
    common_index = adj.index.intersection(mktcap.index)
    adj = adj.loc[common_index]
    mktcap = mktcap.loc[common_index]
    univ_mask = univ_mask.loc[common_index]

    # Daily returns
    returns = adj.pct_change(fill_method=None)

    # We use t-1 market cap to weight returns on day t
    weights = mktcap.shift(1).loc[common_index]

    # Zero out stocks NOT in the investable universe for that day
    weights = weights.where(univ_mask, 0.0)

    # Also mask where returns are NaN (stock not yet trading, halted, etc.)
    weights = weights.where(returns.notna(), 0.0)

    row_sums = weights.sum(axis=1)
    normalized_weights = weights.div(row_sums, axis=0)

    # Weighted average return
    mcw_return = (returns * normalized_weights).sum(axis=1)

    # Seed the first day (since shift(1) makes first day NaN)
    mcw_return.iloc[0] = 0.0

    # Cumulative return to Index Level
    index_level = (1 + mcw_return).cumprod()
    index_level.name = "MCW_Index"

    return index_level


def main():
    print("=" * 70)
    print("  Market-Cap Weighted Index Volatility Analysis")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading and cleaning data...")
    data = load_all_data(DATA_DIR)

    # 2. Filter and prepare — use only tickers present in both price and mktcap
    tickers = data["adjusted"].columns.tolist()
    valid_tickers = [t for t in tickers if t in data["mktcap"].columns]

    prices = prepare_prices(data, valid_tickers)
    adj = prices["adjusted"]
    mktcap = data["mktcap"][valid_tickers]

    # 3. Build a daily universe mask from univ_h (survivorship-bias free)
    print(f"\n[2/4] Building daily universe mask from univ_h.csv...")
    univ_mask = build_universe_mask(adj.index, valid_tickers, data["univ_h"])
    avg_members = univ_mask.sum(axis=1).mean()
    print(
        f"       Average daily universe size: {avg_members:.0f} stocks "
        f"(out of {len(valid_tickers)} total in data)"
    )

    # 4. Build Index
    print("\n[3/5] Building Market-Cap Weighted Index (survivorship-bias free)...")
    index_series = build_mcw_index(adj, mktcap, univ_mask)

    # 5. Calculate Statistics
    print("\n[4/5] Calculating volatility statistics...")
    daily_returns = index_series.pct_change().dropna()
    rolling_vol = daily_returns.rolling(window=ROLLING_WINDOW).std() * np.sqrt(
        ANNUAL_FACTOR
    )

    total_return = (index_series.iloc[-1] / index_series.iloc[0] - 1) * 100
    ann_vol = daily_returns.std() * np.sqrt(ANNUAL_FACTOR) * 100

    # 6. Plotting
    print("\n[5/5] Generating plot...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Index Level
    ax1.plot(index_series.index, index_series.values, color="blue", linewidth=1)
    ax1.set_title(
        f"Market-Cap Weighted Index Level ({index_series.index[0].strftime('%Y-%m-%d')} to {index_series.index[-1].strftime('%Y-%m-%d')})"
    )
    ax1.set_ylabel("Index Level")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling Volatility
    ax2.plot(rolling_vol.index, rolling_vol.values, color="red", linewidth=1)
    ax2.set_title(f"Rolling {ROLLING_WINDOW}-Day Annualized Volatility")
    ax2.set_ylabel("Volatility")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    # Add stats as text
    stats_text = (
        f"Index Statistics:\n"
        f"Total Return: {total_return:.2f}%\n"
        f"Annualized Volatility: {ann_vol:.2f}%"
    )
    plt.figtext(
        0.1, 0.02, stats_text, fontsize=12, fontweight="bold", family="monospace"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(output_dir, "volatility_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nAnalysis complete! Results saved to: {save_path}")
    print("-" * 70)
    print(stats_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
