"""
eda/volatility_deepdive.py
--------------------------
Part 6 – Volatility deep-dive

Analyses:
  1. Realized vs. Parkinson volatility (using OHLC data)
  2. Volatility term structure: 5d vs 20d vs 60d realized vol
  3. Leverage effect: correlation between returns and future volatility

All outputs go to  output/eda/ .
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_all_data, get_always_in_universe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

WINDOW = 20  # 1-month rolling


def main(data=None, output_dir=None):
    # ── Load data ────────────────────────────────────────────────────────────
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    if data is None:
        print("Loading data …")
        data = load_all_data()
    adj = data["adjusted"]
    high = data["high"]
    low = data["low"]
    op = data["open"]
    close = data["close"]
    mktcap = data["mktcap"]

    tickers = get_always_in_universe(data)
    adj = adj[tickers]
    high = high[tickers]
    low = low[tickers]
    op = op[tickers]
    close = close[tickers]
    mktcap = mktcap[tickers]

    returns = adj.pct_change()

    # Market-cap-weighted index
    weights = mktcap.div(mktcap.sum(axis=1), axis=0)
    mkt_ret = (returns * weights.shift(1)).sum(axis=1).dropna()

    # ── 1. Realized vs. Parkinson volatility ────────────────────────────────
    print("1/3  Realized vs Parkinson volatility …")

    realized_var = (mkt_ret**2).rolling(WINDOW, min_periods=15).mean()
    realized_vol = np.sqrt(realized_var * 252) * 100

    log_hl = np.log(high / low)
    parkinson_var_stock = log_hl**2 / (4 * np.log(2))

    park_mkt = (parkinson_var_stock * weights.shift(1)).sum(axis=1).dropna()
    park_mkt_rolling = park_mkt.rolling(WINDOW, min_periods=15).mean()
    parkinson_vol = np.sqrt(park_mkt_rolling * 252) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        realized_vol.index,
        realized_vol.values,
        color="steelblue",
        linewidth=1,
        label="Realized vol (close-to-close)",
    )
    ax.plot(
        parkinson_vol.index,
        parkinson_vol.values,
        color="crimson",
        linewidth=1,
        alpha=0.8,
        label="Parkinson vol (high-low)",
    )
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title(f"Realized vs. Parkinson Volatility ({WINDOW}-day rolling)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "21_realized_vs_parkinson_vol.png"))
    plt.close(fig)
    print("   → saved 21_realized_vs_parkinson_vol.png")

    # ── 2. Volatility term structure: 5d vs 20d vs 60d ──────────────────────
    print("2/3  Volatility term structure …")

    windows = {"5d": 5, "20d": 20, "60d": 60}
    vol_series = {}
    for label, w in windows.items():
        rv = (mkt_ret**2).rolling(w, min_periods=max(3, w // 2)).mean()
        vol_series[label] = np.sqrt(rv * 252) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = {"5d": "#e41a1c", "20d": "#377eb8", "60d": "#4daf4a"}
    for label, vs in vol_series.items():
        axes[0].plot(
            vs.index,
            vs.values,
            color=colors[label],
            linewidth=0.8,
            label=label,
            alpha=0.85,
        )
    axes[0].set_ylabel("Annualized vol (%)")
    axes[0].set_title("Volatility Term Structure: 5d vs 20d vs 60d Realized Vol")
    axes[0].legend(fontsize=9)

    spread = vol_series["5d"] - vol_series["60d"]
    axes[1].fill_between(
        spread.index,
        spread.values,
        0,
        where=spread > 0,
        color="salmon",
        alpha=0.5,
        label="5d > 60d (stress)",
    )
    axes[1].fill_between(
        spread.index,
        spread.values,
        0,
        where=spread <= 0,
        color="lightgreen",
        alpha=0.5,
        label="60d ≥ 5d (calm)",
    )
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Spread (pp)")
    axes[1].set_title("Vol Term Spread: 5d − 60d")
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "22_vol_term_structure.png"))
    plt.close(fig)
    print("   → saved 22_vol_term_structure.png")

    # ── 3. Leverage effect ──────────────────────────────────────────────────
    print("3/3  Leverage effect …")

    fwd_vol_simple = realized_vol.shift(-WINDOW)

    df_lev = pd.DataFrame({"ret": mkt_ret, "fwd_vol": fwd_vol_simple}).dropna()
    rolling_corr = df_lev["ret"].rolling(252, min_periods=126).corr(df_lev["fwd_vol"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    weekly = df_lev.resample("W").last().dropna()
    axes[0].scatter(
        weekly["ret"] * 100, weekly["fwd_vol"], s=8, alpha=0.3, color="steelblue"
    )
    z = np.polyfit(weekly["ret"] * 100, weekly["fwd_vol"], 1)
    x_line = np.linspace(weekly["ret"].min() * 100, weekly["ret"].max() * 100, 100)
    axes[0].plot(
        x_line,
        np.polyval(z, x_line),
        color="crimson",
        linewidth=2,
        label=f"slope = {z[0]:.2f}",
    )
    axes[0].set_xlabel("Weekly return (%)")
    axes[0].set_ylabel("Forward 20d realized vol (%)")
    axes[0].set_title("Leverage Effect: Return vs Future Volatility")
    axes[0].legend(fontsize=9)

    axes[1].plot(rolling_corr.index, rolling_corr.values, color="purple", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(
        rolling_corr.mean(),
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"mean = {rolling_corr.mean():.3f}",
    )
    axes[1].set_ylabel("Rolling 252d correlation")
    axes[1].set_title("Rolling Corr(return, forward 20d vol)")
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "23_leverage_effect.png"))
    plt.close(fig)
    print("   → saved 23_leverage_effect.png")

    # ── Summary stats ────────────────────────────────────────────────────────
    summary = {
        "Avg realized vol (20d, ann %)": f"{realized_vol.mean():.1f}",
        "Avg Parkinson vol (20d, ann %)": f"{parkinson_vol.mean():.1f}",
        "Parkinson / Realized ratio": f"{(parkinson_vol / realized_vol).dropna().mean():.3f}",
        "Avg 5d vol (%)": f"{vol_series['5d'].mean():.1f}",
        "Avg 60d vol (%)": f"{vol_series['60d'].mean():.1f}",
        "Mean leverage corr": f"{rolling_corr.mean():.3f}",
        "Leverage slope (pp vol per 1% ret)": f"{z[0]:.3f}",
    }
    summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "24_volatility_summary.csv"), index=False
    )
    print("   → saved 24_volatility_summary.csv")
    print("\n" + summary_df.to_string(index=False))
    print("\nDone – volatility deep-dive complete.")


if __name__ == "__main__":
    main()
