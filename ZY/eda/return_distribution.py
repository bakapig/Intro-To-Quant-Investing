"""
eda/return_distribution.py
--------------------------
Part 1 – Return distributions & stylized facts

Analyses:
  1. Daily, weekly, monthly return histograms vs normal overlay
  2. QQ plots against normal distribution
  3. Summary statistics table (mean, std, skewness, kurtosis)
  4. Autocorrelation of returns vs autocorrelation of |returns| (volatility clustering)
  5. Rolling kurtosis to show fat-tail dynamics over time

Uses the market-cap weighted index as the representative "market" series,
plus a random sample of individual stocks for cross-sectional colour.

All figures are saved to  output/eda/ .
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_all_data, get_always_in_universe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
    }
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_market_index(data: dict) -> pd.Series:
    """Market-cap-weighted daily return series from adjusted prices."""
    adj = data["adjusted"]
    mktcap = data["mktcap"]

    tickers = get_always_in_universe(data)
    tickers = [t for t in tickers if t in adj.columns and t in mktcap.columns]

    adj = adj[tickers]
    mktcap = mktcap[tickers]

    rets = adj.pct_change()
    weights = mktcap.div(mktcap.sum(axis=1), axis=0)
    idx_ret = (rets * weights).sum(axis=1).dropna()
    idx_ret.name = "market_return"
    return idx_ret


def _to_freq(daily_ret: pd.Series, rule: str) -> pd.Series:
    """Compound daily returns to a lower frequency (W / ME)."""
    return daily_ret.resample(rule).apply(lambda x: (1 + x).prod() - 1).dropna()


# ── 1. Return histograms with normal overlay ────────────────────────────────


def plot_return_histograms(daily_ret: pd.Series) -> None:
    weekly = _to_freq(daily_ret, "W")
    monthly = _to_freq(daily_ret, "ME")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, series, label in zip(
        axes,
        [daily_ret, weekly, monthly],
        ["Daily", "Weekly", "Monthly"],
    ):
        mu, sigma = series.mean(), series.std()
        ax.hist(
            series,
            bins=120,
            density=True,
            alpha=0.65,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
            label="Empirical",
        )
        x = np.linspace(series.min(), series.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.5, label="Normal fit")
        ax.set_title(f"{label} returns")
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Return distributions vs Normal", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_return_histograms.png"))
    plt.close(fig)
    print("  ✓ 01_return_histograms.png")


# ── 2. QQ plots ─────────────────────────────────────────────────────────────


def plot_qq(daily_ret: pd.Series) -> None:
    weekly = _to_freq(daily_ret, "W")
    monthly = _to_freq(daily_ret, "ME")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, series, label in zip(
        axes,
        [daily_ret, weekly, monthly],
        ["Daily", "Weekly", "Monthly"],
    ):
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(f"QQ – {label}")
        ax.get_lines()[0].set(markersize=2, alpha=0.5)

    fig.suptitle("QQ plots against Normal distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_qq_plots.png"))
    plt.close(fig)
    print("  ✓ 02_qq_plots.png")


# ── 3. Summary statistics table ─────────────────────────────────────────────


def summary_statistics(daily_ret: pd.Series) -> pd.DataFrame:
    weekly = _to_freq(daily_ret, "W")
    monthly = _to_freq(daily_ret, "ME")

    rows = []
    for series, label in [
        (daily_ret, "Daily"),
        (weekly, "Weekly"),
        (monthly, "Monthly"),
    ]:
        rows.append(
            {
                "Frequency": label,
                "N": len(series),
                "Mean (%)": series.mean() * 100,
                "Std (%)": series.std() * 100,
                "Skewness": series.skew(),
                "Kurtosis (excess)": series.kurtosis(),  # Fisher = excess
                "Min (%)": series.min() * 100,
                "Max (%)": series.max() * 100,
                "Jarque-Bera stat": stats.jarque_bera(series).statistic,
                "JB p-value": stats.jarque_bera(series).pvalue,
            }
        )

    df = pd.DataFrame(rows).set_index("Frequency")
    df.to_csv(
        os.path.join(OUTPUT_DIR, "03_summary_statistics.csv"), float_format="%.4f"
    )
    print("  ✓ 03_summary_statistics.csv")
    print(df.to_string())
    return df


# ── 4. Autocorrelation: returns vs |returns| (volatility clustering) ────────


def plot_autocorrelation(daily_ret: pd.Series, max_lag: int = 60) -> None:
    lags = range(1, max_lag + 1)
    acf_ret = [daily_ret.autocorr(lag=k) for k in lags]
    acf_abs = [daily_ret.abs().autocorr(lag=k) for k in lags]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, acf_ret, width=0.8, color="steelblue", alpha=0.7, label="Returns")
    ax.plot(
        lags,
        acf_abs,
        color="crimson",
        lw=1.8,
        marker="o",
        markersize=3,
        label="|Returns| (volatility proxy)",
    )
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of returns vs |returns| — volatility clustering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_autocorrelation.png"))
    plt.close(fig)
    print("  ✓ 04_autocorrelation.png")


# ── 5. Rolling kurtosis ─────────────────────────────────────────────────────


def plot_rolling_kurtosis(daily_ret: pd.Series, window: int = 252) -> None:
    roll_kurt = daily_ret.rolling(window).kurt()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(roll_kurt.index, roll_kurt.values, lw=0.9, color="teal")
    ax.axhline(0, color="grey", ls="--", lw=0.5, label="Normal (excess kurt = 0)")
    ax.set_title(f"Rolling {window}-day excess kurtosis of market index returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess kurtosis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_rolling_kurtosis.png"))
    plt.close(fig)
    print("  ✓ 05_rolling_kurtosis.png")


# ── 6. Cross-sectional return distribution snapshot ─────────────────────────


def plot_cross_sectional_snapshot(data: dict) -> None:
    """Distribution of individual stock annualised returns by year."""
    adj = data["adjusted"]
    tickers = get_always_in_universe(data)
    tickers = [t for t in tickers if t in adj.columns]
    rets = adj[tickers].pct_change()

    annual = rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    years = annual.index.year

    fig, ax = plt.subplots(figsize=(14, 5))
    boxdata = [
        annual.loc[annual.index.year == y].values.flatten()
        for y in sorted(years.unique())
    ]
    boxdata = [d[~np.isnan(d)] for d in boxdata]
    bp = ax.boxplot(
        boxdata,
        labels=sorted(years.unique()),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
    )
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Cross-sectional distribution of annual stock returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual return")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_cross_sectional_annual.png"))
    plt.close(fig)
    print("  ✓ 06_cross_sectional_annual.png")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    print("Loading data …")
    data = load_all_data()

    print("Building market-cap weighted index …")
    daily_ret = _build_market_index(data)

    print(
        f"Market index: {daily_ret.index[0].date()} → {daily_ret.index[-1].date()} "
        f"({len(daily_ret)} days)\n"
    )

    print("Running EDA – Return distributions & stylized facts")
    print("=" * 55)
    plot_return_histograms(daily_ret)
    plot_qq(daily_ret)
    summary_statistics(daily_ret)
    plot_autocorrelation(daily_ret)
    plot_rolling_kurtosis(daily_ret)
    plot_cross_sectional_snapshot(data)
    print("\nDone. Outputs saved to", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
