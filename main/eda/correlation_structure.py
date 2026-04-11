"""
eda/correlation_structure.py
----------------------------
Part 3 – Correlation structure analysis

Analyses:
  1. Rolling average pairwise correlation (diversification over time)
  2. Correlation heatmap by GICS sector (from tickers.csv)
  3. Eigenvalue analysis of the return covariance matrix (PCA)

All outputs go to  output/eda/ .
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_all_data, get_always_in_universe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

# GICS sector code → name mapping (top-level 2-digit)
GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Disc.",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Info Tech",
    "50": "Comm. Services",
    "55": "Utilities",
    "60": "Real Estate",
}


def _load_sector_map(data: dict) -> dict:
    """Map ticker → sector name using tickers.csv."""
    tickers_df = data["tickers"]
    col_ticker = tickers_df.columns[0]
    col_gics = tickers_df.columns[1]

    # First row of CSV was treated as header, so include it
    header_ticker = str(col_ticker).strip()
    header_gics = str(col_gics).strip()

    sector_map = {}
    # Header row
    sector_code = header_gics[:2] if len(header_gics) >= 2 else header_gics
    sector_map[header_ticker] = GICS_SECTORS.get(sector_code, f"Sector {sector_code}")

    for _, row in tickers_df.iterrows():
        ticker = str(row[col_ticker]).strip()
        gics = str(row[col_gics]).strip().split(".")[0]  # remove .0 from float
        sector_code = gics[:2] if len(gics) >= 2 else gics
        sector_map[ticker] = GICS_SECTORS.get(sector_code, f"Sector {sector_code}")

    return sector_map


# ── 1. Rolling average pairwise correlation ─────────────────────────────────


def plot_rolling_correlation(data: dict, window: int = 60) -> None:
    """
    Compute rolling average pairwise correlation among universe stocks.
    Uses weekly returns for stability and samples up to 100 stocks for speed.
    """
    adj = data["adjusted"]
    tickers = get_always_in_universe(data)
    tickers = [t for t in tickers if t in adj.columns]

    # Sample for speed
    np.random.seed(42)
    if len(tickers) > 100:
        tickers = list(np.random.choice(tickers, 100, replace=False))

    weekly_ret = (
        adj[tickers]
        .pct_change()
        .resample("W")
        .apply(lambda x: (1 + x).prod() - 1)
        .dropna(how="all")
    )

    # Rolling correlation: for each window, compute mean of upper-triangle corr
    dates = weekly_ret.index[window - 1 :]
    avg_corr = []
    for i in range(window - 1, len(weekly_ret)):
        chunk = weekly_ret.iloc[i - window + 1 : i + 1].dropna(axis=1, how="any")
        if chunk.shape[1] < 10:
            avg_corr.append(np.nan)
            continue
        corr = chunk.corr().values
        mask = np.triu_indices_from(corr, k=1)
        avg_corr.append(np.nanmean(corr[mask]))

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(dates, avg_corr, lw=0.9, color="teal")
    ax.set_title(
        f"Rolling {window}-week average pairwise correlation "
        f"(sample of {len(tickers)} stocks)"
    )
    ax.set_ylabel("Avg pairwise correlation")
    ax.set_xlabel("Date")
    ax.axhline(
        np.nanmean(avg_corr),
        color="grey",
        ls="--",
        lw=0.6,
        label=f"Mean = {np.nanmean(avg_corr):.3f}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "14_rolling_avg_correlation.png"))
    plt.close(fig)
    print("  ✓ 14_rolling_avg_correlation.png")


# ── 2. Sector correlation heatmap ───────────────────────────────────────────


def plot_sector_correlation(data: dict) -> None:
    """
    Compute sector-level returns (equal-weighted within sector),
    then plot cross-sector correlation heatmap.
    """
    adj = data["adjusted"]
    sector_map = _load_sector_map(data)

    tickers = get_always_in_universe(data)
    tickers = [t for t in tickers if t in adj.columns and t in sector_map]

    daily_ret = adj[tickers].pct_change().dropna(how="all")

    # Build sector-level return series
    sector_returns = {}
    for ticker in tickers:
        sec = sector_map.get(ticker)
        if sec:
            sector_returns.setdefault(sec, []).append(ticker)

    sector_ret_df = pd.DataFrame()
    for sec, sec_tickers in sorted(sector_returns.items()):
        sector_ret_df[sec] = daily_ret[sec_tickers].mean(axis=1)

    corr = sector_ret_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = TwoSlopeNorm(vmin=corr.min().min(), vcenter=0.5, vmax=1.0)
    im = ax.imshow(corr.values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_title("GICS sector return correlation (daily, full sample)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "15_sector_correlation_heatmap.png"))
    plt.close(fig)
    print("  ✓ 15_sector_correlation_heatmap.png")


# ── 3. Eigenvalue / PCA analysis ────────────────────────────────────────────


def plot_eigenvalue_analysis(data: dict) -> None:
    """
    PCA on the stock return covariance matrix.
    Shows scree plot and cumulative variance explained.
    """
    adj = data["adjusted"]
    tickers = get_always_in_universe(data)
    tickers = [t for t in tickers if t in adj.columns]

    # Sample for tractable covariance matrix
    np.random.seed(42)
    if len(tickers) > 200:
        tickers = list(np.random.choice(tickers, 200, replace=False))

    daily_ret = adj[tickers].pct_change().dropna(how="all")
    # Drop columns with too many NaNs
    daily_ret = daily_ret.dropna(axis=1, thresh=int(len(daily_ret) * 0.8))
    daily_ret = daily_ret.fillna(0)

    cov = daily_ret.cov().values
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    explained = eigenvalues / eigenvalues.sum()
    cum_explained = np.cumsum(explained)

    # How many PCs for 50%, 70%, 90%?
    n50 = np.searchsorted(cum_explained, 0.50) + 1
    n70 = np.searchsorted(cum_explained, 0.70) + 1
    n90 = np.searchsorted(cum_explained, 0.90) + 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scree plot
    ax = axes[0]
    n_show = min(50, len(eigenvalues))
    ax.bar(
        range(1, n_show + 1),
        explained[:n_show] * 100,
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("Scree plot – variance explained per PC")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% variance explained")
    ax.set_xlim(0.5, n_show + 0.5)
    ax.grid(axis="y", alpha=0.3)

    # Cumulative
    ax = axes[1]
    ax.plot(range(1, len(cum_explained) + 1), cum_explained * 100, lw=1.5, color="teal")
    ax.axhline(50, color="orange", ls="--", lw=0.7, label=f"50% → {n50} PCs")
    ax.axhline(70, color="red", ls="--", lw=0.7, label=f"70% → {n70} PCs")
    ax.axhline(90, color="darkred", ls="--", lw=0.7, label=f"90% → {n90} PCs")
    ax.set_title("Cumulative variance explained")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative % explained")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(cum_explained))

    fig.suptitle(
        f"Eigenvalue analysis of return covariance matrix "
        f"({daily_ret.shape[1]} stocks)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "16_eigenvalue_pca.png"))
    plt.close(fig)

    print("  ✓ 16_eigenvalue_pca.png")
    print(f"    PCs for 50% variance: {n50}")
    print(f"    PCs for 70% variance: {n70}")
    print(f"    PCs for 90% variance: {n90}")
    print(f"    PC1 explains {explained[0]*100:.1f}% of total variance")


# ── main ─────────────────────────────────────────────────────────────────────


def main(data=None, output_dir=None):
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    if data is None:
        print("Loading data …")
        data = load_all_data()

    print("\nRunning EDA – Correlation structure")
    print("=" * 45)

    print("\n1. Rolling average pairwise correlation …")
    plot_rolling_correlation(data)

    print("\n2. Sector correlation heatmap …")
    plot_sector_correlation(data)

    print("\n3. Eigenvalue / PCA analysis …")
    plot_eigenvalue_analysis(data)

    print(f"\nDone. Outputs saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
