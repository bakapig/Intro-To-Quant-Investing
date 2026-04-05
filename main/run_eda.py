"""
run_eda.py
----------
Main entry point for Exploratory Data Analysis (ZY's approach).

Runs all EDA modules sequentially:
  Part 1: Return distribution analysis
  Part 2: Factor analysis (quintile sorts)
  Part 3: Correlation structure (PCA, sector correlations)
  Part 4: (reserved)
  Part 5: Liquidity analysis
  Part 6: Volatility deep-dive
  Part 7: Regime characterization
  Part 8: Survivorship bias analysis

Also reproduces JC's data quality EDA (price coverage, market index construction).

Usage:
    cd main
    python run_eda.py
    python run_eda.py --part 1       # Run only Part 1
    python run_eda.py --part 1,2,3   # Run specific parts
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

from data_loader import load_all_data, build_cap_weighted_market_index

# ===========================================================================
# Configuration
# ===========================================================================

DATA_DIR = "data_cn"
OUTPUT_DIR = os.path.join("output", "eda")


# ===========================================================================
# Part 0: Data Quality & Market Overview (from JC's data_cn_eda.ipynb)
# ===========================================================================


def run_data_quality_eda(data):
    """JC's data quality and market overview analysis."""
    print("\n" + "=" * 60)
    print("  Part 0: Data Quality & Market Overview")
    print("=" * 60)

    df_prices = data["adjusted"]
    df_mcap = data["mktcap"]

    prices = df_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)

    def internal_missing_count(s):
        valid = s.notna().to_numpy()
        if not valid.any():
            return 0
        first = np.argmax(valid)
        last = len(valid) - 1 - np.argmax(valid[::-1])
        return int(s.iloc[first : last + 1].isna().sum())

    def max_drawdown(s):
        s = s.dropna()
        if s.empty:
            return np.nan
        dd = s / s.cummax() - 1
        return dd.min()

    print(f"  Shape: {prices.shape}")
    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Average missingness: {prices.isna().mean().mean() * 100:.2f}%")

    # Coverage analysis
    coverage = pd.DataFrame(
        {
            "n_obs": prices.notna().sum(),
            "missing_pct": prices.isna().mean() * 100,
            "first_date": prices.apply(lambda s: s.first_valid_index()),
            "last_date": prices.apply(lambda s: s.last_valid_index()),
            "internal_missing": prices.apply(internal_missing_count),
        }
    ).sort_values("n_obs", ascending=False)
    coverage["lifespan_days"] = (coverage["last_date"] - coverage["first_date"]).dt.days

    # Active assets over time
    active_assets = prices.notna().sum(axis=1)
    plt.figure(figsize=(12, 8))
    active_assets.plot()
    plt.title("Number of active assets over time")
    plt.ylabel("Count")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "00_active_assets.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Observation distribution
    plt.figure(figsize=(12, 8))
    coverage["n_obs"].hist(bins=40)
    plt.title("Distribution of observations per asset")
    plt.xlabel("Non-missing price observations")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "00_obs_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Normalized price paths
    top10 = coverage.head(10).index
    norm_prices = prices[top10].copy()
    for c in top10:
        s = norm_prices[c].dropna()
        if not s.empty:
            norm_prices[c] = 100 * norm_prices[c] / s.iloc[0]

    plt.figure(figsize=(12, 5))
    norm_prices.plot(ax=plt.gca(), legend=False)
    plt.title("Normalized price paths of 10 longest-history assets (start = 100)")
    plt.ylabel("Normalized level")
    plt.legend()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "00_normalized_paths.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Returns and asset statistics
    simple_ret = prices.div(prices.shift(1)) - 1
    log_ret = np.log1p(simple_ret)

    asset_stats = pd.DataFrame(
        {
            "ann_mean_log_ret": log_ret.mean() * 252,
            "ann_vol": log_ret.std() * np.sqrt(252),
            "skew": log_ret.skew(),
            "excess_kurtosis": log_ret.kurtosis(),
            "max_drawdown": prices.apply(max_drawdown),
        }
    )
    asset_stats["sharpe_proxy"] = (
        asset_stats["ann_mean_log_ret"] / asset_stats["ann_vol"]
    )

    # EW index
    ew_ret = simple_ret.mean(axis=1, skipna=True)
    ew_index = (1 + ew_ret.fillna(0)).cumprod()
    ew_vol_63 = ew_ret.rolling(63).std() * np.sqrt(252)
    ew_drawdown = ew_index / ew_index.cummax() - 1
    xsec_dispersion = simple_ret.std(axis=1)

    for name, series, title, ylabel in [
        ("00_ew_index", ew_index, "Equal-weight universe index", "Index level"),
        (
            "00_ew_volatility",
            ew_vol_63,
            "63-day realized volatility of EW universe",
            "Ann. volatility",
        ),
        ("00_ew_drawdown", ew_drawdown, "Equal-weight universe drawdown", "Drawdown"),
        (
            "00_xsec_dispersion",
            xsec_dispersion,
            "Cross-sectional dispersion",
            "Std. dev.",
        ),
    ]:
        plt.figure(figsize=(12, 8))
        series.plot()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Market diagnostics
    jb = jarque_bera(ew_ret.dropna())
    lb_ret = acorr_ljungbox(ew_ret.dropna(), lags=[5, 10, 20], return_df=True)
    lb_sq = acorr_ljungbox((ew_ret.dropna() ** 2), lags=[5, 10, 20], return_df=True)

    market_diag = pd.Series(
        {
            "mean_daily_return": ew_ret.mean(),
            "daily_volatility": ew_ret.std(),
            "annualized_volatility": ew_ret.std() * np.sqrt(252),
            "skew": ew_ret.skew(),
            "excess_kurtosis": ew_ret.kurtosis(),
            "jarque_bera_pvalue": getattr(jb, "pvalue", jb[1]),
            "lb_ret_pvalue_lag5": lb_ret.loc[5, "lb_pvalue"],
            "lb_ret_pvalue_lag10": lb_ret.loc[10, "lb_pvalue"],
            "lb_sqret_pvalue_lag5": lb_sq.loc[5, "lb_pvalue"],
            "lb_sqret_pvalue_lag10": lb_sq.loc[10, "lb_pvalue"],
        }
    )
    market_diag.to_csv(os.path.join(OUTPUT_DIR, "00_market_diagnostics.csv"))
    print("  Market diagnostics saved.")

    # Correlation heatmap
    corr_cols = coverage.loc[coverage["n_obs"] >= 252].head(40).index
    corr_mat = log_ret[corr_cols].corr(min_periods=252)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_mat, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.title(f"Return correlation heatmap ({len(corr_cols)} assets)")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(corr_cols)), corr_cols, fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "00_correlation_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Cap-weighted market index
    print("  Building cap-weighted market index...")
    market_df, weights, asset_ret = build_cap_weighted_market_index(
        df_prices=df_prices, df_mcap=df_mcap, base=100.0
    )

    for name, col, title, ylabel in [
        ("00_mcw_return", "market_cap_weighted_return", "Daily MCW Return", "Return"),
        (
            "00_mcw_index",
            "market_cap_weighted_index",
            "MCW Index (Base=100)",
            "Index Level",
        ),
        (
            "00_mcw_vol",
            "rolling_20d_ann_vol",
            "20-Day Rolling Ann. Volatility",
            "Volatility",
        ),
    ]:
        plt.figure(figsize=(12, 8))
        market_df[col].plot()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Regime proxy analysis
    regime_proxy = pd.DataFrame(
        {
            "ew_ret": ew_ret,
            "ew_vol_63": ew_vol_63,
            "xsec_dispersion": xsec_dispersion,
            "active_assets": active_assets,
        }
    ).dropna()

    regime_proxy["vol_bucket"] = pd.qcut(
        regime_proxy["ew_vol_63"],
        q=3,
        labels=["Low Vol", "Mid Vol", "High Vol"],
        duplicates="drop",
    )
    regime_summary = regime_proxy.groupby("vol_bucket").agg(
        {
            "ew_ret": ["mean", "std", "min", "max"],
            "xsec_dispersion": ["mean", "median"],
            "active_assets": ["mean", "min"],
        }
    )
    regime_summary.to_csv(os.path.join(OUTPUT_DIR, "00_regime_proxy.csv"))

    print("  Part 0 complete.")


# ===========================================================================
# Main runner
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Run EDA modules")
    parser.add_argument(
        "--part",
        type=str,
        default="all",
        help="Which parts to run: 'all', or comma-separated e.g. '0,1,2,3'",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Exploratory Data Analysis Pipeline")
    print("=" * 70)

    # Determine which parts to run
    if args.part == "all":
        parts = [0, 1, 2, 3, 5, 6, 7, 8]
    else:
        parts = [int(p.strip()) for p in args.part.split(",")]

    # Load data once
    print("\nLoading data...")
    data = load_all_data(DATA_DIR)
    print(f"  Loaded: {list(data.keys())}")

    if 0 in parts:
        run_data_quality_eda(data)

    # Import and run ZY's EDA modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eda"))

    if 1 in parts:
        print("\n  Running Part 1: Return Distribution...")
        from eda.return_distribution import main as run_return_dist

        run_return_dist()

    if 2 in parts:
        print("\n  Running Part 2: Factor Analysis...")
        from eda.factor_analysis import main as run_factor

        run_factor()

    if 3 in parts:
        print("\n  Running Part 3: Correlation Structure...")
        from eda.correlation_structure import main as run_corr

        run_corr()

    if 5 in parts:
        print("\n  Running Part 5: Liquidity Analysis...")
        from eda.liquidity_analysis import main as run_liq

        run_liq()

    if 6 in parts:
        print("\n  Running Part 6: Volatility Deep-Dive...")
        from eda.volatility_deepdive import main as run_vol

        run_vol()

    if 7 in parts:
        print("\n  Running Part 7: Regime Characterization...")
        from eda.regime_characterization import main as run_regime

        run_regime()

    if 8 in parts:
        print("\n  Running Part 8: Survivorship Analysis...")
        from eda.survivorship_analysis import main as run_surv

        run_surv()

    print("\n" + "=" * 70)
    print("  EDA pipeline complete! Check output/eda/ for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
