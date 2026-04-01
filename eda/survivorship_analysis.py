"""
eda/survivorship_analysis.py
----------------------------
Part 8 – Survivorship & universe dynamics

Analyses:
  1. Universe size & entry/exit counts per year
  2. Return characteristics of entering vs. exiting stocks
  3. Survivorship bias: compare full-universe vs. survivors-only index

All outputs go to  output/eda/ .
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_all_data

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


# ── helpers ──────────────────────────────────────────────────────────────────


def _universe_by_year(data: dict) -> dict:
    """Return {year: set(tickers)} from univ_h.csv."""
    univ = data["univ_h"]
    ticker_cols = [c for c in univ.columns if c != "year"]
    result = {}
    for _, row in univ.iterrows():
        yr = int(row["year"])
        tickers = set(t for t in ticker_cols if row[t] == 1)
        result[yr] = tickers
    return result


# ── Analysis 1: Universe dynamics ────────────────────────────────────────────


def plot_universe_dynamics(data: dict) -> pd.DataFrame:
    print("  1. Universe dynamics (entries / exits per year) …")

    univ = _universe_by_year(data)
    years = sorted(univ.keys())

    records = []
    for i, yr in enumerate(years):
        row = {"Year": yr, "Universe Size": len(univ[yr])}
        if i > 0:
            prev = univ[years[i - 1]]
            curr = univ[yr]
            row["Entries"] = len(curr - prev)
            row["Exits"] = len(prev - curr)
            row["Turnover"] = (
                (row["Entries"] + row["Exits"]) / len(prev) if len(prev) > 0 else 0
            )
        else:
            row["Entries"] = len(univ[yr])
            row["Exits"] = 0
            row["Turnover"] = 0
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "28_universe_dynamics.csv"), index=False)
    print("    ✓ 28_universe_dynamics.csv")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: Universe size
    ax = axes[0]
    ax.bar(df["Year"], df["Universe Size"], color="steelblue", alpha=0.8)
    ax.set_ylabel("Number of stocks")
    ax.set_title("Investable Universe Size Over Time")
    ax.grid(alpha=0.3, axis="y")
    for _, r in df.iterrows():
        ax.text(
            r["Year"],
            r["Universe Size"] + 5,
            str(int(r["Universe Size"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 2: Entries and exits
    ax = axes[1]
    ax.bar(df["Year"], df["Entries"], color="#2ca02c", alpha=0.8, label="Entries")
    ax.bar(df["Year"], -df["Exits"], color="#d62728", alpha=0.8, label="Exits")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Count")
    ax.set_xlabel("Year")
    ax.set_title("Annual Universe Entries & Exits")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "28_universe_dynamics.png"))
    plt.close(fig)
    print("    ✓ 28_universe_dynamics.png")
    print(df.to_string(index=False))

    return df


# ── Analysis 2: Return characteristics of entrants vs exiters ────────────────


def plot_entry_exit_returns(data: dict) -> None:
    print("\n  2. Return characteristics of entering vs. exiting stocks …")

    univ = _universe_by_year(data)
    adj = data["adjusted"]
    years = sorted(univ.keys())

    records = []
    for i in range(1, len(years)):
        yr = years[i]
        prev_yr = years[i - 1]
        entries = univ[yr] - univ[prev_yr]
        exits = univ[prev_yr] - univ[yr]
        stayers = univ[yr] & univ[prev_yr]

        # Annual return in the PREVIOUS year for entrants
        # (entrants: first full year is yr, but we look at their return performance)
        # For entries: look at their return in year yr (their first year in)
        # For exits: look at their return in the last year they were in (prev_yr)
        # For stayers: return in year yr

        yr_mask = adj.index.year == yr
        prev_mask = adj.index.year == prev_yr

        def _annual_ret(tickers, year_mask):
            valid = [t for t in tickers if t in adj.columns]
            if not valid or not year_mask.any():
                return np.nan, np.nan
            prices = adj.loc[year_mask, valid]
            if len(prices) < 2:
                return np.nan, np.nan
            rets = (prices.iloc[-1] / prices.iloc[0] - 1).dropna()
            if len(rets) == 0:
                return np.nan, np.nan
            return rets.mean(), rets.median()

        # Entries return in their first year
        entry_mean, entry_med = _annual_ret(entries, yr_mask)
        # Exits return in their last year (the year before leaving)
        exit_mean, exit_med = _annual_ret(exits, prev_mask)
        # Stayers return in yr
        stay_mean, stay_med = _annual_ret(stayers, yr_mask)

        records.append(
            {
                "Year": yr,
                "Entry Mean Ret": entry_mean,
                "Entry Median Ret": entry_med,
                "Exit Mean Ret (prev yr)": exit_mean,
                "Exit Median Ret (prev yr)": exit_med,
                "Stayer Mean Ret": stay_mean,
                "Stayer Median Ret": stay_med,
                "N Entries": len(entries),
                "N Exits": len(exits),
                "N Stayers": len(stayers),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "29_entry_exit_returns.csv"), index=False)
    print("    ✓ 29_entry_exit_returns.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = df["Year"]
    w = 0.25
    ax.bar(
        x - w,
        df["Entry Mean Ret"],
        w,
        color="#2ca02c",
        alpha=0.8,
        label="Entries (first yr)",
    )
    ax.bar(x, df["Stayer Mean Ret"], w, color="steelblue", alpha=0.8, label="Stayers")
    ax.bar(
        x + w,
        df["Exit Mean Ret (prev yr)"],
        w,
        color="#d62728",
        alpha=0.8,
        label="Exits (last yr)",
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Mean Annual Return")
    ax.set_xlabel("Year")
    ax.set_title("Annual Returns: Entrants vs Stayers vs Exiters")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "29_entry_exit_returns.png"))
    plt.close(fig)
    print("    ✓ 29_entry_exit_returns.png")


# ── Analysis 3: Survivorship bias ────────────────────────────────────────────


def plot_survivorship_bias(data: dict) -> None:
    print("\n  3. Survivorship bias comparison …")

    adj = data["adjusted"]
    mktcap = data["mktcap"]
    univ = _universe_by_year(data)
    years = sorted(univ.keys())

    # Full-universe index: at each year, include all tickers in that year's universe
    # Survivors-only index: only tickers in the universe for ALL years
    all_years_tickers = set.intersection(*[univ[y] for y in years])
    survivors = [t for t in all_years_tickers if t in adj.columns]
    print(f"    Survivors in ALL years: {len(survivors)}")

    daily_ret = adj.pct_change()

    # Full-universe equal-weighted index (rebalanced yearly)
    full_idx_ret = pd.Series(0.0, index=daily_ret.index)
    for yr in years:
        yr_tickers = [t for t in univ[yr] if t in adj.columns]
        yr_mask = daily_ret.index.year == yr
        if yr_tickers and yr_mask.any():
            full_idx_ret.loc[yr_mask] = daily_ret.loc[yr_mask, yr_tickers].mean(axis=1)

    # Survivor-only equal-weighted index
    surv_idx_ret = daily_ret[survivors].mean(axis=1)

    # Also do market-cap weighted versions
    full_mcw_ret = pd.Series(0.0, index=daily_ret.index)
    for yr in years:
        yr_tickers = [t for t in univ[yr] if t in adj.columns and t in mktcap.columns]
        yr_mask = daily_ret.index.year == yr
        if yr_tickers and yr_mask.any():
            w = mktcap.loc[yr_mask, yr_tickers]
            w = w.div(w.sum(axis=1), axis=0)
            full_mcw_ret.loc[yr_mask] = (daily_ret.loc[yr_mask, yr_tickers] * w).sum(
                axis=1
            )

    surv_w = mktcap[[s for s in survivors if s in mktcap.columns]]
    surv_w = surv_w.div(surv_w.sum(axis=1), axis=0)
    surv_mcw_ret = (
        daily_ret[[s for s in survivors if s in mktcap.columns]] * surv_w
    ).sum(axis=1)

    # Cumulative
    full_cum = (1 + full_idx_ret).cumprod()
    surv_cum = (1 + surv_idx_ret).cumprod()
    full_mcw_cum = (1 + full_mcw_ret).cumprod()
    surv_mcw_cum = (1 + surv_mcw_ret).cumprod()

    # Drop leading NaNs
    start = full_cum.first_valid_index()
    full_cum = full_cum.loc[start:]
    surv_cum = surv_cum.loc[start:]
    full_mcw_cum = full_mcw_cum.loc[start:]
    surv_mcw_cum = surv_mcw_cum.loc[start:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(
        full_cum.index,
        full_cum.values,
        lw=1.2,
        color="steelblue",
        label="Full universe (EW)",
    )
    ax.plot(
        surv_cum.index,
        surv_cum.values,
        lw=1.2,
        color="#d62728",
        label="Survivors only (EW)",
    )
    ax.set_title("Equal-Weighted: Full Universe vs Survivors")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        full_mcw_cum.index,
        full_mcw_cum.values,
        lw=1.2,
        color="steelblue",
        label="Full universe (MCW)",
    )
    ax.plot(
        surv_mcw_cum.index,
        surv_mcw_cum.values,
        lw=1.2,
        color="#d62728",
        label="Survivors only (MCW)",
    )
    ax.set_title("Market-Cap Weighted: Full Universe vs Survivors")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Survivorship Bias: Impact on Index Returns", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "30_survivorship_bias.png"))
    plt.close(fig)
    print("    ✓ 30_survivorship_bias.png")

    # Summary stats
    def _ann_stats(ret_series):
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + ret_series).prod()
        return {
            "Ann Return": ann_ret,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Total Growth": cum,
        }

    summary = pd.DataFrame(
        {
            "Full Universe (EW)": _ann_stats(full_idx_ret.loc[start:]),
            "Survivors Only (EW)": _ann_stats(surv_idx_ret.loc[start:]),
            "Full Universe (MCW)": _ann_stats(full_mcw_ret.loc[start:]),
            "Survivors Only (MCW)": _ann_stats(surv_mcw_ret.loc[start:]),
        }
    ).T
    summary.index.name = "Index"
    summary.to_csv(os.path.join(OUTPUT_DIR, "30_survivorship_bias_summary.csv"))
    print("    ✓ 30_survivorship_bias_summary.csv")
    print(summary.round(4).to_string())


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    print("Loading data …")
    data = load_all_data()

    plot_universe_dynamics(data)
    plot_entry_exit_returns(data)
    plot_survivorship_bias(data)

    print("\n✓ Part 8 complete – survivorship analysis saved to output/eda/")


if __name__ == "__main__":
    main()
