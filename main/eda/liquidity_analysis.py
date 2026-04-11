"""
eda/liquidity_analysis.py
-------------------------
Part 5 – Liquidity analysis

Analyses:
  1. Amihud illiquidity ratio (|return| / dollar volume) over time
  2. Turnover distribution and its relationship to future returns
  3. Liquidity drying up during drawdowns

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


def main(data=None, output_dir=None):
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    if data is None:
        print("Loading data …")
        data = load_all_data()
    # ── Load data ────────────────────────────────────────────────────────────
    adj = data["adjusted"]
    dv = data["dv"]
    mktcap = data["mktcap"]

    tickers = get_always_in_universe(data)
    adj = adj[tickers]
    dv = dv[tickers]
    mktcap = mktcap[tickers]

    returns = adj.pct_change()

    # Market-cap-weighted index for drawdown analysis
    weights = mktcap.div(mktcap.sum(axis=1), axis=0)
    mkt_ret = (returns * weights.shift(1)).sum(axis=1).dropna()
    mkt_cum = (1 + mkt_ret).cumprod()

    # ── 1. Amihud illiquidity ratio ──────────────────────────────────────────
    print("1/3  Amihud illiquidity ratio …")

    abs_ret = returns.abs()
    dv_safe = dv.replace(0, np.nan)
    amihud = (abs_ret / dv_safe) * 1e6

    amihud_median = amihud.median(axis=1).dropna()
    amihud_smooth = amihud_median.rolling(60, min_periods=30).median()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(
        amihud_smooth.index,
        amihud_smooth.values,
        color="crimson",
        linewidth=1,
        label="Amihud illiquidity (60d median)",
    )
    ax1.set_ylabel("Amihud ratio (×10⁶)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")

    ax2 = ax1.twinx()
    ax2.plot(
        mkt_cum.index,
        mkt_cum.values,
        color="steelblue",
        linewidth=0.8,
        alpha=0.6,
        label="Market index (cum)",
    )
    ax2.set_ylabel("Cumulative market return", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    ax1.set_title("Cross-sectional Median Amihud Illiquidity Ratio Over Time")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "17_amihud_illiquidity.png"))
    plt.close(fig)
    print("   → saved 17_amihud_illiquidity.png")

    # ── 2. Turnover distribution & relationship to future returns ────────────
    print("2/3  Turnover vs future returns …")

    turnover = dv / mktcap
    turnover = turnover.replace([np.inf, -np.inf], np.nan)

    monthly_ret = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_turnover = turnover.resample("ME").mean()

    dates = sorted(set(monthly_ret.index) & set(monthly_turnover.index))
    quintile_returns = {q: [] for q in range(1, 6)}
    quintile_dates = []

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]
        to = monthly_turnover.loc[dt].dropna()
        ret_next = monthly_ret.loc[dt_next]
        common = to.index.intersection(ret_next.dropna().index)
        if len(common) < 50:
            continue
        to = to[common]
        ret_next = ret_next[common]
        quintile_dates.append(dt_next)
        bins = pd.qcut(to, 5, labels=False, duplicates="drop") + 1
        for q in range(1, 6):
            mask = bins == q
            quintile_returns[q].append(ret_next[mask].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for q in range(1, 6):
        cum = (1 + pd.Series(quintile_returns[q], index=quintile_dates)).cumprod()
        axes[0].plot(cum.index, cum.values, label=f"Q{q}")
    axes[0].set_title("Cumulative Returns by Turnover Quintile")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend(title="Turnover\n(Q1=low, Q5=high)", fontsize=8)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ann_rets = {}
    for q in range(1, 6):
        s = pd.Series(quintile_returns[q])
        ann_rets[f"Q{q}"] = ((1 + s.mean()) ** 12 - 1) * 100
    ann_rets["L/S\n(Q1−Q5)"] = ann_rets["Q1"] - ann_rets["Q5"]
    colors = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
    axes[1].bar(ann_rets.keys(), ann_rets.values(), color=colors)
    axes[1].set_title("Annualized Return by Turnover Quintile")
    axes[1].set_ylabel("Annualized return (%)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    for i, (k, v) in enumerate(ann_rets.items()):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "18_turnover_vs_returns.png"))
    plt.close(fig)
    print("   → saved 18_turnover_vs_returns.png")

    # ── 3. Liquidity drying up during drawdowns ─────────────────────────────
    print("3/3  Liquidity during drawdowns …")

    running_max = mkt_cum.cummax()
    drawdown = (mkt_cum / running_max) - 1

    total_dv = dv.sum(axis=1)
    total_dv_smooth = total_dv.rolling(20, min_periods=10).mean()

    regime = pd.cut(
        drawdown,
        bins=[-np.inf, -0.20, -0.05, 0],
        labels=["Bear (<−20%)", "Correction (−20%→−5%)", "Normal (>−5%)"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors_map = {
        "Normal (>−5%)": "#a1d99b",
        "Correction (−20%→−5%)": "#fdae6b",
        "Bear (<−20%)": "#e34a33",
    }
    axes[0].fill_between(
        drawdown.index, drawdown.values, 0, color="lightcoral", alpha=0.4
    )
    axes[0].plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.6)
    axes[0].set_ylabel("Drawdown")
    axes[0].set_title("Market Drawdown & Aggregate Dollar Volume")

    common_idx = total_dv_smooth.dropna().index.intersection(regime.dropna().index)
    for label, color in colors_map.items():
        mask = regime.loc[common_idx] == label
        idx = common_idx[mask]
        axes[1].bar(
            idx,
            total_dv_smooth.loc[idx].values / 1e9,
            width=2,
            color=color,
            alpha=0.7,
            label=label,
        )
    axes[1].set_ylabel("Agg. dollar volume (¥ bn, 20d MA)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "19_liquidity_drawdowns.png"))
    plt.close(fig)
    print("   → saved 19_liquidity_drawdowns.png")

    # ── Summary table ────────────────────────────────────────────────────────
    summary_rows = []
    for label in ["Normal (>−5%)", "Correction (−20%→−5%)", "Bear (<−20%)"]:
        mask = regime == label
        idx = mask[mask].index
        avg_to = turnover.loc[turnover.index.isin(idx)].median(axis=1).mean()
        avg_am = amihud_median.loc[amihud_median.index.isin(idx)].mean()
        avg_dv_bn = total_dv.loc[total_dv.index.isin(idx)].mean() / 1e9
        summary_rows.append(
            {
                "Regime": label,
                "Avg daily turnover (median)": f"{avg_to:.4f}",
                "Avg Amihud (×10⁶)": f"{avg_am:.4f}",
                "Avg agg DV (¥ bn)": f"{avg_dv_bn:.1f}",
                "# days": int(mask.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "20_liquidity_regime_summary.csv"), index=False
    )
    print("   → saved 20_liquidity_regime_summary.csv")
    print("\n" + summary.to_string(index=False))
    print("\nDone – liquidity analysis complete.")


if __name__ == "__main__":
    main()
