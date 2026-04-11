"""
eda/regime_characterization.py
------------------------------
Part 7 - Regime characterisation

Analyses:
  1. Regime overlay on drawdown, volatility, cross-sectional dispersion
  2. Transition matrix heatmap (how often regime X -> regime Y)
  3. Factor performance conditional on regime

All outputs go to  output/eda/ .
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

import config
from data_loader import load_all_data
from hmm_strategy import get_ordered_states
from hmmlearn.hmm import GaussianHMM

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

REGIME_COLORS = {
    "trending": "#2ca02c",
    "mean_reverting": "#d62728",
    "random_walk": "#7f7f7f",
}
REGIME_ORDER = ["trending", "mean_reverting", "random_walk"]


#  helpers 


def _build_market_index(data: dict) -> pd.Series:
    """Market-cap weighted daily index from adjusted prices."""
    adj = data["adjusted"]
    mktcap = data["mktcap"]
    ret = adj.pct_change()
    w = mktcap.div(mktcap.sum(axis=1), axis=0)
    idx_ret = (ret * w).sum(axis=1)
    idx = (1 + idx_ret).cumprod()
    idx.name = "index"
    return idx


def _drawdown(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return (prices - peak) / peak


#  Analysis 1: Regime vs drawdown / vol / dispersion 


def plot_regime_context(regimes: pd.DataFrame, data: dict) -> None:
    print("  1. Regime context (drawdown, vol, dispersion) ...")

    idx = _build_market_index(data)
    dd = _drawdown(idx)
    log_ret = np.log(idx / idx.shift(1))
    vol_20d = log_ret.rolling(20).std() * np.sqrt(252)

    # Cross-sectional dispersion: std of daily stock returns
    adj = data["adjusted"]
    stock_ret = adj.pct_change()
    cs_disp = stock_ret.std(axis=1)
    cs_disp_20d = cs_disp.rolling(20).mean()

    # Align
    common = regimes.index.intersection(idx.index)
    reg = regimes.loc[common]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Panel 1: Index with regime background
    ax = axes[0]
    ax.plot(common, idx.loc[common], lw=0.8, color="black")
    for regime in REGIME_ORDER:
        mask = reg["regime_label"] == regime
        if mask.any():
            ax.fill_between(
                common,
                idx.loc[common].min(),
                idx.loc[common].max(),
                where=mask,
                alpha=0.15,
                color=REGIME_COLORS[regime],
                label=regime.replace("_", " ").title(),
            )
    ax.set_ylabel("Index level")
    ax.set_title("Market Index with HMM Regime Overlay")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 2: Drawdown
    ax = axes[1]
    ax.fill_between(common, dd.loc[common], 0, alpha=0.5, color="navy")
    for regime in REGIME_ORDER:
        mask = reg["regime_label"] == regime
        if mask.any():
            ax.fill_between(
                common,
                dd.loc[common].min(),
                0,
                where=mask,
                alpha=0.1,
                color=REGIME_COLORS[regime],
            )
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown with Regime Overlay")
    ax.grid(alpha=0.3)

    # Panel 3: Volatility
    ax = axes[2]
    ax.plot(common, vol_20d.loc[common], lw=0.8, color="navy")
    for regime in REGIME_ORDER:
        mask = reg["regime_label"] == regime
        if mask.any():
            ax.fill_between(
                common,
                0,
                vol_20d.loc[common].max(),
                where=mask,
                alpha=0.1,
                color=REGIME_COLORS[regime],
            )
    ax.set_ylabel("20d ann. vol")
    ax.set_title("Realised Volatility with Regime Overlay")
    ax.grid(alpha=0.3)

    # Panel 4: Cross-sectional dispersion
    ax = axes[3]
    ax.plot(common, cs_disp_20d.loc[common], lw=0.8, color="navy")
    for regime in REGIME_ORDER:
        mask = reg["regime_label"] == regime
        if mask.any():
            ax.fill_between(
                common,
                0,
                cs_disp_20d.loc[common].max(),
                where=mask,
                alpha=0.1,
                color=REGIME_COLORS[regime],
            )
    ax.set_ylabel("CS dispersion (20d avg)")
    ax.set_title("Cross-sectional Return Dispersion with Regime Overlay")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "25_regime_context.png"))
    plt.close(fig)
    print("    [DONE] 25_regime_context.png")

    # Summary table
    records = []
    for regime in REGIME_ORDER:
        mask = reg["regime_label"] == regime
        if not mask.any():
            continue
        records.append(
            {
                "Regime": regime.replace("_", " ").title(),
                "Days": mask.sum(),
                "Pct of Time": f"{mask.mean():.1%}",
                "Avg Drawdown": dd.loc[common][mask].mean(),
                "Avg 20d Vol (ann)": vol_20d.loc[common][mask].mean(),
                "Avg CS Dispersion": cs_disp_20d.loc[common][mask].mean(),
                "Avg Daily Return": log_ret.loc[common][mask].mean(),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "25_regime_context_summary.csv"), index=False
    )
    print("    [DONE] 25_regime_context_summary.csv")
    print(summary.to_string(index=False))


#  Analysis 2: Transition matrix 


def plot_transition_matrix(regimes: pd.DataFrame) -> None:
    print("\n  2. Transition matrix ...")

    labels = regimes["regime_label"].values
    n = len(REGIME_ORDER)
    counts = np.zeros((n, n), dtype=int)
    idx_map = {r: i for i, r in enumerate(REGIME_ORDER)}

    for t in range(1, len(labels)):
        prev = labels[t - 1]
        curr = labels[t]
        if prev in idx_map and curr in idx_map:
            counts[idx_map[prev], idx_map[curr]] += 1

    # Row-normalise to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(
        counts, row_sums, where=row_sums > 0, out=np.zeros_like(counts, dtype=float)
    )

    nice_labels = [r.replace("_", " ").title() for r in REGIME_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count matrix
    ax = axes[0]
    im = ax.imshow(counts, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nice_labels, fontsize=9)
    ax.set_yticklabels(nice_labels, fontsize=9)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Transition Counts")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Probability matrix
    ax = axes[1]
    im = ax.imshow(probs, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{probs[i, j]:.2%}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nice_labels, fontsize=9)
    ax.set_yticklabels(nice_labels, fontsize=9)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Transition Probabilities")
    fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("HMM Regime Transition Matrix", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "26_transition_matrix.png"))
    plt.close(fig)
    print("    [DONE] 26_transition_matrix.png")

    # Average regime duration
    durations = {r: [] for r in REGIME_ORDER}
    current_regime = labels[0]
    run_len = 1
    for t in range(1, len(labels)):
        if labels[t] == current_regime:
            run_len += 1
        else:
            durations[current_regime].append(run_len)
            current_regime = labels[t]
            run_len = 1
    durations[current_regime].append(run_len)

    print("\n    Average regime duration (trading days):")
    for r in REGIME_ORDER:
        d = durations[r]
        if d:
            print(
                f"      {r:18s}  mean={np.mean(d):5.1f}  median={np.median(d):5.1f}  max={np.max(d):4d}"
            )


#  Analysis 3: Factor performance by regime 


def plot_factor_by_regime(regimes: pd.DataFrame, data: dict) -> None:
    print("\n  3. Factor performance by regime ...")

    adj = data["adjusted"]
    mktcap = data["mktcap"]
    p2b = data["p2b"]
    dv = data["dv"]

    # Monthly returns
    month_end = adj.resample("ME").last()
    monthly_ret = month_end.pct_change()
    fwd_ret = monthly_ret.shift(-1).iloc[:-1]

    # Assign regime to each month: majority regime in that month
    reg_monthly = (
        regimes["regime_label"].resample("ME").agg(lambda x: x.value_counts().idxmax())
    )

    # Signals (month-end)
    sig_pb = p2b.resample("ME").last()
    sig_size = mktcap.resample("ME").last()
    sig_mom = month_end.shift(1) / month_end.shift(12) - 1
    sig_dv = dv.resample("ME").last()

    # Universe mask
    from data_loader import load_all_data as _  # already imported

    univ = data["univ_h"]
    ticker_cols = [c for c in univ.columns if c != "year"]
    univ_mask = pd.DataFrame(0, index=month_end.index, columns=month_end.columns)
    for _, row in univ.iterrows():
        yr = int(row["year"])
        in_tickers = [t for t in ticker_cols if row[t] == 1 and t in month_end.columns]
        univ_mask.loc[univ_mask.index.year == yr, in_tickers] = 1

    factors = {
        "Value (P/B)": (sig_pb, True),  # ascending: low P/B = cheap = Q1
        "Size": (sig_size, True),  # ascending: small cap = Q1
        "Momentum": (sig_mom, False),  # descending: winners = Q1
        "Liquidity": (sig_dv, True),  # ascending: low DV = illiquid = Q1
    }

    common_dates = fwd_ret.index.intersection(reg_monthly.index)

    results = []
    for fname, (signal, ascending) in factors.items():
        for regime in REGIME_ORDER:
            regime_dates = common_dates[reg_monthly.loc[common_dates] == regime]
            if len(regime_dates) < 6:
                continue

            # Compute quintile spread for these months
            ls_rets = []
            for dt in regime_dates:
                if dt not in signal.index or dt not in fwd_ret.index:
                    continue
                sig = signal.loc[dt].dropna()
                ret = fwd_ret.loc[dt].dropna()
                umask = (
                    univ_mask.reindex(index=[dt]).iloc[0]
                    if dt in univ_mask.index
                    else pd.Series(dtype=float)
                )
                valid = sig.index.intersection(ret.index)
                if len(umask) > 0:
                    valid = valid[umask.reindex(valid).fillna(0).astype(bool)]
                if len(valid) < 50:
                    continue
                sig_v = sig[valid]
                ret_v = ret[valid]
                if ascending:
                    ranks = sig_v.rank(method="first")
                else:
                    ranks = (-sig_v).rank(method="first")
                qcut = pd.qcut(ranks, 5, labels=False) + 1
                q1_ret = ret_v[qcut == 1].mean()
                q5_ret = ret_v[qcut == 5].mean()
                ls_rets.append(q1_ret - q5_ret)

            if ls_rets:
                ls_series = pd.Series(ls_rets)
                ann_ret = ls_series.mean() * 12
                ann_vol = ls_series.std() * np.sqrt(12)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                results.append(
                    {
                        "Factor": fname,
                        "Regime": regime.replace("_", " ").title(),
                        "Months": len(ls_rets),
                        "LS Ann Return": ann_ret,
                        "LS Ann Vol": ann_vol,
                        "LS Sharpe": sharpe,
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "27_factor_by_regime.csv"), index=False)
    print("    [DONE] 27_factor_by_regime.csv")

    # Plot: grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    factor_names = df["Factor"].unique()
    regime_names = df["Regime"].unique()
    x = np.arange(len(factor_names))
    width = 0.25

    for i, regime in enumerate(regime_names):
        mask = df["Regime"] == regime
        vals = []
        for f in factor_names:
            row = df[(df["Factor"] == f) & (df["Regime"] == regime)]
            vals.append(row["LS Sharpe"].values[0] if len(row) > 0 else 0)
        color_key = regime.lower().replace(" ", "_")
        ax.bar(
            x + i * width,
            vals,
            width,
            label=regime,
            color=REGIME_COLORS.get(color_key, "#999999"),
            alpha=0.85,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(factor_names)
    ax.set_ylabel("Long-Short Sharpe Ratio")
    ax.set_title("Factor L/S Sharpe by HMM Regime")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "27_factor_by_regime.png"))
    plt.close(fig)
    print("    [DONE] 27_factor_by_regime.png")
    print(df.to_string(index=False))


#  main 


def main(data=None, output_dir=None):
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    if data is None:
        print("Loading data ...")
        data = load_all_data()

    print("Preparing regime labels ...")
    regime_file = os.path.join(os.path.dirname(__file__), "..", "regimes_output.csv")
    
    if os.path.exists(regime_file):
        regimes = pd.read_csv(regime_file, parse_dates=["Date"])
        regimes.set_index("Date", inplace=True)
    else:
        print("  Signal file not found. Fitting HMM on Market Index returns on-the-fly...")
        idx = _build_market_index(data)
        log_ret = np.log(idx / idx.shift(1)).dropna()
        
        # Fit HMM on Market Returns
        X = log_ret.values.reshape(-1, 1)
        model = GaussianHMM(n_components=config.N_STATES, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        
        # Generate labels
        states = get_ordered_states(model, log_ret)
        regimes = pd.DataFrame(index=log_ret.index)
        regimes["regime_label"] = states
        # Map 0/1 to descriptive labels if needed, or just use strings
        label_map = {0: "trending", 1: "mean_reverting"} if config.N_STATES == 2 else {0: "trending", 1: "random_walk", 2: "mean_reverting"}
        regimes["regime_label"] = regimes["regime_label"].map(label_map).fillna("random_walk")

    plot_regime_context(regimes, data)
    plot_transition_matrix(regimes)
    plot_factor_by_regime(regimes, data)

    print("\n[DONE] Part 7 complete - regime characterisation saved to output/eda/")


if __name__ == "__main__":
    main()
