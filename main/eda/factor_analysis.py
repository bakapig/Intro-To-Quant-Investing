"""
eda/factor_analysis.py
----------------------
Part 2 - Cross-sectional factor analysis

Factors analysed (monthly rebalance, quintile sorts):
  1. Value        - price-to-book  (p2b.csv),  low P/B = cheap
  2. Size         - market cap     (mktcap.csv), small = Q1
  3. Momentum     - trailing 12-1 month returns
  4. Liquidity    - dollar volume  (dv.csv),    low DV = illiquid
  5. Analyst Rec  - consensus rec  (recm.csv)
  6. Earnings Yld - EPS / price    (eps.csv / close.csv)

For each factor we:
   At each month-end, rank stocks into 5 equal quintiles on the signal.
   Compute the equal-weighted return of each quintile over the *next* month.
   Plot cumulative quintile returns and the long-short (Q1 - Q5) spread.
   Report annualised return, vol, and Sharpe for each quintile + spread.

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


#  helpers 


def _monthly_returns(adj: pd.DataFrame) -> pd.DataFrame:
    """Month-end to month-end returns for every stock."""
    month_end = adj.resample("ME").last()
    return month_end.pct_change()


def _get_universe_mask(data: dict, adj: pd.DataFrame) -> pd.DataFrame:
    """Binary mask: 1 if a stock is in the investable universe that year."""
    univ = data["univ_h"]
    ticker_cols = [c for c in univ.columns if c != "year"]
    mask = pd.DataFrame(0, index=adj.index, columns=adj.columns)
    for _, row in univ.iterrows():
        yr = int(row["year"])
        in_tickers = [t for t in ticker_cols if row[t] == 1 and t in adj.columns]
        mask.loc[mask.index.year == yr, in_tickers] = 1
    return mask


def _monthly_signal(daily_signal: pd.DataFrame) -> pd.DataFrame:
    """Take the last available value each month."""
    return daily_signal.resample("ME").last()


def _quintile_sort(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    univ_mask: pd.DataFrame,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    For each month, rank stocks into 5 quintiles on *signal*,
    return a DataFrame of equal-weighted quintile returns (columns Q1..Q5).
    ascending=True  -> Q1 = lowest signal value
    ascending=False -> Q1 = highest signal value
    """
    # Align indices
    common_dates = signal.index.intersection(fwd_ret.index).intersection(
        univ_mask.index
    )
    signal = signal.loc[common_dates]
    fwd_ret = fwd_ret.loc[common_dates]
    univ_mask = univ_mask.loc[common_dates]

    records = []
    for dt in common_dates:
        sig = signal.loc[dt].dropna()
        ret = fwd_ret.loc[dt].dropna()
        umask = univ_mask.loc[dt]
        # Keep only universe stocks with both signal and forward return
        valid = sig.index.intersection(ret.index)
        valid = valid[umask.reindex(valid).fillna(0).astype(bool)]
        if len(valid) < 50:
            continue
        sig_v = sig[valid]
        ret_v = ret[valid]
        if ascending:
            ranks = sig_v.rank(method="first")
        else:
            ranks = (-sig_v).rank(method="first")
        n = len(ranks)
        qcut = pd.qcut(ranks, 5, labels=False) + 1  # 1..5
        row = {}
        for q in range(1, 6):
            q_mask = qcut == q
            row[f"Q{q}"] = ret_v[q_mask].mean()
        row["LS"] = row["Q1"] - row["Q5"]
        row["date"] = dt
        records.append(row)

    df = pd.DataFrame(records).set_index("date")
    return df


def _plot_factor(
    quintile_ret: pd.DataFrame,
    factor_name: str,
    q1_label: str,
    q5_label: str,
    filename: str,
) -> None:
    """Plot cumulative quintile returns and long-short spread."""
    cum = (1 + quintile_ret).cumprod()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Quintile cumulative returns
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
    for i, q in enumerate([f"Q{j}" for j in range(1, 6)]):
        label = q
        if q == "Q1":
            label += f" ({q1_label})"
        elif q == "Q5":
            label += f" ({q5_label})"
        ax.plot(cum.index, cum[q], lw=1.2, color=colors[i], label=label)
    ax.set_title(f"{factor_name} - Quintile cumulative returns")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Long-short spread
    ax = axes[1]
    cum_ls = (1 + quintile_ret["LS"]).cumprod()
    ax.plot(cum_ls.index, cum_ls.values, lw=1.3, color="navy")
    ax.axhline(1, color="grey", ls="--", lw=0.5)
    ax.set_title(f"{factor_name} - Long Q1 / Short Q5")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  [Done] {filename}")


def _print_stats(quintile_ret: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Annualised return, vol, Sharpe for each quintile + LS."""
    ann = quintile_ret.mean() * 12
    vol = quintile_ret.std() * np.sqrt(12)
    sharpe = ann / vol
    stats = pd.DataFrame({"Ann Return": ann, "Ann Vol": vol, "Sharpe": sharpe})
    stats.index.name = "Quintile"
    print(f"\n  {factor_name} - Quintile statistics (annualised)")
    print(stats.round(4).to_string())
    return stats


#  Factor signal builders 


def _build_momentum_signal(adj: pd.DataFrame) -> pd.DataFrame:
    """Trailing 12-month return, skipping the most recent month (12-1)."""
    monthly_price = adj.resample("ME").last()
    # 12-month return = price[t-1] / price[t-12] - 1  (skip most recent month)
    ret_12_1 = monthly_price.shift(1) / monthly_price.shift(12) - 1
    return ret_12_1


def _build_earnings_yield(data: dict) -> pd.DataFrame:
    """
    Build daily earnings-yield = EPS / price.
    eps.csv stores pairs of rows per ticker:
      row 0: report dates (as YYYYMMDD floats)
      row 1: EPS values
    We forward-fill EPS to daily dates using the close price index.
    """
    eps_raw = data["eps"]
    close = data["close"]

    # Parse eps.csv: alternating ticker-date / EPS rows
    first_col = eps_raw.columns[0]  # first ticker
    date_cols = eps_raw.columns[1:]  # report-date columns

    daily_eps = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    # First ticker is in the column header
    # Then alternating rows: even index = (ticker, report_dates), odd index = (EPS, values)
    # But first row (index 0) is 'EPS' row for the first ticker (header ticker)
    # Let's handle the header ticker separately
    header_ticker = first_col.strip()
    eps_vals = eps_raw.iloc[0, 1:].values.astype(float)  # EPS for header ticker
    # Report dates for header ticker need to come from the column names
    report_dates = pd.to_datetime(date_cols, format="%Y%m%d", errors="coerce")
    valid = ~pd.isna(report_dates) & ~np.isnan(eps_vals)
    if valid.any() and header_ticker in close.columns:
        ts = pd.Series(eps_vals[valid], index=report_dates[valid]).sort_index()
        ts = ts[~ts.index.duplicated(keep="last")]
        ts = ts.reindex(close.index, method="ffill")
        daily_eps[header_ticker] = ts

    # Remaining tickers: rows come in pairs
    i = 1
    while i + 1 < len(eps_raw):
        ticker = str(eps_raw.iloc[i, 0]).strip()
        if ticker == "EPS" or ticker not in close.columns:
            i += 2
            continue
        # Row i has report dates as floats, Row i+1 has EPS values
        raw_dates = eps_raw.iloc[i, 1:].values.astype(float)
        raw_eps = eps_raw.iloc[i + 1, 1:].values.astype(float)
        report_dates_t = pd.to_datetime(
            [str(int(d)) for d in raw_dates if not np.isnan(d)],
            format="%Y%m%d",
            errors="coerce",
        )
        eps_v = raw_eps[~np.isnan(raw_dates)]
        valid = ~pd.isna(report_dates_t) & ~np.isnan(eps_v)
        if valid.any():
            ts = pd.Series(eps_v[valid], index=report_dates_t[valid]).sort_index()
            ts = ts[~ts.index.duplicated(keep="last")]
            ts = ts.reindex(close.index, method="ffill")
            daily_eps[ticker] = ts
        i += 2

    ey = daily_eps / close
    return ey


#  main 


def main(data=None, output_dir=None):
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    if data is None:
        print("Loading data ...")
        data = load_all_data()

    adj = data["adjusted"]
    close = data["close"]
    mktcap = data["mktcap"]
    p2b = data["p2b"]
    dv = data["dv"]
    recm = data["recm"]

    print("Computing monthly returns & universe mask ...")
    monthly_ret = _monthly_returns(adj)
    fwd_ret = monthly_ret.shift(-1)  # next-month return
    univ_mask_daily = _get_universe_mask(data, adj)
    univ_mask = univ_mask_daily.resample("ME").last()

    # Drop last month (no forward return)
    fwd_ret = fwd_ret.iloc[:-1]

    all_stats = {}

    #  1. Value (P/B) 
    print("\n1. Value factor (P/B) ...")
    sig = _monthly_signal(p2b)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=True)
    _plot_factor(
        qr,
        "Value (P/B)",
        "Low P/B (cheap)",
        "High P/B (expensive)",
        "07_factor_value_pb.png",
    )
    all_stats["Value (P/B)"] = _print_stats(qr, "Value (P/B)")

    #  2. Size (Market Cap) 
    print("\n2. Size factor (Market Cap) ...")
    sig = _monthly_signal(mktcap)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=True)
    _plot_factor(qr, "Size (Mkt Cap)", "Small cap", "Large cap", "08_factor_size.png")
    all_stats["Size"] = _print_stats(qr, "Size (Mkt Cap)")

    #  3. Momentum (12-1) 
    print("\n3. Momentum factor (12-1 month) ...")
    sig = _build_momentum_signal(adj)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=False)
    _plot_factor(qr, "Momentum (12-1)", "Winners", "Losers", "09_factor_momentum.png")
    all_stats["Momentum"] = _print_stats(qr, "Momentum (12-1)")

    #  4. Liquidity (Dollar Volume) 
    print("\n4. Liquidity factor (Dollar Volume) ...")
    sig = _monthly_signal(dv)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=True)
    _plot_factor(
        qr,
        "Liquidity (DV)",
        "Low liquidity",
        "High liquidity",
        "10_factor_liquidity.png",
    )
    all_stats["Liquidity"] = _print_stats(qr, "Liquidity (DV)")

    #  5. Analyst Recommendation 
    print("\n5. Analyst recommendation ...")
    sig = _monthly_signal(recm)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=False)
    _plot_factor(
        qr, "Analyst Rec", "Highest rated", "Lowest rated", "11_factor_analyst.png"
    )
    all_stats["Analyst Rec"] = _print_stats(qr, "Analyst Rec")

    #  6. Earnings Yield (EPS / Price) 
    print("\n6. Earnings yield (EPS / price) ...")
    ey = _build_earnings_yield(data)
    sig = _monthly_signal(ey)
    qr = _quintile_sort(sig, fwd_ret, univ_mask, ascending=False)
    _plot_factor(
        qr,
        "Earnings Yield",
        "High EY (cheap)",
        "Low EY (expensive)",
        "12_factor_earnings_yield.png",
    )
    all_stats["Earnings Yield"] = _print_stats(qr, "Earnings Yield")

    #  Summary table 
    summary_rows = []
    for name, st in all_stats.items():
        summary_rows.append(
            {
                "Factor": name,
                "Q1 Ann Ret": st.loc["Q1", "Ann Return"],
                "Q5 Ann Ret": st.loc["Q5", "Ann Return"],
                "LS Ann Ret": st.loc["LS", "Ann Return"],
                "LS Sharpe": st.loc["LS", "Sharpe"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("Factor")
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "13_factor_summary.csv"), float_format="%.4f"
    )
    print("\n  [DONE] 13_factor_summary.csv")
    print("\n  Factor summary (long-short = Q1 - Q5):")
    print(summary_df.round(4).to_string())

    print(f"\nDone. Outputs saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
