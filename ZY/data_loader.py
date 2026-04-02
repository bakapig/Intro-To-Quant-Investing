"""
data_loader.py
--------------
Loads, cleans, and filters the Chinese equity data from data_cn/.
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Raw loading
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV and strip whitespace from column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def load_all_data(data_dir: str = "data_cn") -> dict:
    """
    Load every CSV in data_cn/ into a dict keyed by filename stem.
    Price / volume files get Date parsed as datetime index.
    """
    # Only include files that have a Date column as first column
    FILES = [
        "close", "adjusted", "open", "high", "low",
        "dv", "mktcap", "p2b", "recm",
    ]
    META_FILES = ["tickers", "dateline", "in_univ", "univ_h", "eps"]

    data = {}

    for name in FILES:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df.columns = df.columns.astype(str).str.strip()
        data[name] = df

    for name in META_FILES:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        data[name] = load_csv(path)

    return data


# ---------------------------------------------------------------------------
# 2. Universe helpers
# ---------------------------------------------------------------------------

def get_universe_tickers(data: dict, year: int) -> list:
    """
    Return list of tickers that are in the investable universe for *year*
    using univ_h.csv (has a 'year' column + one column per ticker).
    """
    univ = data["univ_h"]
    row = univ[univ["year"] == year]
    if row.empty:
        return []
    ticker_cols = [c for c in univ.columns if c != "year"]
    mask = row[ticker_cols].iloc[0] == 1
    return mask[mask].index.tolist()


def get_always_in_universe(data: dict, start_year: int = 2006, end_year: int = 2025) -> list:
    """Return tickers that are in the universe for EVERY year in the range."""
    univ = data["univ_h"]
    ticker_cols = [c for c in univ.columns if c != "year"]
    mask = univ[(univ["year"] >= start_year) & (univ["year"] <= end_year)]
    always_in = (mask[ticker_cols] == 1).all(axis=0)
    return always_in[always_in].index.tolist()


# ---------------------------------------------------------------------------
# 3. Filtering
# ---------------------------------------------------------------------------

def filter_stocks(
    data: dict,
    min_history_days: int = 252 * 5,
    min_median_dv: float = 1e7,
    top_n: int = 30,
) -> list:
    """
    Select a subset of stocks that:
      1. Have at least `min_history_days` of non-NaN adjusted prices
      2. Have median daily dollar volume >= `min_median_dv`
      3. Are in the universe for most years
    Returns the top_n tickers sorted by median dollar volume (descending).
    """
    adj = data["adjusted"]
    dv = data["dv"]

    # Count non-NaN days per ticker
    valid_days = adj.notna().sum()
    candidates = valid_days[valid_days >= min_history_days].index.tolist()

    # Median dollar volume filter
    med_dv = dv[candidates].median()
    candidates = med_dv[med_dv >= min_median_dv].sort_values(ascending=False).index.tolist()

    # Intersect with universe tickers (present in at least 15 of 20 years)
    univ = data["univ_h"]
    ticker_cols = [c for c in univ.columns if c != "year"]
    years_in = (univ[ticker_cols] == 1).sum()
    stable_tickers = years_in[years_in >= 15].index.tolist()
    candidates = [t for t in candidates if t in stable_tickers]

    return candidates[:top_n]


# ---------------------------------------------------------------------------
# 4. Prepare clean price panel
# ---------------------------------------------------------------------------

def prepare_prices(data: dict, tickers: list) -> dict:
    """
    Return a dict with cleaned DataFrames for the selected tickers:
      - adjusted, close, open, high, low, dv
    Small gaps (<=5 days) within each stock's alive period are forward-filled.
    """
    result = {}
    for key in ["adjusted", "close", "open", "high", "low", "dv"]:
        df = data[key][tickers].copy()
        # Forward-fill small gaps (up to 5 days)
        df = df.ffill(limit=5)
        result[key] = df
    return result


# ---------------------------------------------------------------------------
# 5. Build an equal-weight index (for index-level HMM)
# ---------------------------------------------------------------------------

def build_index(adj: pd.DataFrame) -> pd.Series:
    """
    Build a simple equal-weight price index from adjusted prices.
    Uses daily returns averaged across all non-NaN stocks, then cumulated.
    """
    returns = adj.pct_change(fill_method=None)
    avg_ret = returns.mean(axis=1)  # equal-weight average return
    index = (1 + avg_ret).cumprod()
    index.iloc[0] = 1.0  # start at 1
    index.name = "EW_Index"
    return index


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_all_data("data_cn")
    print("Loaded keys:", list(data.keys()))
    print("Adjusted shape:", data["adjusted"].shape)
    print("Date range:", data["adjusted"].index[0], "->", data["adjusted"].index[-1])

    tickers = filter_stocks(data, top_n=30)
    print(f"\nSelected {len(tickers)} tickers: {tickers}")

    prices = prepare_prices(data, tickers)
    print("Adjusted (filtered) shape:", prices["adjusted"].shape)

    idx = build_index(prices["adjusted"])
    print(f"\nEW Index: start={idx.iloc[1]:.4f}, end={idx.iloc[-1]:.4f}")
