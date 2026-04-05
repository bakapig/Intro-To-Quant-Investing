"""
data_loader.py
--------------
Loads, cleans, and filters the Chinese equity data from data_cn/.
Adapted from ZY's modular loader + JC's cleaning logic.
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
    FILES = [
        "close",
        "adjusted",
        "open",
        "high",
        "low",
        "dv",
        "mktcap",
        "p2b",
        "recm",
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
    """Return list of tickers in the investable universe for a given year."""
    univ = data["univ_h"]
    row = univ[univ["year"] == year]
    if row.empty:
        return []
    ticker_cols = [c for c in univ.columns if c != "year"]
    mask = row[ticker_cols].iloc[0] == 1
    return mask[mask].index.tolist()


def get_always_in_universe(
    data: dict, start_year: int = 2006, end_year: int = 2025
) -> list:
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
    Select top_n stocks by median dollar volume that meet minimum history
    and universe stability requirements.
    """
    adj = data["adjusted"]
    dv = data["dv"]

    valid_days = adj.notna().sum()
    candidates = valid_days[valid_days >= min_history_days].index.tolist()

    med_dv = dv[candidates].median()
    candidates = (
        med_dv[med_dv >= min_median_dv].sort_values(ascending=False).index.tolist()
    )

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
    Return a dict with cleaned DataFrames for the selected tickers.
    Small gaps (<=5 days) are forward-filled.
    """
    result = {}
    for key in ["adjusted", "close", "open", "high", "low", "dv"]:
        df = data[key][tickers].copy()
        df = df.ffill(limit=5)
        result[key] = df
    return result


# ---------------------------------------------------------------------------
# 5. Build an equal-weight index
# ---------------------------------------------------------------------------


def build_index(adj: pd.DataFrame) -> pd.Series:
    """Build a simple equal-weight price index from adjusted prices."""
    returns = adj.pct_change(fill_method=None)
    avg_ret = returns.mean(axis=1)
    index = (1 + avg_ret).cumprod()
    index.iloc[0] = 1.0
    index.name = "EW_Index"
    return index


# ---------------------------------------------------------------------------
# 6. Market-Cap Weighted Index (from JC's EDA notebook)
# ---------------------------------------------------------------------------


def build_cap_weighted_market_index(
    df_prices, df_mcap, base=100.0, vol_window=20, annualization=252
):
    """
    Build a market-cap-weighted return series and index level.

    Returns
    -------
    out : DataFrame with columns market_cap_weighted_return, market_cap_weighted_index, rolling_20d_ann_vol
    weights : DataFrame of daily lagged cap weights
    asset_ret : DataFrame of constituent daily simple returns
    """
    prices = df_prices.copy()
    mcaps = df_mcap.copy()

    prices.index = pd.to_datetime(prices.index)
    mcaps.index = pd.to_datetime(mcaps.index)

    prices = prices.sort_index()
    mcaps = mcaps.sort_index()

    prices = prices[~prices.index.duplicated(keep="last")]
    mcaps = mcaps[~mcaps.index.duplicated(keep="last")]

    prices = prices.apply(pd.to_numeric, errors="coerce")
    mcaps = mcaps.apply(pd.to_numeric, errors="coerce")

    common_dates = prices.index.intersection(mcaps.index)
    common_tickers = prices.columns.intersection(mcaps.columns)

    prices = prices.loc[common_dates, common_tickers]
    mcaps = mcaps.loc[common_dates, common_tickers]

    mcaps = mcaps.where(mcaps > 0)

    asset_ret = prices.pct_change(fill_method=None)
    lagged_mcaps = mcaps.shift(1)

    valid = asset_ret.notna() & lagged_mcaps.notna()
    effective_caps = lagged_mcaps.where(valid)

    weights = effective_caps.div(effective_caps.sum(axis=1), axis=0)

    market_ret = (weights * asset_ret).sum(axis=1, min_count=1)
    market_index = base * (1 + market_ret.fillna(0.0)).cumprod()

    rolling_vol = market_ret.rolling(
        vol_window, min_periods=vol_window
    ).std() * np.sqrt(annualization)

    out = pd.DataFrame(
        {
            "market_cap_weighted_return": market_ret,
            "market_cap_weighted_index": market_index,
            "rolling_20d_ann_vol": rolling_vol,
        }
    )

    return out, weights, asset_ret


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
