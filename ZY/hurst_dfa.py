"""
hurst_dfa.py
------------
Detrended Fluctuation Analysis (DFA) to compute the Hurst exponent.

Interpretation:
  H > 0.5  →  persistent / trending
  H < 0.5  →  anti-persistent / mean-reverting
  H ≈ 0.5  →  random walk
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _dfa_fluctuation(profile: np.ndarray, window_size: int) -> float:
    """
    Compute the RMS fluctuation F(n) for a given window size n.
    1. Split profile into non-overlapping windows of size n.
    2. Fit a linear trend to each window.
    3. Compute root-mean-square of detrended residuals.
    """
    n = window_size
    N = len(profile)
    num_windows = N // n

    if num_windows == 0:
        return np.nan

    fluctuations = np.zeros(num_windows)
    x = np.arange(n)

    for i in range(num_windows):
        segment = profile[i * n : (i + 1) * n]
        # Linear fit (polyfit degree 1)
        coeffs = np.polyfit(x, segment, 1)
        trend = np.polyval(coeffs, x)
        residuals = segment - trend
        fluctuations[i] = np.sqrt(np.mean(residuals ** 2))

    return np.sqrt(np.mean(fluctuations ** 2))


def compute_dfa(
    series: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Hurst exponent via DFA.

    Parameters
    ----------
    series : 1-D array of prices or log-prices
    min_window : smallest window size
    max_window : largest window size (default: len/4)
    num_windows : number of log-spaced window sizes

    Returns
    -------
    hurst : float, the DFA exponent (α)
    window_sizes : array of window sizes used
    fluctuations : array of F(n) values
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]

    if len(series) < 2 * min_window:
        return np.nan, np.array([]), np.array([])

    # Step 1: compute log returns
    log_ret = np.diff(np.log(series))

    # Step 2: cumulative sum of deviations from mean (the "profile")
    profile = np.cumsum(log_ret - np.mean(log_ret))

    # Step 3: define window sizes
    if max_window is None:
        max_window = len(profile) // 4

    if max_window <= min_window:
        max_window = len(profile) // 2

    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=num_windows,
        ).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= 4]  # need at least 4 for polyfit

    # Step 4: compute fluctuations
    fluctuations = np.array([_dfa_fluctuation(profile, w) for w in window_sizes])

    # Remove NaN / zero
    valid = (fluctuations > 0) & ~np.isnan(fluctuations)
    window_sizes = window_sizes[valid]
    fluctuations = fluctuations[valid]

    if len(window_sizes) < 3:
        return np.nan, window_sizes, fluctuations

    # Step 5: log-log regression to get slope = Hurst exponent
    log_n = np.log(window_sizes)
    log_f = np.log(fluctuations)
    coeffs = np.polyfit(log_n, log_f, 1)
    hurst = coeffs[0]

    return hurst, window_sizes, fluctuations


def rolling_hurst(
    prices: pd.Series,
    lookback: int = 252,
    step: int = 1,
    min_window: int = 10,
    num_windows: int = 15,
) -> pd.Series:
    """
    Compute a rolling Hurst exponent over a price series.

    Parameters
    ----------
    prices : pd.Series with DatetimeIndex
    lookback : number of observations in the rolling window
    step : step size (1 = compute at every bar)
    min_window : min DFA window
    num_windows : number of DFA windows

    Returns
    -------
    pd.Series of Hurst exponents (NaN where insufficient data)
    """
    n = len(prices)
    hurst_vals = pd.Series(np.nan, index=prices.index)

    for i in range(lookback, n, step):
        window = prices.iloc[i - lookback : i].dropna()
        if len(window) < lookback * 0.8:  # require 80% non-NaN
            continue
        h, _, _ = compute_dfa(
            window.values,
            min_window=min_window,
            num_windows=num_windows,
        )
        hurst_vals.iloc[i] = h

    # Forward-fill to cover steps > 1
    if step > 1:
        hurst_vals = hurst_vals.ffill()

    return hurst_vals


# ---------------------------------------------------------------------------
# Validation on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- White noise (should give H ≈ 0.5) ---
    wn_prices = np.cumsum(np.random.randn(2000)) + 100
    wn_prices = np.abs(wn_prices)  # keep positive
    h_wn, _, _ = compute_dfa(wn_prices)
    print(f"White noise random walk  -> H = {h_wn:.4f}  (expect ~0.50)")

    # --- Trending series (should give H > 0.5) ---
    trend = np.cumsum(np.random.randn(2000) + 0.05) + 100
    trend = np.abs(trend)
    h_trend, _, _ = compute_dfa(trend)
    print(f"Trending series          -> H = {h_trend:.4f}  (expect > 0.50)")

    # --- Mean-reverting (Ornstein–Uhlenbeck approximation) ---
    mr = [100.0]
    theta = 0.1
    for _ in range(1999):
        mr.append(mr[-1] + theta * (100 - mr[-1]) + np.random.randn() * 0.5)
    mr = np.array(mr)
    h_mr, _, _ = compute_dfa(mr)
    print(f"Mean-reverting (OU)      -> H = {h_mr:.4f}  (expect < 0.50)")
