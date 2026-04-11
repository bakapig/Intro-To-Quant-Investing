"""
hmm_strategy.py
---------------
HMM + Hurst strategy logic.

Core components:
  - HMM regime fitting (2 states) with BIC selection
  - Rolling Hurst exponent calculation
  - Signal generation: regime + Hurst + momentum
  - Sector-neutral transformation
  - Market-cap weighted portfolio construction
  - Backtest evaluation
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1. HMM Regime Fitting
# ---------------------------------------------------------------------------


def _fit_hmm_model(X, n_states):
    """Fits a Gaussian HMM and returns the model and log-likelihood. Set random_state for reproducibility."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
        tol=1e-4,
        min_covar=1e-6,
    )
    model.fit(X)
    log_likelihood = model.score(X)
    return model, log_likelihood


def fit_hmm_regimes(returns, n_states):
    """
    Fits a Gaussian HMM with the given number of states.
    Returns BIC DataFrame and the fitted model.
    """
    X = (returns.values * 100).reshape(-1, 1)
    n_samples = len(X)

    bic_records = []

    try:
        model, log_likelihood = _fit_hmm_model(X, n_states)

        k = (n_states**2 - n_states) + n_states + n_states + (n_states - 1)
        bic = k * np.log(n_samples) - 2 * log_likelihood

        bic_records.append(
            {"n_states": n_states, "log_likelihood": log_likelihood, "bic": bic}
        )

    except Exception:
        pass

    bic_df = pd.DataFrame(bic_records)

    if bic_df.empty:
        return None, None

    return bic_df, model


# ---------------------------------------------------------------------------
# 2. State Ordering (by volatility)
# ---------------------------------------------------------------------------


def get_ordered_states(model, returns):
    """Maps HMM states to a consistent order based on volatility (low to high)."""
    X = (returns.values * 100).reshape(-1, 1)
    hidden_states = model.predict(X)

    n_states = model.n_components

    variances = np.array([np.diag(model.covars_[i])[0] for i in range(n_states)])
    sorted_idx = np.argsort(variances)

    state_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
    ordered_states = pd.Series(hidden_states).map(state_map).values

    return ordered_states


# ---------------------------------------------------------------------------
# 3. Rolling Hurst Exponent
# ---------------------------------------------------------------------------


@njit
def _fast_linear_fit_residuals(x, y):
    """Blazing fast 1D linear regression to find residuals (bypasses slow np.polyfit)."""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0:
        return np.zeros(n)
        
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    trend = slope * x + intercept
    return y - trend

@njit
def _mfdfa_fluctuation_numba(profile, window_size, q=2.0):
    """Computes MFDFA fluctuation F_q(s) optimized with Numba."""
    n = window_size
    N = len(profile)
    num_windows = N // n
    if num_windows == 0:
        return np.nan
        
    variances = np.zeros(num_windows)
    x = np.arange(float(n))
    
    for i in range(num_windows):
        segment = profile[i * n : (i + 1) * n]
        residuals = _fast_linear_fit_residuals(x, segment)
        variances[i] = np.mean(residuals**2)

    # Filter out zero/negative variances safely in numba
    valid_count = 0
    for v in variances:
        if v > 0: valid_count += 1
        
    if valid_count == 0:
        return np.nan

    valid_vars = np.zeros(valid_count)
    idx = 0
    for v in variances:
        if v > 0:
            valid_vars[idx] = v
            idx += 1

    if q == 0.0:
        return np.exp(0.5 * np.mean(np.log(valid_vars)))
    else:
        return np.power(np.mean(valid_vars ** (q / 2.0)), 1.0 / q)


def compute_mfdfa_hurst(log_returns, q=2, min_window=10, num_windows=15):
    """Calculates Generalized Hurst Exponent h(q) using MFDFA."""
    profile = np.cumsum(log_returns - np.mean(log_returns))
    max_window = len(profile) // 4
    if max_window <= min_window:
        max_window = len(profile) // 2

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_windows).astype(
            int
        )
    )
    window_sizes = window_sizes[window_sizes >= 4]

    # CALLING THE NEW NUMBA FUNCTION HERE
    fluctuations = np.array([_mfdfa_fluctuation_numba(profile, w, float(q)) for w in window_sizes])
    valid = (fluctuations > 0) & ~np.isnan(fluctuations)

    if np.sum(valid) < 3:
        return np.nan

    log_n = np.log(window_sizes[valid])
    log_f = np.log(fluctuations[valid])
    coeffs = np.polyfit(log_n, log_f, 1)
    return coeffs[0]


def calculate_rolling_hurst(log_return_series, window=100, q=2):
    """Calculates rolling Generalized Hurst Exponent using MFDFA analysis."""
    hurst_values = np.full(len(log_return_series), np.nan)

    for i in range(window, len(log_return_series)):
        window_data = log_return_series.iloc[i - window : i].values
        try:
            H = compute_mfdfa_hurst(window_data, q=q)
            hurst_values[i] = H
        except Exception:
            hurst_values[i] = np.nan

    return hurst_values


# ---------------------------------------------------------------------------
# 4. Signal Generation
# ---------------------------------------------------------------------------


def _apply_strategy_rules(df, model, n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days):
    """Apply HMM regime + Hurst + Momentum signal integration."""
    df["Regime"] = get_ordered_states(model, df["Log_Return"])

    # Micro: Hurst Exponent
    df["Hurst"] = calculate_rolling_hurst(df["Log_Return"], window=hurst_window, q=2)

    # Tactical: Momentum
    df["Momentum"] = df["Close"].pct_change(periods=momentum_periods)

    # Signal Integration Logic (VECTORIZED)
    df["Signal"] = 0

    is_low_vol = df["Regime"] == 0
    is_med_vol = df["Regime"] == 1
    is_high_vol = df["Regime"] == (n_states - 1)

    # Thresholds
    high_hurst, low_hurst = df["Hurst"] > hurst_upper, df["Hurst"] < hurst_lower
    pos_mom, neg_mom = df["Momentum"] > 0, df["Momentum"] < 0

    # RULES - State 0 (Always Active: Trend + Mean Reversion)
    df.loc[is_low_vol & high_hurst & pos_mom, "Signal"] = 1
    df.loc[is_low_vol & low_hurst, "Signal"] = -np.sign(df["Momentum"])

    # RULES - State 1 (Tactical: Only if 3 states exist)
    if n_states == 3:
        # In medium vol, we only take trend-following signals with momentum confirmation
        df.loc[is_med_vol & high_hurst & pos_mom, "Signal"] = 1

    # RULES - Highest State (SAFE HARBOR: Always Cash)
    df.loc[is_high_vol, "Signal"] = 0 

    # Clean up: NAs in Hurst/Momentum don't trigger signals
    df.loc[df["Hurst"].isna() | df["Momentum"].isna(), "Signal"] = 0

    # --- Signal Holding Period Filter (VECTORIZED) ---
    # Only allow a signal change if the new signal persists for `holding_days` consecutive days.
    rolling_min = df["Signal"].rolling(window=holding_days).min()
    rolling_max = df["Signal"].rolling(window=holding_days).max()
    # 2. If min == max, the signal has been identical for 'holding_days' straight
    is_stable = rolling_min == rolling_max
    
    # 3. Keep the signal where stable, otherwise forward-fill the last stable signal
    df["Signal"] = df["Signal"].where(is_stable).ffill().fillna(df["Signal"].iloc[0])

    # Shift signal to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Strategy_Return"] = df["Position"] * df["Log_Return"]

    return df

# ---------------------------------------------------------------------------
# 4.5 Risk Forecasting (GARCH 1-Day VaR)
# ---------------------------------------------------------------------------


def calculate_var_forecast(log_returns):
    """Calculates 1-day 99% VaR using EWMA volatility."""
    # Center returns
    returns = log_returns.dropna()
    mean_return = returns.mean()
    
    # Calculate EWMA variance (lambda=0.94 is RiskMetrics standard for daily data)
    ewma_var = (returns - mean_return)**2
    ewma_var = ewma_var.ewm(alpha=1-0.94, adjust=False).mean()
    
    cond_vol = np.sqrt(ewma_var).shift(-1)
    
    var_95 = 1.645 * cond_vol
    return var_95.reindex(log_returns.index).ffill().fillna(0)

# ---------------------------------------------------------------------------
# 5. Single Asset Processing
# ---------------------------------------------------------------------------


def process_single_asset(price_series, log_return_series, n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days):
    """Applies the strategy logic to a single asset."""
    valid_price = price_series.dropna()
    valid_log_return = log_return_series.dropna()

    if len(valid_log_return) < 100 or len(valid_price) < 100:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame({"Log_Return": valid_log_return, "Close": valid_price})
    df.fillna(0, inplace=True)

    # NOTE: JC's original passes the full DataFrame (Log_Return + Close) to
    # fit_hmm_regimes.  Inside, .values.reshape(-1,1) interleaves both columns
    # into a single 1-D sequence fed to the HMM.  This is what produced JC's
    # published results, so we replicate it exactly here.
    bic_df, model = fit_hmm_regimes(df["Log_Return"], n_states=n_states)

    if model is None:
        return pd.DataFrame(), pd.DataFrame()

    df = _apply_strategy_rules(df, model, n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days)
    df["VaR"] = calculate_var_forecast(df["Log_Return"])

    return df[["Log_Return", "Strategy_Return", "Signal", "VaR"]], bic_df


# ---------------------------------------------------------------------------
# 6. Universe Processing (Sector-Neutral)
# ---------------------------------------------------------------------------


def process_ticker(ticker, price_series, log_return_series, n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days):
    """Helper function to process a single ticker for parallel execution."""
    asset_data, bic_df = process_single_asset(price_series, log_return_series, n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days)
    return ticker, asset_data, bic_df


def process_universe(df_prices, n_states, tickers_csv_path, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days):
    """
    Processes all assets with sector-neutral transformation.

    Parameters
    ----------
    df_prices : DataFrame of adjusted prices
    n_states : number of HMM states
    tickers_csv_path : path to tickers.csv for sector info

    Returns
    -------
    strat_returns, bh_returns, signals, bic_all : DataFrames
    """
    print(f"Processing assets for states={n_states}, H={hurst_window}, M={momentum_periods}, U={hurst_upper}, L={hurst_lower}, HD={holding_days}...")

    tickers = pd.read_csv(tickers_csv_path, header=None)
    tickers.columns = ["ticker", "gics"]
    tickers["sector"] = tickers["gics"].apply(
        lambda x: str(x)[:2] if x is not None else None
    )
    tickers["industry"] = tickers["gics"].apply(
        lambda x: str(x)[:6] if x is not None else None
    )

    gics_map = tickers.set_index("ticker")["sector"]

    log_return = np.log(df_prices / df_prices.shift(1))
    log_return_T = log_return.T
    log_return_T["sector"] = gics_map

    log_return_neutral_T = log_return_T.groupby("sector").transform(
        lambda x: x - x.mean()
    )

    df_log_return_neutral = log_return_neutral_T.T

    all_strat_ret, all_bh_ret, all_signals, all_var = {}, {}, {}, {}
    all_bic = []

    print(f"  Dispatching {len(df_prices.columns)} assets to CPU worker pool...")
    results = Parallel(n_jobs=-1)(
        delayed(process_ticker)(
            ticker, df_prices[ticker], df_log_return_neutral[ticker], n_states, hurst_window, momentum_periods, hurst_upper, hurst_lower, holding_days
        )
        for ticker in tqdm(df_prices.columns, desc="Dispatching Tasks")
    )

    for ticker, asset_data, bic_df in tqdm(results, desc="Aggregating Results"):
        if not asset_data.empty:
            all_strat_ret[ticker] = asset_data["Strategy_Return"]
            all_bh_ret[ticker] = asset_data["Log_Return"]
            all_signals[ticker] = asset_data["Signal"]
            all_var[ticker] = asset_data["VaR"]

            bic_df["ticker"] = ticker
            all_bic.append(bic_df)
        else:
            print(f"Skipping {ticker} due to insufficient data.")

    return (
        pd.DataFrame(all_strat_ret),
        pd.DataFrame(all_bh_ret),
        pd.DataFrame(all_signals),
        pd.concat(all_bic, ignore_index=True),
        pd.DataFrame(all_var),
    )


# ---------------------------------------------------------------------------
# 7. Portfolio Aggregation (Market-Cap Weighted)
# ---------------------------------------------------------------------------


def calculate_mcap_weighted_returns(strat_log_returns, bh_log_returns, mcap_df):
    """Aggregates individual asset returns into a Market Cap weighted portfolio."""
    print("Calculating Market Cap weighted portfolio returns...")

    mcap_df = mcap_df.reindex(
        index=bh_log_returns.index, columns=bh_log_returns.columns
    )
    shifted_mcap = mcap_df.shift(1)

    valid_returns_mask = ~bh_log_returns.isna()
    active_mcap = shifted_mcap.where(valid_returns_mask)

    weights = active_mcap.div(active_mcap.sum(axis=1), axis=0).fillna(0)

    simple_bh = np.exp(bh_log_returns) - 1
    simple_strat = np.exp(strat_log_returns) - 1

    port_simple_bh = (simple_bh * weights).sum(axis=1)
    port_simple_strat = (simple_strat * weights).sum(axis=1)

    port_log_bh = np.log(1 + port_simple_bh)
    port_log_strat = np.log(1 + port_simple_strat)

    return (
        pd.DataFrame({"Log_Return": port_log_bh, "Strategy_Return": port_log_strat})
        .replace(0, np.nan)
        .dropna()
    )


# ---------------------------------------------------------------------------
# 8. Backtest Evaluation
# ---------------------------------------------------------------------------


def evaluate_backtest(df, risk_free_rate=0.02):
    """Calculates standard quantitative performance metrics."""
    df["Simple_Market_Ret"] = np.exp(df["Log_Return"]) - 1
    df["Simple_Strat_Ret"] = np.exp(df["Strategy_Return"]) - 1

    df["Cum_Market"] = (1 + df["Simple_Market_Ret"]).cumprod()
    df["Cum_Strat"] = (1 + df["Simple_Strat_Ret"]).cumprod()

    trading_days = len(df.dropna())
    years = trading_days / 252

    cagr_market = (df["Cum_Market"].iloc[-1]) ** (1 / years) - 1
    cagr_strat = (df["Cum_Strat"].iloc[-1]) ** (1 / years) - 1

    vol_market = df["Simple_Market_Ret"].std() * np.sqrt(252)
    vol_strat = df["Simple_Strat_Ret"].std() * np.sqrt(252)

    sharpe_market = (cagr_market - risk_free_rate) / vol_market
    sharpe_strat = (cagr_strat - risk_free_rate) / vol_strat

    mdd_market = (
        (df["Cum_Market"] - df["Cum_Market"].cummax()) / df["Cum_Market"].cummax()
    ).min()
    mdd_strat = (
        (df["Cum_Strat"] - df["Cum_Strat"].cummax()) / df["Cum_Strat"].cummax()
    ).min()

    results = pd.DataFrame(
        {
            "Metric": ["CAGR", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Benchmark (Buy & Hold)": [
                f"{cagr_market*100:.2f}%",
                f"{vol_market*100:.2f}%",
                f"{sharpe_market:.2f}",
                f"{mdd_market*100:.2f}%",
            ],
            "HMM-Hurst Strategy": [
                f"{cagr_strat*100:.2f}%",
                f"{vol_strat*100:.2f}%",
                f"{sharpe_strat:.2f}",
                f"{mdd_strat*100:.2f}%",
            ],
        }
    ).set_index("Metric")

    return results
