"""
regime_hmm.py
-------------
Hidden Markov Model with 3 states for regime detection.

States are mapped post-hoc to:
  - Trending    (highest mean Hurst)
  - Mean-reverting (lowest mean Hurst)
  - Random-walk / stay-out (middle Hurst)

Features used:
  - Rolling Hurst exponent
  - Rolling return (20-day)
  - Rolling volatility (20-day)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from hurst_dfa import rolling_hurst


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(
    index_prices: pd.Series,
    hurst_lookback: int = 252,
    ret_lookback: int = 20,
    vol_lookback: int = 20,
    hurst_step: int = 5,
) -> pd.DataFrame:
    """
    Compute feature matrix for the HMM from an index price series.

    Features:
      1. Rolling Hurst exponent (via DFA)
      2. Rolling return (log, 20-day)
      3. Rolling volatility (20-day, annualised std of daily log returns)

    Returns DataFrame with columns ['hurst', 'ret_20d', 'vol_20d'].
    """
    # Hurst
    hurst = rolling_hurst(index_prices, lookback=hurst_lookback, step=hurst_step)

    # Log returns
    log_ret = np.log(index_prices / index_prices.shift(1))

    # Rolling 20-day return
    ret_20d = index_prices.pct_change(ret_lookback)

    # Rolling 20-day volatility (annualised)
    vol_20d = log_ret.rolling(vol_lookback).std() * np.sqrt(252)

    features = pd.DataFrame({
        "hurst": hurst,
        "ret_20d": ret_20d,
        "vol_20d": vol_20d,
    }, index=index_prices.index)

    return features


# ---------------------------------------------------------------------------
# HMM fitting & prediction
# ---------------------------------------------------------------------------

def fit_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> Tuple[GaussianHMM, StandardScaler, pd.Index]:
    """
    Fit a Gaussian HMM on the feature matrix.

    Parameters
    ----------
    features : DataFrame with columns ['hurst', 'ret_20d', 'vol_20d']
    n_states : number of hidden states
    n_iter : max EM iterations
    random_state : for reproducibility

    Returns
    -------
    model : fitted GaussianHMM
    scaler : fitted StandardScaler (needed for transform at prediction time)
    valid_index : the DatetimeIndex of rows actually used (after dropping NaN)
    """
    # Drop NaN rows
    clean = features.dropna()
    if len(clean) < 100:
        raise ValueError(f"Not enough valid observations ({len(clean)}) to fit HMM.")

    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)

    return model, scaler, clean.index


def predict_regime(
    model: GaussianHMM,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Predict regime labels and posterior probabilities.

    Returns
    -------
    labels : pd.Series of regime labels (0, 1, 2)
    probs  : pd.DataFrame of posterior probabilities per state
    """
    clean = features.dropna()
    X = scaler.transform(clean.values)

    labels = model.predict(X)
    probs = model.predict_proba(X)

    labels_series = pd.Series(labels, index=clean.index, name="regime")
    probs_df = pd.DataFrame(
        probs,
        index=clean.index,
        columns=[f"prob_state_{i}" for i in range(probs.shape[1])],
    )

    return labels_series, probs_df


# ---------------------------------------------------------------------------
# Map HMM states → semantic regime labels
# ---------------------------------------------------------------------------

REGIME_TREND = "trending"
REGIME_MEAN_REV = "mean_reverting"
REGIME_RANDOM = "random_walk"


def label_regimes(
    model: GaussianHMM,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Predict regimes and map integer states to semantic labels based on
    the mean Hurst exponent in each state:
      - Highest Hurst  → trending
      - Lowest  Hurst  → mean_reverting
      - Middle  Hurst  → random_walk (stay out)

    Returns pd.Series with values in {REGIME_TREND, REGIME_MEAN_REV, REGIME_RANDOM}.
    """
    labels, _ = predict_regime(model, scaler, features)

    clean = features.dropna()
    # Compute mean Hurst per integer state
    state_hurst_mean = {}
    for s in range(model.n_components):
        mask = labels == s
        if mask.sum() > 0:
            state_hurst_mean[s] = clean.loc[mask.index[mask], "hurst"].mean()
        else:
            state_hurst_mean[s] = 0.5  # fallback

    # Sort states by mean Hurst: lowest → mean_rev, middle → random, highest → trend
    sorted_states = sorted(state_hurst_mean, key=state_hurst_mean.get)
    mapping = {
        sorted_states[0]: REGIME_MEAN_REV,
        sorted_states[1]: REGIME_RANDOM,
        sorted_states[2]: REGIME_TREND,
    }

    regime_labels = labels.map(mapping)
    regime_labels.name = "regime_label"
    return regime_labels


# ---------------------------------------------------------------------------
# Walk-forward regime prediction (to avoid look-ahead bias)
# ---------------------------------------------------------------------------

def walk_forward_regimes(
    features: pd.DataFrame,
    train_end: str,
    refit_every: int = 63,  # quarterly re-fit
    n_states: int = 3,
    min_train_size: int = 504,  # ~2 years
) -> pd.Series:
    """
    Expanding-window walk-forward regime prediction.

    1. Fit HMM on data up to `train_end`.
    2. Predict one step at a time on out-of-sample data.
    3. Every `refit_every` days, refit with all data up to that point.

    Returns pd.Series of regime labels for the entire period (NaN for early training).
    """
    clean = features.dropna()
    all_labels = pd.Series(np.nan, index=clean.index, dtype=object)

    train_mask = clean.index <= pd.Timestamp(train_end)
    train_data = clean[train_mask]
    test_data = clean[~train_mask]

    if len(train_data) < min_train_size:
        raise ValueError(f"Training data too small: {len(train_data)} < {min_train_size}")

    # Initial fit on training data
    model, scaler, _ = fit_hmm(train_data, n_states=n_states)
    regime_mapping = _get_regime_mapping(model, scaler, train_data)

    # Label training period
    train_labels, _ = predict_regime(model, scaler, train_data)
    all_labels.loc[train_data.index] = train_labels.map(regime_mapping)

    # Walk forward through test period
    days_since_refit = 0
    expanding_data = train_data.copy()

    for i, (date, row) in enumerate(test_data.iterrows()):
        # Predict with current model
        row_df = pd.DataFrame([row], index=[date])
        X = scaler.transform(row_df.values)
        pred = model.predict(X)[0]
        all_labels.loc[date] = regime_mapping.get(pred, REGIME_RANDOM)

        # Add to expanding window
        expanding_data = pd.concat([expanding_data, row_df])
        days_since_refit += 1

        # Refit periodically
        if days_since_refit >= refit_every:
            try:
                model, scaler, _ = fit_hmm(expanding_data, n_states=n_states)
                regime_mapping = _get_regime_mapping(model, scaler, expanding_data)
                days_since_refit = 0
            except Exception:
                pass  # keep using old model if refit fails

    all_labels.name = "regime_label"
    return all_labels


def _get_regime_mapping(model, scaler, features_df):
    """Helper: compute the state → semantic label mapping."""
    labels, _ = predict_regime(model, scaler, features_df)
    state_hurst = {}
    for s in range(model.n_components):
        mask = labels == s
        if mask.sum() > 0:
            state_hurst[s] = features_df.loc[mask.index[mask], "hurst"].mean()
        else:
            state_hurst[s] = 0.5
    sorted_states = sorted(state_hurst, key=state_hurst.get)
    return {
        sorted_states[0]: REGIME_MEAN_REV,
        sorted_states[1]: REGIME_RANDOM,
        sorted_states[2]: REGIME_TREND,
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import load_all_data, filter_stocks, prepare_prices, build_index

    data = load_all_data("data_cn")
    tickers = filter_stocks(data, top_n=30)
    prices = prepare_prices(data, tickers)
    idx = build_index(prices["adjusted"])

    print("Computing features...")
    feats = compute_features(idx, hurst_lookback=252, hurst_step=5)
    print(f"Features shape: {feats.shape}")
    print(feats.dropna().describe())

    print("\nFitting HMM...")
    model, scaler, valid_idx = fit_hmm(feats)
    print(f"HMM converged: {model.monitor_.converged}")
    print(f"Log-likelihood: {model.score(scaler.transform(feats.dropna().values)):.2f}")

    regimes = label_regimes(model, scaler, feats)
    print(f"\nRegime distribution:\n{regimes.value_counts()}")
