"""
strategies.py
-------------
Pluggable trading strategies that generate signals based on the current regime.

Each strategy has a `generate_signal(prices, idx) -> int` method:
  +1 = long
   0 = flat (stay out)
  -1 = short
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract base class for pluggable strategies."""

    @abstractmethod
    def generate_signal(self, prices: pd.Series, idx: int) -> int:
        """
        Generate a trading signal at position `idx` in the price series.

        Parameters
        ----------
        prices : pd.Series of adjusted close prices (full history up to idx)
        idx : integer position in the series

        Returns
        -------
        signal : int in {-1, 0, +1}
        """
        pass


# ---------------------------------------------------------------------------
# Trend Following: Dual Moving Average Crossover
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(BaseStrategy):
    """
    Dual Moving Average crossover.

    Buy (+1) when fast MA > slow MA  (uptrend)
    Sell (-1) when fast MA < slow MA (downtrend)
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 60):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, prices: pd.Series, idx: int) -> int:
        if idx < self.slow_period:
            return 0

        window = prices.iloc[max(0, idx - self.slow_period) : idx + 1]
        if len(window) < self.slow_period:
            return 0

        fast_ma = window.iloc[-self.fast_period:].mean()
        slow_ma = window.mean()

        if fast_ma > slow_ma:
            return 1
        elif fast_ma < slow_ma:
            return -1
        else:
            return 0


# ---------------------------------------------------------------------------
# Mean Reversion: Bollinger Band / Z-Score
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Z-score based mean reversion.

    Buy  (+1) when z-score < -entry_z  (price is too low → expect reversion up)
    Sell (-1) when z-score > +entry_z  (price is too high → expect reversion down)
    Flat (0)  when |z-score| < exit_z  (price near mean → close position)
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signal(self, prices: pd.Series, idx: int) -> int:
        if idx < self.lookback:
            return 0

        window = prices.iloc[idx - self.lookback : idx + 1]
        if len(window) < self.lookback:
            return 0

        mean = window.mean()
        std = window.std()
        if std == 0 or np.isnan(std):
            return 0

        z = (prices.iloc[idx] - mean) / std

        if z < -self.entry_z:
            return 1   # buy — price is below mean
        elif z > self.entry_z:
            return -1  # sell — price is above mean
        elif abs(z) < self.exit_z:
            return 0   # close — price near mean
        else:
            # Maintain current direction (hysteresis zone)
            # We return 0 here for simplicity; the backtrader strategy
            # will hold the existing position in the hysteresis zone
            return 0


# ---------------------------------------------------------------------------
# Stay-Out Strategy
# ---------------------------------------------------------------------------

class StayOutStrategy(BaseStrategy):
    """Always returns 0 — stay flat."""

    def generate_signal(self, prices: pd.Series, idx: int) -> int:
        return 0


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def get_strategy(regime: str) -> BaseStrategy:
    """
    Return the appropriate strategy for the given regime label.

    Parameters
    ----------
    regime : one of 'trending', 'mean_reverting', 'random_walk'

    Returns
    -------
    BaseStrategy instance
    """
    strategies = {
        "trending": TrendFollowingStrategy(fast_period=20, slow_period=60),
        "mean_reverting": MeanReversionStrategy(lookback=20, entry_z=1.5, exit_z=0.5),
        "random_walk": StayOutStrategy(),
    }
    return strategies.get(regime, StayOutStrategy())


# ---------------------------------------------------------------------------
# Vectorised signal generation (for pre-computation)
# ---------------------------------------------------------------------------

def generate_signals_vectorised(
    prices: pd.Series,
    regimes: pd.Series,
) -> pd.Series:
    """
    Generate signals for a full price series given regime labels at each date.
    Uses the appropriate sub-strategy based on the regime.

    Returns pd.Series of signals (-1, 0, +1).
    """
    signals = pd.Series(0, index=prices.index, name="signal")

    # Pre-compute indicators for efficiency
    fast_ma = prices.rolling(20).mean()
    slow_ma = prices.rolling(60).mean()
    z_mean = prices.rolling(20).mean()
    z_std = prices.rolling(20).std()
    z_score = (prices - z_mean) / z_std

    for date in regimes.index:
        if date not in prices.index:
            continue

        regime = regimes.loc[date]

        if regime == "trending":
            if pd.notna(fast_ma.loc[date]) and pd.notna(slow_ma.loc[date]):
                if fast_ma.loc[date] > slow_ma.loc[date]:
                    signals.loc[date] = 1
                else:
                    signals.loc[date] = -1

        elif regime == "mean_reverting":
            z = z_score.loc[date] if pd.notna(z_score.loc[date]) else 0
            if z < -1.5:
                signals.loc[date] = 1
            elif z > 1.5:
                signals.loc[date] = -1
            else:
                signals.loc[date] = 0

        else:  # random_walk / stay out
            signals.loc[date] = 0

    return signals
