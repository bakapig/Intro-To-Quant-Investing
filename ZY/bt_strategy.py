"""
bt_strategy.py
--------------
Backtrader Strategy that uses pre-computed regime labels and pluggable
sub-strategies (trend following / mean reversion / stay out).
"""

import backtrader as bt
import pandas as pd
import numpy as np


class RegimeSwitchingStrategy(bt.Strategy):
    """
    A backtrader Strategy that:
      1. Reads pre-computed regime labels from a line in the data feed.
      2. Dispatches to the appropriate sub-strategy logic (trend / mean-rev / flat).
      3. Manages positions with simple equal-weight sizing.

    Parameters
    ----------
    regime_series : pd.Series  — pre-computed regime labels indexed by date
    fast_period   : int        — fast MA period for trend following
    slow_period   : int        — slow MA period for trend following
    mr_lookback   : int        — lookback for mean-reversion z-score
    mr_entry_z    : float      — z-score threshold to enter mean-reversion trade
    mr_exit_z     : float      — z-score threshold to exit mean-reversion trade
    pct_size      : float      — fraction of portfolio to risk per trade
    """

    params = dict(
        regime_series=None,
        fast_period=20,
        slow_period=60,
        mr_lookback=20,
        mr_entry_z=1.5,
        mr_exit_z=0.5,
        pct_size=0.95,  # use 95% of portfolio value
    )

    def __init__(self):
        self.regime_map = self.p.regime_series
        # Indicators (for trend following)
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        # For mean reversion
        self.sma_mr = bt.indicators.SMA(self.data.close, period=self.p.mr_lookback)
        self.std_mr = bt.indicators.StdDev(self.data.close, period=self.p.mr_lookback)

        self.order = None
        self.trade_count = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # Uncomment for debugging:
        # print(f"[{dt}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY  @ {order.executed.price:.2f}, size={order.executed.size:.0f}")
            else:
                self.log(f"SELL @ {order.executed.price:.2f}, size={order.executed.size:.0f}")
            self.trade_count += 1
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE P&L: gross={trade.pnl:.2f}, net={trade.pnlcomm:.2f}")

    def next(self):
        if self.order:
            return  # pending order, skip

        current_date = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(current_date)

        # Look up regime
        regime = self._get_regime(dt_ts)

        if regime == "trending":
            self._trend_following_logic()
        elif regime == "mean_reverting":
            self._mean_reversion_logic()
        else:
            # random_walk → stay out / close positions
            self._stay_out_logic()

    def _get_regime(self, dt: pd.Timestamp) -> str:
        """Look up regime for the given date (with fallback)."""
        if self.regime_map is None:
            return "random_walk"
        # Find the most recent regime label <= dt
        valid = self.regime_map.loc[:dt]
        if valid.empty:
            return "random_walk"
        return valid.iloc[-1]

    def _target_size(self) -> int:
        """Compute target position size (number of shares)."""
        price = self.data.close[0]
        if price <= 0:
            return 0
        value = self.broker.getvalue() * self.p.pct_size
        return int(value / price)

    # --- Trend Following Logic ---
    def _trend_following_logic(self):
        target = self._target_size()
        if self.sma_fast[0] > self.sma_slow[0]:
            # Uptrend — go long
            if self.position.size <= 0:
                if self.position.size < 0:
                    self.close()
                self.order = self.buy(size=target)
        elif self.sma_fast[0] < self.sma_slow[0]:
            # Downtrend — go short (or flat if no shorting)
            if self.position.size > 0:
                self.order = self.close()
            # Optionally go short:
            # if self.position.size >= 0:
            #     if self.position.size > 0:
            #         self.close()
            #     self.order = self.sell(size=target)

    # --- Mean Reversion Logic ---
    def _mean_reversion_logic(self):
        if self.std_mr[0] == 0:
            return
        z = (self.data.close[0] - self.sma_mr[0]) / self.std_mr[0]
        target = self._target_size()

        if z < -self.p.mr_entry_z:
            # Price too low → buy
            if self.position.size <= 0:
                if self.position.size < 0:
                    self.close()
                self.order = self.buy(size=target)
        elif z > self.p.mr_entry_z:
            # Price too high → sell / close long
            if self.position.size > 0:
                self.order = self.close()
        elif abs(z) < self.p.mr_exit_z:
            # Near mean → close any position
            if self.position.size != 0:
                self.order = self.close()

    # --- Stay Out Logic ---
    def _stay_out_logic(self):
        if self.position.size != 0:
            self.order = self.close()


# ---------------------------------------------------------------------------
# Custom PandasData feed (supports standard OHLCV)
# ---------------------------------------------------------------------------

class PandasDataFeed(bt.feeds.PandasData):
    """Standard OHLCV data feed from a DataFrame."""
    params = (
        ("datetime", None),  # use index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )
