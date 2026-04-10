"""
backtest_engine.py
------------------
Backtrader-based execution engine replicated from JC's notebook.

Components:
  - SignalData: Custom data feed with pre-calculated target weights
  - DetailedExecutionStrategy: Executes rebalancing + detailed logging
  - generate_target_weights: Convert signals -> MCap-weighted target weights
  - run_backtrader_engine: Full Cerebro setup, execution, and KPI extraction
"""

import logging
import numpy as np
import pandas as pd
import backtrader as bt


# ---------------------------------------------------------------------------
# 1. Target Weight Generation
# ---------------------------------------------------------------------------


def generate_target_weights(
    signals_df, df_mcap, is_buy_and_hold=False, df_var=None, target_risk=0.01
):
    """
    Converts signals and market caps into target portfolio weights.
    If is_buy_and_hold=True, ignores signals and buys the market-cap weighted index.
    If df_var is provided, uses Inverse VaR (Risk Parity) weighting instead of Market Cap.
    """
    print("Generating Target Weights for Backtrader...")

    if is_buy_and_hold:
        execution_signals = signals_df.copy()
        execution_signals[:] = 1
    else:
        execution_signals = signals_df.shift(1).fillna(0)

    if df_var is not None:
        # Approach B: VaR as the Core Weighting Engine (Risk Parity)
        # Target Weight = Target Risk Budget / Forecasted VaR
        execution_var = df_var.shift(1).reindex(
            index=signals_df.index, columns=signals_df.columns
        )
        safe_var = execution_var.replace(0, np.nan)

        raw_weights = target_risk / safe_var
        active_weights = raw_weights.where(execution_signals != 0)

        # Normalize to ensure the sum of absolute weights <= 1.0 (no margin leverage used)
        total_weight = active_weights.abs().sum(axis=1)
        leverage_cap = total_weight.where(total_weight > 1.0, 1.0)
        return (active_weights.div(leverage_cap, axis=0)).fillna(0) * execution_signals

    # Approach A: Standard Market-Cap Weighting
    if 'Date' in df_mcap.columns:
        # Convert YYYYMMDD integers/strings to actual Datetime objects
        df_mcap['Date'] = pd.to_datetime(df_mcap['Date'].astype(str), format='%Y%m%d')
        # Set the Date column as the true index
        df_mcap = df_mcap.set_index('Date')
    
    # 2. Ensure the index is a DatetimeIndex to perfectly match signals_df
    df_mcap.index = pd.to_datetime(df_mcap.index)
    
    # Now reindex will align the dates properly instead of generating NaNs
    df_mcap = df_mcap.reindex(index=signals_df.index, columns=signals_df.columns)
    execution_mcap = df_mcap.shift(1)

    active_mcap = execution_mcap.where(execution_signals != 0)
    normalized_weights = active_mcap.div(active_mcap.sum(axis=1), axis=0).fillna(0)
    target_weights = normalized_weights * execution_signals

    return target_weights


# ---------------------------------------------------------------------------
# 2. Custom Data Feed
# ---------------------------------------------------------------------------


class ChinaAShareCommission(bt.CommInfoBase):
    """
    Realistic Chinese A-share commission structure:
      - Broker commission: 0.025% per side (buy & sell)
      - Stamp duty:        0.05%  sell-only (since Aug 2023)
      - Transfer fee:      0.005% per side
    Total buy:  0.03%   (commission + transfer)
    Total sell: 0.08%   (commission + stamp duty + transfer)
    """

    params = (
        ("commission", 0.0003),  # buy-side: 0.03%
        ("stamp_duty", 0.0005),  # sell-only: 0.05%
        ("transfer_fee", 0.0),  # already included in commission above
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        cost = turnover * self.p.commission  # buy-side cost
        if size < 0:  # sell
            cost += turnover * self.p.stamp_duty
        return cost

class CustomTradeLogAnalyzer(bt.Analyzer):
    """
    Custom Analyzer to cleanly extract all transaction details,
    including the asset ticker, which standard Backtrader omits.
    """
    def __init__(self):
        self.transactions = []

    def notify_order(self, order):
        # Only record completed orders (executed trades)
        if order.status == order.Completed:
            self.transactions.append({
                'datetime': self.strategy.datetime.datetime(),
                'ticker': order.data._name, # Safely grabs the exact ticker
                'size': order.executed.size,
                'price': order.executed.price,
                'value': order.executed.value,
                'commission': order.executed.comm
            })

    def get_analysis(self):
        return self.transactions
    
class SignalData(bt.feeds.PandasData):
    """Custom Data Feed that includes pre-calculated Target Weight."""

    lines = ("target_weight",)
    params = (
        ("datetime", None),
        ("open", "Close"),
        ("high", "Close"),
        ("low", "Close"),
        ("close", "Close"),
        ("volume", -1),
        ("openinterest", -1),
        ("target_weight", "Target_Weight"),
    )


# ---------------------------------------------------------------------------
# 3. Execution Strategy
# ---------------------------------------------------------------------------


class DetailedExecutionStrategy(bt.Strategy):
    """Executes trades and tracks deeply detailed logging."""

    params = (
        ("logger", None),
        ("print_logs", False),
        ("use_elder_rules", False),
        ("elder_max_pos", 0.02),  # 2% Rule (max allocation per asset)
        ("elder_max_dd", 0.06),  # 6% Rule (max monthly drawdown)
        ("rebalance_day", 0),  # 0=Monday. Only rebalance on this weekday.
        ("min_weight_change", 0.005),  # 0.5% threshold to trigger a trade
    )

    def __init__(self):
        self.logger = self.p.logger
        self.month_start_value = None
        self.current_month = None
        self.trading_halted = False

        # Used to ensure we rebalance exactly once per week, ignoring holidays
        self.last_rebalance_week = None 
    
    def prenext(self):
        """
        Backtrader defaults to waiting until ALL 900+ assets have data.
        Calling next() here forces it to run immediately from 2006,
        even if some stocks don't IPO until 2023.
        """
        self.next()

    def log(self, txt, dt=None):
        if self.p.print_logs:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} | {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                msg = (
                    f"Datetime: {self.datas[0].datetime.date(0)} | BUY EXECUTED  | "
                    f"Ticker: {order.data._name} | Price: ${order.executed.price:.2f} | "
                    f"Size: {order.executed.size}"
                )
            elif order.issell():
                msg = (
                    f"Datetime: {self.datas[0].datetime.date(0)} | SELL EXECUTED | "
                    f"Ticker: {order.data._name} | Price: ${order.executed.price:.2f} | "
                    f"Size: {order.executed.size}"
                )

            if self.logger:
                self.logger.info(msg)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(
                f"ORDER FAILED  | Ticker: {order.data._name} | Status: {order.getstatusname()}"
            )

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        msg = (
            f"Datetime: {self.datas[0].datetime.date(0)} | TRADE CLOSED. | "
            f"Ticker: {trade.data._name} | Gross PnL: ${trade.pnl:.2f} | "
            f"Net PnL: ${trade.pnlcomm:.2f}"
        )
        if self.logger:
            self.logger.info(msg)

    def next(self):
        current_date = self.datas[0].datetime.date(0)

        if self.p.use_elder_rules:
            # Evaluate at the start of a new month
            if self.current_month != current_date.month:
                self.current_month = current_date.month
                self.month_start_value = self.broker.getvalue()
                if self.trading_halted:
                    self.log("ELDER 6% RULE: New month started. Trading resumed.")
                self.trading_halted = False

            # Monitor the 6% Rule monthly drawdown threshold
            if (
                not self.trading_halted
                and self.month_start_value is not None
                and self.month_start_value > 0
            ):
                current_dd = (
                    self.broker.getvalue() - self.month_start_value
                ) / self.month_start_value
                if current_dd <= -self.p.elder_max_dd:
                    self.log(
                        f"ELDER 6% RULE TRIGGERED: Monthly DD {current_dd*100:.2f}%. Halting trading."
                    )
                    if self.logger:
                        self.logger.info(
                            f"{current_date.isoformat()} | ELDER 6% RULE: Liquidating portfolio for the month."
                        )
                    self.trading_halted = True
                    for data in self.datas:
                        if self.getposition(data).size != 0:
                            self.order_target_percent(data, target=0.0)

            if self.trading_halted:
                return
        
        # --------------------------------------------------
        # WEEKLY REBALANCE GATE (Holiday Safe)
        # --------------------------------------------------
        # isocalendar()[1] returns the week number of the year (1-52).
        # This ensures that if Monday is a holiday, it will rebalance on Tuesday.
        current_week = current_date.isocalendar()[1]
        
        if current_week == self.last_rebalance_week:
            return  # We already rebalanced this week, skip today.

        # Mark this week as rebalanced!
        self.last_rebalance_week = current_week

        for data in self.datas:
            target_pct = data.target_weight[0]
            if not np.isnan(target_pct):
                if self.p.use_elder_rules:
                    target_pct = max(
                        min(target_pct, self.p.elder_max_pos), -self.p.elder_max_pos
                    )
                
                # --- Chinese A-Share 100-Lot Constraint ---
                pos = self.getposition(data)
                portfolio_value = self.broker.getvalue()

                # MUST STOP if portfolio is burnt to zero to prevent reverse-margin shorting bugs
                if portfolio_value <= 0:
                    return

                if data.close[0] > 0:
                    target_value = portfolio_value * target_pct
                    raw_shares = target_value / data.close[0]
                    # Round down to the nearest 100 shares (1 Board Lot = 100 shares)
                    target_shares = int(raw_shares // 100) * 100
                else:
                    target_shares = 0

                current_shares = pos.size if pos else 0

                # Only issue an order if the lot-adjusted target differs from current holdings
                if target_shares == current_shares:
                    continue

                # --- 5% Drift Tolerance ---
                # Only rebalance if the shares needed to buy/sell are greater than a 5% difference
                # from our current standing holdings to avoid micro-transaction spam in Buy & Hold
                if current_shares != 0:
                    change_pct = abs(target_shares - current_shares) / abs(current_shares)
                    if change_pct < 0.05:
                        continue

                self.order_target_size(data, target=target_shares)


# ---------------------------------------------------------------------------
# 4. Logger Setup
# ---------------------------------------------------------------------------


def setup_logger(test_name, output_dir="output", console_out=True):
    """Create a logger that writes to both file and console."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(output_dir, f"{test_name}.log"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)

    if console_out:
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# 5. Backtrader Execution Engine
# ---------------------------------------------------------------------------


def run_backtrader_engine(
    df_prices,
    target_weights_df,
    test_name="Strategy",
    starting_cash=100_000_000.0,
    logger=None,
    print_logs=False,
    output_dir="output",
    use_elder_rules=False,
    console_out=True,
):
    """Initializes Cerebro, runs the engine, and calculates KPIs."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    if logger is None:
        logger = setup_logger(test_name, output_dir=output_dir, console_out=console_out)

    if console_out:
        print(f"\nInitializing Backtrader Engine for: {test_name} ...")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.addcommissioninfo(ChinaAShareCommission())

    for ticker in df_prices.columns:
        asset_df = pd.DataFrame(
            {"Close": df_prices[ticker], "Target_Weight": target_weights_df[ticker]}
        ).dropna(subset=["Close"])

        if asset_df.empty:
            continue

        data = SignalData(dataname=asset_df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        DetailedExecutionStrategy,
        logger=logger,
        print_logs=print_logs,
        use_elder_rules=use_elder_rules,
    )

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02, annualize=True
    )
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe_a",
        riskfreerate=0.02,
        annualize=True,
        timeframe=bt.TimeFrame.Days,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
    cerebro.addanalyzer(CustomTradeLogAnalyzer, _name="transactions")
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positions_value")

    logger.info("Starting Backtest execution...")
    logger.info(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    net_pnl = final_value - starting_cash

    # ========================
    # Trade Analysis
    # ========================
    ta = strat.analyzers.trades.get_analysis()

    def safe_get(d, *keys):
        for k in keys:
            d = d.get(k, {})
        return d if d != {} else 0

    total_open = safe_get(ta, "total", "open")
    total_closed = safe_get(ta, "total", "closed")

    won_total = safe_get(ta, "won", "total")
    lost_total = safe_get(ta, "lost", "total")

    win_rate = (won_total / total_closed * 100) if total_closed > 0 else 0

    # Streaks
    win_streak_current = safe_get(ta, "streak", "won", "current")
    win_streak_longest = safe_get(ta, "streak", "won", "longest")
    loss_streak_current = safe_get(ta, "streak", "lost", "current")
    loss_streak_longest = safe_get(ta, "streak", "lost", "longest")

    # PnL
    pnl_total = safe_get(ta, "pnl", "net", "total")
    pnl_avg = safe_get(ta, "pnl", "net", "average")

    # Won/Lost stats
    won_pnl_total = safe_get(ta, "won", "pnl", "total")
    won_pnl_avg = safe_get(ta, "won", "pnl", "average")
    won_pnl_max = safe_get(ta, "won", "pnl", "max")

    lost_pnl_total = safe_get(ta, "lost", "pnl", "total")
    lost_pnl_avg = safe_get(ta, "lost", "pnl", "average")
    lost_pnl_max = safe_get(ta, "lost", "pnl", "max")

    # Long/Short
    long_total = safe_get(ta, "long", "total")
    long_pnl_total = safe_get(ta, "long", "pnl", "total")
    long_pnl_avg = safe_get(ta, "long", "pnl", "average")
    long_pnl_max = safe_get(ta, "long", "pnl", "max")

    short_total = safe_get(ta, "short", "total")
    short_pnl_total = safe_get(ta, "short", "pnl", "total")
    short_pnl_avg = safe_get(ta, "short", "pnl", "average")
    short_pnl_max = safe_get(ta, "short", "pnl", "max")

    # Length (bars)
    len_total = safe_get(ta, "len", "total")
    len_avg = safe_get(ta, "len", "average")
    len_max = safe_get(ta, "len", "max")
    len_min = safe_get(ta, "len", "min")

    # ========================
    # Drawdown
    # ========================
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = safe_get(dd, "max", "drawdown")

    # ========================
    # Sharpe
    # ========================
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0)
    sharpe_a_ratio = strat.analyzers.sharpe_a.get_analysis().get("sharperatio", 0)

    # ========================
    # Time Returns → Annual Return
    # ========================
    returns_dict = strat.analyzers.returns.get_analysis()
    returns_series = pd.Series(returns_dict)

    total_return = (1 + returns_series).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1

    # ========================
    # Calmar Ratio
    # ========================
    calmar = (annual_return / (max_dd / 100)) if max_dd != 0 else 0

    # ========================
    # Positions Value
    # ========================
    pos_val = strat.analyzers.positions_value.get_analysis()
    pos_df = pd.DataFrame(pos_val)

    # ========================
    # Logging
    # ========================
    logger.info("\n" + "=" * 60)
    logger.info(f"BACKTRADER RESULTS: {test_name.upper()}")
    logger.info("=" * 60)

    logger.info(f"Final Portfolio Value : ${final_value:,.2f}")
    logger.info(f"Net PnL               : ${net_pnl:,.2f}")

    logger.info("--- Returns ---")
    logger.info(f"Total Return          : {total_return:.2%}")
    logger.info(f"Annual Return         : {annual_return:.2%}")
    logger.info(f"Calmar Ratio          : {calmar:.2f}")

    logger.info("--- Trades ---")
    logger.info(f"Open Trades           : {total_open}")
    logger.info(f"Closed Trades         : {total_closed}")
    logger.info(f"Win Rate              : {win_rate:.2f}%")

    logger.info("--- Streaks ---")
    logger.info(f"Win Streak (Cur/Max)  : {win_streak_current}/{win_streak_longest}")
    logger.info(f"Loss Streak (Cur/Max) : {loss_streak_current}/{loss_streak_longest}")

    logger.info("--- PnL ---")
    logger.info(f"Total PnL             : {pnl_total:,.2f}")
    logger.info(f"Avg PnL               : {pnl_avg:,.2f}")

    logger.info("--- Won Trades ---")
    logger.info(f"Count                 : {won_total}")
    logger.info(f"Total/Avg/Max         : {won_pnl_total:.2f} / {won_pnl_avg:.2f} / {won_pnl_max:.2f}")

    logger.info("--- Lost Trades ---")
    logger.info(f"Count                 : {lost_total}")
    logger.info(f"Total/Avg/Max         : {lost_pnl_total:.2f} / {lost_pnl_avg:.2f} / {lost_pnl_max:.2f}")

    logger.info("--- Long Trades ---")
    logger.info(f"Count                 : {long_total}")
    logger.info(f"Total/Avg/Max         : {long_pnl_total:.2f} / {long_pnl_avg:.2f} / {long_pnl_max:.2f}")

    logger.info("--- Short Trades ---")
    logger.info(f"Count                 : {short_total}")
    logger.info(f"Total/Avg/Max         : {short_pnl_total:.2f} / {short_pnl_avg:.2f} / {short_pnl_max:.2f}")

    logger.info("--- Trade Length (bars) ---")
    logger.info(f"Total/Avg/Max/Min     : {len_total} / {len_avg:.2f} / {len_max} / {len_min}")

    logger.info("--- Risk ---")
    logger.info(f"Max Drawdown          : {max_dd:.2f}%")
    logger.info(f"Sharpe Ratio          : {sharpe_ratio:.2f}")
    logger.info(f"Sharpe Ratio (Annual) : {sharpe_a_ratio:.2f}")

    logger.info("=" * 60)

    metrics = {
        "Strategy": test_name,
        "Final Value": final_value,
        "Net PnL": net_pnl,
        "Trades": total_closed,
        "Win Rate (%)": win_rate,
        "Max DD (%)": max_dd,
        "Sharpe": sharpe_ratio,
    }

    return cerebro, strat, metrics
