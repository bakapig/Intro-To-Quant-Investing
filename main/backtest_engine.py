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

        for data in self.datas:
            target_pct = data.target_weight[0]
            if not np.isnan(target_pct):
                if self.p.use_elder_rules:
                    target_pct = max(
                        min(target_pct, self.p.elder_max_pos), -self.p.elder_max_pos
                    )

                # --- Weekly rebalance gate ---
                # Only rebalance on the designated weekday (default Monday=0)
                if current_date.weekday() != self.p.rebalance_day:
                    continue

                # --- Weight change threshold ---
                # Skip if the change vs current position is below min threshold
                pos = self.getposition(data)
                portfolio_value = self.broker.getvalue()
                if portfolio_value > 0 and pos:
                    current_pct = (pos.size * data.close[0]) / portfolio_value
                else:
                    current_pct = 0.0
                if abs(target_pct - current_pct) < self.p.min_weight_change:
                    continue

                self.order_target_percent(data, target=target_pct)


# ---------------------------------------------------------------------------
# 4. Logger Setup
# ---------------------------------------------------------------------------


def setup_logger(test_name, output_dir="output"):
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
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# 5. Backtrader Execution Engine
# ---------------------------------------------------------------------------


def run_backtrader_engine(
    df_prices,
    target_weights_df,
    test_name="Strategy",
    starting_cash=1_000_000.0,
    logger=None,
    print_logs=False,
    output_dir="output",
    use_elder_rules=False,
):
    """Initializes Cerebro, runs the engine, and calculates KPIs."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    if logger is None:
        logger = setup_logger(test_name, output_dir=output_dir)

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
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")

    logger.info("Starting Backtest execution...")
    logger.info(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    net_pnl = final_value - starting_cash

    trade_analysis = strat.analyzers.trades.get_analysis()
    try:
        total_closed = trade_analysis.total.closed
    except (KeyError, AttributeError):
        total_closed = 0
    try:
        won_trades = trade_analysis.won.total
    except (KeyError, AttributeError):
        won_trades = 0
    win_rate = (won_trades / total_closed * 100) if total_closed > 0 else 0

    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    try:
        max_dd = drawdown_analysis.max.drawdown
    except (KeyError, AttributeError):
        max_dd = 0.0

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio") or 0

    txn = strat.analyzers.transactions.get_analysis()
    txn_df = pd.DataFrame(
        [
            {"datetime": k, "ticker": v[0][0], "size": v[0][1], "price": v[0][2]}
            for k, v in txn.items()
        ]
    )
    txn_df.to_csv(
        os.path.join(output_dir, f"{test_name}_transactions.csv"), index=False
    )

    logger.info("\n" + "=" * 50)
    logger.info(f"BACKTRADER RESULTS: {test_name.upper()}")
    logger.info("=" * 50)
    logger.info(f"Final Portfolio Value : ${final_value:,.2f}")
    logger.info(f"Net PnL               : ${net_pnl:,.2f}")
    logger.info(f"Total Trades Closed   : {total_closed:,}")
    logger.info(f"Win Rate (Strike Rate): {win_rate:.2f}%")
    logger.info(f"Max Drawdown          : {max_dd:.2f}%")
    logger.info(f"Sharpe Ratio          : {sharpe_ratio:.2f}")
    logger.info("=" * 50)

    return cerebro, strat
