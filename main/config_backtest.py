"""
config_backtest.py
------------------
Overrides for the Backtesting Execution Stage (Stage 2).
This file inherits from config.py and allows you to customize the execution 
without regenerating signals.
"""

from config import *

# ===========================================================================
# 1. Parameter Grid Overrides
# ===========================================================================
# These allow you to pick specific windows for backtesting.
# By default, they inherit everything from config.py.
HURST_WINDOWS = [200]           # e.g., only run the best window from research
MOMENTUM_PERIODS = [21]   # e.g., test all 3 momentum periods

# ===========================================================================
# 2. Execution Scope Overrides
# ===========================================================================
# You might want a different timeline for the backtest execution.
START_YEAR = 2016 #Set this to whenever you want trades to start

# ===========================================================================
# 2. Execution Frequencies
# ===========================================================================
REBALANCE_FREQ_STRAT = "W"  # W=Weekly, M=Monthly
REBALANCE_FREQ_BENCH = "Q"  # Typically Quarterly for index replication

# ===========================================================================
# 3. Risk Management Limits (Elder Rules)
# ===========================================================================
ELDER_MAX_POS = 0.02  # 2% max per asset
ELDER_MAX_DD = 0.06   # 6% max monthly drawdown
