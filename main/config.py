"""
config.py
---------
Centralized configuration for the HMM + Hurst Backtesting Pipeline.
Modify these parameters to control the scope and behavior of the strategy.
"""

import pandas as pd

# ===========================================================================
# 1. Backtest Scope
# ===========================================================================
# The year the backtest should start. 
# It will automatically be max(START_YEAR, Benchmark_Launch_Date).
START_YEAR = 2006

# ===========================================================================
# 2. Strategy Parameters
# ===========================================================================
# HMM States to fit
N_STATES = 3

# Rolling window sizes for the Hurst exponent calculation
# HURST_WINDOWS = [150, 200, 250]
HURST_WINDOWS = [200]

# Momentum calculation period (months)
# MOMENTUM_PERIODS = [5, 10, 21]  # Test these
MOMENTUM_PERIODS = [21]  # Test these
HURST_UPPER = 0.6              # Test 0.55, 0.60
HURST_LOWER = 0.4              # Test 0.45, 0.40
HOLDING_DAYS = 10                # Test 3, 5, 10

# ===========================================================================
# 3. Path Configuration
# ===========================================================================
DATA_DIR = "data_cn"
OUTPUT_DIR = "output"
CACHE_DIR = f"{OUTPUT_DIR}/cache"

# ===========================================================================
# 4. Helper Functions
# ===========================================================================
def get_start_date(df_prices=None):
    """Calculates the best start date based on config and data availability."""
    config_start = pd.Timestamp(year=START_YEAR, month=1, day=1)
    return config_start
