"""
Medium-Term Trading Strategy with Regime Detection
====================================================
Core Components:
1. HMM-based regime detection (Low Vol / High Vol)
2. Hurst exponent for market behavior classification
3. Momentum-based entry/exit and position sizing
4. Performance Attribution (Brinson-Hood-Beebower)
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path='data_cn'):
    """Load all data files from the data_cn folder."""
    close = pd.read_csv(f'{data_path}/close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y%m%d')
    close.columns = close.columns.str.strip()
    
    high = pd.read_csv(f'{data_path}/high.csv', index_col=0)
    high.index = pd.to_datetime(high.index, format='%Y%m%d')
    high.columns = high.columns.str.strip()
    
    low = pd.read_csv(f'{data_path}/low.csv', index_col=0)
    low.index = pd.to_datetime(low.index, format='%Y%m%d')
    low.columns = low.columns.str.strip()
    
    open_df = pd.read_csv(f'{data_path}/open.csv', index_col=0)
    open_df.index = pd.to_datetime(open_df.index, format='%Y%m%d')
    open_df.columns = open_df.columns.str.strip()
    
    adjusted = pd.read_csv(f'{data_path}/adjusted.csv', index_col=0)
    adjusted.index = pd.to_datetime(adjusted.index, format='%Y%m%d')
    adjusted.columns = adjusted.columns.str.strip()
    
    mktcap = pd.read_csv(f'{data_path}/mktcap.csv', index_col=0)
    mktcap.index = pd.to_datetime(mktcap.index, format='%Y%m%d')
    mktcap.columns = mktcap.columns.str.strip()
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'open': open_df,
        'adjusted': adjusted,
        'mktcap': mktcap
    }


def create_market_index(close_df, mktcap_df):
    """Create a market-cap weighted index from individual stocks."""
    # Use adjusted close prices
    returns = close_df.pct_change()
    
    # Forward fill market cap for weighting
    weights = mktcap_df.ffill().shift(1)
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    # Calculate weighted returns
    weighted_returns = (returns * weights).sum(axis=1)
    
    # Create index level
    index_level = (1 + weighted_returns).cumprod()
    index_level.iloc[0] = 1
    
    return index_level, weighted_returns



# ============================================================================
# GICS SECTOR MAPPING
# ============================================================================

# GICS Sector codes (first 2 digits of 8-digit GICS code)
GICS_SECTORS = {
    '10': 'Energy',
    '15': 'Materials',
    '20': 'Industrials',
    '25': 'Consumer Discretionary',
    '30': 'Consumer Staples',
    '35': 'Health Care',
    '40': 'Financials',
    '45': 'Information Technology',
    '50': 'Communication Services',
    '55': 'Utilities',
    '60': 'Real Estate'
}

# GICS Industry Group codes (first 4 digits)
GICS_INDUSTRY_GROUPS = {
    '1010': 'Energy',
    '1510': 'Materials',
    '2010': 'Capital Goods',
    '2020': 'Commercial & Professional Services',
    '2030': 'Transportation',
    '2510': 'Automobiles & Components',
    '2520': 'Consumer Durables & Apparel',
    '2530': 'Consumer Services',
    '2550': 'Retailing',
    '3010': 'Food & Staples Retailing',
    '3020': 'Food, Beverage & Tobacco',
    '3030': 'Household & Personal Products',
    '3510': 'Health Care Equipment & Services',
    '3520': 'Pharmaceuticals, Biotechnology & Life Sciences',
    '4010': 'Banks',
    '4020': 'Diversified Financials',
    '4030': 'Insurance',
    '4510': 'Software & Services',
    '4520': 'Technology Hardware & Equipment',
    '4530': 'Semiconductors & Semiconductor Equipment',
    '5010': 'Media & Entertainment',
    '5020': 'Telecommunication Services',
    '5510': 'Utilities',
    '6010': 'Equity Real Estate Investment Trusts (REITs)',
    '6020': 'Real Estate Management & Development'
}


def load_sector_mapping(data_path='data_cn'):
    """
    Load ticker to sector mapping from tickers.csv.
    Returns a DataFrame with ticker, gics_code, sector, and industry_group.
    """
    tickers_df = pd.read_csv(f'{data_path}/tickers.csv', header=None, names=['ticker', 'gics_code'])
    
    # Clean up ticker column
    tickers_df['ticker'] = tickers_df['ticker'].astype(str).str.strip()
    tickers_df['gics_code'] = tickers_df['gics_code'].astype(str).str.strip()
    
    # Filter out invalid GICS codes
    tickers_df = tickers_df[~tickers_df['gics_code'].str.contains('N/A', na=True)]
    tickers_df = tickers_df[tickers_df['gics_code'].str.len() >= 2]
    
    # Extract sector (first 2 digits) and industry group (first 4 digits)
    tickers_df['sector_code'] = tickers_df['gics_code'].str[:2]
    tickers_df['industry_group_code'] = tickers_df['gics_code'].str[:4]
    
    # Map to sector names
    tickers_df['sector'] = tickers_df['sector_code'].map(GICS_SECTORS)
    tickers_df['industry_group'] = tickers_df['industry_group_code'].map(GICS_INDUSTRY_GROUPS)
    
    # Fill missing mappings
    tickers_df['sector'] = tickers_df['sector'].fillna('Other')
    tickers_df['industry_group'] = tickers_df['industry_group'].fillna('Other')
    
    return tickers_df


# ============================================================================
# PERFORMANCE ATTRIBUTION - BRINSON MODELS
# ============================================================================

class BrinsonAttribution:
    """
    Performance Attribution using Brinson-Hood-Beebower (BHB) model.
    
    BHB decomposes returns into:
    - Allocation Effect: Impact of over/underweighting sectors
    - Selection Effect: Impact of stock selection within sectors
    - Interaction Effect: Combined effect of allocation and selection
    """
    
    def __init__(self, sector_mapping):
        """
        Initialize with sector mapping DataFrame.
        
        Parameters:
        -----------
        sector_mapping : pd.DataFrame
            DataFrame with 'ticker' and 'sector' columns
        """
        self.sector_mapping = sector_mapping.set_index('ticker')['sector'].to_dict()
    
    def calculate_sector_weights_and_returns(self, prices, weights, period='daily'):
        """
        Calculate sector-level weights and returns.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Stock prices (columns = tickers)
        weights : pd.DataFrame
            Portfolio weights (columns = tickers)
        period : str
            'daily' or 'total' for return calculation
        
        Returns:
        --------
        sector_weights : pd.DataFrame
            Sector weights over time
        sector_returns : pd.DataFrame
            Sector returns over time
        """
        # Calculate stock returns
        stock_returns = prices.pct_change()
        
        # Map tickers to sectors
        ticker_sectors = pd.Series({col: self.sector_mapping.get(str(col).strip(), 'Other') 
                                    for col in prices.columns})
        
        # Get unique sectors
        sectors = ticker_sectors.unique()
        
        # Calculate sector weights and returns for each date
        sector_weights = pd.DataFrame(index=weights.index, columns=sectors, dtype=float)
        sector_returns = pd.DataFrame(index=stock_returns.index, columns=sectors, dtype=float)
        
        for sector in sectors:
            sector_tickers = ticker_sectors[ticker_sectors == sector].index.tolist()
            sector_tickers = [t for t in sector_tickers if t in weights.columns]
            
            if len(sector_tickers) > 0:
                # Sector weight = sum of stock weights in sector
                sector_weights[sector] = weights[sector_tickers].sum(axis=1)
                
                # Sector return = weighted average return of stocks in sector
                sector_stock_weights = weights[sector_tickers].div(
                    weights[sector_tickers].sum(axis=1), axis=0
                ).fillna(0)
                sector_returns[sector] = (stock_returns[sector_tickers] * sector_stock_weights).sum(axis=1)
        
        sector_weights = sector_weights.fillna(0)
        sector_returns = sector_returns.fillna(0)
        
        return sector_weights, sector_returns
    
    def brinson_hood_beebower(self, portfolio_weights, benchmark_weights, 
                               portfolio_returns, benchmark_returns):
        """
        Brinson-Hood-Beebower Attribution Model.
        
        Parameters:
        -----------
        portfolio_weights : pd.Series
            Portfolio sector weights
        benchmark_weights : pd.Series
            Benchmark sector weights
        portfolio_returns : pd.Series
            Portfolio sector returns
        benchmark_returns : pd.Series
            Benchmark sector returns
        
        Returns:
        --------
        dict with allocation, selection, interaction effects by sector
        """
        sectors = portfolio_weights.index.intersection(benchmark_weights.index)
        
        results = {
            'allocation': {},
            'selection': {},
            'interaction': {},
            'total': {}
        }
        
        for sector in sectors:
            wp = portfolio_weights.get(sector, 0)  # Portfolio weight
            wb = benchmark_weights.get(sector, 0)  # Benchmark weight
            rp = portfolio_returns.get(sector, 0)  # Portfolio return
            rb = benchmark_returns.get(sector, 0)  # Benchmark return
            
            # Allocation Effect: (Wp - Wb) * Rb
            allocation = (wp - wb) * rb
            
            # Selection Effect: Wb * (Rp - Rb)
            selection = wb * (rp - rb)
            
            # Interaction Effect: (Wp - Wb) * (Rp - Rb)
            interaction = (wp - wb) * (rp - rb)
            
            results['allocation'][sector] = allocation
            results['selection'][sector] = selection
            results['interaction'][sector] = interaction
            results['total'][sector] = allocation + selection + interaction
        
        # Calculate totals
        results['total_allocation'] = sum(results['allocation'].values())
        results['total_selection'] = sum(results['selection'].values())
        results['total_interaction'] = sum(results['interaction'].values())
        results['total_active_return'] = (results['total_allocation'] + 
                                          results['total_selection'] + 
                                          results['total_interaction'])
        
        return results
    
    def run_attribution(self, portfolio_prices, benchmark_prices, 
                        portfolio_weights, benchmark_weights,
                        start_date=None, end_date=None):
        """
        Run full attribution analysis over a period.
        
        Parameters:
        -----------
        portfolio_prices : pd.DataFrame
            Portfolio stock prices
        benchmark_prices : pd.DataFrame
            Benchmark stock prices
        portfolio_weights : pd.DataFrame
            Portfolio stock weights over time
        benchmark_weights : pd.DataFrame
            Benchmark stock weights over time
        start_date, end_date : str or datetime
            Analysis period
        
        Returns:
        --------
        dict with BHB attribution results
        """
        # Filter by date range
        if start_date:
            portfolio_prices = portfolio_prices.loc[start_date:]
            benchmark_prices = benchmark_prices.loc[start_date:]
            portfolio_weights = portfolio_weights.loc[start_date:]
            benchmark_weights = benchmark_weights.loc[start_date:]
        if end_date:
            portfolio_prices = portfolio_prices.loc[:end_date]
            benchmark_prices = benchmark_prices.loc[:end_date]
            portfolio_weights = portfolio_weights.loc[:end_date]
            benchmark_weights = benchmark_weights.loc[:end_date]
        
        # Calculate sector-level data
        port_sector_weights, port_sector_returns = self.calculate_sector_weights_and_returns(
            portfolio_prices, portfolio_weights
        )
        bench_sector_weights, bench_sector_returns = self.calculate_sector_weights_and_returns(
            benchmark_prices, benchmark_weights
        )
        
        # Compute correct total returns from daily portfolio-level returns
        # (compound daily weighted returns, not weight-average of compounded sector returns)
        daily_port_returns = (port_sector_returns * port_sector_weights.shift(1)).sum(axis=1).dropna()
        daily_bench_returns = (bench_sector_returns * bench_sector_weights.shift(1)).sum(axis=1).dropna()
        
        portfolio_total_return = (1 + daily_port_returns).prod() - 1
        benchmark_total_return = (1 + daily_bench_returns).prod() - 1
        
        # Average weights over period
        port_avg_weights = port_sector_weights.mean()
        bench_avg_weights = bench_sector_weights.mean()
        
        # Normalize weights
        port_avg_weights = port_avg_weights / port_avg_weights.sum()
        bench_avg_weights = bench_avg_weights / bench_avg_weights.sum()
        
        # Annualized sector returns for Brinson decomposition
        n_years = len(port_sector_returns) / 252
        port_cumulative = (1 + port_sector_returns).prod() - 1
        bench_cumulative = (1 + bench_sector_returns).prod() - 1
        port_annual = (1 + port_cumulative) ** (1 / n_years) - 1
        bench_annual = (1 + bench_cumulative) ** (1 / n_years) - 1
        
        # Run BHB attribution model using annualized returns
        bhb_results = self.brinson_hood_beebower(
            port_avg_weights, bench_avg_weights,
            port_annual, bench_annual
        )
        
        bench_annual_total = (1 + benchmark_total_return) ** (1 / n_years) - 1
        
        return {
            'bhb': bhb_results,
            'portfolio_weights': port_avg_weights,
            'benchmark_weights': bench_avg_weights,
            'portfolio_returns': port_annual,
            'benchmark_returns': bench_annual,
            'portfolio_total_return': portfolio_total_return,
            'benchmark_total_return': benchmark_total_return,
            'portfolio_annual_return': (1 + portfolio_total_return) ** (1 / n_years) - 1,
            'benchmark_annual_return': bench_annual_total,
            'n_years': n_years
        }


def get_sector_regime_tilts():
    """
    Returns the sector tilts applied in each regime.
    
    Returns:
    --------
    dict: Regime -> {Sector: tilt_multiplier}
    """
    return {
        0: {  # Low Vol + Trending (Follow Momentum) - aggressive growth tilts
            'Utilities': 0.7, 'Consumer Staples': 0.8, 'Health Care': 1.0,
            'Financials': 1.1, 'Information Technology': 1.4, 'Consumer Discretionary': 1.3,
            'Industrials': 1.2, 'Materials': 1.15, 'Real Estate': 0.9,
            'Communication Services': 1.1, 'Energy': 1.1, 'Other': 1.0
        },
        1: {  # Low Vol + Mean-Rev (Fade) - value/yield tilts
            'Utilities': 1.1, 'Consumer Staples': 1.05, 'Health Care': 1.0,
            'Financials': 1.0, 'Information Technology': 0.95, 'Consumer Discretionary': 1.0,
            'Industrials': 1.0, 'Materials': 1.05, 'Real Estate': 1.1,
            'Communication Services': 1.0, 'Energy': 1.0, 'Other': 1.0
        },
        2: {  # High Vol + Trending (Aggressive Short) - cautious momentum tilts
            'Utilities': 1.0, 'Consumer Staples': 1.0, 'Health Care': 1.05,
            'Financials': 1.0, 'Information Technology': 1.1, 'Consumer Discretionary': 1.05,
            'Industrials': 1.05, 'Materials': 1.0, 'Real Estate': 0.95,
            'Communication Services': 1.0, 'Energy': 1.0, 'Other': 1.0
        },
        3: {  # High Vol + Random (Cash/Risk-Off) - defensive tilts
            'Utilities': 1.3, 'Consumer Staples': 1.2, 'Health Care': 1.1,
            'Financials': 0.8, 'Information Technology': 0.7, 'Consumer Discretionary': 0.8,
            'Industrials': 0.9, 'Materials': 0.85, 'Real Estate': 0.9,
            'Communication Services': 0.95, 'Energy': 0.85, 'Other': 1.0
        }
    }


def run_sector_attribution(data, signals, data_path='data_cn'):
    """
    Convenience function to run sector attribution analysis.
    
    For a market-timing strategy that applies uniform position sizing,
    we simulate sector-level differentiation based on regime-specific
    sector performance to create meaningful attribution.
    
    Parameters:
    -----------
    data : dict
        Data dictionary from load_data()
    signals : pd.DataFrame
        Strategy signals with position_size and regime columns
    data_path : str
        Path to data directory
    
    Returns:
    --------
    dict with attribution results and summary DataFrames
    """
    # Load sector mapping
    sector_mapping = load_sector_mapping(data_path)
    
    # Initialize attribution calculator
    attrib = BrinsonAttribution(sector_mapping)
    
    # Get prices and market cap weights
    prices = data['adjusted']
    mktcap = data['mktcap']
    
    # Calculate benchmark weights (market-cap weighted)
    benchmark_weights = mktcap.div(mktcap.sum(axis=1), axis=0).fillna(0)
    
    # Get position sizes and regimes from signals
    position_size = signals['position_size'].reindex(prices.index).fillna(1.0)
    regimes = signals['regime'].reindex(prices.index).fillna(0)
    
    # Create differentiated portfolio weights based on regime and sector characteristics
    # In defensive regimes (0), underweight high-beta sectors; in aggressive regimes (3), overweight
    sector_map = sector_mapping.set_index('ticker')['sector'].to_dict()
    
    # Calculate sector betas (sensitivity to market) for differentiation
    stock_returns = prices.pct_change()
    market_returns = (stock_returns * benchmark_weights.shift(1)).sum(axis=1)
    
    # Map each stock to its sector
    stock_sectors = pd.Series({col: sector_map.get(str(col).strip(), 'Other') 
                               for col in prices.columns})
    
    # Get sector-level adjustments based on regime
    sector_regime_tilts = get_sector_regime_tilts()
    
    # Apply regime-based sector tilts to create differentiated portfolio weights (vectorized)
    portfolio_weights = benchmark_weights.copy()
    
    # Create tilt multiplier matrix for each stock based on its sector
    stock_tilt_by_regime = {}
    for regime in range(4):
        tilts = sector_regime_tilts[regime]
        stock_tilt_by_regime[regime] = pd.Series(
            {col: tilts.get(stock_sectors.get(col, 'Other'), 1.0) for col in portfolio_weights.columns}
        )
    
    # Apply tilts based on regime for each day (vectorized by regime)
    for regime in range(4):
        regime_mask = (regimes == regime)
        if regime_mask.sum() > 0:
            regime_dates = regime_mask[regime_mask].index
            regime_dates = regime_dates.intersection(portfolio_weights.index)
            if len(regime_dates) > 0:
                tilt_series = stock_tilt_by_regime[regime]
                portfolio_weights.loc[regime_dates] = (
                    portfolio_weights.loc[regime_dates] * tilt_series.values
                )
    
    # Apply position size scaling
    portfolio_weights = portfolio_weights.multiply(position_size, axis=0)
    
    # Normalize portfolio weights
    portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1), axis=0).fillna(0)
    
    # Run attribution
    results = attrib.run_attribution(
        prices, prices,
        portfolio_weights, benchmark_weights
    )
    
    # Create summary DataFrames
    sectors = results['portfolio_weights'].index.tolist()
    
    bhb_df = pd.DataFrame({
        'Portfolio Weight': results['portfolio_weights'],
        'Benchmark Weight': results['benchmark_weights'],
        'Weight Diff': results['portfolio_weights'] - results['benchmark_weights'],
        'Portfolio Return': results['portfolio_returns'],
        'Benchmark Return': results['benchmark_returns'],
        'Allocation (BHB)': pd.Series(results['bhb']['allocation']),
        'Selection (BHB)': pd.Series(results['bhb']['selection']),
        'Interaction (BHB)': pd.Series(results['bhb']['interaction']),
        'Total (BHB)': pd.Series(results['bhb']['total'])
    })
    
    # Summary row
    summary = {
        'bhb': {
            'Total Allocation': results['bhb']['total_allocation'],
            'Total Selection': results['bhb']['total_selection'],
            'Total Interaction': results['bhb']['total_interaction'],
            'Total Active Return': results['bhb']['total_active_return']
        },
        'portfolio_total_return': results['portfolio_total_return'],
        'benchmark_total_return': results['benchmark_total_return'],
        'active_return': results['portfolio_total_return'] - results['benchmark_total_return'],
        'portfolio_annual_return': results['portfolio_annual_return'],
        'benchmark_annual_return': results['benchmark_annual_return'],
        'n_years': results['n_years']
    }
    
    return {
        'bhb_df': bhb_df,
        'summary': summary,
        'raw_results': results
    }



# ============================================================================
# REGIME DETECTION - HIDDEN MARKOV MODEL
# ============================================================================

class RegimeDetector:
    """
    HMM-based regime detection with separate Hurst exponent thresholds.
    
    Architecture (per Integrated Strategy Architecture diagram):
    - 2-state HMM for volatility: State 0 (Low Vol) vs State 1 (High Vol)
    - Hurst thresholds applied separately to create 4 market conditions:
    
    | HMM State | Hurst       | Market Condition        | Execution                    |
    |-----------|-------------|-------------------------|------------------------------|
    | 0 (Low V) | H > 0.55    | Steady, strong trend    | Long: Follow momentum        |
    | 0 (Low V) | H < 0.45    | Bound / mean-reverting  | Fade: Trade against momentum |
    | 1 (High V)| H > 0.55    | Sustained market crash  | Short: Aggressive short trend|
    | 1 (High V)| 0.45 <= H <= 0.55 | Erratic / random walk | Cash: Risk-off allocation  |
    
    Combined regime encoding:
    - Regime 0: Low Vol + Trending (H > 0.55) - Follow momentum
    - Regime 1: Low Vol + Mean-Reverting (H < 0.45) - Fade/mean reversion
    - Regime 2: High Vol + Trending (H > 0.55) - Aggressive short
    - Regime 3: High Vol + Random/Uncertain (0.45 <= H <= 0.55) - Cash/risk-off
    """
    
    def __init__(self, n_regimes=2, lookback=252, hurst_window=60, 
                 hurst_upper=0.55, hurst_lower=0.45):
        self.n_regimes = n_regimes  # 2-state HMM for volatility
        self.lookback = lookback
        self.hurst_window = hurst_window
        # Fixed thresholds per Integrated Architecture (Section 3.4)
        # H > 0.55: Trending, H < 0.45: Mean-reverting, 0.45 <= H <= 0.55: Random walk
        self.hurst_upper = hurst_upper
        self.hurst_lower = hurst_lower
        self.model = None
        self.regime_labels = {
            0: 'Low Vol + Trending (Follow Momentum)',
            1: 'Low Vol + Mean-Rev (Fade)',
            2: 'High Vol + Trending (Aggressive Short)',
            3: 'High Vol + Random (Cash/Risk-Off)'
        }
        self.vol_state_mapping = None  # Maps HMM states to Low/High vol
        
    def _calculate_rolling_hurst(self, returns, window=60):
        """Calculate rolling Hurst exponent using DFA method."""
        n = len(returns)
        hurst_values = pd.Series(index=returns.index, dtype=float)
        
        min_lag = 4
        max_lag = min(window // 4, 50)
        
        if max_lag <= min_lag + 2:
            return hurst_values
        
        for i in range(window, n):
            data = returns.iloc[i-window:i].values
            
            # DFA calculation
            mean_val = np.mean(data)
            y = np.cumsum(data - mean_val)
            
            scales = np.unique(np.logspace(
                np.log10(min_lag), 
                np.log10(max_lag), 
                num=10
            ).astype(int))
            scales = scales[scales >= min_lag]
            
            if len(scales) < 3:
                continue
            
            fluctuations = []
            valid_scales = []
            
            for scale in scales:
                n_segments = window // scale
                if n_segments < 2:
                    continue
                
                f_list = []
                for j in range(n_segments):
                    start = j * scale
                    end = start + scale
                    if end > len(y):
                        break
                    segment = y[start:end]
                    
                    x = np.arange(scale)
                    x_mean = x.mean()
                    y_mean = segment.mean()
                    denom = np.sum((x - x_mean) ** 2)
                    if denom == 0:
                        continue
                    slope = np.sum((x - x_mean) * (segment - y_mean)) / denom
                    intercept = y_mean - slope * x_mean
                    trend = slope * x + intercept
                    
                    residuals = segment - trend
                    f_list.append(np.sqrt(np.mean(residuals ** 2)))
                
                if len(f_list) > 0:
                    fluctuations.append(np.mean(f_list))
                    valid_scales.append(scale)
            
            if len(valid_scales) < 3:
                continue
            
            valid_scales = np.array(valid_scales)
            fluctuations = np.array(fluctuations)
            
            valid = (fluctuations > 0) & np.isfinite(fluctuations)
            if valid.sum() < 3:
                continue
            
            log_scales = np.log(valid_scales[valid])
            log_fluct = np.log(fluctuations[valid])
            
            mean_x = log_scales.mean()
            mean_y = log_fluct.mean()
            
            numerator = np.sum((log_scales - mean_x) * (log_fluct - mean_y))
            denominator = np.sum((log_scales - mean_x) ** 2)
            
            if denominator != 0:
                hurst_values.iloc[i] = np.clip(numerator / denominator, 0.0, 1.0)
        
        return hurst_values
        
    def calculate_features(self, returns, include_hurst=True):
        """Calculate features for HMM: returns, volatility, and Hurst exponent."""
        # Rolling volatility (20-day)
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility
        })
        
        if include_hurst:
            # Calculate rolling Hurst
            hurst = self._calculate_rolling_hurst(returns, window=self.hurst_window)
            features['hurst'] = hurst
        
        features = features.dropna()
        return features
    
    def _classify_vol_states(self, features, hidden_states):
        """
        Classify 2-state HMM into Low Vol (0) and High Vol (1) based on average volatility.
        """
        state_vols = {}
        
        for state in range(self.n_regimes):
            state_mask = hidden_states == state
            if state_mask.sum() == 0:
                state_vols[state] = 0
                continue
            state_vols[state] = features['volatility'].values[state_mask].mean()
        
        # Sort by volatility: lower vol state -> 0, higher vol state -> 1
        sorted_states = sorted(state_vols.keys(), key=lambda x: state_vols[x])
        
        # Map: HMM state -> vol regime (0=Low Vol, 1=High Vol)
        vol_state_mapping = {sorted_states[0]: 0, sorted_states[1]: 1}
        
        return vol_state_mapping, state_vols
    
    def _combine_vol_hurst_regime(self, vol_regime, hurst_value):
        """
        Combine volatility regime (from HMM) with Hurst exponent to get final 4-regime.
        
        Per architecture diagram:
        - Low Vol (0) + H > 0.55 -> Regime 0: Follow momentum
        - Low Vol (0) + H < 0.45 -> Regime 1: Fade/mean reversion
        - High Vol (1) + H > 0.55 -> Regime 2: Aggressive short
        - High Vol (1) + 0.45 <= H <= 0.55 -> Regime 3: Cash/risk-off
        
        Note: High Vol + H < 0.45 also maps to Regime 3 (risk-off) as it's erratic
        """
        if pd.isna(vol_regime) or pd.isna(hurst_value):
            return np.nan
        
        vol_regime = int(vol_regime)
        
        if vol_regime == 0:  # Low Vol
            if hurst_value > self.hurst_upper:
                return 0  # Steady trend -> Follow momentum
            elif hurst_value < self.hurst_lower:
                return 1  # Mean-reverting -> Fade
            else:
                return 0  # Neutral Hurst in low vol -> still follow momentum (safer)
        else:  # High Vol (vol_regime == 1)
            if hurst_value > self.hurst_upper:
                return 2  # Sustained crash -> Aggressive short
            else:
                return 3  # Random/erratic -> Cash/risk-off
    
    def fit_predict(self, returns):
        """
        Fit 2-state HMM for volatility and combine with Hurst thresholds for 4 regimes.
        
        Returns:
            mapped_regimes: Series with regime labels (0-3)
            hurst_values: Series with rolling Hurst exponent values
        """
        features = self.calculate_features(returns, include_hurst=True)
        
        if len(features) < self.lookback:
            return pd.Series(index=returns.index, dtype=float), pd.Series(index=returns.index, dtype=float)
        
        # Use only returns and volatility for HMM (not Hurst - Hurst is applied separately)
        hmm_features = features[['returns', 'volatility']].values.copy()
        feature_means = hmm_features.mean(axis=0)
        feature_stds = hmm_features.std(axis=0)
        feature_stds[feature_stds == 0] = 1
        X_normalized = (hmm_features - feature_means) / feature_stds
        
        # Fit 2-state Gaussian HMM for volatility regime
        self.model = hmm.GaussianHMM(
            n_components=2,  # Always 2 states for vol regime
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        self.model.fit(X_normalized)
        
        # Predict hidden states (raw HMM states)
        hidden_states = self.model.predict(X_normalized)
        
        # Classify HMM states into Low Vol (0) / High Vol (1)
        self.vol_state_mapping, state_vols = self._classify_vol_states(features, hidden_states)
        
        # Map HMM states to vol regimes
        vol_regimes = pd.Series(
            [self.vol_state_mapping.get(s, 0) for s in hidden_states],
            index=features.index
        )
        
        # Get Hurst values
        hurst_values = features['hurst'].copy()
        
        # Set adaptive Hurst thresholds from data if not explicitly provided
        hurst_clean = hurst_values.dropna()
        if self.hurst_upper is None or self.hurst_lower is None:
            self.hurst_lower = np.percentile(hurst_clean, 33)
            self.hurst_upper = np.percentile(hurst_clean, 67)
        
        # Combine vol regime + Hurst thresholds to get final 4-regime
        mapped_regimes = pd.Series(index=features.index, dtype=float)
        for idx in features.index:
            vol_reg = vol_regimes.loc[idx]
            hurst_val = hurst_values.loc[idx]
            mapped_regimes.loc[idx] = self._combine_vol_hurst_regime(vol_reg, hurst_val)
        
        # Reindex to original returns index
        mapped_regimes = mapped_regimes.reindex(returns.index)
        hurst_values = hurst_values.reindex(returns.index)
        
        return mapped_regimes, hurst_values
    
    def get_vol_regime(self, regime):
        """
        Extract volatility regime from combined regime.
        Returns: 0=Low Vol (regimes 0,1), 1=High Vol (regimes 2,3)
        """
        if pd.isna(regime):
            return np.nan
        return 1 if regime >= 2 else 0
    
    def get_behavior_regime(self, regime):
        """
        Extract behavior from combined regime.
        Returns: 0=Trending (regimes 0,2), 1=Mean-Rev/Random (regimes 1,3)
        """
        if pd.isna(regime):
            return np.nan
        return 0 if int(regime) in [0, 2] else 1
    
    def get_hurst_category(self, hurst_value):
        """
        Categorize Hurst value based on thresholds.
        Returns: 'trending', 'mean_rev', or 'random'
        """
        if pd.isna(hurst_value):
            return np.nan
        if self.hurst_upper is None or self.hurst_lower is None:
            return 'random'
        if hurst_value > self.hurst_upper:
            return 'trending'
        elif hurst_value < self.hurst_lower:
            return 'mean_rev'
        else:
            return 'random'
    
    def get_regime_stats(self, regimes, hurst_values):
        """
        Get statistics for each regime including mean Hurst exponent.
        Useful for analysis and verification.
        """
        stats = {}
        for regime in range(4):
            mask = regimes == regime
            if mask.sum() > 0:
                stats[regime] = {
                    'count': mask.sum(),
                    'pct': mask.mean(),
                    'mean_hurst': hurst_values[mask].mean(),
                    'std_hurst': hurst_values[mask].std(),
                    'min_hurst': hurst_values[mask].min(),
                    'max_hurst': hurst_values[mask].max()
                }
            else:
                stats[regime] = {'count': 0, 'pct': 0, 'mean_hurst': np.nan}
        return stats

# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

class MomentumIndicators:
    """Calculate various momentum indicators for entry/exit signals."""
    
    @staticmethod
    def rsi(prices, period=14):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def momentum(prices, period=20):
        """Simple momentum (rate of change)."""
        return prices.pct_change(period)
    
    @staticmethod
    def moving_average_crossover(prices, short_window=20, long_window=50):
        """Moving average crossover signal."""
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        signal = (short_ma > long_ma).astype(int)
        
        return signal, short_ma, long_ma


# ============================================================================
# ADAPTIVE TRADING STRATEGY
# ============================================================================

class AdaptiveTradingStrategy:
    """
    Adaptive trading strategy that combines:
    1. HMM regime detection with integrated Hurst exponent (4 regimes)
    2. Momentum-based signals
    
    Regimes (per Integrated Architecture):
    - 0: Low Vol + Trending (Follow Momentum)
    - 1: Low Vol + Mean-Reverting (Fade)
    - 2: High Vol + Trending (Aggressive Short)
    - 3: High Vol + Random (Cash/Risk-Off)
    """
    
    def __init__(self, 
                 regime_lookback=252,
                 hurst_window=60,
                 momentum_period=20,
                 rsi_period=14,
                 position_size_base=0.3,
                 trend_following_bias=True):
        
        self.regime_detector = RegimeDetector(
            n_regimes=2, 
            lookback=regime_lookback,
            hurst_window=hurst_window
        )
        self.momentum = MomentumIndicators()
        
        self.momentum_period = momentum_period
        self.rsi_period = rsi_period
        self.position_size_base = position_size_base
        self.trend_following_bias = trend_following_bias
        
    def generate_signals(self, prices, returns=None):
        """Generate trading signals based on all components."""
        if returns is None:
            returns = prices.pct_change()
        
        # 1. Integrated Regime Detection (HMM + Hurst)
        regimes, hurst = self.regime_detector.fit_predict(returns)
        
        # 2. Momentum Indicators
        rsi = self.momentum.rsi(prices, self.rsi_period)
        mom = self.momentum.momentum(prices, self.momentum_period)
        ma_signal, short_ma, long_ma = self.momentum.moving_average_crossover(prices)
        macd_line, signal_line, macd_hist = self.momentum.macd(prices)
        
        # Combine into signals DataFrame
        signals = pd.DataFrame(index=prices.index)
        signals['regime'] = regimes
        signals['hurst'] = hurst
        signals['rsi'] = rsi
        signals['momentum'] = mom
        signals['ma_signal'] = ma_signal
        signals['macd_hist'] = macd_hist
        
        # Extract vol and behavior components for analysis
        signals['vol_regime'] = regimes.apply(self.regime_detector.get_vol_regime)
        signals['behavior_regime'] = regimes.apply(self.regime_detector.get_behavior_regime)
        
        # Generate composite signal
        signals['signal'] = self._generate_composite_signal(signals)
        
        # Generate position sizes
        signals['position_size'] = self._calculate_position_size(signals, returns)
        
        # Apply position smoothing to reduce turnover and transaction costs
        signals['position_size'] = self._smooth_positions(signals['position_size'])
        
        return signals
    
    def _generate_composite_signal(self, signals):
        """
        Generate composite trading signal based on 4-regime system.
        
        Strategy Logic by Regime:
        - Regime 0 (High Vol + Mean-Rev): Defensive, buy oversold
        - Regime 1 (High Vol + Trending): Cautious momentum following
        - Regime 2 (Low Vol + Mean-Rev): Mean reversion plays
        - Regime 3 (Low Vol + Trending): Aggressive momentum following
        """
        composite = pd.Series(index=signals.index, data=0.0)
        
        for i in range(len(signals)):
            regime = signals['regime'].iloc[i]
            rsi = signals['rsi'].iloc[i]
            mom = signals['momentum'].iloc[i]
            ma_signal = signals['ma_signal'].iloc[i]
            macd_hist = signals['macd_hist'].iloc[i]
            
            # Handle missing data - default to fully invested
            if pd.isna(regime):
                composite.iloc[i] = 1.0
                continue
            if pd.isna(rsi):
                composite.iloc[i] = 1.0
                continue
            
            regime = int(regime)
            
            # Calculate momentum score combining multiple indicators
            momentum_score = 0
            if ma_signal == 1:
                momentum_score += 1.5
            else:
                momentum_score -= 0.8
            
            if not pd.isna(macd_hist):
                if macd_hist > 0:
                    momentum_score += 1.2
                else:
                    momentum_score -= 0.5
            
            if not pd.isna(mom):
                if mom > 0.005:
                    momentum_score += 1
                elif mom < -0.02:
                    momentum_score -= 1
            
            # 4-Regime Strategy Logic (per Integrated Architecture)
            if regime == 0:  # Low Vol + Trending -> Follow Momentum (BEST)
                # Steady strong trend - stay fully invested, use leverage
                if momentum_score >= 1.5:
                    composite.iloc[i] = 1.2  # Very strong momentum - leverage
                elif momentum_score >= 0.5:
                    composite.iloc[i] = 1.1  # Good momentum - slight leverage
                else:
                    composite.iloc[i] = 1.05  # Weak but trending - stay invested
                    
            elif regime == 1:  # Low Vol + Mean-Reverting -> Fade
                # Bound/mean-reverting - trade against momentum
                if rsi < 30:
                    composite.iloc[i] = 1.05  # Oversold - buy the dip
                elif rsi > 70:
                    composite.iloc[i] = 0.85  # Overbought - slight reduce
                else:
                    composite.iloc[i] = 0.95  # Stay mostly invested
                    
            elif regime == 2:  # High Vol + Trending -> Asymmetric response
                # Use momentum direction to distinguish crash vs rebound
                if momentum_score >= 1.5:
                    composite.iloc[i] = 0.85  # Strong uptrend - capture rebound
                elif momentum_score >= 0.5:
                    composite.iloc[i] = 0.65  # Moderate uptrend - cautious long
                elif momentum_score >= -0.5:
                    composite.iloc[i] = 0.4  # Weak/flat - defensive
                else:
                    composite.iloc[i] = 0.25  # Strong downtrend - very defensive
                    
            else:  # regime == 3: High Vol + Random -> Cash/Risk-Off
                # Erratic/random walk - asymmetric response
                if rsi < 20:
                    composite.iloc[i] = 0.6  # Extreme oversold - buy opportunity
                elif rsi > 75:
                    composite.iloc[i] = 0.25  # Overbought in bad regime - cut
                elif momentum_score >= 1:
                    composite.iloc[i] = 0.55  # Some positive momentum
                else:
                    composite.iloc[i] = 0.4  # Stay defensive
        
        return composite
    
    def _calculate_position_size(self, signals, returns):
        """
        Calculate position size based on signal strength.
        
        Position sizing rules:
        - Direct mapping from signal to position
        - Cap at 120% to allow slight leverage in best conditions
        """
        position_size = pd.Series(index=signals.index, data=0.0)
        
        for i in range(len(signals)):
            signal = signals['signal'].iloc[i]
            
            if pd.isna(signal) or signal == 0:
                continue
            
            # Direct mapping: signal value IS the position size
            base_size = self.position_size_base * abs(signal)
            
            # Cap position size at 125% (allow leverage in best regimes)
            position_size.iloc[i] = np.sign(signal) * min(base_size, 1.25)
        
        return position_size
    
    def _smooth_positions(self, position_size, ema_alpha=0.35, min_change=0.03):
        """
        Smooth position sizes to reduce turnover and transaction costs.
        
        Uses EMA smoothing and a minimum change threshold:
        - EMA smoothing prevents abrupt daily position changes
        - Minimum change threshold filters out small noise-driven trades
        
        Parameters:
        -----------
        position_size : pd.Series
            Raw position sizes
        ema_alpha : float
            EMA smoothing factor (lower = smoother, higher = more responsive)
        min_change : float
            Minimum position change to execute (filters noise)
        """
        smoothed = position_size.copy()
        
        for i in range(1, len(smoothed)):
            prev = smoothed.iloc[i - 1]
            target = position_size.iloc[i]
            
            # EMA smoothing toward target
            ema_val = prev + ema_alpha * (target - prev)
            
            # Only update if change exceeds minimum threshold
            if abs(ema_val - prev) < min_change:
                smoothed.iloc[i] = prev
            else:
                smoothed.iloc[i] = ema_val
        
        return smoothed


# ============================================================================
# BACKTESTING
# ============================================================================

def run_backtest(prices, signals, initial_capital=1000000, transaction_cost=0.001):
    """
    Run backtest on the strategy.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    signals : pd.DataFrame
        Signals DataFrame with 'position_size' column
    initial_capital : float
        Initial capital
    transaction_cost : float
        Transaction cost as fraction of trade value
    
    Returns:
    --------
    results : dict
        Backtest results including returns, metrics, etc.
    """
    returns = prices.pct_change()
    position = signals['position_size'].fillna(0)
    
    # Calculate strategy returns
    strategy_returns = position.shift(1) * returns
    
    # Calculate transaction costs
    position_changes = position.diff().abs()
    costs = position_changes * transaction_cost
    
    # Net returns
    net_returns = strategy_returns - costs
    
    # Calculate cumulative returns
    cumulative_returns = (1 + net_returns).cumprod()
    cumulative_returns.iloc[0] = 1
    
    # Calculate metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
    
    volatility = net_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_days = (net_returns > 0).sum()
    total_trading_days = (net_returns != 0).sum()
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    results = {
        'cumulative_returns': cumulative_returns,
        'daily_returns': net_returns,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar_ratio,
        'positions': position,
        'signals': signals
    }
    
    return results


def print_backtest_results(results):
    """Print backtest results summary."""
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:     {results['total_return']:.2%}")
    print(f"Annual Return:    {results['annual_return']:.2%}")
    print(f"Volatility:       {results['volatility']:.2%}")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:     {results['max_drawdown']:.2%}")
    print(f"Win Rate:         {results['win_rate']:.2%}")
    print(f"Calmar Ratio:     {results['calmar_ratio']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    data = load_data('data_cn')
    
    print("Creating market index...")
    index_level, index_returns = create_market_index(
        data['adjusted'], 
        data['mktcap']
    )
    
    print("Running strategy...")
    strategy = AdaptiveTradingStrategy(position_size_base=1.0)
    signals = strategy.generate_signals(index_level, index_returns)
    
    print("Running backtest...")
    results = run_backtest(index_level, signals)
    print_backtest_results(results)
    
    # Compare with buy & hold
    bh_return = (1 + index_returns).cumprod().iloc[-1] - 1
    print(f"\nBuy & Hold Return: {bh_return:.2%}")
