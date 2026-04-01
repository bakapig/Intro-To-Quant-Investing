# Exploratory Data Analysis Report
## Chinese A-Share Universe (CSI 300 Constituents, 2006–2025)

**Dataset:** ~300 stocks per period (CSI 300 constituents), daily frequency, 2006-01 to 2025-03  
**Data fields:** OHLC prices, adjusted close, dollar volume, market cap, P/B, EPS, analyst recommendations, GICS sector codes, universe membership flags

---

## Table of Contents

1. [Return Distributions & Stylized Facts](#1-return-distributions--stylized-facts)
2. [Cross-Sectional Factor Analysis](#2-cross-sectional-factor-analysis)
3. [Correlation Structure](#3-correlation-structure)
4. [Liquidity Analysis](#4-liquidity-analysis)
5. [Volatility Deep-Dive](#5-volatility-deep-dive)
6. [Regime Characterization](#6-regime-characterization)
7. [Survivorship & Universe Dynamics](#7-survivorship--universe-dynamics)
8. [Key Takeaways for Strategy Design](#8-key-takeaways-for-strategy-design)

---

## 1. Return Distributions & Stylized Facts

**Objective:** Characterize the statistical properties of market-cap-weighted index returns and test departures from normality.

### Summary Statistics

| Frequency | N | Mean (%) | Std (%) | Skewness | Excess Kurtosis | Min (%) | Max (%) | JB p-value |
|-----------|-------|----------|---------|----------|-----------------|---------|---------|------------|
| Daily | 4,859 | 0.09 | 1.59 | −0.18 | **4.74** | −9.63 | 9.97 | 0.000 |
| Weekly | 1,044 | 0.41 | 3.39 | 0.25 | **3.35** | −14.46 | 17.75 | 0.000 |
| Monthly | 240 | 1.86 | 7.95 | −0.01 | **1.98** | −28.81 | 30.98 | 0.000 |

### Key Findings

- **Fat tails are persistent across all frequencies.** Daily excess kurtosis of 4.74 means extreme moves are ~5× more frequent than a normal distribution would predict. The Jarque-Bera test rejects normality at all frequencies (p ≈ 0).
- **Slight negative skew at daily frequency** (−0.18) flips to positive at weekly (+0.25), suggesting intraday mean-reversion dampens left-tail events over multi-day horizons.
- **Volatility clustering is clearly present.** Autocorrelation of absolute returns remains significantly positive out to 100+ lags, while raw return autocorrelation decays to near-zero after lag 1 — the classic signature of GARCH-type dynamics.
- **Rolling kurtosis is time-varying**, spiking during the 2008 GFC, 2015 bubble burst, and 2020 COVID crash, confirming that tail risk is regime-dependent.

### Figures

| Figure | Description |
|--------|-------------|
| `01_return_histograms.png` | Daily, weekly, monthly return histograms with normal overlay |
| `02_qq_plots.png` | QQ plots — departure from the diagonal line shows fat tails |
| `03_summary_statistics.csv` | Full summary table |
| `04_autocorrelation.png` | ACF of returns vs ACF of |returns| (volatility clustering) |
| `05_rolling_kurtosis.png` | 252-day rolling excess kurtosis over time |
| `06_cross_sectional_annual.png` | Box plots of individual stock annual returns by year |

---

## 2. Cross-Sectional Factor Analysis

**Objective:** Test whether classic equity factors (value, size, momentum, liquidity, analyst sentiment, earnings yield) are priced in this universe by constructing monthly-rebalanced quintile portfolios and computing long/short spreads.

### Factor Summary (Full Sample)

| Factor | Q1 Ann. Ret. | Q5 Ann. Ret. | L/S Ann. Ret. | L/S Sharpe |
|-----------------|-------------|-------------|---------------|------------|
| **Earnings Yield** | 19.0% | 11.4% | **+7.7%** | **0.43** |
| Analyst Rec | 17.3% | 12.5% | +4.8% | 0.37 |
| Momentum (12-1) | 13.8% | 6.7% | +7.1% | 0.35 |
| Value (P/B) | 17.2% | 11.1% | +6.1% | 0.26 |
| Liquidity (DV) | 16.6% | 12.2% | +4.3% | 0.24 |
| Size (Mkt Cap) | 16.7% | 13.6% | +3.1% | 0.15 |

### Key Findings

- **Earnings yield is the strongest single factor** (Sharpe 0.43), consistent with a value premium driven by actual profitability rather than book value alone.
- **Analyst recommendations carry alpha** (Sharpe 0.37) — highest-rated stocks outperform lowest-rated by ~4.8% p.a. This is notable since analyst coverage in China is heavily sell-side driven.
- **Momentum works** (Sharpe 0.35, 7.1% L/S return) but has episodic crashes (see regime analysis below).
- **Size effect is weak** (Sharpe 0.15) — within the CSI 300 universe, all stocks are already large-cap, leaving little room for a small-cap premium.
- **Liquidity premium exists** (Sharpe 0.24) — less liquid stocks earn higher returns, consistent with a compensation for illiquidity risk.

### Figures

| Figure | Description |
|--------|-------------|
| `07_factor_value_pb.png` | Value quintile cumulative returns + L/S spread |
| `08_factor_size.png` | Size quintile returns |
| `09_factor_momentum.png` | Momentum (12-1 month) quintile returns |
| `10_factor_liquidity.png` | Dollar volume quintile returns |
| `11_factor_analyst.png` | Analyst recommendation quintile returns |
| `12_factor_earnings_yield.png` | Earnings yield quintile returns |
| `13_factor_summary.csv` | Full factor comparison table |

---

## 3. Correlation Structure

**Objective:** Understand the dimensionality and co-movement structure of the cross-section.

### Key Findings

- **PC1 alone explains 42.8% of return variance** — the market factor dominates. Only 3 PCs are needed for 50%, and 22 PCs for 90%. This is a highly concentrated risk structure compared to developed markets (where PC1 typically explains ~25–30%).
- **Rolling average pairwise correlation varies substantially** over time, peaking during stress episodes (2008, 2015, 2020) and declining during calm periods — correlation rises when diversification is most needed.
- **All sector-pair correlations are positive**, ranging from ~0.4 to ~0.8. GICS sectors provide limited diversification within Chinese A-shares. Financials and Industrials are most correlated; Healthcare and Consumer Staples are least correlated with the market.

### Implications for Portfolio Construction

The high concentration of variance in PC1 means that:
1. Market timing / regime detection matters more than stock selection on a risk-adjusted basis
2. Sector-neutral construction has limited benefit (sectors themselves are highly correlated)
3. A statistical risk model needs at least ~20 factors to capture the cross-section adequately

### Figures

| Figure | Description |
|--------|-------------|
| `14_rolling_avg_correlation.png` | Rolling 60-week average pairwise correlation |
| `15_sector_correlation_heatmap.png` | GICS sector cross-correlation heatmap |
| `16_eigenvalue_pca.png` | Scree plot + cumulative variance explained |

---

## 4. Liquidity Analysis

**Objective:** Characterize market liquidity dynamics and their relationship to returns and drawdowns.

### Liquidity by Market Regime

| Regime | Avg Daily Turnover (median) | Avg Amihud (×10⁶) | Avg Agg. DV (¥ bn) | # Days |
|--------|----------------------------|-------------------|--------------------|--------|
| Normal (DD > −5%) | 5,414 | 0.10 | 26.3 | 836 |
| Correction (−20% to −5%) | 5,152 | 0.05 | 35.6 | 1,160 |
| Bear (DD < −20%) | 4,052 | 0.11 | 19.0 | 2,863 |

### Key Findings

- **Liquidity dries up in bear markets.** Aggregate dollar volume falls 27% in bear regimes (¥19bn vs ¥26bn in normal). This is a critical constraint for any strategy that needs to trade during stress.
- **Corrections actually see _higher_ aggregate volume** (¥35.6bn) — this captures the panic selling / capitulation phase before a sustained bear market, when turnover spikes.
- **Low-turnover stocks outperform high-turnover stocks**, consistent with an illiquidity premium. This is exploitable but requires careful execution cost modeling.
- **Amihud illiquidity ratio spikes during market stress**, confirming that price impact costs increase precisely when portfolio rebalancing is most needed.

### Figures

| Figure | Description |
|--------|-------------|
| `17_amihud_illiquidity.png` | Cross-sectional Amihud ratio vs market index |
| `18_turnover_vs_returns.png` | Turnover quintile cumulative returns |
| `19_liquidity_drawdowns.png` | Drawdown chart with dollar volume colored by regime |
| `20_liquidity_regime_summary.csv` | Summary statistics by drawdown regime |

---

## 5. Volatility Deep-Dive

**Objective:** Decompose volatility using OHLC data and test for asymmetric volatility (leverage effect).

### Summary

| Metric | Value |
|--------|-------|
| Avg realized vol (20d, annualized) | 22.4% |
| Avg Parkinson vol (20d, annualized) | 32.0% |
| Parkinson / Realized ratio | **1.56** |
| Avg 5d realized vol | 21.2% |
| Avg 60d realized vol | 23.0% |
| Mean leverage correlation | **−0.027** |
| Leverage slope (pp vol per 1% return) | 0.064 |

### Key Findings

- **Parkinson volatility (using high-low range) is 56% higher than close-to-close realized vol.** This is unusually large (typical ratio in developed markets is ~1.2–1.3) and indicates significant intraday price movement that close prices do not capture. This has implications for risk management: close-to-close VaR underestimates true intraday risk.
- **The leverage effect is essentially absent** (correlation = −0.027). Unlike developed markets where negative returns amplify future volatility (due to financial leverage increasing), Chinese A-shares show near-zero asymmetry. This is consistent with a retail-investor-dominated market where leverage mechanisms differ (margin trading, structured products are less prevalent).
- **Volatility term structure inversion (5d vol > 60d vol) is a reliable stress indicator.** When short-term vol exceeds long-term vol, the market is in or entering a crisis regime.

### Figures

| Figure | Description |
|--------|-------------|
| `21_realized_vs_parkinson_vol.png` | Close-to-close vs Parkinson volatility overlay |
| `22_vol_term_structure.png` | 5d / 20d / 60d realized vol + term spread |
| `23_leverage_effect.png` | Return vs forward vol scatter + rolling leverage correlation |
| `24_volatility_summary.csv` | Summary statistics |

---

## 6. Regime Characterization

**Objective:** Evaluate HMM-detected regimes (Trending, Mean-Reverting, Random Walk) against market conditions and factor performance.

### Regime Comparison

| Regime | % of Time | Avg Drawdown | Avg 20d Vol (ann.) | Avg CS Dispersion | Avg Daily Return |
|---------------|-----------|-------------|-------------------|------------------|-----------------|
| Trending | 28.5% | −13.4% | 14.6% | 2.03% | +0.022% |
| Mean Reverting | 24.3% | −15.2% | 14.7% | 2.02% | +0.034% |
| **Random Walk** | **47.2%** | −14.4% | **24.8%** | **2.61%** | +0.116% |

### Factor Performance by Regime (L/S Sharpe)

| Factor | Trending | Mean Reverting | Random Walk |
|-----------------|----------|----------------|-------------|
| Value (P/B) | **0.64** | 0.41 | 0.04 |
| Size | −0.45 | −0.86 | **0.83** |
| Momentum | 0.65 | **1.46** | −0.17 |
| Liquidity | −0.59 | 0.55 | 0.58 |

### Key Findings

- **Random walk is the dominant regime** (47% of time) and has the highest volatility (24.8% ann.) and cross-sectional dispersion. It corresponds to high-uncertainty periods where neither trend-following nor mean-reversion works well.
- **Momentum thrives in mean-reverting regimes** (Sharpe 1.46!) but fails in random walk (−0.17). This is counterintuitive — it suggests momentum captures persistence of _cross-sectional_ relative performance even when the _market_ is mean-reverting.
- **Value works best in trending regimes** (Sharpe 0.64) and collapses in random walk (0.04).
- **Size factor flips sign across regimes**: small caps win in random walk (0.83 Sharpe) but lose in trending (−0.45) and mean-reverting (−0.86). This is a strong argument for regime-conditional factor allocation.
- **The transition matrix shows very short regime durations** for trending and mean-reverting (avg 1.4–1.6 days) — these are transient states within a dominant random walk backdrop.

### Figures

| Figure | Description |
|--------|-------------|
| `25_regime_context.png` | 4-panel: index, drawdown, volatility, dispersion with regime overlay |
| `25_regime_context_summary.csv` | Summary by regime |
| `26_transition_matrix.png` | Transition count and probability heatmaps |
| `27_factor_by_regime.png` / `.csv` | Factor L/S Sharpe by regime |

---

## 7. Survivorship & Universe Dynamics

**Objective:** Quantify how universe membership changes over time and measure the impact of survivorship bias.

### Universe Turnover

The CSI 300 universe is maintained at exactly **300 stocks** with annual turnover averaging ~30% (40–50 stocks rotate each year). Only **34 stocks survived the entire 2006–2025 period**.

| Year | Universe | Entries | Exits | Turnover |
|------|----------|---------|-------|----------|
| 2007 | 300 | 49 | 49 | 32.7% |
| 2008 | 300 | 62 | 62 | 41.3% |
| 2015 | 300 | 48 | 48 | 32.0% |
| 2020 | 300 | 35 | 35 | 23.3% |
| 2024 | 300 | 23 | 23 | 15.3% |

### Entry vs. Exit Stock Characteristics

Stocks entering the index tend to have **strong recent performance** (selection bias), while exiting stocks typically had **poor prior-year returns**. For example:
- In 2019, entrants returned +46.6% in their entry year vs stayers at +34.7%; exiters had returned −34.3% the year before removal.
- In 2008 (bear market), the pattern reversed: entrants still fell −67.9% alongside the market.

### Survivorship Bias

| Index | Ann. Return | Ann. Vol | Sharpe | Total Growth (×) |
|-------|-------------|----------|--------|-------------------|
| Full Universe (EW) | 15.4% | 27.9% | 0.55 | 9.1× |
| **Survivors Only (EW)** | **21.8%** | **25.8%** | **0.85** | **34.9×** |
| Full Universe (MCW) | 19.5% | 23.4% | 0.83 | 25.3× |
| **Survivors Only (MCW)** | **22.3%** | **25.3%** | **0.88** | **39.5×** |

**Survivorship bias overstates equal-weighted returns by 6.4% annually** (15.4% → 21.8%) and the Sharpe ratio from 0.55 to 0.85. Even market-cap-weighted, survivors outperform by ~2.8% annually. This is a critical calibration: any backtest run on current constituents only will be materially overstated.

### Figures

| Figure | Description |
|--------|-------------|
| `28_universe_dynamics.png` / `.csv` | Universe size and entry/exit counts by year |
| `29_entry_exit_returns.png` / `.csv` | Return characteristics: entrants vs stayers vs exiters |
| `30_survivorship_bias.png` / `.csv` | Full universe vs survivors-only cumulative index |

---

## 8. Key Takeaways for Strategy Design

### What Works
1. **Earnings yield is the strongest standalone factor** (Sharpe 0.43) — use EPS/Price as a primary signal
2. **Momentum is strong but regime-dependent** — combine with regime detection to avoid drawdowns in random walk periods
3. **Regime-conditional factor allocation** is promising: value in trending, momentum in mean-reverting, size in random walk
4. **Illiquidity premium is real** but needs execution cost modeling

### What to Watch
1. **Survivorship bias is severe** (+6.4% annual overstatement). Always use point-in-time universe membership for backtesting
2. **Fat tails matter** — daily kurtosis of 4.74 means normal-distribution-based risk models will severely underestimate drawdown risk
3. **Parkinson vol >> realized vol** — intraday risk is ~56% higher than close-to-close measures suggest
4. **Concentration risk** — PC1 explains 43% of variance; undiversified market exposure dwarfs factor returns
5. **No leverage effect** — volatility targeting strategies designed for developed markets (which exploit asymmetric vol) may not work here

### Data Limitations
- Universe is CSI 300 (large-cap only) — small/mid-cap factors are untested
- No transaction cost data — liquidity factor returns may erode after costs
- Analyst recommendations may have look-ahead bias if data timestamps are not point-in-time
- EPS data had duplicate dates requiring deduplication; accuracy should be cross-validated
