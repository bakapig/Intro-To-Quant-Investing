# Regime Change

> Reference: Chen & Tsang, *Detecting Regime Change in Computational Finance* (2021), Chapter 2.1  
> Project paper: `JC/FE5214_Group_Project.tex` §Introduction

## What Is Regime Change?

In computational finance, **regime change** refers to a significant shift in the **collective behaviour** of market participants, reflected in price dynamics. It is not ordinary noise — it is a structural change that persists for a period of time, effectively a new “rule set” for the market.

Typical examples:

- **Steady / Normal regime**: low volatility, economic growth, relatively stable asset correlations
- **Turbulent / Abnormal regime**: high volatility, economic contraction, shifts in mean, volatility, and correlation structure

Macro events such as the 2008 financial crisis, the Brexit referendum, and the 2020 pandemic shock are often treated as regime-change episodes.

## Why It Matters

Traditional models often assume stationary returns and constant parameters. Empirically, financial data commonly exhibits:

- Volatility clustering
- Fat tails, skewness, time-varying correlations
- Structural breaks

A strategy calibrated under one regime may fail under another. **Detecting and tracking regimes** is therefore directly useful for risk management, asset allocation, and algorithmic trading.

## Two Common Market Regimes

Historical data often supports a simple two-regime view (Ang & Timmermann; Hamilton):

| Regime | Characteristics | Common labels |
|--------|-----------------|---------------|
| Low volatility | Stable growth, lower vol | Bull / Normal / Steady |
| High volatility | Panic selling, elevated vol | Bear / Crash / Abnormal |

In this project, after HMM fitting, states are usually ordered by **estimated variance** $\sigma_k^{2}$ from low to high. State 0 maps to the low-volatility regime and state $N - 1$ to the high-volatility regime, addressing the label-switching problem.

## Detection Methods Overview

### 1. Markov / Regime-Switching Models

Hamilton (1989) introduced the **Markov switching model**, a classic framework:

- An unobservable state $S_t$ drives the data-generating process
- Transitions follow a **first-order Markov chain**: the next state depends only on the current state
- Well suited to discrete, occasional shifts (e.g. recession ↔ expansion)

### 2. Time Series Methods

The conventional approach samples at fixed intervals (daily, weekly, or monthly closes), monitors return statistics (mean, volatility, etc.), and applies regime-switching or change-point detection.

**Limitation**: the sampling interval is arbitrary and may discard important intra-period information; different researchers may use incompatible time scales.

### 3. Directional Change (DC) Approach

The book proposes an alternative: **event-based sampling** instead of physical-time sampling. Price turning points are recorded only when a move exceeds a preset threshold $\theta$ (see book §2.2). Full details: [directional-change.md](./directional-change.md).

DC indicators (TMV, T, R) measure trend magnitude, completion time, and return per unit time, complementing the time-series view. The book describes this as “seeing with two eyes” rather than one.

## Drivers of Regime Change

Regime shifts may be caused by:

- **External shocks**: 1973 oil crisis, Lehman bankruptcy in 2008, etc.
- **Policy and expectations**: shifts in monetary policy stance
- **Endogenous buildup**: business cycles, collective shifts in investor behaviour

## Link to This Project

This project uses a layered framework of **Gaussian HMM + Hurst exponent + momentum signals**:

1. **Macro layer**: HMM identifies low / medium / high volatility regimes (`main/hmm_strategy.py`, `ZY/regime_hmm.py`)
2. **Micro layer**: rolling Hurst distinguishes trending vs mean-reverting behaviour
3. **Tactical layer**: momentum signals are gated by regime to produce trade direction

This aligns with the book’s idea of inferring hidden states with machine learning while observing the market through statistical features — though our implementation focuses on cross-sectional equity portfolios and sector-neutral return inputs.

## Further Reading

- Hamilton (1989, 1994) — Markov switching
- Ang & Timmermann — regime surveys
- Book Chapters 3–5 — DC detection, normal/abnormal classification, real-time tracking
