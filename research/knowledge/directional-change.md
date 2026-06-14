# Directional Change (DC)

> Reference: Chen & Tsang (2021) §2.2, Appendix A; Tsang et al. [47]  
> Related: [regime-change.md](./regime-change.md), [naive-bayes-classifier.md](./naive-bayes-classifier.md)

## What Is Directional Change?

**Directional Change (DC)** is an **event-based** way to summarise financial prices. Instead of sampling at fixed clock times (daily close, hourly bar, etc.), you record prices only when the market makes a **significant** move.

The idea has roots in technical analysis (the **Zig Zag indicator**) and was formalised for scientific use by Guillaume et al. and later Olsen/Tsang's research group. Chen & Tsang's book uses DC as the primary data representation for regime change detection.

### Time Series vs DC

| | Time Series | Directional Change |
|---|-------------|------------------|
| **When to sample** | Fixed intervals (1 day, 1 hour, …) | When price reverses by threshold $\theta$ |
| **Time spacing** | Regular (calendar-driven) | Irregular (data-driven) |
| **What gets captured** | Everything at each tick of the clock | Peaks, troughs, and significant trends only |
| **Noise** | All small moves included | Small moves filtered out by $\theta$ |
| **Best for** | Standard econometrics, daily equity studies | High-frequency data, regime monitoring, event-driven analysis |

**Key insight from the book:** financial time does not flow uniformly — a quiet lunch hour and a frantic open are both "one hour" on the clock, but not equally meaningful. DC lets **the data decide** when a observation is worth recording.

---

## Core Idea in One Picture

A price path is split into alternating **uptrends** and **downtrends**. Each full trend = one **DC Event** + one **Overshoot (OS) Event**.

```
Price
  │         C ●  ← EXT (peak); next downtrend starts
  │        ╱ ╲
  │       ╱   ╲
  │      ╱     ╲
  │  B ●       ╲    B = DCC (directional change confirmation)
  │ ╱           ╲
  │╱             ╲
  ● A             ╲
  EXT (trough)     ● D  ← next DCC

  |←── DC Event ──→|←── OS Event ──→|
  |←──────── one uptrend ──────────→|
```

- **A** — Extreme point (EXT): start of uptrend (trough)
- **B** — Directional Change Confirmation (DCC): price has fallen $\theta$% from the running high → uptrend's DC leg ends
- **B → C** — Overshoot (OS): continuation in the same direction until the reversal is confirmed at C
- **C** — Next EXT (peak); downtrend begins

---

## Key Terminology

| Term | Abbrev. | Meaning |
|------|---------|---------|
| **Threshold** | $\theta$ | Minimum % price move you treat as "significant" (chosen by the analyst) |
| **Extreme point** | EXT | Confirmed peak or trough — retrospectively labelled when the next DC fires |
| **DC Confirmation** | DCC | The price point where the reversal reaches threshold $\theta$ |
| **DC Event** | DC | The reversal leg itself (from EXT to DCC) |
| **Overshoot Event** | OS | Price continuation from DCC to the next EXT |
| **Upward / Downward Run** | — | Full trend = DC Event + following OS Event |
| **Last High / Last Low** | — | Running extremum updated tick-by-tick until DC confirms |

### The Threshold $\theta$

$\theta$ is a **percentage** set by the observer (e.g. 0.3%, 0.6%, 3%). It controls the resolution:

| Smaller $\theta$ | Larger $\theta$ |
|------------------|-----------------|
| More events, more detail | Fewer events, smoother picture |
| Sensitive to noise | Focus on large moves only |
| Good for high-frequency / short horizons | Good for macro trends |

Different $\theta$ values reveal different **scales** of market structure. Glattfelder et al. showed that many DC properties follow **scaling laws** across thresholds — a fractal-like regularity not visible in fixed-interval time series.

---

## Formal Definition (Appendix A)

At any moment the market is in either an **Upward Run** or a **Downward Run**.

### Updating extremes

- **Downward Run:** track **Last Low** = min(current price, Last Low)
- **Upward Run:** track **Last High** = max(current price, Last High)

### Confirming a directional change

**End of Downward Run → start of Upward Run:**  
When price $P$ rises more than $\theta$ above Last Low:

$$
\frac{P - P_{\text{low}}}{P_{\text{low}}} \geq \theta
$$

This is an **Upward DC Event**. Last Low becomes a confirmed **Trough** (EXT). $P$ is the **Upward DCC**.

**End of Upward Run → start of Downward Run:**  
When price $P'$ falls more than $\theta$ below Last High:

$$
\frac{P_{\text{high}} - P'}{P_{\text{high}}} \geq \theta
$$

This is a **Downward DC Event**. Last High becomes a confirmed **Peak** (EXT). $P'$ is the **Downward DCC**.

### Overshoot events

- **Downward OS:** price path from Downward DCC to the next Trough
- **Upward OS:** price path from Upward DCC to the next Peak

### Event cycle

The market cycles through four event types:

```
… → Downward DC → Downward OS → Upward DC → Upward OS → Downward DC → …
```

### Ex-ante vs post-ante (important for trading)

| Moment | What you know |
|--------|---------------|
| At a peak/trough (EXT) | You do **not** yet know the trend has ended |
| At DCC | The DC Event is **confirmed**; the EXT is retrospectively labelled |
| During OS | You know OS has started, but not when it ends |
| At next DCC | Previous OS ends; new DC confirmed |

DC is inherently **confirmatory** — you always know a significant reversal **after** it has moved by $\theta$. This is why the book pairs DC with **unfinished trend tracking** (Chapter 5) for earlier regime warnings.

---

## DC Indicators: TMV, T, and R

Volatility is not directly observable from prices. Tsang et al. [47] proposed DC-based indicators to profile each completed trend. The book uses three core indicators for regime change research.

All three are computed **per trend** (from one EXT to the next EXT), i.e. one DC Event + one OS Event combined.

---

### 1. Total Price Movement (TMV)

**What it measures:** Total absolute percentage price change over the full trend, **normalised by** $\theta$.

$$
\mathrm{TMV}(n) = \frac{\left| P_{\text{EXT}}(n) - P_{\text{EXT}}(n-1) \right|}{P_{\text{EXT}}(n-1) \times \theta}
$$

| Interpretation | |
|----------------|--|
| $\mathrm{TMV} \approx 1$ | Trend is barely larger than the threshold — minimal move |
| $\mathrm{TMV} \gg 1$ | Large swing relative to $\theta$ — strong trend / high activity |
| Rising TMV across trends | Market becoming more volatile or directional |

**Example:** $\theta = 0.5\%$, price moves from 100 to 103 (3% total):

$$
\mathrm{TMV} = \frac{3}{0.5} = 6
$$

The trend was six times the minimum significant move.

> The book uses **absolute values** of TMV throughout.

---

### 2. Time for Completion (T)

**What it measures:** Physical (clock) time elapsed between two consecutive extreme points.

$$
T(n) = t_{\text{EXT}}(n) - t_{\text{EXT}}(n-1)
$$

| Interpretation | |
|----------------|--|
| Small $T$, large TMV | Fast, violent move (crisis-like) |
| Large $T$, small TMV | Slow, grinding trend |
| Very small $T$ during stress | Rapid regime transition |

Units: seconds, minutes, hours, or days depending on data frequency.

---

### 3. Time-Adjusted Return (R)

**What it measures:** Speed of the trend — absolute return per unit of physical time.

$$
R(n) = \frac{\left| \mathrm{TMV}(n) \right| \times \theta}{T(n)}
$$

Equivalently: absolute percentage price change per unit time.

| Interpretation | |
|----------------|--|
| High $R$ | Price moving fast — panic, rush, liquidity stress |
| Low $R$ | Sluggish drift |
| Spike in $R$ | Often coincides with abnormal / crisis regimes |

> The book uses **absolute values** of $R$ throughout.

---

### How the Three Indicators Work Together

| Indicator | Captures | Regime signal |
|-----------|----------|---------------|
| **TMV** | *How far* price moved | Magnitude of market activity |
| **T** | *How long* it took | Pace and duration of the move |
| **R** | *How fast* per unit time | Intensity / urgency of the move |

**Normal regime (book Ch.4):** TMV and T cluster in a compact region of indicator space — calm, moderate trends.

**Abnormal regime:** TMV and T shift to higher values; $R$ spikes — large, fast, volatile moves dominate.

These features feed into:

- **HMM** (Ch.3–4) — detect regime switches from indicator distributions
- **Naïve Bayes** (Ch.5) — classify current $(\mathrm{TMV}, T)$ pair as normal vs abnormal in real time

---

## Unfinished Trends (Chapter 5)

A completed trend gives TMV and T only **after** the next EXT is confirmed. For **real-time tracking**, the book also monitors **in-progress (unfinished) DC trends**:

- While OS is running, TMV and T are partially observed and updated tick-by-tick
- NBC uses these live values to estimate $P(\text{normal} \mid \mathbf{x})$ and $P(\text{abnormal} \mid \mathbf{x})$

This is how DC-based regime tracking can fire **before** a full trend completes — reducing lag compared to waiting for all EXT labels.

---

## DC vs Time Series for Regime Detection (Chapter 3)

The book compares regime changes detected under DC vs conventional time series on EUR–GBP, GBP–USD, and EUR–USD. Key findings:

| Aspect | DC-based detection | Time-series detection |
|--------|-------------------|----------------------|
| Event timing | Tied to significant price moves | Tied to calendar dates |
| Sensitivity | Catches sharp, fast transitions | May smooth over intra-period moves |
| Information | TMV, T, R add structure | Standard vol / return stats only |
| Conclusion | **Complementary** — "two eyes are better than one" | Still useful as baseline |

Neither view is strictly superior; DC adds information that fixed-interval sampling can miss.

---

## Choosing the Threshold $\theta$

The book experiments with multiple thresholds (e.g. 0.006, 0.009, 0.03 in Ch.6 trading tests). Guidelines:

| Consideration | Guidance |
|---------------|----------|
| Asset volatility | Higher-vol assets need larger $\theta$ to filter noise |
| Data frequency | Tick data → smaller $\theta$; daily data → larger $\theta$ |
| Research goal | Regime detection → moderate $\theta$; HFT → small $\theta$ |
| Robustness | Test sensitivity across 2–3 $\theta$ values |

$\theta$ is not learned from data in the basic framework — it is an **analyst choice**, like choosing a bar size in candlestick charts.

---

## How DC Fits the Full Pipeline

```
Tick / daily prices
        ↓
DC summarisation (threshold θ)
        ↓
Events: DC + OS  →  Indicators: TMV, T, R
        ↓
   ┌────┴────┐
   ▼         ▼
 HMM      Naïve Bayes
(detect)   (track online)
   ↓         ↓
Regime labels → Trading / risk rules (Ch.6)
```

| Book chapter | DC role |
|--------------|---------|
| Ch.2 | Theory, indicators defined |
| Ch.3 | Detect regime changes; compare DC vs time series |
| Ch.4 | Classify normal vs abnormal regimes in indicator space |
| Ch.5 | Track transitions with NBC on $(\mathrm{TMV}, T)$ |
| Ch.6 | Feed tracking signals into JC1/JC2 trading algorithms |

---

## Relation to This Project

The FE5214 codebase currently uses **time-series-based** inputs for HMM (sector-neutral daily returns, rolling Hurst, volatility) rather than a full DC pipeline. The concepts still align:

| Book (DC) | This project (time series) |
|-----------|---------------------------|
| TMV — trend magnitude | Rolling volatility, return dispersion |
| T — time to complete trend | Lookback windows (20d, 60d, 252d) |
| R — speed of move | Short-horizon momentum |
| Threshold $\theta$ | Hurst upper/lower bands, vol state cutoffs |
| Normal vs abnormal indicator space | HMM states ordered by $\sigma_k^2$ |

A natural extension would be to compute TMV, T, R from index or portfolio prices and use them as HMM/NBC features alongside existing signals.

---

## Worked Example (Simplified)

**Setup:** $\theta = 1\%$. Index rises from 4000 to 4200 over 10 days, then falls to 4100 where DC confirms.

| Step | Event | TMV (partial) | T (partial) |
|------|-------|---------------|-------------|
| Day 0 | EXT at 4000 (trough) | — | — |
| Day 10 | DCC at ~4158 (first 1% drop from high 4200) | DC leg ends | — |
| Day 14 | EXT at 4200 (peak confirmed) | $\mathrm{TMV} = \lvert 4200-4000 \rvert / (4000 \times 0.01) = 5$ | $T = 14$ days |
| Day 14+ | $R = 5 \times 0.01 / 14 \approx 0.0036$ per day | | |

If during a crisis the same 5% move happens in 2 days instead of 14, TMV is similar but $R$ is ~7× higher — a strong abnormal-regime signal.

---

## Other DC Indicators (Mentioned in Literature)

| Indicator | Source | Use |
|-----------|--------|-----|
| **SMQ** (Scale of Market Quakes) | Bisig et al. | Probabilistic activity measure; crisis detection |
| **Scaling laws** | Glattfelder et al. | Power-law relationships across $\theta$ scales |
| **Alpha Engine** | Golub et al. | DC-based automated trading |

The book's regime research focuses on **TMV, T, R** as the core trio.

---

## Summary

| Concept | Takeaway |
|---------|----------|
| **DC** | Sample prices at significant reversals, not fixed clock times |
| **$\theta$** | Analyst-chosen % threshold defining "significant" |
| **DC + OS** | Every trend = reversal leg + continuation leg |
| **TMV** | How large was the move (normalised by $\theta$) |
| **T** | How long did the move take |
| **R** | How fast was the move (return per unit time) |
| **Regime use** | Normal regimes cluster in low TMV/T/R space; crises push all three up |
| **Limitation** | EXT labels are retrospective; use unfinished trends for live tracking |

---

## Further Reading

- Book §2.2 — DC concept and indicators
- Book Appendix A — formal event definitions
- Tsang et al. [47] — full indicator set
- Glattfelder et al. [27] — DC scaling laws
- [regime-change.md](./regime-change.md) — what regime change means
- [hidden-markov-model.md](./hidden-markov-model.md) — HMM on DC features
- [naive-bayes-classifier.md](./naive-bayes-classifier.md) — online tracking with TMV and T
