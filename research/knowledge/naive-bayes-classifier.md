# Naïve Bayes Classifier

> Reference: Chen & Tsang (2021) §2.3.2, Chapter 5  
> Purpose: **real-time tracking** of whether the market is entering a new regime, given historical regime information

## Why Naïve Bayes?

HMMs are strong at **discovering** hidden states from an observation sequence, but trading practice also requires:

> Given the DC indicators observed right now, what is the probability that the market is in each regime?

Chapter 5 uses the **Naïve Bayes Classifier (NBC)** for **regime tracking**: it combines historical regime labels with current market features to compute posterior regime probabilities as early-warning signals.

This complements the project’s “HMM for macro state + Hurst/momentum for tactical decisions” — NBC focuses on **online classification and probability output**.

## Bayes’ Theorem

$$
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}
$$

- $P(A)$ — prior probability of event $A$
- $P(B \mid A)$ — likelihood of observing $B$ given $A$
- $P(A \mid B)$ — posterior probability of $A$ after observing $B$

## Classifier Form

For class $C_i$ and feature vector $\mathbf{x}$:

$$
P(C_i \mid \mathbf{x}) = \frac{P(C_i)\, P(\mathbf{x} \mid C_i)}{P(\mathbf{x})}
$$

| Term | Meaning |
|------|---------|
| $P(C_i \mid \mathbf{x})$ | Posterior probability of class $C_i$ given features $\mathbf{x}$ |
| $P(\mathbf{x} \mid C_i)$ | Likelihood of features $\mathbf{x}$ under class $C_i$ |
| $P(C_i)$ | Prior probability of class $C_i$ |
| $P(\mathbf{x})$ | Marginal probability of $\mathbf{x}$ (normalizing constant) |

Prediction picks the class with highest posterior:

$$
\hat{C} = \arg\max_{i} P(C_i \mid \mathbf{x})
$$

## What Makes It “Naïve”?

NBC assumes **class conditional independence**: given the class, all features are independent:

$$
P(\mathbf{x} \mid C_i) = \prod_{j=1}^{d} P(x_j \mid C_i)
$$

This is a strong assumption — financial features (TMV, T, R) are often correlated. In practice, NBC still tends to be robust, simple to compute, and data-efficient, making it a practical proof-of-concept regime tracker.

## Training and Prediction

### Training Phase

Given labelled samples $(\mathbf{x}, C)$:

1. Estimate class priors $P(C)$
2. Estimate per-feature conditional distributions $P(x_j \mid C)$
3. Store parameters for inference

### Testing Phase

For unlabelled sample $\mathbf{x}$:

1. Compute each $P(C_i \mid \mathbf{x})$ via Bayes’ rule
2. Output the most likely class and its probability (usable as an alert threshold)

## Application to Regime Tracking (Book Chapter 5)

### Problem Setup

- **Classes**: Regime 1 (normal, low vol) vs Regime 2 (abnormal, high vol)
- **Features**: DC indicators (TMV, T, R) and information on unfinished trends
- **Goal**: estimate $p(C_1 \mid \mathbf{x})$ and $p(C_2 \mid \mathbf{x})$ to detect a slide from normal to abnormal conditions

### Decision Rules (Book Experiments)

| Rule | Meaning |
|------|---------|
| **B-Simple** | Looser classification threshold, fewer alerts |
| **B-Strict** | Stricter threshold, earlier regime-shift warnings |

On DJIA, FTSE 100, and S&P 500, NBC tracking is compared against HMM-detected “ground truth” regime periods (Figures 5.9–5.11).

### Division of Labour with HMM

| Method | Role |
|--------|------|
| **HMM** | Offline / batch: learn hidden states and transition structure from observations |
| **Naïve Bayes** | Online: given current features and historical regime knowledge, output instant class probabilities |

Chapter 6 feeds regime-tracking signals into trading algorithms (JC1, JC2 vs control CT1) to test whether tracking improves performance.

## Timing, Causality, and What NBC Can (and Cannot) Do

This section addresses a common question: if regime change is the **result** of collective trader behaviour, and traders have already acted, how can NBC help you **react** — isn't it too late?

### What you are actually observing

You never observe trader psychology directly. The chain looks like this:

```
Trader behaviour shifts (fear, deleveraging, herding, etc.)
        ↓
Price / volatility / liquidity change
        ↓
DC indicators change (TMV, T, R, unfinished trends)
        ↓
NBC outputs P(normal | x) and P(abnormal | x)
        ↓
You adjust risk, exposure, or strategy rules
```

NBC is **not** reading minds. It is a **symptom detector**: “current market features resemble patterns that historically occurred during abnormal regimes.”

### NBC is a monitoring signal, not a prophecy

| Role | What it does |
|------|----------------|
| **HMM (offline)** | Labels past regimes from full return history — “what regime were we in back then?” |
| **NBC (online)** | Classifies the **current** feature vector — “given what we see right now, how likely is each regime?” |

So yes: NBC acts as a **regime-tracking signal**. It outputs a probability that the market is entering (or already in) an abnormal state. It is an **early-warning / risk-off switch**, not a guarantee that you catch the regime change before any price move.

### Why reacting “after” traders move can still matter

Regime changes are usually **persistent**, not one-day events. A crisis regime can last weeks or months. You do not need to trade the exact first tick of a shift to benefit:

1. **Most of the damage happens during the regime, not only at the switch** — reducing exposure while $P(\text{abnormal} \mid \mathbf{x})$ is elevated can still cut drawdown over the full episode.
2. **Transitions are gradual in the data** — volatility rises, trends lengthen, DC indicators shift **before** the regime is obvious in hindsight. NBC is trained to spot feature combinations that tended to appear **during** transitions, including **unfinished DC trends** (in-progress moves, not only completed ones).
3. **Risk management ≠ perfect prediction** — the goal is often “exit or de-risk when conditions look like past crises,” not “predict the crisis before anyone trades.”

### The lag problem (be honest about it)

There is **inherent lag**:

- Price moves first → indicators update → classifier fires → you trade

You cannot react to a regime change **before** it leaves a footprint in prices. NBC does not solve causality backwards. What it offers is:

- **Faster recognition** than waiting for a human to label “we are in a crash”
- **Probabilistic output** — e.g. $P(\text{abnormal})$ rising from 0.15 → 0.6 → 0.9 lets you scale risk gradually instead of binary on/off
- **Tick-by-tick monitoring** under DC (book’s framing), which can flag abnormal conditions earlier than slow monthly macro indicators

### How traders “react” in practice

Typical uses when $P(\text{abnormal} \mid \mathbf{x})$ rises:

| Action | Rationale |
|--------|-----------|
| Cut position size | Strategy calibrated for normal vol may be oversized in crisis |
| Switch rule set | e.g. stop trend-following, go flat or defensive (as in book Ch.6 JC1/JC2) |
| Tighten stops / raise cash | Limit tail risk while regime is uncertain |
| Pause new entries | Avoid adding exposure into an unstable regime |

You are not undoing what other traders already did. You are **adapting your own positioning** to the environment their actions created.

### Summary

- **Regime change** = collective behaviour change, visible only through prices and derived indicators.
- **NBC** = online classifier that maps current indicators → regime probability, using patterns learned from past labelled regimes (from HMM).
- **Value** = timely risk adjustment during transitions and crises, not clairvoyance before any market move.
- **Limit** = always some lag; works best when regimes persist and when you use probabilities to scale risk, not as a single perfect binary trigger.

---

## Relation to This Project

The codebase **primarily implements Gaussian HMM** (`main/hmm_strategy.py`, `ZY/regime_hmm.py`) and does not yet include a standalone NBC regime tracker. A natural extension would be:

1. Label historical regimes offline with HMM + DC/time-series indicators
2. Train NBC on TMV, T, R (or Hurst, rolling volatility) as features
3. In walk-forward backtests, output $P(\text{high vol} \mid \mathbf{x}_t)$ daily as a risk switch

This is consistent with the “macro HMM + micro Hurst gating” design in `wenlin/strategy.py`; NBC could serve as a lightweight online supplement.

## Pros and Cons

**Pros**

- Simple to implement, fast inference, suitable for real-time monitoring
- Natural probability output for risk thresholds
- Often works with limited data

**Cons**

- Feature independence rarely holds in finance
- Sensitive to prior and feature distribution estimates
- Does not explicitly model state transition dynamics like HMM

## Minimal Code Example

```python
from sklearn.naive_bayes import GaussianNB

# X: feature matrix (n_samples, n_features), e.g. [TMV, T, R]
# y: historical regime labels (0=normal, 1=abnormal)
clf = GaussianNB()
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)  # posterior per class
pred = clf.predict(X_test)
```

## Further Reading

- Book §5.2.2 — Use of a Naïve Bayes Classifier
- Book §5.4 — DJIA / FTSE / S&P 500 empirical results
- Mitchell (1997) — NBC survey in machine learning
