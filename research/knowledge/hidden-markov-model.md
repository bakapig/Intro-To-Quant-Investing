# Hidden Markov Model (HMM)

> Reference: Chen & Tsang (2021) §2.3.1; Hamilton (1989)  
> Project implementation: `main/hmm_strategy.py`, `ZY/regime_hmm.py`, `wenlin/strategy.py`

## Core Idea

Financial markets involve two sequences:

1. **Observation sequence** $O_t$: prices, returns, volatility, etc. — visible to market participants
2. **Hidden state sequence** $S_t$: the true market regime — not directly observable

The **HMM** infers the most likely hidden state sequence and its transition dynamics from the observed data.

```
Observed:  r_t series     (returns, volatility, etc.)
Hidden:    S_t states     (low vol / high vol / ...)
```

## Markov Assumption

The state sequence follows a **first-order Markov chain**:

$$
P(q_i = a \mid q_1, \ldots, q_{i-1}) = P(q_i = a \mid q_{i-1})
$$

The next state depends only on the current state, not on earlier history. This is appropriate for modelling occasional, discrete regime jumps.

## Three Model Components

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Initial distribution | $\pi_i$ | Probability of starting in state $i$ |
| Transition matrix | $A = [a_{ij}]$ | Probability of moving from state $i$ to state $j$ |
| Emission probabilities | $B = b_i(O_t)$ | Probability of observing $O_t$ given state $i$ |

### Gaussian HMM in This Project

Conditional on state $k$, sector-neutral returns are assumed normal:

$$
\tilde{r}_t \mid S_t = k \;\sim\; \mathcal{N}(\mu_k,\, \sigma_k^{2})
$$

Implementation uses `hmmlearn.GaussianHMM` with `diag` or `full` covariance.

## Three Classic Problems

Once an HMM is defined by parameters $\lambda = (A, B, \pi)$, there are three fundamental questions you can ask. Rabiner (1989) groups them as the **three classic problems**. In regime detection, each problem answers a different practical question.

### Intuition First

Think of the market as a machine with hidden gears (regimes). You only see the output (returns), not which gear is engaged.

| Problem | Plain-English question | What you get |
|---------|------------------------|--------------|
| Evaluation | “How likely is this return history under my model?” | A single probability score |
| Decoding | “Which regime was the market in on each day?” | A state label per time step |
| Learning | “What model parameters best explain the data?” | Fitted $A$, $B$, $\pi$, $\mu_k$, $\sigma_k$ |

You almost always solve **Learning** first (fit the model), then use **Decoding** for trading signals. **Evaluation** is mainly for model comparison and diagnostics.

---

### Problem 1: Evaluation (Scoring)

**Question:** Given a fixed model $\lambda$ and an observation sequence $O = (O_1, O_2, \ldots, O_T)$, how likely is this data?

$$
P(O \mid \lambda)
$$

**Why it matters:** This is the model’s **goodness of fit**. A higher score means the observed returns are more consistent with the assumed regime structure (e.g. low-vol and high-vol Gaussian states with Markov transitions).

**Algorithm:** **Forward algorithm** — efficiently sums over all possible hidden state paths without enumerating them one by one.

**In this project:** After fitting, `model.score(X)` in `hmmlearn` returns the **log-likelihood** $\ln P(O \mid \lambda)$. That value feeds directly into the BIC calculation below.

---

### Problem 2: Decoding (State Inference)

**Question:** Given observations and a fitted model, what is the **most likely sequence of hidden states**?

$$
S^{*} = \arg\max_{S} \, P(S \mid O, \lambda)
$$

**Why it matters:** This is what you actually use for trading. You need a daily label: “today the market is in low-vol regime” or “high-vol regime.” Decoding turns the abstract HMM into a **regime time series**.

**Algorithms:**

- **Viterbi algorithm** — finds the single best state path (globally most likely sequence)
- **`model.predict(X)`** in `hmmlearn` — assigns the most likely state at each $t$ (slightly different objective, but standard in practice)

**In this project:** `get_ordered_states()` in `main/hmm_strategy.py` calls `model.predict(X)`, then re-orders states by variance so that label 0 always means lowest volatility.

**Example:** If returns are small and stable for months, the decoder mostly assigns state 0. After a crash, when $\tilde{r}_t$ becomes large and volatile, assignments shift toward state $N - 1$.

---

### Problem 3: Learning (Parameter Estimation)

**Question:** The hidden states are never observed. How do we **learn** $\lambda = (A, B, \pi)$ from data alone?

$$
\lambda^{*} = \arg\max_{\lambda} \, P(O \mid \lambda)
$$

**Why it’s hard:** This is like fitting a mixture model with **time dependence**. You don’t know which days belong to which regime, so you cannot simply compute sample means and variances per regime. The state labels and the parameters must be solved **jointly**.

**Algorithm:** **Expectation–Maximization (EM)**, also called **Baum–Welch** in the HMM literature. `hmmlearn` runs this internally when you call `model.fit(X)`.

**In this project:** Every asset in the universe gets its own HMM fit on sector-neutral log returns (`fit_hmm_regimes()` → `_fit_hmm_model()`).

---

## EM Algorithm (Expectation–Maximization)

### Why EM Exists

Direct maximum-likelihood estimation of HMM parameters has no closed-form solution because the regime labels $S_t$ are **latent** (missing data). EM handles this by alternating between:

1. Guessing which state each observation likely came from
2. Updating parameters as if those guesses were true

Each iteration is guaranteed (for this class of models) to **increase or maintain** the log-likelihood, until convergence.

### The Two Steps

**E-step (Expectation):**  
Given current parameters $\lambda^{(old)}$, compute the **posterior probability** that observation $O_t$ was generated by state $k$:

$$
\gamma_t(k) = P(S_t = k \mid O, \lambda^{(old)})
$$

In words: “Day $t$’s return — how likely does it belong to regime $k$?”  
Days with small, calm returns get high $\gamma_t(\text{low vol})$; crash days get high $\gamma_t(\text{high vol})$.

**M-step (Maximization):**  
Using those soft assignments $\gamma_t(k)$, re-estimate parameters as **weighted averages**:

- **Emission parameters** $\mu_k$, $\sigma_k$: weighted mean and variance of returns in state $k$
- **Transition matrix** $A$: count how often the model transitions $i \to j$, weighted by state posteriors
- **Initial distribution** $\pi$: posterior probability of each state at $t = 1$

Then set $\lambda^{(new)}$ and repeat until the log-likelihood change falls below a tolerance (`tol=1e-4` in our code).

### Finance Example (2-State Gaussian HMM)

Suppose you fit on daily sector-neutral returns:

1. **Start** with random $\mu_k$, $\sigma_k$, and transition probabilities
2. **E-step:** A day with return $+0.1\%$ gets ~90% weight on low-vol state; a day with $-4\%$ gets ~95% weight on high-vol state
3. **M-step:** Recompute $\mu_0$, $\sigma_0$ from all low-vol-weighted days; same for state 1; update $a_{01}$ (probability of jumping from calm to crisis)
4. **Repeat** until parameters stabilize

After convergence you have interpretable regimes: state 0 with $\mu_0 \approx 0$, small $\sigma_0$; state 1 with $\mu_1$ near zero or negative, large $\sigma_1$.

### Baum–Welch = EM for HMM

Baum–Welch is the HMM-specific name for EM. The Forward–Backward algorithm computes the $\gamma_t(k)$ terms needed in the E-step. You do not implement this manually — `GaussianHMM.fit()` handles it.

### Practical Caveats

| Issue | What to do |
|-------|------------|
| Local optima | Fix `random_state=42`; try multiple restarts if results are unstable |
| Too few observations | Need enough data per state; our code skips assets with insufficient history |
| Numerical stability | Scale returns (we multiply by 100) and set `min_covar=1e-6` |
| Convergence | Increase `n_iter` (we use 1000); check that log-likelihood plateaus |

---

## Choosing the Number of States (BIC)

### The Core Tradeoff

More states $N$ almost always **fit the data better** (higher likelihood), but risk **overfitting** — modelling noise as if it were extra regimes.

| Too few states ($N = 1$) | Too many states ($N = 5+$) |
|--------------------------|----------------------------|
| Miss real crisis periods | Fragment data into meaningless micro-regimes |
| Underfit structural change | Unstable, hard to interpret |
| Simple but biased | High in-sample fit, poor out-of-sample |

**BIC** balances fit and complexity: it rewards likelihood but **penalizes** the number of free parameters.

### The Formula

$$
\mathrm{BIC} = k \cdot \ln(T) - 2 \cdot \ln(\mathcal{L})
$$

| Symbol | Meaning |
|--------|---------|
| $\mathcal{L}$ | Maximized likelihood of the data under the fitted model |
| $\ln(\mathcal{L})$ | Log-likelihood from `model.score(X)` |
| $T$ | Number of observations (trading days) |
| $k$ | Number of **free** parameters estimated |

**Rule: choose the $N$ with the lowest BIC** (among candidates you fit).

Lower BIC = preferred model, assuming the same data and likelihood definition.

### Counting Parameters ($k$)

For a Gaussian HMM with $N$ states and $d$-dimensional observations, a common count (as in our project paper) is:

$$
k = \underbrace{N^2 - N}_{\text{transition matrix}} + \underbrace{N - 1}_{\text{initial distribution}} + \underbrace{N \cdot d}_{\text{means}} + \underbrace{N \cdot d}_{\text{variances (diag)}}
$$

For **univariate** returns ($d = 1$), this simplifies to:

$$
k = (N^2 - N) + (N - 1) + N + N = N^2 + 2N - 1
$$

This matches `main/hmm_strategy.py`:

```python
k = (n_states**2 - n_states) + n_states + n_states + (n_states - 1)
bic = k * np.log(n_samples) - 2 * log_likelihood
```

**Intuition for the penalty:** Each extra state adds rows to $A$, adds $\mu_k$ and $\sigma_k$, and the $\ln(T)$ factor grows with sample size — so with enough data, BIC increasingly punishes unnecessary states.

### Step-by-Step Selection Procedure

1. **Choose candidates** — typically $N \in \{2, 3\}$ for finance (bull/crash, or low/medium/high vol). Rarely go beyond 4 for daily equity data.
2. **Fit each model** — run EM (`model.fit`) separately for each $N$.
3. **Record log-likelihood** — `model.score(X)` for each fit.
4. **Compute BIC** — plug into the formula above.
5. **Pick minimum BIC** — that $N$ is the statistical choice.
6. **Sanity-check economically** — does each state have a clear meaning (e.g. sorted by $\sigma_k^2$)? If $N = 3$ splits “normal” into two nearly identical states, prefer $N = 2$ even if BIC is slightly lower.

### Worked Example (Conceptual)

| $N$ | $\ln(\mathcal{L})$ | $k$ | $T = 2520$ | BIC | Verdict |
|-----|-------------------|-----|------------|-----|---------|
| 2 | $-3100$ | 7 | 10-year daily | $6200 + 7 \ln(2520) \approx 6720$ | Competitive |
| 3 | $-3080$ | 14 | same | $6160 + 14 \ln(2520) \approx 6725$ | Slightly worse BIC → prefer $N=2$ |
| 4 | $-3070$ | 23 | same | $6140 + 23 \ln(2520) \approx 6790$ | Overfitting penalty dominates |

*(Illustrative numbers only — always compute from your actual fit.)*

### How This Project Uses BIC

- `fit_hmm_regimes()` computes BIC per asset for a **given** `n_states` (set in `config.N_STATES`, often 2 or 3).
- `process_universe()` collects BIC across all tickers; `generate_signals.py` aggregates mean/std BIC by `n_states` for comparison runs.
- The project often **fixes** $N$ based on economic interpretation (2 = normal/crash per Bae et al.; 3 = low/medium/high vol) and uses BIC as a **diagnostic**, not always as an automatic selector.

**Recommended workflow for your own experiments:**

```python
results = []
for n in [2, 3, 4]:
    bic_df, model = fit_hmm_regimes(returns, n_states=n)
    if model is not None:
        results.append(bic_df)
# Pick n_states with lowest mean BIC across universe, then validate out-of-sample
```

Always pair BIC with **walk-forward** testing (`ZY/regime_hmm.py`) — the best in-sample $N$ is not always best for trading.


## Label Switching and State Ordering

After fitting, state indices are arbitrary. This project orders states by **variance**:

- Sort estimated variances $\sigma_k^{2}$ in ascending order
- State 0 → lowest volatility (Steady Bull)
- State $N - 1$ → highest volatility (Crash / Panic)

See `get_ordered_states()` in `main/hmm_strategy.py`.

## Financial Applications

| Study | Application |
|-------|-------------|
| Hamilton (1989) | US GNP / business cycles, recession vs expansion |
| Ghysels | Seasonal patterns in economic recoveries |
| Kritzman et al. | Regime forecasting and asset allocation from macro variables |
| Bae et al.; Mulvey & Liu | Two-state classification: normal vs crash |

## Implementation in This Project

### Feature Inputs

| Module | Features |
|--------|----------|
| `main/hmm_strategy.py` | Sector-neutral returns (single column) |
| `ZY/regime_hmm.py` | Hurst, 20-day return, 20-day volatility |
| `wenlin/strategy.py` | Returns + volatility (2-state HMM), combined with Hurst into 4 regimes |

### Avoiding Look-Ahead Bias

`ZY/regime_hmm.py` provides **walk-forward** fitting:

- Predict one step at a time after the training cutoff
- Refit every 63 trading days (~one quarter) on an expanding window

### Division of Labour with Hurst

- **HMM**: macro volatility environment (risk-on / risk-off)
- **Hurst**: local price memory structure (trend / mean reversion / random walk)

Final signals combine both via gating rules (see project paper §Methodology).

## Limitations and Caveats

- Assumes constant transition matrix and fixed emission form (e.g. Gaussian); real markets may be more complex
- EM converges to local optima — fix `random_state` and check stability
- Too few states underfit; too many overfit — use BIC and economic interpretation
- Out-of-sample work requires walk-forward or rolling refit; do not backtest on full-sample labels

## Minimal Code Example

```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

X = (returns.values * 100).reshape(-1, 1)  # scale for numerical stability

model = GaussianHMM(
    n_components=2,
    covariance_type="diag",
    n_iter=1000,
    random_state=42,
)
model.fit(X)
hidden_states = model.predict(X)
```

## Further Reading

- Rabiner (1989) — HMM tutorial
- Book §4.2.2, §4.3.3 — DC features + HMM regime detection
- Project `JC/FE5214_Group_Project.tex` §Macroeconomic Regime Detection
