# Research Knowledge Base

This directory collects theory and references on **regime change detection**, supporting strategy development for the FE5214 project.

## Primary Reference

- **Book**: Chen, Jun & Tsang, Edward P. K. (2021). *Detecting Regime Change in Computational Finance: Data Science, Machine Learning and Algorithmic Trading*. CRC Press.  
  Local PDF: [Detecting-Regime-Change-in-Computational-Finance-Data-Science-Machine-Learning-and-Algorithmic-Trading.pdf](./Detecting-Regime-Change-in-Computational-Finance-Data-Science-Machine-Learning-and-Algorithmic-Trading.pdf)

## Document Index

| Document | Contents |
|----------|----------|
| [regime-change.md](./regime-change.md) | Regime change concept, detection methods, link to this project |
| [directional-change.md](./directional-change.md) | DC framework, threshold, TMV / T / R indicators |
| [hidden-markov-model.md](./hidden-markov-model.md) | HMM definition, EM learning, project implementation |
| [naive-bayes-classifier.md](./naive-bayes-classifier.md) | Naïve Bayes classifier and real-time regime tracking |

## How the Methods Fit Together

```
Price data
   │
   ├─ Time Series sampling ──→ monitor statistics ──→ Regime-Switching / HMM
   │
   └─ Directional Change sampling ──→ TMV, T, R indicators  [directional-change.md]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              HMM (discover hidden states)   Naïve Bayes (online class probabilities)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          Regime Tracking → trading / risk control
```

## Project Code Entry Points

| File | Description |
|------|-------------|
| `main/hmm_strategy.py` | Gaussian HMM fitting, BIC, Hurst, signal generation |
| `ZY/regime_hmm.py` | 3-state HMM + walk-forward |
| `wenlin/strategy.py` | 2-state HMM × Hurst → 4 regimes |
| `main/eda/regime_characterization.py` | Regime characterization in EDA |

## Book Chapter Quick Reference

| Chapter | Topic |
|---------|-------|
| Ch.2 | Background: Regime Change, DC (§2.2), HMM, Naïve Bayes |
| Ch.3 | Regime detection using DC indicators |
| Ch.4 | Normal vs abnormal regime classification |
| Ch.5 | Naïve Bayes regime tracking |
| Ch.6 | Algorithmic trading based on regime tracking |
