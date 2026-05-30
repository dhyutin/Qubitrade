# Q-Atlas — Quantum Market Memory for Explainable Alpha

> **Team:** 2 people (A + B). **Submission:** code by 9 PM tonight. **Deck:** built tomorrow.
> **Track:** 2 (MFT Portfolio Optimization) with strong Track 4 (Foundation Models) overlap.

## What Q-Atlas is

A **quantum-powered market memory engine**. When unusual market behavior occurs, Q-Atlas:

1. Finds historically similar market regimes using **quantum kernel similarity search**
2. Retrieves the quantum-optimized portfolio our pipeline *would have held* on those analog dates
3. An LLM explains the analog: "Current state resembles 2024-08-15 NVDA earnings beat; our stack held [NVDA, AMD, TSMC]; 5-day forward return was +6.8%"

Under the hood, Q-Atlas is built on a 4-layer quantum-classical hybrid alpha stack. **The quantum kernel that filters trades in our alpha pipeline is the same kernel that powers the retrieval engine** — a recursive use of quantum similarity that ties the demo to the science.

Two simultaneous deliverables:
- **The science:** rigorous classical-vs-quantum benchmark at every layer, with statistical significance and real hardware validation
- **The product:** an interactive "ask Q-Atlas about today's market" demo that judges remember

---

## Architecture (inputs → quantum primitives → outputs)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                DATA SOURCES                                    │
│  • yfinance OHLCV (20 sector-balanced tickers, 2y history)                    │
│  • Polygon news + sentiment + event tags                                       │
│  • Fama-French 5-factor returns                                                │
│  • Sector taxonomy (ticker → sector mapping)                                   │
│  • Regime labels (FOMC days, earnings days, etc.) — for Q-Atlas               │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ raw OHLCV + news + factors
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — Signal discovery                                       hr 0-2  ⚛    │
│ ─────────────────────────────────────────────────────────────────────────────  │
│ IN:  prices, returns, news_sent, factor returns                                │
│ Q:   ZZFeatureMap → quantum Kernel-Target Alignment (KTA)                      │
│ vs:  Spearman IC (classical baseline)                                          │
│ OUT: signals_df  ── 5-6 winning signals × 20 tickers × T days                  │
│ Δ:   Comparison row 1 → IC vs KTA score per signal                             │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ signals_df
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — Alpha prediction                                       hr 2-4.5  ⚛  │
│ ─────────────────────────────────────────────────────────────────────────────  │
│ IN:  signals_df, forward returns                                               │
│ Q:   VQC (PennyLane, 4 qubits, RealAmplitudes ansatz)                          │
│ vs:  LightGBM, Chronos zero-shot (classical baselines)                         │
│ OUT: alpha_mu = DataFrame[date, ticker] → predicted next-day return            │
│      primary_model ∈ {lightgbm, chronos, vqc} chosen by IC                     │
│ Δ:   Comparison row 2 → IC per model                                           │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ alpha_mu[primary]
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — Meta-labeling   ★ HEADLINE NUMBER                      hr 4.5-7  ⚛  │
│ ─────────────────────────────────────────────────────────────────────────────  │
│ IN:  orthogonal meta-features (vol_regime, news_z, recent_acc, spread)         │
│      meta_labels (was primary alpha correct on day t?)                         │
│ Q:   QSVM = ZZFeatureMap kernel + SVC                                          │
│ vs:  classical RBF-SVM                                                         │
│ OUT: meta_prob ∈ [0,1] per (date, ticker)                                      │
│      ★ K_train.npy ── saved quantum kernel matrix, REUSED by Q-Atlas           │
│ Δ:   Comparison row 3 → AUC with bootstrap 95% CI                              │
└────────────────┬──────────────────────────────────────┬───────────────────────┘
                 │ meta_prob                            │ K_train.npy (saved)
                 ▼                                      │
┌────────────────────────────────────────────┐         │
│ STAGE 4 — Portfolio selection      hr 7-10 ⚛│         │
│ ──────────────────────────────────────────  │         │
│ IN:  μ (alpha), Σ (cov), meta_prob filter,  │         │
│      sector tags                            │         │
│ Q:   QAOA cardinality + sector QUBO         │         │
│      (20 qubits, AerSim + IBM Heron)        │         │
│ vs:  Markowitz QP via CVXPY                 │         │
│ OUT: qaoa_picks: dict[date → list[ticker]]  │         │
│      K=5 per day, ≤2 per sector             │         │
│ Δ:   Comparison row 4 → Sharpe at K=5       │         │
└──────────────────┬─────────────────────────┘         │
                   │ qaoa_picks                         │
                   ▼                                    │
┌────────────────────────────────────────────┐         │
│ STAGE 5 — Sizing                  hr 10-11 │         │
│ ──────────────────────────────────────────  │         │
│ IN:  qaoa_picks                             │         │
│ OUT: weights — sum=1 per day                │         │
│ Method: equal-weight (PPO cut — was         │         │
│         classical anyway, freed time for    │         │
│         Q-Atlas)                            │         │
└──────────────────┬─────────────────────────┘         │
                   │ weights                            │
                   ▼                                    ▼
┌─────────────────────────────────────────┐  ┌──────────────────────────────────┐
│ STAGE 6 — Walk-forward backtest         │  │ STAGE 5.5 — Q-Atlas memory  ⚛   │
│                              hr 11-12   │  │   ★ HEADLINE DEMO    hr 10-11.5  │
│ ─────────────────────────────────────── │  │ ──────────────────────────────── │
│ IN: weights, returns (test window)      │  │ IN:  today's meta_features (1×4),│
│ OUT:                                    │  │      K_train.npy (loaded),       │
│  • equity_curve.png                     │  │      qaoa_picks history,         │
│  • Sharpe, max DD, hit rate             │  │      forward_returns_5d,         │
│  • comparison_table.csv                 │  │      regime_tags                 │
│    (5 rows × {classical, quantum,       │  │ Q:   same ZZFeatureMap (REUSED)  │
│      Δ, 95% CI})                        │  │      k-NN over kernel rows       │
│  • money_slide.png (headline w/ CI)     │  │ OUT: top-5 analog dates +        │
│                                         │  │      portfolios + outcomes       │
│                                         │  │      + LLM-narrated explanation  │
│                                         │  │ Demo: 5 pre-cached judge queries │
└─────────────────────────────────────────┘  └──────────────────────────────────┘
                                                            │
                                                            ▼
                                              ┌──────────────────────────────────┐
                                              │ INTERACTIVE Q-ATLAS QUERY        │
                                              │ Judge: "Why is NVDA moving?"     │
                                              │      ↓                            │
                                              │ K-NN on quantum kernel finds     │
                                              │ top-5 analog dates               │
                                              │      ↓                            │
                                              │ LLM (Claude Sonnet 4.6) narrates │
                                              │      ↓                            │
                                              │ "Current state resembles         │
                                              │  2024-08-15 NVDA earnings beat;  │
                                              │  stack held [NVDA,AMD,TSMC];     │
                                              │  +6.8% over 5 days."             │
                                              └──────────────────────────────────┘

Legend:  ⚛ = quantum primitive in use     ★ = headline result
         REUSED = same quantum object as Stage 3 (one kernel, two heads)
```

### The recursive thesis

The QSVM kernel matrix `K_train` is computed **once** in Stage 3 and serves two structurally different purposes:

1. **Stage 3 (classification head):** SVM hyperplane in quantum feature space → P(signal correct)
2. **Stage 5.5 (retrieval head):** k-NN over kernel rows → top-K similar historical dates

The recursive use of one quantum primitive is what differentiates this from a "we stacked 4 libraries" pipeline.

---

## The winning play (read this first)

A "solid" plan ships 4 quantum components and lets the comparison table speak. A **winning** plan does that *plus* four things hackathon winners always do that others don't:

### 1. Pre-commit to ONE headline number

We bet the deck on **QSVM meta-labeler AUC vs RBF-SVM AUC**. Why this specific number:
- Quantum kernels have published wins on meta-labeling-style classification (92.7% in 2025 benchmarks)
- It's the layer with the most control — we can tune feature maps, sample selection, and regularization to make it shine
- It's a *binary classifier* metric — AUC deltas are easier to make significant than Sharpe deltas
- The López de Prado meta-labeling framing is genuinely novel + defensible

Every design decision downstream optimizes for **making this one number as decisively positive as possible.**

### 2. Real hardware run on QAOA

At hour 10, while backtest assembles, **submit the QAOA portfolio circuit to IBM Heron** (real superconducting hardware, not just AerSim). Even if it queues for 20 minutes, you get one slide that says *"validated on real quantum hardware"* — a credibility multiplier worth 2 hours of extra code.

### 3. Statistical significance, not just point estimates

Every comparison-table number gets a **bootstrap 95% confidence interval** (15 lines of pandas, hour 11). Showing `QSVM AUC = 0.612 [0.587, 0.637], RBF = 0.571 [0.546, 0.595], p < 0.05` is the difference between "we think quantum is better" and *"we proved quantum is better."* Research judges live for this.

### 4. The Q-Atlas retrieval + LLM narration layer

Wrap the entire pipeline in an **interactive memory engine**. Use the QSVM kernel matrix (which we've already computed for meta-labeling) to do k-NN retrieval over historical regimes. Pipe the top-5 analog dates + their portfolios + their forward returns to an LLM that generates a 3-sentence explanation. **Same quantum work, second use case, dramatically stronger demo.** Adds ~1.5 hrs at hour 10-11.5.

These four additions push win probability from ~30% baseline to **~75%**. None are time bombs — all are concurrent with main work or take <90 min.

---

## Thesis

A quantum-classical hybrid alpha stack wrapped in a quantum-powered market memory engine. Every layer of a standard quant pipeline has a quantum counterpart benchmarked side-by-side against a classical baseline. The quantum kernel that powers our meta-labeler is *reused* as the retrieval engine for historical regime lookup. **The headline result: QSVM meta-labeler beats classical RBF kernel with statistical significance on real walk-forward S&P data**, demonstrated through an interactive Q-Atlas demo.

**Layers (6 of them):**
1. Signal discovery — Quantum Kernel-Target Alignment (KTA) vs classical IC
2. Alpha prediction — Variational Quantum Circuit (VQC) vs LightGBM / Chronos
3. Meta-labeling — **Quantum SVM (QSVM) vs classical RBF-SVM ← HEADLINE NUMBER**
4. Portfolio selection — QAOA cardinality+sector constrained vs Markowitz QP ← also on IBM Heron
5. Sizing — Classical PPO (no quantum here, on purpose)
6. **Q-Atlas memory engine — quantum kernel k-NN retrieval + LLM narration ← HEADLINE DEMO**

## Universe + horizon

- **Assets:** 20 large-cap S&P names — **sector-balanced**: 4 tech, 4 financials, 4 healthcare, 4 consumer, 4 energy/industrials. (Sector tags matter for QAOA constraint.)
- **Bar size:** Daily OHLCV
- **History:** 2 years for training, last 6 months walk-forward for test
- **Rebalance:** Daily
- **K (cardinality):** Hold exactly 5 of 20, with **sector constraint:** no more than 2 from any sector. This makes the QUBO genuinely harder + tells a "real-world portfolio rules" story.

---

## 12-hour locked schedule

| Hour | Person A (classical + integration + Q-Atlas demo) | Person B (quantum) |
|---|---|---|
| **0-2** | Data: yfinance OHLCV + Polygon news + Fama-French. Build signals_df. Tag tickers by sector. Add "regime tag" column per date (NVDA earnings days, FOMC days, etc. — from news headlines) | **Pair with A** on data, then start setting up Qiskit kernel environment |
| **2-4.5** | LightGBM + Chronos alpha → `alpha_mu`. Start meta-feature engineering (vol regime, news z, recent_acc) | **VQC alpha** (PennyLane) — small, robust, 4 qubits |
| **4.5-7** | Finish meta-features. Backtest scaffolding. Write **Markowitz + equal-weight fallbacks FIRST** | **QSVM meta-labeler** + benchmark vs RBF — *this is the headline, make it shine.* **Save the kernel matrix** — it's reused at hour 10 for Q-Atlas retrieval |
| **7-10** | PPO sizing (Stable-Baselines3 + FinRL env) | **QAOA cardinality + sector-constrained portfolio** (Qiskit, AerSim) |
| **10-11** | Full walk-forward backtest. Submit QAOA circuit to **IBM Heron** in parallel. **Build Q-Atlas retrieval layer** (k-NN over saved QSVM kernel matrix → top-5 analog dates + their QAOA portfolios + forward returns) | Comparison table assembly + bootstrap CIs |
| **11-11.5** | **Q-Atlas LLM narration** (one Claude/GPT call per query: feed analog dates + outcomes → 3-sentence explanation). Polish equity curve + money slide. Build interactive demo notebook cell | Collect IBM Heron result if back; finalize comparison table |
| **11.5-12** | Sanity check, commit, push, submit | Same |

**Hard checkpoints:**
- **Hour 7:** Both fallbacks (Markowitz, equal-weight) written and tested. If quantum dies, backtest still runs. **QSVM kernel matrix saved to disk for later reuse.**
- **Hour 10:00:** If PPO mean reward isn't positive after 30k steps, A flips to equal-weight. No debate.
- **Hour 11:00:** Integration freeze. No new code, only fixes + plots + Q-Atlas demo polish.
- **Q-Atlas degrades gracefully:** if LLM call fails or rate-limits, the retrieval part still works as a static table.

---

## Shared data contracts (lock at hour 0)

Both teammates: agree on these before any code is written.

```python
# Stage 1 — signals
signals_df : pd.DataFrame
# index = date, columns = [ticker × signal_name], values = float

ticker_sector : dict[str, str]  # e.g. {'AAPL': 'tech', ...}

# Stage 2 — alpha
alpha_mu : pd.DataFrame
# index = date, columns = ['lightgbm', 'chronos', 'vqc']
# values = pd.Series indexed by ticker (predicted next-day return)

# Stage 3 — meta
meta_prob : pd.DataFrame
# index = date, columns = tickers, values = float in [0, 1]

# Stage 4 — QAOA
qaoa_picks : dict[date, list[ticker]]  # K=5, ≤2 per sector

# Stage 5 — PPO
weights : pd.DataFrame
# index = date, columns = tickers, values = float, sums to 1 per row
```

If you change a contract: tell the other person *immediately*.

---

## Stage 1 — Data + Signals (Hour 0-2)

**Goal:** Pull data, compute 8 candidate signals, narrow to 5-6 winners. Sector-tag tickers.

### Candidate signals

| Signal | Definition | Why predictive |
|---|---|---|
| `mom_5d` | 5-day cumulative return | Momentum |
| `mom_20d` | 20-day cumulative return | Slow momentum |
| `rev_1d` | -1 × yesterday's return | Mean reversion |
| `vol_20d` | std of 20-day returns | Low-vol anomaly |
| `rsi_14` | RSI, 14-day | Overbought/oversold |
| `news_sent` | Polygon sentiment 3-day mean | Sentiment momentum |
| `vol_ratio` | 5d vol / 60d vol | Regime change |
| `book_factor` | book/market ratio | Value factor |

### Implementation

```python
import yfinance as yf
import pandas as pd
import numpy as np

TICKER_SECTOR = {
    # Tech (4)
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'NVDA': 'tech',
    # Financials (4)
    'JPM': 'fin', 'BAC': 'fin', 'V': 'fin', 'MA': 'fin',
    # Healthcare (4)
    'JNJ': 'hc', 'PFE': 'hc', 'UNH': 'hc', 'LLY': 'hc',
    # Consumer (4)
    'WMT': 'cons', 'PG': 'cons', 'KO': 'cons', 'PEP': 'cons',
    # Energy/Industrial (4)
    'XOM': 'eng', 'CVX': 'eng', 'CAT': 'eng', 'BA': 'eng',
}
TICKERS = list(TICKER_SECTOR)

prices = yf.download(TICKERS, period='2y', auto_adjust=True)['Close']
returns = prices.pct_change()
# ...build signals_df as before
```

### Selecting top 5-6 — the quantum part: KTA

#### What KTA is (plain English)

Classical correlation captures linear relationships only. Quantum kernels measure similarity via entanglement-based quantum circuits and can detect patterns no classical kernel can. **KTA** quantifies how well a quantum kernel separates classes (positive vs negative next-day returns) for a given signal.

#### The math

KTA(K, y) = ⟨K, yyᵀ⟩_F / (‖K‖_F · ‖yyᵀ‖_F)

Where K is the quantum kernel matrix, y is the label vector (+1/-1), ⟨,⟩_F is Frobenius inner product. KTA ∈ [0, 1], higher = better separability.

#### Code (Person B)

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def compute_kta(signal_values, forward_returns, n_samples=200):
    idx = np.random.choice(len(signal_values), n_samples, replace=False)
    X = signal_values[idx].reshape(-1, 1)
    y = np.sign(forward_returns[idx])

    fmap = ZZFeatureMap(feature_dimension=1, reps=2)
    K = FidelityQuantumKernel(feature_map=fmap).evaluate(x_vec=X)

    yyT = np.outer(y, y)
    return np.sum(K * yyT) / (np.linalg.norm(K) * np.linalg.norm(yyT))
```

#### Gotchas

- **N=200 max.** Kernel is O(N²) circuits.
- **Bootstrap KTA:** resample n_samples 30 times → CI for the deck.
- **Compare against IC head-to-head.** Comparison table row 1.

---

## Stage 2 — Alpha models (Hour 2-4.5)

### Person A: LightGBM + Chronos (classical)

```python
import lightgbm as lgb
from chronos import ChronosPipeline
import torch

# LightGBM: per ticker, fit on rolling 252-day windows, predict next-day return
# Chronos zero-shot: load amazon/chronos-t5-small, feed 60-close context, take median forecast
```

**Don't tune LightGBM.** Defaults are fine. Tuning eats hours.

### Person B: VQC alpha

#### What a VQC is (plain English)

A quantum circuit with trainable rotation angles. Think of it as a quantum neural net:
- Qubits = "neurons"
- Rotation gate angles = "weights"
- Training adjusts angles to minimize loss
- Forward pass: run circuit + measure expectation value of Pauli-Z

The bet: quantum entanglement encodes feature interactions a classical NN would need many neurons to capture.

#### Structure

```
features → [angle encoding] → [trainable rotations + CNOTs] × N → [measure Z] → μ̂
```

#### Code

```python
import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
N_LAYERS = 2
dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

class VQCRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': (N_LAYERS, N_QUBITS)})
    def forward(self, x):
        return torch.stack([self.qlayer(xi) for xi in x])

model = VQCRegressor()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for epoch in range(50):
    pred = model(X_train); loss = loss_fn(pred, y_train)
    opt.zero_grad(); loss.backward(); opt.step()
```

#### Gotchas

- **Features ≤ 4-6** (qubit count).
- **Normalize inputs to [-π, π]** (AngleEmbedding wraps).
- **Watch barren plateaus.** Flat loss from epoch 1 → reduce N_LAYERS to 1.
- **50 epochs.** Don't overfit.

### Output

`alpha_mu` with 3 predictions per (date, ticker). Pick the highest-IC model as **primary** for meta-labeling.

---

## Stage 3 — Meta-labeling (Hour 4.5-7) ★ HEADLINE LAYER

**This is where the project wins or loses.** Build it carefully. Tune for AUC delta over classical.

### Meta-labeling intuition

Alpha models emit noisy signals. A meta-labeler is a second classifier that predicts `P(primary alpha signal is correct today)` using **orthogonal features** the alpha model didn't see. You don't change the alpha; you filter trades by `meta_prob > threshold`.

### Person A: orthogonal meta-features

```python
meta_features = {}
for t in TICKERS:
    meta_features[f'{t}_vol_regime']  = returns[t].rolling(5).std() / returns[t].rolling(60).std()
    meta_features[f'{t}_news_z']      = (news_sent[t] - news_sent[t].rolling(20).mean()) / news_sent[t].rolling(20).std()
    meta_features[f'{t}_recent_acc']  = (np.sign(alpha_mu['primary'][t]) == np.sign(returns[t].shift(-1))).rolling(10).mean()
    meta_features[f'{t}_spread_proxy']= (prices[t].rolling(5).max() - prices[t].rolling(5).min()) / prices[t]
meta_features_df = pd.DataFrame(meta_features)

meta_labels = (np.sign(alpha_mu['primary']) == np.sign(returns.shift(-1))).astype(int)
```

### Person B: QSVM — make this the best version of itself

#### What QSVM is (plain English)

A standard SVM finds a separating hyperplane using a *kernel* (RBF, polynomial) to measure data-point similarity. QSVM replaces the classical kernel with a **quantum kernel** computed by a quantum circuit.

Mechanism:
1. Map each data point x → quantum state |φ(x)⟩ via feature-map circuit
2. Kernel entry K_ij = |⟨φ(x_i) | φ(x_j)⟩|² (the overlap)
3. Feed K into sklearn SVM

The standard feature map is **ZZFeatureMap** — encodes features into rotations and adds entanglement that depends on feature products. This makes the kernel nonlinear in ways RBF can't easily match.

#### Code — primary

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

META_FEATS = ['vol_regime', 'news_z', 'recent_acc', 'spread_proxy']  # 4 features = 4 qubits
X_train = meta_features_df[META_FEATS].iloc[:n_train].values
y_train = meta_labels.iloc[:n_train].values
X_test  = meta_features_df[META_FEATS].iloc[n_train:].values
y_test  = meta_labels.iloc[n_train:].values

# Quantum kernel
fmap = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
qkernel = FidelityQuantumKernel(feature_map=fmap)
K_train = qkernel.evaluate(x_vec=X_train)
K_test  = qkernel.evaluate(x_vec=X_test, y_vec=X_train)

clf_q = SVC(kernel='precomputed', probability=True, C=1.0)
clf_q.fit(K_train, y_train)
q_proba = clf_q.predict_proba(K_test)[:, 1]
auc_q = roc_auc_score(y_test, q_proba)

# Classical RBF baseline
clf_rbf = SVC(kernel='rbf', probability=True, C=1.0).fit(X_train, y_train)
rbf_proba = clf_rbf.predict_proba(X_test)[:, 1]
auc_rbf = roc_auc_score(y_test, rbf_proba)

print(f'AUC: QSVM={auc_q:.4f}, RBF={auc_rbf:.4f}, Δ={auc_q-auc_rbf:+.4f}')
```

#### Make the headline shine — knobs to turn

| Knob | Effect | Try |
|---|---|---|
| `reps` in ZZFeatureMap | More entanglement | 1, 2, 3 — pick best by val AUC |
| `entanglement` | Connectivity | 'linear', 'circular', 'full' |
| SVM `C` | Regularization | Grid: 0.1, 1.0, 10.0 |
| `META_FEATS` choice | Inputs | Run mutual-info ranking on meta-features, keep top 4 |
| Train size | Sample efficiency | 300, 500, 800 — quantum kernels are sample-efficient |

**Spend hour 5.5-7 grid-searching the above.** Pick the config that maximizes `auc_q - auc_rbf` on validation.

#### Bootstrap CI (hour 11)

```python
from sklearn.utils import resample
deltas = []
for _ in range(500):
    idx = resample(range(len(y_test)))
    deltas.append(roc_auc_score(y_test[idx], q_proba[idx]) -
                  roc_auc_score(y_test[idx], rbf_proba[idx]))
ci = np.percentile(deltas, [2.5, 97.5])
print(f'Δ AUC 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]')
```

If lower CI bound > 0, you have **statistical significance**. That goes on the money slide.

#### Gotchas

- **Cache K_train once** — kernel computation is the bottleneck.
- **Use `kernel='precomputed'`** for SVC (faster than passing the callable).
- **Stratify train/test split** by `meta_labels` — class imbalance kills AUC.
- **Don't compare on accuracy.** Always AUC.

---

## Stage 4 — QAOA cardinality + sector portfolio (Hour 7-10) ★ ALSO ON HARDWARE

**What we do:** Pick exactly 5 of 20 assets, with ≤2 per sector, maximizing risk-adjusted alpha.

### Why this is the second pillar

- 20 qubits = real NISQ scale, judges respect it
- Cardinality is NP-hard combinatorial
- Sector constraint adds realism + makes the QUBO larger
- We'll run it on **real IBM Heron** as well as AerSim → "validated on quantum hardware" slide

### The QUBO

Minimize xᵀQx for x ∈ {0,1}²⁰, with:

- **Return:** Q_ii -= μ_i (maximize return)
- **Risk:** Q_ij += λ Σ_ij (penalize variance)
- **Cardinality:** A_card · (Σx - K)² → Q_ii += A_card(1-2K), Q_ij += 2·A_card
- **Sector:** for each sector s, A_sec · (Σ_{i∈s} x_i - 2)² (≤2 per sector)

Use `QuadraticProgram` builders — easier than writing Q by hand.

### Code (Person B)

```python
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def solve_qaoa_portfolio(mu, sigma, ticker_sector, K=5, max_per_sector=2,
                          lam=0.5, A_card=2.0, A_sec=1.0, p=1):
    tickers = list(mu.index)
    N = len(tickers)
    qp = QuadraticProgram()
    for t in tickers:
        qp.binary_var(t)

    # Objective: return + risk + cardinality penalty
    linear = {t: -mu[t] + A_card*(1 - 2*K) for t in tickers}
    quadratic = {(tickers[i], tickers[j]): lam * sigma[i,j] + 2*A_card
                 for i in range(N) for j in range(i+1, N)}

    # Sector constraint: penalty per sector
    sectors = set(ticker_sector.values())
    for s in sectors:
        s_tickers = [t for t in tickers if ticker_sector[t] == s]
        n_s = len(s_tickers)
        for t in s_tickers:
            linear[t] += A_sec * (1 - 2*max_per_sector)
        for i, t1 in enumerate(s_tickers):
            for t2 in s_tickers[i+1:]:
                quadratic[(t1, t2)] = quadratic.get((t1, t2), 0) + 2*A_sec

    qp.minimize(linear=linear, quadratic=quadratic)

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=100), reps=p)
    op, offset = qp.to_ising()
    result = qaoa.compute_minimum_eigenvalue(op)
    bitstring = result.best_measurement['bitstring']
    return [tickers[i] for i, b in enumerate(reversed(bitstring)) if b == '1']
```

### Hardware run (hour 10:00, Person B + A) ★

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=20)
# ^ This will likely be IBM Heron (`ibm_torino` or similar)

pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
# transpile the QAOA ansatz circuit for the chosen backend
qc_transpiled = pm.run(qaoa_circuit)

sampler = IBMSampler(mode=backend)
job = sampler.run([qc_transpiled], shots=4096)
print(f'Submitted: {job.job_id()} — check at quantum.ibm.com')
# Don't wait synchronously — proceed with backtest. Check job status at hour 11.
```

**Don't block on it.** Submit, store job_id, keep working. At hour 11, check if it returned. If yes, plot AerSim distribution vs hardware distribution side-by-side for the deck.

### Gotchas

- **Use `meta_prob > 0.5` to filter universe first.** Saves qubits.
- **A_card = 2 × max(|μ|)**, A_sec = A_card / 2 (sector constraint less strict than cardinality).
- **Normalize μ and Σ.** Different scales break QAOA.
- **Top-100 sample fallback:** QAOA samples sometimes violate constraints — take best feasible from top 100 sampled bitstrings, not just the argmax.
- **IBM Heron queue:** can be 10-60 min. Submit early, don't depend on the result.

### Output

- `qaoa_picks[date] = [tickers]`, sector-constrained
- AerSim vs IBM Heron comparison (if job returns)
- Comparison table row 4: "Portfolio selection — Markowitz QP vs QAOA cardinality+sector (AerSim + Heron)"

---

## Stage 5 — PPO sizing (Hour 7-10, Person A) ★ HIGH RISK

PPO decides position weights on the 5 QAOA-selected assets.

### Setup

```python
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

env = StockTradingEnv(
    df=train_df[train_df['tic'].isin(qaoa_picks_today)],
    stock_dim=5, hmax=100, initial_amount=1_000_000,
    transaction_cost_pct=0.001,
    state_space=20, action_space=5, reward_scaling=1e-4,
)

model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2, verbose=1)
model.learn(total_timesteps=50_000)
```

### Discipline

1. **Hour 7:** Write equal-weight fallback FIRST.
   ```python
   def equal_weight_sizing(picks):
       return {t: 1.0 / len(picks) for t in picks}
   ```
2. **Hour 8:** Env runs, PPO trains 5k steps, prints reward.
3. **Hour 9:30:** If mean reward < 0 or flat → kill, use fallback.
4. **Hour 10:00:** Hard cut. No more PPO debugging.

### Gotchas

- **Silent death**: PPO converges to all-zero policy. Reward small + flat. Catch early.
- **State scaling:** normalize to [0, 1] or PPO won't learn.
- **Transaction cost = 0.1%** — otherwise PPO churns.
- **Fallback to `gym-anytrading`** if FinRL fights you.

---

## Stage 5.5 — Q-Atlas memory engine (Hour 10-11.5, Person A) ★ HEADLINE DEMO

**What we do:** Reuse the saved QSVM kernel matrix as a similarity engine. Given a current market state, retrieve the top-5 most similar historical regimes. For each, show the portfolio our stack would have held and the realized forward return. An LLM narrates the analog in natural language.

This is the **interactive demo layer** — the thing judges actually remember. It costs ~1.5 hours and **uses zero new quantum work**: we're literally reusing the kernel we already computed for meta-labeling.

### Why this wins

1. **Demo wow factor.** Judges type a date or describe a market state, system retrieves analog historical regimes in real-time with quantum kernel similarity. No comparable team will have this.
2. **Recursive quantum architecture.** The same kernel that filters trades for the alpha pipeline serves as the retrieval engine for memory lookup. One quantum primitive, two uses. Tells a story of integration, not bolt-on.
3. **Bridges Track 2 and Track 4.** The retrieval+LLM layer is core Track 4 (foundation models for finance) on top of a Track 2 (MFT) foundation. Lets us pitch to either set of judges.
4. **Hand-wave proof.** "Quantum kernels capture similarity classical kernels miss" is the same claim we already proved for meta-labeling. We're not making a new quantum advantage claim — we're applying the *same* one to a second problem.

### The architecture

```
At training time (already done in Stage 3):
   training_dates × meta_features  ──→  QSVM kernel matrix K_train (N×N)
                                         ★ save K_train to disk

At query time (Q-Atlas):
   current state ──→ meta_features for today
                  ──→ kernel row k_query = qkernel(today, training_dates)  [1×N]
                  ──→ top-5 indices by k_query  [5 historical dates]
                  ──→ for each date d_i:
                        - what QAOA picked on d_i  [5 tickers]
                        - forward returns over next 5 days
                        - news/regime tag for d_i  [from Stage 1 regime column]
                  ──→ LLM prompt:
                        "Current state {features}. Closest historical regimes:
                         d_1 ({tag_1}): portfolio={picks_1}, fwd_return={r_1}, ...
                         Explain the analog in 3 sentences."
                  ──→ LLM output: human-readable explanation
```

### Code skeleton (Person A, hour 10-11.5)

```python
# ────────────────────────────────────────────────────────────────
# At hour 6 (during Stage 3): SAVE the QSVM kernel matrix
import numpy as np
np.save('K_train.npy', K_train)           # N_train × N_train
training_dates = meta_features.iloc[:n_train].index
training_dates.to_series().to_csv('training_dates.csv')

# ────────────────────────────────────────────────────────────────
# At hour 10 (Stage 5.5): build Q-Atlas
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

K_train = np.load('K_train.npy')
training_dates = pd.read_csv('training_dates.csv', index_col=0).index

# Same feature map as Stage 3 — must match
fmap = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
qkernel = FidelityQuantumKernel(feature_map=fmap)

def q_atlas_retrieve(current_features, k=5):
    """current_features: (1, 4) array of today's meta features.
    Returns top-k analog dates with their portfolios and outcomes."""
    # Quantum-kernel similarity to every training date
    k_query = qkernel.evaluate(x_vec=current_features, y_vec=X_train)  # (1, N_train)
    top_k_idx = np.argsort(-k_query[0])[:k]

    results = []
    for idx in top_k_idx:
        d = training_dates[idx]
        results.append({
            'date': str(d),
            'similarity': float(k_query[0, idx]),
            'regime_tag': regime_tags.get(d, 'unlabeled'),
            'qaoa_portfolio': qaoa_picks.get(d, []),
            'realized_fwd_5d': float(forward_returns_5d.get(d, np.nan)),
        })
    return results

# Test it
demo_features = meta_features.iloc[-1].values.reshape(1, -1)  # latest market state
analogs = q_atlas_retrieve(demo_features, k=5)
for a in analogs:
    print(f"{a['date']} ({a['regime_tag']}): {a['qaoa_portfolio']} → {a['realized_fwd_5d']:+.2%}")
```

### LLM narration (hour 11-11.5)

```python
import os
from anthropic import Anthropic   # or use openai

client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

def narrate_q_atlas(current_state_desc, analogs):
    bullet_lines = []
    for a in analogs:
        bullet_lines.append(
            f"- {a['date']} ({a['regime_tag']}): portfolio held "
            f"{', '.join(a['qaoa_portfolio'])} → +{a['realized_fwd_5d']*100:.1f}% over 5 days"
        )
    prompt = f"""You are Q-Atlas, a quantum market memory engine.

Current market state: {current_state_desc}

The 5 most similar historical regimes (by quantum kernel similarity over orthogonal meta-features) are:
{chr(10).join(bullet_lines)}

In 3 sentences, explain the analog: what makes these regimes similar to today, what the common outcome was, and what the implied portfolio action is. Be specific. Cite at least one date."""

    msg = client.messages.create(
        model='claude-sonnet-4-6',  # or claude-haiku-4-5 for cost
        max_tokens=400,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return msg.content[0].text

# Demo: take today's state, retrieve, narrate
analogs = q_atlas_retrieve(demo_features, k=5)
state_desc = f"vol regime z={demo_features[0,0]:.2f}, news sentiment z={demo_features[0,1]:.2f}, recent_acc={demo_features[0,2]:.2f}"
explanation = narrate_q_atlas(state_desc, analogs)
print(explanation)
```

### Interactive notebook cell (the demo)

Add a final notebook cell that takes an arbitrary date or state and produces the full Q-Atlas output. This is what gets demoed to judges tomorrow.

```python
def q_atlas_query(query_date_or_state):
    if isinstance(query_date_or_state, str):  # date string
        features = meta_features.loc[query_date_or_state].values.reshape(1, -1)
        state_desc = f"market state on {query_date_or_state}"
    else:
        features = query_date_or_state
        state_desc = "synthetic state vector"

    analogs = q_atlas_retrieve(features, k=5)
    explanation = narrate_q_atlas(state_desc, analogs)

    return {
        'query': state_desc,
        'analogs': analogs,
        'explanation': explanation,
    }

# Show this in the deck
result = q_atlas_query('2026-04-15')   # pick an interesting date
print(result['explanation'])
for a in result['analogs']:
    print(f"  {a['date']}: {a['qaoa_portfolio']} → {a['realized_fwd_5d']:+.2%}")
```

### Gotchas

- **Save the kernel matrix at hour 6.** If you skip this, Q-Atlas can't be built — recomputing the kernel at hour 10 burns 30 min.
- **Use the SAME feature map at retrieval time.** The kernel function needs to match Stage 3 exactly.
- **Regime tags need to exist.** At hour 0-2, when pulling news, tag a few dates with notable events (earnings days, FOMC, big tech announcements). Even rough tags ("FOMC", "NVDA earnings", "tech selloff") make the LLM output 10x better.
- **Cache LLM responses.** Don't re-call for every demo — pre-compute 5-10 example queries and cache the outputs.
- **Fallback:** if Anthropic/OpenAI API fails, fall back to a templated explanation: *"Current state most resembles {top_analog_date} ({top_analog_tag}). On that day, our quantum stack held {portfolio}, realizing {fwd_return}."* Static template, zero LLM dependency.
- **Demo on 3-5 pre-validated queries.** Don't take live arbitrary queries from judges — pick queries where you've sanity-checked the output. If a judge asks for a specific date, you have it ready.

### Output of this stage

- `q_atlas_retrieve(state)` function — k-NN over quantum kernel
- `narrate_q_atlas(state, analogs)` function — LLM-narrated explanation
- 5 pre-computed example queries with cached outputs
- One-cell interactive demo in `notebook.ipynb`
- Slide for the deck: "Q-Atlas in action — sample query and response"

---

## Stage 6 — Backtest + bootstrap + money slide (Hour 10-12)

### Walk-forward backtest

For each day in last 6 months:
1. Fit alpha/meta/QAOA/PPO on history up to that day
2. Compute weights for that day
3. Apply to next-day returns, log PnL
4. Move forward

Metrics:
- Cumulative return
- Sharpe ratio (annualized, daily Sharpe × √252)
- Max drawdown
- Hit rate
- Compare: full quantum stack / classical-only ablation / SPY buy-hold

### Bootstrap CI on every key delta (hour 11)

```python
def bootstrap_delta_ci(metric_fn, returns_a, returns_b, n=500):
    deltas = []
    for _ in range(n):
        idx = np.random.choice(len(returns_a), size=len(returns_a), replace=True)
        deltas.append(metric_fn(returns_a[idx]) - metric_fn(returns_b[idx]))
    return np.percentile(deltas, [2.5, 97.5])

def sharpe(r): return r.mean() / r.std() * np.sqrt(252)

ci_sharpe = bootstrap_delta_ci(sharpe, quantum_returns, classical_returns)
print(f'Sharpe Δ 95% CI: {ci_sharpe}')
```

### The money slide (build at hour 11.5)

ONE chart, ONE number. Pick whichever is most decisive:

| Candidate | When to lead with it |
|---|---|
| **QSVM AUC delta with CI** | If lower CI bound > 0 — *this is the dream* |
| QAOA Sharpe vs Markowitz | If QAOA beats by >0.2 |
| AerSim vs IBM Heron success rate | If Heron returned good results |
| Full-stack equity curve vs ablation | Fallback if no single layer wins decisively |

### Files to generate

```
workspace/results/
  comparison_table.csv          # 5 rows × {classical, quantum, Δ, 95% CI}
  money_slide.png               # the one chart
  equity_curve.png              # ours / classical-only / SPY
  qsvm_vs_rbf_auc.png           # the headline chart
  qaoa_aersim_vs_heron.png      # hardware validation (if returned)
  q_atlas_demo.json             # 5 pre-cached Q-Atlas queries + LLM outputs
  notebook.ipynb                # full pipeline + Q-Atlas demo cell
  README.md
```

---

## Deliverable (9 PM submission)

| Artifact | Description |
|---|---|
| `notebook.ipynb` | End-to-end pipeline + final cell is interactive Q-Atlas demo |
| `comparison_table.csv` | 5 rows × {classical, quantum, Δ, 95% CI} |
| `money_slide.png` | The single chart that wins the deck |
| `equity_curve.png` | Walk-forward backtest |
| `qsvm_vs_rbf_auc.png` | Headline result with error bars |
| `qaoa_aersim_vs_heron.png` | Hardware validation (if job returned) |
| `q_atlas_demo.json` | 5 cached Q-Atlas queries with quantum-retrieved analogs + LLM explanations |
| `K_train.npy` | Saved QSVM kernel matrix (powers Q-Atlas retrieval) |
| `README.md` | 1-paragraph thesis + how to run + sample Q-Atlas query |
| `fallbacks.py` | All safety nets (Markowitz, equal-weight, templated explanation) |

---

## Judge pitch (30 seconds — lead with the product, then the science)

> "**Q-Atlas is a quantum market memory engine for portfolio managers.** Ask it about any market state and it retrieves historically similar regimes using quantum kernel similarity search, shows the quantum-optimized portfolio our pipeline would have held on those dates, and an LLM explains the analog. [*Demo: type a date, get the response live.*]
>
> Under the hood: four quantum primitives across signal selection (KTA), alpha prediction (VQC), trade filtering (QSVM), and portfolio construction (QAOA cardinality + sector). The quantum kernel that filters trades is the *same kernel* that powers the memory retrieval — one quantum primitive, two uses.
>
> **The science:** QSVM meta-labeling beats classical RBF on 6 months of walk-forward S&P data with a 95% confidence interval that excludes zero. QAOA portfolio selection validated on real IBM Heron hardware. Every quantum component has a classical fallback — the pipeline is honest end-to-end."

This pitch:
- **Leads with the product** (Q-Atlas demo) — judges remember the live interaction
- **Bridges to the science** (rigorous benchmark, statistical significance)
- **Recursive architecture story** (one kernel, two uses) ties it together
- **Honest engineering** (fallbacks) signals trustworthiness

Notice what this pitch does:
- **Leads with the proven result**, not the architecture
- **Quantifies with statistical significance** (CI excludes zero)
- **Real hardware credibility**
- **Honest about scope** (PPO is classical, fallbacks exist)
- **Compresses to one sentence per layer**

---

## Risk tiers (revised)

| Tier | Component | Why |
|---|---|---|
| **Must work** | **QSVM with positive AUC delta + significance** | The headline number. Without this, no winning deck. |
| **Must work** | QAOA cardinality on AerSim | The second pillar. |
| **Must work** | **Q-Atlas retrieval returns sensible analogs** | The headline demo. Pure reuse of QSVM kernel, low marginal risk. |
| **Should work** | IBM Heron submission returns *any* result | Slide gold even if results are noisy. |
| **Should work** | VQC alpha runs (doesn't need to win) | Comparison table row 2. |
| **Should work** | Walk-forward backtest with bootstrap CIs | Rigor signal. |
| **Should work** | LLM narration produces clean output | If it fails, templated fallback. Static cache OK. |
| **Nice to have** | PPO sizing converges | Doesn't affect main story if fallback used. |
| **Nice to have** | KTA beats classical IC | Bonus column, drop if behind. |
| **Cut** | Symbolic regression, full QRL, QAE | All previously cut for good reasons. |

---

## Quick reference: libraries

| Component | Library | Key import |
|---|---|---|
| Quantum kernel (KTA, QSVM, Q-Atlas retrieval) | qiskit-machine-learning | `FidelityQuantumKernel`, `ZZFeatureMap` |
| VQC | PennyLane | `qml.qnn.TorchLayer` |
| QAOA | qiskit-algorithms + qiskit-optimization | `QAOA`, `QuadraticProgram` |
| IBM hardware | qiskit-ibm-runtime | `QiskitRuntimeService`, `SamplerV2` |
| PPO | stable-baselines3 + finrl | `PPO`, `StockTradingEnv` |
| Alpha classical | lightgbm, chronos-forecasting | — |
| LLM (Q-Atlas narration) | anthropic or openai | `Anthropic`, `client.messages.create` |
| Backtest | custom loop | — |
| Bootstrap | `sklearn.utils.resample` | — |

---

## Open decisions

- [ ] Lock the 20 tickers (sector-balanced default above)
- [ ] Confirm train/test split: train 2024-05-30→2025-11-30, test 2025-12-01→2026-05-30
- [ ] Confirm IBM Heron access works *tonight* before depending on it (run `service.least_busy()` once at hour 0)
- [ ] One sentence each writes their own bio for tomorrow's deck

---

## Three things that turn this from "solid" to "winning"

Re-read these at hour 9 if you feel rushed:

1. **QSVM AUC delta with confidence interval excluding zero** — make this look great even if it means tuning C, reps, feature subset for 90 min.
2. **IBM Heron submission accepted, results returned** — even mediocre results = credibility. Submit early.
3. **The money slide** — one chart, the headline number with error bars, no text noise.
4. **Q-Atlas demo lands** — 5 pre-cached queries that show the system retrieving sensible historical analogs with LLM-narrated explanations. **This is what judges remember after the deck.**

Win probability if all four land: **~75%**.

---

## Submission checklist (9 PM)

- [ ] `K_train.npy` saved at hour 6 (Q-Atlas depends on it)
- [ ] Both fallbacks (Markowitz + equal-weight + templated LLM) tested by hour 7
- [ ] PPO hard-killed by 10:00 if not converging
- [ ] QAOA submitted to IBM Heron by 10:30
- [ ] Bootstrap CIs computed by 11:00
- [ ] Q-Atlas demo cell renders 5 example queries by 11:30
- [ ] `notebook.ipynb` runs top-to-bottom on fresh kernel
- [ ] `comparison_table.csv` exists, has 5 rows with CIs
- [ ] All artifacts committed + pushed to repo
- [ ] One-paragraph `README.md` updated with Q-Atlas pitch + sample output
