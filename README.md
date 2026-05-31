# Qubitrade - Quantum Market Memory for Explainable Alpha

> **Track 2 · MFT Portfolio Optimisation** · with Track 4 (Foundation Models) overlap  
> Quantum hardware: **xpyq cloud QPU** · LLM: **Google Gemini 2.5 Flash** · UI: **FastAPI + Plotly**

---

## What it does

Qubitrade is a full quant trading pipeline that uses **quantum computing at every major stage** — from alpha generation through portfolio sizing to trade explanation. Given any ticker and date, it asks:

> *"What historical market regime does this most resemble, and what happened next?"*

A quantum kernel searches the full price history, retrieves the 5 most structurally similar regimes, and Gemini narrates the plausible cause in plain English.

---

## Pipeline Overview

```
Raw OHLCV + News
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · Data & Signals          01_data_signals.ipynb    │
│  20 tickers · 2013–2020 · 47 features                       │
│  Signals: momentum, volatility, RSI, news sentiment, regime │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 · Alpha Models            02_alpha_models.ipynb    │
│  14 models: LightGBM, RF, XGB, LSTM, VQC ⚛ …               │
│  Best IC = 0.0916 (RandomForest)                            │
│  ⚛ VQC (Variational Quantum Classifier) as alpha model      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 · QSVM Meta-Labeling      04_meta_labeling.ipynb   │
│  ⚛ ZZFeatureMap (4 qubits, reps=2) quantum kernel           │
│  QSVM outperforms classical SVM on 5 / 14 models            │
│  Best improvement: +0.2136 F1 on LightGBM                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 · PPO Portfolio Sizing    05_ppo_portfolio.ipynb   │
│  Proximal Policy Optimisation with Differential Sharpe       │
│  reward · Sharpe 1.038 vs 0.859 baseline (+21%)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 · Q-Atlas                 06_q_atlas.ipynb         │
│                        08_interactive_q_atlas.ipynb         │
│  ⚛ xpyq cloud QPU · quantum covariance kernel               │
│  Joint signal + news latent space (8D) via Gemini embeddings │
│  K-nearest analog retrieval → Gemini narrative synthesis     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6 · Results Dashboard       07_results_dashboard.ipynb│
│  Dark-theme master figure · all-stage scorecard             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Interactive UI                    ui/                      │
│  FastAPI + Plotly · Chart / PPO Trader / Q-Atlas tabs       │
│  http://localhost:8000                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quantum Components

| Stage | Technique | Hardware |
|-------|-----------|----------|
| Stage 2 - Alpha | **VQC** (Variational Quantum Classifier) | Local Qiskit simulation |
| Stage 3 - Meta-labeling | **QSVM** · ZZFeatureMap (4 qubits, reps=2) | Local Qiskit simulation |
| Stage 5 - Q-Atlas KNN | **Quantum covariance kernel** · xpyq API | **xpyq cloud QPU** ⚛ |
| Stage 5 - Joint latent | **8-qubit ZZFeatureMap** (signals + news PCA) | Local Qiskit simulation |

The Q-Atlas kernel uses a **two-stage approach**: classical cosine pre-filter (top-100) → xpyq quantum covariance rerank. This keeps QPU usage efficient while capturing entangled cross-feature correlations that cosine similarity misses.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best alpha IC (RandomForest) | **0.0916** |
| QSVM wins vs classical SVM | **5 / 14 models** |
| Best QSVM improvement | **+0.2136 F1** (LightGBM) |
| PPO Sharpe ratio | **1.038** vs 0.859 baseline |
| Q-Atlas quantum vs cosine avg 5d fwd | **+1.04%** vs +0.63% |
| COVID analog signal (5/5 positive) | **+45.3% avg** 5d fwd (BA, Mar 2020) |

---

## Notebooks

| Notebook | Stage | Description |
|----------|-------|-------------|
| `01_data_signals.ipynb` | 1 | OHLCV download, feature engineering, news sentiment |
| `02_alpha_models.ipynb` | 2 | 14 alpha models including VQC ⚛ |
| `03_inference.ipynb` | 2.5 | Run trained models on test set |
| `04_meta_labeling.ipynb` | 3 | QSVM ⚛ meta-labeling, quantum vs classical comparison |
| `05_ppo_portfolio.ipynb` | 4 | PPO agent with Differential Sharpe reward |
| `06_q_atlas.ipynb` | 5 | Q-Atlas core: quantum KNN, Gemini narration |
| `07_results_dashboard.ipynb` | 6 | Master results dashboard, all-stage figures |
| `08_interactive_q_atlas.ipynb` | 5 | **Interactive demo** — ask any question about any ticker |

---

## Q-Atlas: Joint Signal + News Latent Space

The centrepiece of Stage 5. Every market date is represented as an **8-dimensional joint vector**:

```
4D signals  [vol_20d · news_sent · mom_20d · dist_to_high]  ─┐
                                                               ├─▶  8D joint state
4D news PCA  Gemini text-embedding-004 → 768D → PCA(4D)     ─┘
```

When a user asks *"Why is NVDA moving?"*:
1. Build / load cached joint embedding for all NVDA dates
2. Find top-5 nearest historical regimes in joint space (xpyq QPU)
3. Retrieve raw news text from each analog
4. Gemini synthesises: `[WHY] · [SIGNAL] · [RISK]`

---

## Running the UI

```bash
cd ui
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-key"
export XPYQ_KEY="your-xpyq-key"          # optional — falls back to local Qiskit
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — three tabs:
- **Chart View** — quantum analog chart for any ticker + timeframe
- **PPO Trader** — step through the PPO agent's historical decisions
- **⚛ Q-Atlas** — ask any question, get a quantum-retrieved narrative

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Quantum kernels | Qiskit · ZZFeatureMap · FidelityStatevectorKernel |
| Quantum cloud | **xpyq** (`quant.covariance` QPU operator) |
| RL portfolio | Stable-Baselines3 PPO |
| Alpha models | LightGBM · XGBoost · RandomForest · LSTM · VQC |
| LLM narration | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| News embeddings | Gemini `text-embedding-004` (768D → PCA 4D) |
| Price data | yfinance |
| News sources | Polygon.io · NewsAPI · DuckDuckGo News |
| News datasets | Kaggle financial headlines |
| UI backend | FastAPI · uvicorn |
| UI charts | Plotly.js |
| Notebook runtime | Jupyter · Python 3.11 |
