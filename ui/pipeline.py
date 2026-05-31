"""
Inference pipeline: yfinance OHLCV -> notebook-01 signals -> LightGBM alpha.

Mirrors the signal construction in `01_data_signals.ipynb` (14 base signals +
14 cross-sectional rank versions = 28 features) and loads the model trained in
`02_alpha_models.ipynb` (`workspace/models/lgb_alpha.txt`). The model predicts
5-day forward returns per (date, ticker).

The cross-sectional rank features require the whole 20-ticker universe to be
fetched together — `get_panel()` does that once and caches the result.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
MODELS_DIR = WORKSPACE / "models"

with open(MODELS_DIR / "signal_cols.json") as f:
    _META = json.load(f)
SIGNAL_COLS: list[str] = _META["signal_cols"]
TICKERS: list[str] = _META["tickers"]

BASE_SIGNALS = [
    "mom_5d", "mom_20d", "mom_60d", "rev_1d",
    "vol_20d", "vol_of_vol", "rsi_14",
    "news_sent", "vol_ratio", "spread_prox",
    "vol_adj_mom", "dist_to_high", "dist_to_low", "pct_above_200ma",
]

# Each predicted value covers 5 forward business days.
FORECAST_HORIZON_DAYS = 5

# Models the UI dropdown exposes. Keep in sync with the dispatcher below.
AVAILABLE_MODELS: list[dict] = [
    {"key": "lightgbm", "label": "LightGBM",  "desc": "Gradient-boosted trees on 28 signals"},
    {"key": "vqc",      "label": "VQC",       "desc": "Variational Quantum Circuit (4 qubits)"},
    {"key": "momentum", "label": "Momentum",  "desc": "Classical 5-day momentum baseline"},
]

_LGB_MODEL: lgb.Booster | None = None
_VQC_BUNDLE: dict | None = None


def get_lgb_model() -> lgb.Booster:
    global _LGB_MODEL
    if _LGB_MODEL is None:
        _LGB_MODEL = lgb.Booster(model_file=str(MODELS_DIR / "lgb_alpha.txt"))
    return _LGB_MODEL


def get_vqc_bundle() -> dict:
    """Lazy-load the VQC model, scaler, and feature list. Heavy import (torch +
    pennylane), so we only pay it the first time the user picks VQC."""
    global _VQC_BUNDLE
    if _VQC_BUNDLE is not None:
        return _VQC_BUNDLE

    import torch
    import torch.nn as nn
    import pennylane as qml

    with open(MODELS_DIR / "vqc_features.json") as f:
        meta = json.load(f)
    features = meta["features"]
    n_qubits = meta["n_qubits"]
    n_layers = meta["n_layers"]

    with open(MODELS_DIR / "vqc_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Same circuit as 02_alpha_models.ipynb cell `vqc`.
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def vqc_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    class VQCRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(
                vqc_circuit, {"weights": (n_layers, n_qubits)}
            )

        def forward(self, x):
            return torch.stack([self.qlayer(xi) for xi in x]).reshape(-1, 1)

    model = VQCRegressor()
    state = torch.load(MODELS_DIR / "vqc_alpha.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    _VQC_BUNDLE = {
        "torch": torch,
        "model": model,
        "scaler": scaler,
        "features": features,
    }
    return _VQC_BUNDLE


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = -delta.where(delta < 0, 0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# --- Universe panel cache -------------------------------------------------
# Need 200+ days of history to make pct_above_200ma well-defined. yfinance
# "2y" gives ~500 trading days, comfortably enough.
_PANEL_CACHE: tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
_PANEL_TTL = 60 * 30  # 30 min


def get_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (close, high, low) wide DataFrames for the full ticker universe."""
    global _PANEL_CACHE
    now = time.time()
    if _PANEL_CACHE and now - _PANEL_CACHE[0] < _PANEL_TTL:
        return _PANEL_CACHE[1], _PANEL_CACHE[2], _PANEL_CACHE[3]

    raw = yf.download(
        TICKERS, period="2y", interval="1d",
        progress=False, auto_adjust=True, group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no panel data")

    close = raw["Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    for df in (close, high, low):
        df.columns = [str(c) for c in df.columns]
    close = close.reindex(columns=TICKERS).astype(float)
    high = high.reindex(columns=TICKERS).astype(float)
    low = low.reindex(columns=TICKERS).astype(float)
    close.index = pd.to_datetime(close.index)
    high.index = pd.to_datetime(high.index)
    low.index = pd.to_datetime(low.index)

    _PANEL_CACHE = (now, close, high, low)
    return close, high, low


# --- Signals --------------------------------------------------------------
def compute_signals(
    prices: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Per-ticker 28-column signal DataFrames. Mirrors notebook 01 cell `signals`."""
    returns = prices.pct_change()
    panels: dict[str, pd.DataFrame] = {}

    for t in prices.columns:
        r = returns[t]
        p = prices[t]
        h = highs[t]
        l = lows[t]
        sig = pd.DataFrame(index=prices.index)
        sig["mom_5d"]          = r.rolling(5).sum()
        sig["mom_20d"]         = r.rolling(20).sum()
        sig["mom_60d"]         = r.rolling(60).sum()
        sig["rev_1d"]          = -r
        sig["vol_20d"]         = r.rolling(20).std()
        sig["vol_of_vol"]      = sig["vol_20d"].rolling(20).std()
        sig["rsi_14"]          = _rsi(p)
        sig["news_sent"]       = 0.0  # no realtime news source at inference time
        sig["vol_ratio"]       = r.rolling(5).std() / r.rolling(60).std().replace(0, np.nan)
        sig["spread_prox"]     = (h - l) / p
        sig["vol_adj_mom"]     = sig["mom_20d"] / sig["vol_20d"].replace(0, np.nan)
        sig["dist_to_high"]    = (p - p.rolling(60).max()) / p.rolling(60).max()
        sig["dist_to_low"]     = (p - p.rolling(60).min()) / p.rolling(60).min().replace(0, np.nan)
        sig["pct_above_200ma"] = (p - p.rolling(200).mean()) / p.rolling(200).mean()
        panels[t] = sig

    # Cross-sectional percentile rank for each base signal (the big IC lift).
    for s in BASE_SIGNALS:
        wide = pd.DataFrame({t: panels[t][s] for t in prices.columns})
        wide_rank = wide.rank(axis=1, pct=True)
        for t in prices.columns:
            panels[t][s + "_rank"] = wide_rank[t]

    # Ensure column order matches the trained model.
    for t in panels:
        panels[t] = panels[t][SIGNAL_COLS]
    return panels


# --- Per-model prediction backends ---------------------------------------
# Cache: (model_key, ticker, panel_vintage_ts) -> Series. Vintage keys ensure
# we recompute when the underlying yfinance panel cache refreshes.
_PRED_CACHE: dict[tuple[str, str, pd.Timestamp], pd.Series] = {}


def _panel_vintage(close: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(close.index.max())


def _predict_lightgbm(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    feats = panels[ticker].dropna()
    if feats.empty:
        return pd.Series(dtype=float)
    preds = get_lgb_model().predict(feats.values)
    return pd.Series(preds, index=feats.index, name="pred_fwd5")


def _predict_vqc(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    bundle = get_vqc_bundle()
    torch = bundle["torch"]
    feats = panels[ticker][bundle["features"]].dropna()
    if feats.empty:
        return pd.Series(dtype=float)

    # Same input pre-processing as the notebook: StandardScaler -> * (pi/3),
    # clipped to [-pi, pi].
    x = bundle["scaler"].transform(feats.values) * (np.pi / 3.0)
    x = np.clip(x, -np.pi, np.pi)
    x_t = torch.tensor(x, dtype=torch.float32)

    preds = []
    batch_size = 200
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            preds.append(bundle["model"](x_t[i:i + batch_size]).numpy())
    arr = np.concatenate(preds).flatten()
    return pd.Series(arr, index=feats.index, name="pred_fwd5")


def _predict_momentum(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Classical baseline: predict the trailing 5-day return continues. No
    model file — pure feature read-through, gives the UI an instant contrast."""
    if "mom_5d" not in panels[ticker].columns:
        return pd.Series(dtype=float)
    s = panels[ticker]["mom_5d"].dropna()
    return s.rename("pred_fwd5")


_BACKENDS = {
    "lightgbm": _predict_lightgbm,
    "vqc":      _predict_vqc,
    "momentum": _predict_momentum,
}


def predict_fwd5_returns(ticker: str, model: str = "lightgbm") -> pd.Series:
    """Predicted 5-day forward returns for `ticker` from the chosen model."""
    backend = _BACKENDS.get(model)
    if backend is None:
        raise ValueError(f"Unknown model: {model!r} (have: {list(_BACKENDS)})")

    close, high, low = get_panel()
    vintage = _panel_vintage(close)
    cache_key = (model, ticker, vintage)
    cached = _PRED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    panels = compute_signals(close, high, low)
    if ticker not in panels:
        return pd.Series(dtype=float)

    result = backend(ticker, panels)
    _PRED_CACHE[cache_key] = result
    return result
