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

import importlib
import importlib.util
import json
import os
import pickle
import time
from pathlib import Path

# torch must come BEFORE lightgbm — otherwise lightgbm's OpenMP runtime claims
# libomp and torch ops segfault silently on the first forward pass (macOS).
import torch  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
MODELS_DIR = WORKSPACE / "models"
KRONOS_REPO = WORKSPACE / "external" / "Kronos"  # cloned repo (see README)
DATA_DIR = WORKSPACE / "data"

# Fixed historical window — matches notebooks 01/02 exactly.
DATA_START      = "2013-01-01"
DATA_END        = "2020-06-12"   # exclusive upper bound for yfinance (data thru 2020-06-11)
TRAIN_END_DATE  = "2018-12-31"
TEST_START_DATE = "2019-01-01"

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

def _has_module(name: str) -> bool:
    """Is the package importable? Uses find_spec so we don't actually execute
    the module — some packages (timesfm in particular) have heavy side effects
    on import that destabilize the rest of the process."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _xpyq_key() -> str | None:
    return os.environ.get("XPYQ_KEY") or os.environ.get("XPYQ_API_KEY")


def _sklearn_available(name: str) -> bool:
    return (MODELS_DIR / f"{name}_alpha.pkl").exists()


def _kronos_available() -> bool:
    return (KRONOS_REPO / "model" / "__init__.py").exists()


_NN_ARTIFACTS = ["nn_mlp.pt", "nn_transformer.pt", "nn_weekly.pt", "nn_monthly.pt",
                 "nn_quarterly.pt", "nn_meta_learner.pkl", "nn_scaler.pkl",
                 "nn_features.json"]


def _nn_available() -> bool:
    return all((MODELS_DIR / f).exists() for f in _NN_ARTIFACTS)


# Registry order = dropdown order. The `available` predicate gates whether the
# key shows up in the UI at all — we never expose a model whose backend can't
# load on this machine.
_MODEL_REGISTRY = [
    ("lightgbm",      "LightGBM",      "Gradient-boosted trees on 28 signals",       lambda: True),
    ("random_forest", "RandomForest",  "150-tree random forest on 28 signals",       lambda: _sklearn_available("random_forest")),
    ("ridge",         "Ridge",         "L2-regularized linear regression",           lambda: _sklearn_available("ridge")),
    ("linear_reg",    "LinearReg",     "Ordinary least squares",                     lambda: _sklearn_available("linear_reg")),
    ("ensemble",      "Ensemble",      "Mean of LightGBM + Linear + Ridge + RF",     lambda: _sklearn_available("ridge")),
    ("agent_ensemble","Wtd Ensemble","Static weights: 0.4 LGB · 0.3 XpyQ · 0.2 VQC · 0.1 Chronos", lambda: True),
    # Neural ensemble (notebook 02.2). All six load from the same nn_*.pt /
    # nn_meta_learner.pkl bundle, so one availability check covers them all.
    ("nn_mlp",        "NN-MLP",        "3-layer MLP on 28 signals (Stage 2.2)",         _nn_available),
    ("nn_transformer","NN-Transformer","4-head Transformer, 20d lookback (Stage 2.2)",  _nn_available),
    ("nn_weekly",     "NN-Weekly",     "LSTM, 5d lookback — weekly momentum",           _nn_available),
    ("nn_monthly",    "NN-Monthly",    "LSTM, 21d lookback — monthly trend",            _nn_available),
    ("nn_quarterly",  "NN-Quarterly",  "LSTM, 63d lookback — quarterly regime",         _nn_available),
    ("nn_meta",       "NN-Meta",       "Ridge meta-learner over all 5 neural models",   _nn_available),
    ("momentum",      "Momentum",      "Naive 5-day momentum baseline",              lambda: True),
    ("vqc",           "VQC",           "Variational Quantum Circuit (4 qubits)",     lambda: _has_module("pennylane")),
    ("chronos",       "Chronos",       "Amazon zero-shot time-series transformer",   lambda: _has_module("chronos")),
    ("kronos",        "Kronos",        "Finance-native OHLCV foundation model",      _kronos_available),
    # TimesFM 2.0 segfaults during load on some setups (e.g. Apple Silicon
    # without CUDA). Opt-in via TIMESFM_ENABLED=1 once you've verified it
    # loads in your env.
    ("timesfm",       "TimesFM",       "Google 500M time-series foundation model",
     lambda: _has_module("timesfm") and os.environ.get("TIMESFM_ENABLED") == "1"),
    ("xpyq",          "XpyQ-EWMA",     "xpyq remote: EWMA + factor exposure",        lambda: _xpyq_key() is not None),
]


# Notebook label (left) -> registry key (right). The comparison CSV uses the
# notebook's display names; we want metrics keyed by the dropdown key.
_METRIC_LABEL_TO_KEY = {
    "lightgbm":       "lightgbm",
    "linearreg":      "linear_reg",
    "ridge":          "ridge",
    "randomforest":   "random_forest",
    "ensemble":       "ensemble",        # classical equal-weight ensemble
    # Notebook 02.2 names. The neural comparison numbers aren't in the CSV yet;
    # these mappings let the metrics drop into place automatically once they are.
    "mlp":            "nn_mlp",
    "transformer":    "nn_transformer",
    "nn-weekly":      "nn_weekly",
    "nn-monthly":     "nn_monthly",
    "nn-quarterly":   "nn_quarterly",
    "metalearner":    "nn_meta",
    "mom_5d (naive)": "momentum",
    "vqc":            "vqc",
    "chronos":        "chronos",
    "kronos":         "kronos",
    "xpyq-ewma":      "xpyq",
    "timesfm":        "timesfm",
}


def _load_notebook_metrics() -> dict[str, dict]:
    """Per-model IC / dir_acc / Sharpe keyed by registry key. Prefers the
    richer stage2_model_comparison.csv (11 models), falls back to
    stage2_primary.json (6 models)."""
    out: dict[str, dict] = {}

    csv_path = DATA_DIR / "stage2_model_comparison.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path).set_index("model")
            for label, row in df.iterrows():
                key = _METRIC_LABEL_TO_KEY.get(str(label).lower())
                if not key:
                    continue
                out[key] = {
                    "IC":                row.get("IC"),
                    "dir_acc":           row.get("dir_acc"),
                    "long_short_sharpe": row.get("long_short_sharpe"),
                }
        except Exception:
            pass

    json_path = DATA_DIR / "stage2_primary.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                blob = json.load(f)
            for metric_key, by_model in blob.get("all_results", {}).items():
                for label, val in by_model.items():
                    key = _METRIC_LABEL_TO_KEY.get(label.lower())
                    if key and key not in out:
                        out.setdefault(key, {})[metric_key] = val
        except Exception:
            pass

    return out


_NOTEBOOK_METRICS = _load_notebook_metrics()


def _build_available_models() -> list[dict]:
    out: list[dict] = []
    for key, label, desc, avail in _MODEL_REGISTRY:
        if not avail():
            continue
        entry = {"key": key, "label": label, "desc": desc}
        metrics = _NOTEBOOK_METRICS.get(key)
        if metrics:
            entry["metrics"] = {
                "ic":      _safe_float(metrics.get("IC")),
                "dir_acc": _safe_float(metrics.get("dir_acc")),
                "sharpe":  _safe_float(metrics.get("long_short_sharpe")),
            }
        out.append(entry)
    return out


def _safe_float(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)  # nan guard
    except (TypeError, ValueError):
        return None


AVAILABLE_MODELS: list[dict] = _build_available_models()

_LGB_MODEL: lgb.Booster | None = None
_VQC_BUNDLE: dict | None = None
_CHRONOS_PIPE = None
_TIMESFM_MODEL = None
_SKLEARN_CACHE: dict[str, object] = {}
_KRONOS_BUNDLE: dict | None = None


def _get_sklearn_model(name: str):
    if name not in _SKLEARN_CACHE:
        with open(MODELS_DIR / f"{name}_alpha.pkl", "rb") as f:
            _SKLEARN_CACHE[name] = pickle.load(f)
    return _SKLEARN_CACHE[name]


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
        TICKERS, start=DATA_START, end=DATA_END, interval="1d",
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


# --- Chronos (Amazon, zero-shot) -----------------------------------------
def _get_chronos_pipeline():
    """Lazy-load chronos-t5-small once. First call downloads ~500 MB."""
    global _CHRONOS_PIPE
    if _CHRONOS_PIPE is not None:
        return _CHRONOS_PIPE
    from chronos import ChronosPipeline
    import torch as _torch
    if _torch.backends.mps.is_available() and _torch.backends.mps.is_built():
        device = "mps"
    elif _torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    try:
        _CHRONOS_PIPE = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map=device, dtype=_torch.float32,
        )
    except Exception:
        _CHRONOS_PIPE = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map="cpu", dtype=_torch.float32,
        )
    return _CHRONOS_PIPE


def _predict_chronos(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Per-date 5d-fwd return from chronos forecasts of the close path. Strides
    every 5 dates and forward-fills between, matching notebook 02."""
    close, _, _ = get_panel()
    if ticker not in close.columns:
        return pd.Series(dtype=float)
    import torch as _torch
    pipe = _get_chronos_pipeline()
    series = close[ticker].dropna()
    if len(series) < 25:
        return pd.Series(dtype=float)

    CONTEXT_LEN = 60
    PRED_HORIZON = FORECAST_HORIZON_DAYS
    STRIDE = 5
    # Start where there's enough history for a context window.
    dates = list(series.index[CONTEXT_LEN::STRIDE])
    preds: dict[pd.Timestamp, float] = {}
    for d in dates:
        end_loc = series.index.get_loc(d)
        ctx = series.iloc[max(0, end_loc - CONTEXT_LEN):end_loc]
        if len(ctx) < 20:
            continue
        ctx_t = _torch.tensor(ctx.values, dtype=_torch.float32).unsqueeze(0)
        try:
            fc = pipe.predict(ctx_t, prediction_length=PRED_HORIZON)
            med = fc[0].median(dim=0).values.cpu().numpy()
            current = float(ctx.iloc[-1])
            preds[d] = (float(med[-1]) - current) / current
        except Exception:
            continue
    if not preds:
        return pd.Series(dtype=float)
    s = pd.Series(preds).sort_index()
    # Forward-fill onto the full date index so the predictor sees one value/day.
    return s.reindex(series.index).ffill().dropna().rename("pred_fwd5")


# --- TimesFM (Google, zero-shot) -----------------------------------------
def _get_timesfm_model():
    """Lazy-load timesfm-2.0-500m-pytorch."""
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL
    import timesfm as _tfm
    backend = "cpu"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            backend = "gpu"
    except Exception:
        pass
    _TIMESFM_MODEL = _tfm.TimesFm(
        hparams=_tfm.TimesFmHparams(
            backend=backend, per_core_batch_size=32, horizon_len=16,
            num_layers=50, use_positional_embedding=False, context_len=128,
        ),
        checkpoint=_tfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    return _TIMESFM_MODEL


def _predict_timesfm(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Same shape as Chronos: stride-5 forecasts, ffill between."""
    close, _, _ = get_panel()
    if ticker not in close.columns:
        return pd.Series(dtype=float)
    model = _get_timesfm_model()
    series = close[ticker].dropna()
    if len(series) < 25:
        return pd.Series(dtype=float)

    CONTEXT_LEN = 60
    PRED_HORIZON = FORECAST_HORIZON_DAYS
    STRIDE = 5

    # Batch contexts for this ticker; TimesFM is much faster batched.
    contexts: list[np.ndarray] = []
    ctx_dates: list[pd.Timestamp] = []
    for d in series.index[CONTEXT_LEN::STRIDE]:
        end_loc = series.index.get_loc(d)
        ctx = series.iloc[max(0, end_loc - CONTEXT_LEN):end_loc].values.astype("float32")
        if len(ctx) < 20:
            continue
        contexts.append(ctx)
        ctx_dates.append(d)
    if not contexts:
        return pd.Series(dtype=float)
    try:
        point_fcast, _ = model.forecast(inputs=contexts, freq=[0] * len(contexts))
    except Exception:
        return pd.Series(dtype=float)

    preds: dict[pd.Timestamp, float] = {}
    for d, ctx, fc in zip(ctx_dates, contexts, point_fcast):
        current = float(ctx[-1])
        preds[d] = (float(fc[PRED_HORIZON - 1]) - current) / current
    s = pd.Series(preds).sort_index()
    return s.reindex(series.index).ffill().dropna().rename("pred_fwd5")


# --- XpyQ-EWMA (remote: EWMA + factor exposure on xpyq hardware) ---------
def _run_xpyq_ewma_panel() -> dict[str, pd.Series]:
    """Send the returns panel to xpyq, get back per-(date, ticker) forecasts.
    One REST call covers all 20 tickers; cached per panel vintage by callers."""
    import requests

    key = _xpyq_key()
    if not key:
        return {}
    base = "https://xpyq-lib-production.up.railway.app"
    close, _, _ = get_panel()
    returns = close.pct_change().fillna(0).astype(np.float32)
    if returns.empty:
        return {}

    tickers_order = list(returns.columns)
    returns_list = returns.values.tolist()
    n_lookback = 0  # use the whole window — emit forecasts for all dates

    code = (
        "import numpy as _np\n"
        "import json as _json\n"
        f"returns_arr = _np.array({returns_list}, dtype=_np.float32)\n"
        "R = from_numpy(returns_arr)\n"
        "T, N = returns_arr.shape\n"
        "cov = quant.covariance(R).numpy()\n"
        "eigvals, eigvecs = linalg.eigh(from_numpy(cov))\n"
        "eigvals = eigvals.numpy(); eigvecs = eigvecs.numpy()\n"
        "factor = eigvecs[:, -1]\n"
        "if factor.sum() < 0: factor = -factor\n"
        "alpha = 0.10\n"
        "ewma = _np.zeros_like(returns_arr)\n"
        "ewma[0] = returns_arr[0]\n"
        "for i in range(1, T): ewma[i] = alpha * returns_arr[i] + (1 - alpha) * ewma[i-1]\n"
        "factor_signal = ewma @ factor\n"
        "factor_signal_centered = factor_signal - _np.mean(factor_signal)\n"
        "forecast = 5.0 * ewma + 0.3 * (factor_signal_centered[:, None] * factor[None, :])\n"
        f"forecast_emit = forecast[{n_lookback}:]\n"
        'print("DATA_START")\n'
        "print(_json.dumps(forecast_emit.tolist()))\n"
        'print("DATA_END")\n'
    )

    H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        submit = requests.post(
            f"{base}/api/v1/compute/runs", headers=H,
            json={"code": code, "name": "xpyq_ewma_ui"}, timeout=15,
        )
        if submit.status_code not in (200, 202):
            return {}
        run_id = submit.json().get("run_id")
        if not run_id:
            return {}
        deadline = time.time() + 90
        poll: dict | None = None
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{base}/api/v1/compute/runs/{run_id}", headers=H, timeout=10
                )
                if r.status_code not in (200, 202):
                    time.sleep(1); continue
                p = r.json()
                if isinstance(p, dict):
                    poll = p
                    if poll.get("status") in ("completed", "failed", "timed_out", "cancelled"):
                        break
            except Exception:
                time.sleep(1); continue
            time.sleep(0.5)
        if not isinstance(poll, dict) or poll.get("status") != "completed":
            return {}
        stdout = poll.get("stdout") or ""
        if "DATA_START" not in stdout or "DATA_END" not in stdout:
            return {}
        payload = stdout.split("DATA_START")[1].split("DATA_END")[0].strip()
        arr = np.array(json.loads(payload), dtype=np.float32)
    except Exception:
        return {}

    # Reshape to per-ticker Series indexed by the returns dates.
    dates = returns.index[n_lookback:]
    if arr.shape != (len(dates), len(tickers_order)):
        return {}
    out: dict[str, pd.Series] = {}
    for j, t in enumerate(tickers_order):
        out[t] = pd.Series(arr[:, j], index=dates, name="pred_fwd5").dropna()
    return out


def _predict_xpyq(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Hit xpyq once, slice out this ticker. Cache writes the full 20-ticker
    result so the next call (any ticker) is free."""
    close, _, _ = get_panel()
    vintage = _panel_vintage(close)
    all_key = ("xpyq", "__ALL__", vintage)
    bundle = _PRED_CACHE.get(all_key)
    if bundle is None:
        bundle = _run_xpyq_ewma_panel()
        _PRED_CACHE[all_key] = bundle
    return bundle.get(ticker, pd.Series(dtype=float)).copy()


# --- sklearn classical baselines (Linear, Ridge, RandomForest) -----------
def _predict_sklearn(ticker: str, panels: dict[str, pd.DataFrame], name: str) -> pd.Series:
    feats = panels[ticker].dropna()
    if feats.empty:
        return pd.Series(dtype=float)
    preds = _get_sklearn_model(name).predict(feats.values)
    return pd.Series(preds, index=feats.index, name="pred_fwd5")


def _predict_linear(ticker, panels):        return _predict_sklearn(ticker, panels, "linear_reg")
def _predict_ridge(ticker, panels):         return _predict_sklearn(ticker, panels, "ridge")
def _predict_random_forest(ticker, panels): return _predict_sklearn(ticker, panels, "random_forest")


def _predict_ensemble(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight mean of the four classical models. Most ensembles outperform
    any single member by IC on this dataset (see notebook comparison)."""
    parts = []
    for fn in (_predict_lightgbm, _predict_linear, _predict_ridge, _predict_random_forest):
        s = fn(ticker, panels)
        if not s.empty:
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1).rename("pred_fwd5")


# Weights from `02.1_langchain_agent.ipynb` cell 26 — the agent's final
# heuristic ensemble. Any backend missing from this machine drops out and the
# remaining weights are renormalized so the call still works.
_AGENT_ENSEMBLE_WEIGHTS = {
    "lightgbm": 0.4,
    "xpyq":     0.3,
    "vqc":      0.2,
    "chronos":  0.1,
}


def _predict_agent_ensemble(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Reproduces the LangChain notebook's final 4-model weighted ensemble.

    Predictions come from each member's own backend (XpyQ and Chronos do their
    own caching), so a switch to this model after picking others is cheap."""
    parts: list[pd.Series] = []
    weights: list[float] = []
    for key, w in _AGENT_ENSEMBLE_WEIGHTS.items():
        backend = _BACKENDS.get(key)
        if backend is None:
            continue
        try:
            s = backend(ticker, panels)
        except Exception:
            continue
        if s.empty:
            continue
        parts.append(s)
        weights.append(w)
    if not parts:
        return pd.Series(dtype=float)
    wsum = sum(weights)
    df = pd.concat(parts, axis=1).fillna(0.0)
    df.columns = range(len(parts))
    blended = sum(df[i] * (w / wsum) for i, w in enumerate(weights))
    return blended.rename("pred_fwd5")


# --- Kronos (finance-native foundation model) ----------------------------
def _get_kronos():
    """Lazy-load Kronos. Repo is cloned under workspace/external/Kronos; we
    insert it on sys.path so `from model import …` resolves to the Kronos repo
    rather than any stray local module."""
    global _KRONOS_BUNDLE
    if _KRONOS_BUNDLE is not None:
        return _KRONOS_BUNDLE
    import sys
    repo = str(KRONOS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # `model` is a generic module name; do not let the import linger if it
    # would collide with anything else later in the process.
    from model import Kronos, KronosTokenizer, KronosPredictor

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    # CPU only — MPS path is unstable in upstream Kronos.
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    _KRONOS_BUNDLE = {"predictor": predictor}
    return _KRONOS_BUNDLE


def _predict_kronos(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    """Per-date 5d-fwd return from Kronos OHLCV forecasts. Same striding /
    ffill strategy as Chronos."""
    close, high, low = get_panel()
    if ticker not in close.columns:
        return pd.Series(dtype=float)
    bundle = _get_kronos()
    predictor = bundle["predictor"]

    px = close[ticker].dropna()
    hi = high[ticker].reindex(px.index)
    lo = low[ticker].reindex(px.index)
    if len(px) < 25:
        return pd.Series(dtype=float)

    CONTEXT_LEN = 60
    PRED_HORIZON = FORECAST_HORIZON_DAYS
    STRIDE = 5

    preds: dict[pd.Timestamp, float] = {}
    for d in px.index[CONTEXT_LEN::STRIDE]:
        end_loc = px.index.get_loc(d)
        sl = slice(max(0, end_loc - CONTEXT_LEN), end_loc)
        ctx_close = px.iloc[sl]
        if len(ctx_close) < 20:
            continue
        ctx_high = hi.iloc[sl].fillna(ctx_close)
        ctx_low = lo.iloc[sl].fillna(ctx_close)
        ctx_open = ctx_close.shift(1).fillna(ctx_close)
        ohlcv = pd.DataFrame({
            "open":   ctx_open.values,
            "high":   ctx_high.values,
            "low":    ctx_low.values,
            "close":  ctx_close.values,
            "volume": np.ones(len(ctx_close), dtype="float32"),
        }, index=ctx_close.index)
        y_ts = pd.bdate_range(ohlcv.index[-1], periods=PRED_HORIZON + 1)[1:]
        try:
            out = predictor.predict(
                df=ohlcv,
                x_timestamp=pd.Series(ohlcv.index),
                y_timestamp=pd.Series(y_ts),
                pred_len=PRED_HORIZON, T=1.0, top_p=0.9, sample_count=1,
            )
            pred_close = float(out["close"].values[-1])
            current = float(ctx_close.iloc[-1])
            preds[d] = (pred_close - current) / current
        except Exception:
            continue
    if not preds:
        return pd.Series(dtype=float)
    s = pd.Series(preds).sort_index()
    return s.reindex(px.index).ffill().dropna().rename("pred_fwd5")


# --- Neural ensemble (notebook 02.2): MLP, Transformer, 3 LSTMs, Ridge meta ---
_NN_BUNDLE: dict | None = None


def _get_nn_bundle() -> dict:
    """Lazy-load all 6 neural artifacts in one shot. Heavy torch import."""
    global _NN_BUNDLE
    if _NN_BUNDLE is not None:
        return _NN_BUNDLE

    import torch as _torch
    import torch.nn as _nn

    # Force CPU. MPS path is unstable for these custom nets under uvicorn —
    # silent segfaults inside Transformer/LSTM forward on Apple Silicon.
    _torch.set_default_device("cpu")

    with open(MODELS_DIR / "nn_features.json") as f:
        meta = json.load(f)
    signal_cols = meta["signal_cols"]
    n_features = len(signal_cols)
    dropout = meta["dropout"]

    # Net classes — must match the training script's architectures exactly.
    class MLPForecaster(_nn.Module):
        def __init__(self, n_in, hidden):
            super().__init__()
            layers, dim = [], n_in
            for h in hidden:
                layers += [_nn.Linear(dim, h), _nn.LayerNorm(h), _nn.GELU(), _nn.Dropout(dropout)]
                dim = h
            layers.append(_nn.Linear(dim, 1))
            self.net = _nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class TemporalTransformer(_nn.Module):
        def __init__(self, n_features, d_model, nhead, num_layers):
            super().__init__()
            self.input_proj = _nn.Linear(n_features, d_model)
            self.pos_enc = _nn.Parameter(_torch.zeros(1, 512, d_model))
            enc = _nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = _nn.TransformerEncoder(enc, num_layers=num_layers)
            self.head = _nn.Sequential(_nn.LayerNorm(d_model), _nn.Linear(d_model, 1))

        def forward(self, x):
            T = x.size(1)
            x = self.input_proj(x) + self.pos_enc[:, :T, :]
            x = self.encoder(x)
            return self.head(x[:, -1, :]).squeeze(-1)

    class ScaleLSTM(_nn.Module):
        def __init__(self, n_features, hidden, n_layers):
            super().__init__()
            self.lstm = _nn.LSTM(
                input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = _nn.Sequential(
                _nn.LayerNorm(hidden), _nn.Dropout(dropout), _nn.Linear(hidden, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def _load(model, fname):
        model = model.to("cpu")
        state = _torch.load(MODELS_DIR / fname, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.eval()

    mlp = _load(MLPForecaster(n_features, meta["mlp_hidden"]), "nn_mlp.pt")
    transformer = _load(
        TemporalTransformer(
            n_features, meta["transformer_d_model"],
            meta["transformer_nhead"], meta["transformer_layers"],
        ),
        "nn_transformer.pt",
    )
    lstms = {
        name: _load(
            ScaleLSTM(n_features, meta["lstm_hidden"], meta["lstm_layers"]),
            f"nn_{name}.pt",
        )
        for name in meta["scales"]
    }
    with open(MODELS_DIR / "nn_meta_learner.pkl", "rb") as f:
        meta_learner = pickle.load(f)
    with open(MODELS_DIR / "nn_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    _NN_BUNDLE = {
        "torch": _torch,
        "signal_cols": signal_cols,
        "scaler": scaler,
        "mlp": mlp,
        "transformer": transformer,
        "transformer_lookback": meta["transformer_lookback"],
        "lstms": lstms,                 # {name: model}
        "scales": meta["scales"],       # {name: lookback}
        "meta_learner": meta_learner,
        "meta_features": meta["meta_features"],
    }
    return _NN_BUNDLE


def _nn_scaled_feats(ticker: str, panels: dict[str, pd.DataFrame]) -> tuple:
    """Returns (scaled_df, dates_idx) for the ticker, in notebook column order."""
    bundle = _get_nn_bundle()
    cols = bundle["signal_cols"]
    feats = panels[ticker][cols].dropna()
    if feats.empty:
        return None, None
    x = bundle["scaler"].transform(feats.values)
    return pd.DataFrame(x, index=feats.index, columns=cols), feats.index


def _predict_mlp(ticker: str, panels: dict[str, pd.DataFrame]) -> pd.Series:
    bundle = _get_nn_bundle()
    torch = bundle["torch"]
    scaled, idx = _nn_scaled_feats(ticker, panels)
    if scaled is None:
        return pd.Series(dtype=float)
    with torch.no_grad():
        preds = bundle["mlp"](torch.tensor(scaled.values, dtype=torch.float32, device="cpu")).numpy()
    return pd.Series(preds, index=idx, name="pred_fwd5")


def _predict_seq_model(ticker, panels, key: str, lookback: int) -> pd.Series:
    """Shared sequence inference for Transformer + the 3 LSTMs."""
    bundle = _get_nn_bundle()
    torch = bundle["torch"]
    scaled, idx = _nn_scaled_feats(ticker, panels)
    if scaled is None or len(scaled) < lookback:
        return pd.Series(dtype=float)

    # Build (n_dates_with_history, lookback, n_features) tensor.
    arr = scaled.values
    seqs, out_idx = [], []
    for i in range(lookback, len(arr) + 1):
        seqs.append(arr[i - lookback:i])
        out_idx.append(idx[i - 1])
    seq_tensor = torch.tensor(np.array(seqs, dtype=np.float32), device="cpu")

    model = bundle["transformer"] if key == "transformer" else bundle["lstms"][key]
    preds = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(seq_tensor), batch):
            preds.append(model(seq_tensor[i:i + batch]).numpy())
    arr_out = np.concatenate(preds) if preds else np.array([])
    return pd.Series(arr_out, index=out_idx, name="pred_fwd5")


def _predict_transformer(ticker, panels):
    bundle = _get_nn_bundle()
    return _predict_seq_model(ticker, panels, "transformer", bundle["transformer_lookback"])


def _predict_nn_weekly(ticker, panels):
    return _predict_seq_model(ticker, panels, "weekly", _get_nn_bundle()["scales"]["weekly"])


def _predict_nn_monthly(ticker, panels):
    return _predict_seq_model(ticker, panels, "monthly", _get_nn_bundle()["scales"]["monthly"])


def _predict_nn_quarterly(ticker, panels):
    return _predict_seq_model(ticker, panels, "quarterly", _get_nn_bundle()["scales"]["quarterly"])


def _predict_nn_meta(ticker, panels):
    """Ridge meta-learner over (MLP + Transformer + 3 LSTMs). Only emits a
    prediction on dates where every sub-model has one."""
    bundle = _get_nn_bundle()
    members = {
        "mlp":          _predict_mlp(ticker, panels),
        "transformer":  _predict_transformer(ticker, panels),
        "nn_weekly":    _predict_nn_weekly(ticker, panels),
        "nn_monthly":   _predict_nn_monthly(ticker, panels),
        "nn_quarterly": _predict_nn_quarterly(ticker, panels),
    }
    df = pd.DataFrame(members).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    df = df[bundle["meta_features"]]  # enforce column order saved at training
    preds = bundle["meta_learner"].predict(df.values)
    return pd.Series(preds, index=df.index, name="pred_fwd5")



# ── Neural models (Stage 2.2) ─────────────────────────────────────────────────
_NEURAL_BUNDLE: dict | None = None


def _get_neural_bundle() -> dict:
    """Lazy-load all Stage 2.2 neural models and the shared scaler."""
    global _NEURAL_BUNDLE
    if _NEURAL_BUNDLE is not None:
        return _NEURAL_BUNDLE

    import torch
    import torch.nn as nn

    _DROP = 0.2
    n_feat = len(SIGNAL_COLS)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, dim = [], n_feat
            for h in [256, 128, 64]:
                layers += [nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(_DROP)]
                dim = h
            layers.append(nn.Linear(dim, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_feat, hidden_size=64,
                                num_layers=2, batch_first=True, dropout=_DROP)
            self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(_DROP), nn.Linear(64, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class _Transformer(nn.Module):
        # Attribute names match the keys saved by ui/train_neural_models.py
        # (input_proj / pos_enc / encoder) — don't rename without retraining.
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_feat, 64)
            self.pos_enc   = nn.Parameter(torch.zeros(1, 512, 64))
            enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                                              dropout=_DROP, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))
        def forward(self, x):
            T = x.size(1)
            x = self.input_proj(x) + self.pos_enc[:, :T, :]
            return self.head(self.encoder(x)[:, -1, :]).squeeze(-1)

    bundle: dict = {"torch": torch}
    with open(MODELS_DIR / "nn_scaler.pkl", "rb") as f:
        bundle["scaler"] = pickle.load(f)

    _CLASSES = {
        "nn_mlp":        _MLP,
        "nn_weekly":     _LSTM,
        "nn_monthly":    _LSTM,
        "nn_quarterly":  _LSTM,
        "nn_transformer":_Transformer,
    }
    for key, cls in _CLASSES.items():
        path = MODELS_DIR / f"{key}.pt"
        if path.exists():
            m = cls()
            m.load_state_dict(torch.load(str(path), map_location="cpu"))
            m.eval()
            bundle[key] = m

    meta_path = MODELS_DIR / "nn_meta_learner.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            bundle["nn_meta"] = pickle.load(f)

    _NEURAL_BUNDLE = bundle
    return _NEURAL_BUNDLE


def _build_sequences(scaled_arr: np.ndarray, dates, lookback: int):
    seqs, idx = [], []
    for i in range(lookback, len(scaled_arr) + 1):
        window = scaled_arr[i - lookback:i]
        if np.isnan(window).any():
            continue
        seqs.append(window)
        idx.append(dates[i - 1])
    if not seqs:
        return np.empty((0, lookback, scaled_arr.shape[1]), dtype=np.float32), []
    return np.array(seqs, dtype=np.float32), idx


def _predict_nn_cross(ticker: str, panels: dict, model_key: str) -> pd.Series:
    bundle = _get_neural_bundle()
    if model_key not in bundle:
        return pd.Series(dtype=float)
    torch = bundle["torch"]
    feat  = panels[ticker].fillna(0)
    if feat.empty:
        return pd.Series(dtype=float)
    X = bundle["scaler"].transform(feat.values).astype(np.float32)
    with torch.no_grad():
        preds = bundle[model_key](torch.tensor(X)).numpy()
    return pd.Series(preds, index=feat.index, name="pred_fwd5")


def _predict_nn_seq(ticker: str, panels: dict, model_key: str, lookback: int) -> pd.Series:
    bundle = _get_neural_bundle()
    if model_key not in bundle:
        return pd.Series(dtype=float)
    torch = bundle["torch"]
    feat  = panels[ticker].fillna(0)
    if len(feat) < lookback:
        return pd.Series(dtype=float)
    scaled = bundle["scaler"].transform(feat.values).astype(np.float32)
    seqs, idx = _build_sequences(scaled, list(feat.index), lookback)
    if len(seqs) == 0:
        return pd.Series(dtype=float)
    with torch.no_grad():
        preds = bundle[model_key](torch.tensor(seqs)).numpy()
    return pd.Series(preds, index=idx, name="pred_fwd5")


def _predict_nn_mlp(ticker, panels):
    return _predict_nn_cross(ticker, panels, "nn_mlp")

def _predict_nn_transformer(ticker, panels):
    return _predict_nn_seq(ticker, panels, "nn_transformer", 20)

def _predict_nn_weekly(ticker, panels):
    return _predict_nn_seq(ticker, panels, "nn_weekly", 5)

def _predict_nn_monthly(ticker, panels):
    return _predict_nn_seq(ticker, panels, "nn_monthly", 21)

def _predict_nn_quarterly(ticker, panels):
    return _predict_nn_seq(ticker, panels, "nn_quarterly", 63)


def _predict_nn_meta(ticker: str, panels: dict) -> pd.Series:
    bundle = _get_neural_bundle()
    if "nn_meta" not in bundle:
        return pd.Series(dtype=float)
    parts = {
        "mlp":         _predict_nn_mlp(ticker, panels),
        "transformer": _predict_nn_transformer(ticker, panels),
        "nn_weekly":   _predict_nn_weekly(ticker, panels),
        "nn_monthly":  _predict_nn_monthly(ticker, panels),
        "nn_quarterly":_predict_nn_quarterly(ticker, panels),
    }
    df = pd.DataFrame(parts).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    preds = bundle["nn_meta"].predict(df.values)
    return pd.Series(preds, index=df.index, name="pred_fwd5")

_BACKENDS = {
    "lightgbm":      _predict_lightgbm,
    "linear_reg":    _predict_linear,
    "ridge":         _predict_ridge,
    "random_forest": _predict_random_forest,
    "ensemble":      _predict_ensemble,
    "agent_ensemble":_predict_agent_ensemble,
    "momentum":      _predict_momentum,
    "vqc":           _predict_vqc,
    "chronos":       _predict_chronos,
    "kronos":        _predict_kronos,
    "timesfm":       _predict_timesfm,
    "xpyq":          _predict_xpyq,
    # Neural ensemble (notebook 02.2)
    "nn_mlp":         _predict_mlp,
    "nn_transformer": _predict_transformer,
    "nn_weekly":      _predict_nn_weekly,
    "nn_monthly":     _predict_nn_monthly,
    "nn_quarterly":   _predict_nn_quarterly,
    "nn_meta":        _predict_nn_meta,
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
