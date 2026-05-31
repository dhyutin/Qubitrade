"""
Q-Atlas dashboard backend (FastAPI).

Serves real OHLCV prices (black line) from yfinance and a prediction overlay
(green line) from `predictor.predict`. Tickers are selected on top, timeframe
on the bottom — matching the dashboard sketch.

Run:
    cd ui && uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import predictor

# --- Universe (matches PLAN.md Stage 1) -----------------------------------
TICKER_SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "NVDA": "tech",
    "JPM": "fin", "BAC": "fin", "V": "fin", "MA": "fin",
    "JNJ": "hc", "PFE": "hc", "UNH": "hc", "LLY": "hc",
    "WMT": "cons", "PG": "cons", "KO": "cons", "PEP": "cons",
    "XOM": "eng", "CVX": "eng", "CAT": "eng", "BA": "eng",
}
SECTOR_LABEL = {
    "tech": "Tech", "fin": "Financials", "hc": "Healthcare",
    "cons": "Consumer", "eng": "Energy/Ind.",
}

# Timeframe options shown in the UI.
# "ALL"   -> full 2013-2020 window
# "TRAIN" -> training period  2013-2018
# "TEST"  -> test period      2019-2020
# integers -> trailing N calendar days from the end of the dataset
TIMEFRAMES = {
    "ALL":   None,
    "TRAIN": "train",
    "TEST":  "test",
    "1Y":    365,
    "6M":    180,
    "3M":    90,
}

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Qubitrade Dashboard")

# In-memory price cache: ticker -> (fetched_at, DataFrame of full history).
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 60 * 30  # 30 min — fine for a daily-bar demo.


def _fetch_full_history(ticker: str) -> pd.DataFrame:
    """Fetch (and cache) the full available daily history for a ticker."""
    now = time.time()
    cached = _CACHE.get(ticker)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    raw = yf.download(ticker, start=pipeline.DATA_START, end=pipeline.DATA_END,
                      interval="1d", progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise HTTPException(status_code=502, detail=f"No price data for {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # multiindex when yf wraps single ticker
        close = close.iloc[:, 0]
    df = pd.DataFrame({"close": close.astype(float)}).dropna()
    df.index = pd.to_datetime(df.index)

    _CACHE[ticker] = (now, df)
    return df


def _series_stats(dates: list[str], closes: list[float], pred: dict) -> dict:
    """Header-strip numbers: today's move, predicted move, direction, hit-rate."""
    last = closes[-1]
    prev = closes[-2] if len(closes) > 1 else last
    day_change = (last / prev - 1.0) if prev else 0.0

    pc = pred.get("pred_close", [])

    # Directional hit-rate in the overlap region only (real bars from split_date
    # onward): did the predicted day-over-day move agree in sign with the real?
    split = pred.get("split_date")
    real_overlap = closes[dates.index(split):] if split in dates else []

    # Predicted return over the *forecast horizon only*: from today (last real
    # bar = last overlap point) to the end of the green line.
    anchor_idx = min(len(real_overlap) - 1, len(pc) - 1) if real_overlap else 0
    pred_return = (
        (pc[-1] / pc[anchor_idx] - 1.0)
        if len(pc) >= 2 and pc[anchor_idx] else 0.0
    )
    hits = total = 0
    for i in range(1, min(len(real_overlap), len(pc))):
        real_dir = real_overlap[i] - real_overlap[i - 1]
        pred_dir = pc[i] - pc[i - 1]
        total += 1
        if (real_dir >= 0) == (pred_dir >= 0):
            hits += 1
    hit_rate = (hits / total) if total else None

    return {
        "last_price": round(last, 2),
        "day_change_pct": round(day_change * 100, 2),
        "pred_return_pct": round(pred_return * 100, 2),
        "direction": "UP" if pred_return >= 0 else "DOWN",
        "hit_rate_pct": round(hit_rate * 100) if hit_rate is not None else None,
    }


def _analogs(df: pd.DataFrame, k: int = 5) -> list[dict]:
    """Classical analog retrieval over the ticker's own history.

    Builds a small feature vector per day (5d momentum, 20d momentum, 20d vol),
    standardizes it, and finds the historical days most similar to *today*. For
    each, reports the realized 5-day forward return and a coarse regime tag.

    This is the honest classical version of the Q-Atlas idea — swap the
    euclidean distance for the QSVM quantum-kernel similarity to get the
    quantum retrieval described in PLAN.md Stage 5.5.
    """
    close = df["close"]
    feats = pd.DataFrame({
        "mom_5d": close.pct_change(5),
        "mom_20d": close.pct_change(20),
        "vol_20d": close.pct_change().rolling(20).std(),
    }).dropna()
    if len(feats) < 40:
        return []

    fwd_5d = close.shift(-5) / close - 1.0  # realized 5-day forward return

    z = (feats - feats.mean()) / (feats.std() + 1e-9)
    cur = z.iloc[-1]

    # Candidates: exclude the last 10 days (too close to now) and any day
    # without a realized 5-day forward return.
    cand = z.iloc[:-10]
    dist = ((cand - cur) ** 2).sum(axis=1) ** 0.5
    dist = dist[fwd_5d.reindex(dist.index).notna()]
    top = dist.nsmallest(k)

    out = []
    for date, d in top.items():
        row = feats.loc[date]
        vol_z = z.loc[date, "vol_20d"]
        m5 = row["mom_5d"]
        if m5 > 0.03:
            move = "rally"
        elif m5 < -0.03:
            move = "selloff"
        else:
            move = "drift"
        tag = ("high-vol " if vol_z > 0.8 else "") + move
        out.append({
            "date": date.strftime("%Y-%m-%d"),
            "similarity": round(float(2.718281828 ** (-d)), 3),
            "regime_tag": tag,
            "fwd_5d_pct": round(float(fwd_5d.loc[date]) * 100, 2),
        })
    return out


# ── Q-Atlas: quantum kernel analog retrieval ──────────────────────────────────
# Uses QAtlasEngine (q_atlas_engine.py) — same logic as notebook 08.
# Features: vol_20d, news_sent, mom_20d, dist_to_high (from master_dataset.parquet)
# Kernel:   xpyq remote QPU → Qiskit FidelityStatevectorKernel fallback
# Narration: Gemini with [WHY]/[ANALOGS]/[SIGNAL]/[POSITION] structure

_QATLAS_ENGINE = None   # lazy-loaded on first request
_QATLAS_CACHE: dict[tuple[str, str], dict] = {}
_QATLAS_TTL = 3600  # 1 hour


def _get_qatlas_engine():
    global _QATLAS_ENGINE
    if _QATLAS_ENGINE is None:
        from q_atlas_engine import QAtlasEngine
        _QATLAS_ENGINE = QAtlasEngine()
    return _QATLAS_ENGINE


def _price_window_from_close(close: pd.Series, anchor_date, look_back=10, look_fwd=20):
    """Normalized price window ±days around anchor_date for the regime chart."""
    all_dates = close.index
    try:
        anchor_pos = all_dates.get_loc(anchor_date)
    except KeyError:
        pos_arr = np.argmin(np.abs((all_dates - anchor_date).total_seconds()))
        anchor_pos = int(pos_arr)
    start = max(0, anchor_pos - look_back)
    end   = min(len(all_dates) - 1, anchor_pos + look_fwd)
    window = close.iloc[start : end + 1]
    base   = float(close.iloc[anchor_pos])
    if base == 0:
        return []
    return [
        {"day": i - (anchor_pos - start), "pct": round((float(p) / base - 1) * 100, 3)}
        for i, p in enumerate(window)
    ]


def _qatlas_features(close: pd.Series) -> pd.DataFrame:
    """4-dim ticker-specific feature vector for ZZFeatureMap kernel.

    Features chosen to capture distinct market regimes in 4 qubits:
    mom_5d  — short-term drift direction
    vol_z   — volatility regime (z-score vs 60d mean)
    rsi_14  — mean-reversion signal
    price_z — where price sits in its 60-day range
    """
    r = close.pct_change()
    delta = r
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    vol_5  = r.rolling(5).std()
    vol_60_mean = r.rolling(60).std().rolling(20).mean()
    vol_60_std  = r.rolling(60).std().rolling(20).std()
    vol_z = (vol_5 - vol_60_mean) / (vol_60_std + 1e-9)

    return pd.DataFrame({
        "mom_5d":  r.rolling(5).sum(),
        "vol_z":   vol_z,
        "rsi_14":  rsi,
        "price_z": (close - close.rolling(60).mean()) / (close.rolling(60).std() + 1e-9),
    }).dropna()


def _qatlas_compute(ticker: str, query_date: str | None) -> dict:
    """ZZFeatureMap quantum kernel analog retrieval for one ticker."""
    now = time.time()
    cache_key = (ticker, query_date or "__latest__")
    cached = _QATLAS_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _QATLAS_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    df   = _fetch_full_history(ticker)
    close = df["close"]
    feats = _qatlas_features(close)

    if len(feats) < 50:
        return {"error": "Not enough history", "ticker": ticker}

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    # ── Resolve query date ─────────────────────────────────────────────────
    if query_date:
        ts = pd.Timestamp(query_date)
        q_idx = int(np.argmin(np.abs((feats.index - ts).total_seconds())))
    else:
        q_idx = len(feats) - 1
    ts = feats.index[q_idx]
    query_vec = X[[q_idx]]

    # ── Candidate pool: exclude ±30 days around query ──────────────────────
    lo   = ts - pd.Timedelta(days=30)
    hi   = ts + pd.Timedelta(days=30)
    mask = (feats.index < lo) | (feats.index > hi)
    cand_X     = X[mask]
    cand_dates = feats.index[mask]

    # ── Quantum or classical kernel ────────────────────────────────────────
    method = "RBF kernel (classical fallback)"
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector as _SV

        class _ZZKernel:
            def __init__(self, fm):
                self.fm = fm
                self._p = list(fm.parameters)
            def _svs(self, X):
                return np.array([_SV(self.fm.assign_parameters(dict(zip(self._p, x)))).data for x in X])
            def evaluate(self, x_vec, y_vec):
                return np.abs(self._svs(x_vec) @ self._svs(y_vec).conj().T) ** 2

        fm   = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")
        kern = _ZZKernel(fm)
        sim    = kern.evaluate(x_vec=query_vec, y_vec=cand_X)[0]
        method = "ZZFeatureMap quantum kernel · 4 qubits · reps=2"
    except Exception:
        from sklearn.metrics.pairwise import rbf_kernel
        sim = rbf_kernel(query_vec, cand_X, gamma=0.5)[0]

    fwd_raw = close.pct_change(5).shift(-5)
    top_idx = np.argsort(sim)[::-1][:5]

    def _price_window(anchor_date, look_back=10, look_fwd=20):
        """Return list of {day, pct} rebased to 0 at anchor_date."""
        all_dates = close.index
        anchor_pos = all_dates.get_loc(anchor_date) if anchor_date in all_dates else None
        if anchor_pos is None:
            return []
        start = max(0, anchor_pos - look_back)
        end   = min(len(all_dates) - 1, anchor_pos + look_fwd)
        window = close.iloc[start : end + 1]
        base   = float(close.iloc[anchor_pos])
        if base == 0:
            return []
        return [
            {"day": i - (anchor_pos - start), "pct": round((float(p) / base - 1) * 100, 3)}
            for i, p in enumerate(window)
        ]

    analogs_out = []
    for rank, idx in enumerate(top_idx):
        d   = cand_dates[idx]
        fwd = fwd_raw.get(d)
        analogs_out.append({
            "rank":       rank + 1,
            "date":       d.strftime("%Y-%m-%d"),
            "similarity": round(float(sim[idx]), 4),
            "fwd_5d_pct": (round(float(fwd) * 100, 2)
                           if fwd is not None and not np.isnan(float(fwd)) else None),
            "state": {k: round(float(v), 3) for k, v in feats.loc[d].to_dict().items()},
            "window": _price_window(d),
        })

    result = {
        "ticker":        ticker,
        "query_date":    ts.strftime("%Y-%m-%d"),
        "query_state":   {k: round(float(v), 3) for k, v in feats.loc[ts].to_dict().items()},
        "query_window":  _price_window(ts),
        "analogs":       analogs_out,
        "method":        method,
        "n_candidates":  int(len(cand_dates)),
        "_ts":           now,
    }
    _QATLAS_CACHE[cache_key] = result
    return {k: v for k, v in result.items() if k != "_ts"}


def _gemini_narrate(ticker: str, query_date: str, query_state: dict,
                    analogs: list) -> str:
    """Call Gemini to narrate the Q-Atlas analog result."""
    fwds  = [a["fwd_5d_pct"] for a in analogs if a["fwd_5d_pct"] is not None]
    avg   = sum(fwds) / len(fwds) if fwds else 0.0
    n_pos = sum(1 for f in fwds if f > 0)

    def _fallback():
        direction = ("cautiously bullish" if avg > 0.5
                     else "bearish" if avg < -0.5 else "neutral")
        best = analogs[0] if analogs else {}
        return (
            f"Quantum analog signal for {ticker} is {direction}. "
            f"In {n_pos}/{len(fwds)} similar historical regimes {ticker} rose over "
            f"the next 5 days (avg {avg:+.1f}%). "
            f"Closest analog: {best.get('date','n/a')} "
            f"(sim={best.get('similarity',0):.3f}, "
            f"5d={best.get('fwd_5d_pct',0):+.1f}%)."
        )

    try:
        import google.generativeai as genai
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return _fallback()
        genai.configure(api_key=api_key)

        rows = "\n".join(
            f"  #{a['rank']} {a['date']}: sim={a['similarity']:.3f}  "
            f"5d_fwd={a['fwd_5d_pct']:+.1f}%  "
            f"mom={a['state']['mom_5d']:+.3f}  "
            f"rsi={a['state']['rsi_14']:.0f}  "
            f"vol_z={a['state']['vol_z']:+.2f}"
            for a in analogs if a["fwd_5d_pct"] is not None
        )

        prompt = (
            f"You are Q-Atlas, a quantum market intelligence engine for "
            f"institutional quant traders.\n\n"
            f"Ticker: {ticker}  |  Query date: {query_date}\n"
            f"Current state: mom_5d={query_state['mom_5d']:+.3f}  "
            f"vol_z={query_state['vol_z']:+.2f}  "
            f"rsi={query_state['rsi_14']:.0f}  "
            f"price_z={query_state['price_z']:+.2f}\n\n"
            f"The ZZFeatureMap quantum kernel (4 qubits, reps=2) retrieved "
            f"these top-5 historical analogs from {ticker}'s own price history:\n"
            f"{rows}\n\n"
            f"Quantum consensus: {n_pos}/5 analogs were followed by positive "
            f"5-day returns. Avg 5d forward return: {avg:+.1f}%.\n\n"
            f"In 2–3 punchy sentences, narrate the quantum analog signal for "
            f"{ticker}. Name specific dates and return figures. "
            f"End with: \"Stack held [{ticker}, <1-2 peers>]; <avg>% over 5 days.\""
        )

        for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
            try:
                model    = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception:
                continue
        return _fallback()
    except Exception:
        return _fallback()


@app.get("/api/qatlas")
def qatlas_query(ticker: str = Query(...), date: str = Query(None),
                 live: bool = Query(False)):
    """Interactive Q-Atlas: quantum kernel analog retrieval + Gemini narration.

    Uses QAtlasEngine (same as notebook 08):
      - Features: vol_20d, news_sent, mom_20d, dist_to_high (master_dataset.parquet)
      - Kernel:   xpyq QPU → Qiskit FidelityStatevectorKernel fallback
      - Narration: Gemini [WHY]/[ANALOGS]/[SIGNAL]/[POSITION]
    """
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")

    # Check cache
    cache_key = (ticker, date or "__latest__")
    now = time.time()
    if cache_key in _QATLAS_CACHE and now - _QATLAS_CACHE[cache_key].get("_ts", 0) < _QATLAS_TTL:
        cached = _QATLAS_CACHE[cache_key]
        return {k: v for k, v in cached.items() if k != "_ts"}

    engine = _get_qatlas_engine()
    if live:
        result, err = engine.query_live(ticker, top_k=5)
    else:
        result, err = engine.query(ticker, date=date, top_k=5)
    if err:
        raise HTTPException(status_code=503, detail=err)

    # Build price windows for the regime comparison chart
    df    = _fetch_full_history(ticker)
    close = df["close"]

    query_ts  = result["query_date"]
    query_date_str = pd.Timestamp(query_ts).strftime("%Y-%m-%d")

    def _kernel_label(ku: str) -> str:
        if ku == "xpyq":
            return "xpyq remote quantum covariance kernel · QPU hardware"
        if "Statevector" in ku:
            return "ZZFeatureMap · FidelityStatevectorKernel · 4 qubits · reps=2"
        return f"ZZFeatureMap · {ku} · 4 qubits · reps=2"

    analogs_out = []
    for a in result["analogs"]:
        ctx = a["context"]
        fwd = ctx.get("fwd_return_5d")
        d   = pd.Timestamp(a["date"])
        analogs_out.append({
            "rank":       a["rank"],
            "date":       d.strftime("%Y-%m-%d"),
            "similarity": round(float(a["similarity"]), 4),
            "fwd_5d_pct": round(float(fwd) * 100, 2) if fwd is not None else None,
            "state":      {k: round(float(v), 4) for k, v in ctx.get("state", {}).items()},
            "headline":   ctx.get("headline", ""),
            "news_items": ctx.get("news_items", []),
            "regime":     ctx.get("regime", ""),
            "window":     _price_window_from_close(close, d),
        })

    qs = result["query_state"]
    out = {
        "ticker":          ticker,
        "query_date":      query_date_str,
        "is_live":         result.get("is_live", False),
        "live_headlines":  result.get("live_headlines", []),
        "query_state":     {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                            for k, v in qs.items()},
        "query_window":    _price_window_from_close(close, pd.Timestamp(query_ts)),
        "analogs":         analogs_out,
        "method":          _kernel_label(result.get("kernel_used", "")),
        "n_candidates":    result["n_candidates"],
        "kernel_time":     round(result.get("kernel_time", 0), 2),
        "_ts":             now,
    }
    out["narrative"] = engine.narrate(result, original_query=f"Why is {ticker} moving?")

    _QATLAS_CACHE[cache_key] = out
    return {k: v for k, v in out.items() if k != "_ts"}


@app.get("/api/analogs")
def analogs(ticker: str = Query(...)):
    """Top-5 historically similar regimes for the ticker (Q-Atlas panel)."""
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")
    df = _fetch_full_history(ticker)
    items = _analogs(df)
    narration = None
    if items:
        top = items[0]
        verb = "gained" if top["fwd_5d_pct"] >= 0 else "fell"
        narration = (
            f"Today's {ticker} regime most resembles {top['date']} "
            f"({top['regime_tag']}); over the next 5 days it {verb} "
            f"{abs(top['fwd_5d_pct'])}%."
        )
    return {"ticker": ticker, "analogs": items, "narration": narration,
            "method": "classical cosine (swap in QSVM kernel for quantum)"}


@app.get("/api/tickers")
def list_tickers():
    """Universe grouped by sector, for the chip selector on top."""
    sectors: dict[str, list[str]] = {}
    for t, s in TICKER_SECTOR.items():
        sectors.setdefault(s, []).append(t)
    return {
        "timeframes": list(TIMEFRAMES.keys()),
        "sectors": [
            {"key": s, "label": SECTOR_LABEL.get(s, s), "tickers": ts}
            for s, ts in sectors.items()
        ],
    }


@app.get("/api/models")
def list_models():
    """Available prediction models for the dropdown."""
    return {"models": pipeline.AVAILABLE_MODELS}


_VALID_MODELS = {m["key"] for m in pipeline.AVAILABLE_MODELS}


@app.get("/api/series")
def series(
    ticker: str = Query(...),
    timeframe: str = Query("3M"),
    forecast_days: int = Query(10, ge=0, le=60),
    model: str = Query("lightgbm"),
):
    """Real prices (black) + prediction overlay (green) for one ticker."""
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Unknown timeframe {timeframe}")
    if model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {model}")

    df = _fetch_full_history(ticker)

    tf_val = TIMEFRAMES[timeframe]
    if tf_val == "train":
        df = df[df.index <= pd.Timestamp(pipeline.TRAIN_END_DATE)]
    elif tf_val == "test":
        df = df[df.index >= pd.Timestamp(pipeline.TEST_START_DATE)]
    elif isinstance(tf_val, int):
        cutoff = df.index.max() - pd.Timedelta(days=tf_val)
        df = df[df.index >= cutoff]
    # None -> full window, no filter
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in this window")

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    closes = [round(float(c), 4) for c in df["close"].tolist()]

    pred = predictor.predict(
        ticker, dates, closes, forecast_days=forecast_days, model=model,
    )

    return {
        "ticker":      ticker,
        "sector":      SECTOR_LABEL.get(TICKER_SECTOR[ticker], ""),
        "timeframe":   timeframe,
        "model":       model,
        "real":        {"dates": dates, "close": closes},
        "prediction":  pred,
        "stats":       _series_stats(dates, closes, pred),
        "train_end":   pipeline.TRAIN_END_DATE,
        "test_start":  pipeline.TEST_START_DATE,
        "data_start":  pipeline.DATA_START,
        "data_end":    "2020-06-11",
    }


# --- PPO Trader ------------------------------------------------------------
@app.get("/api/ppo/trajectory")
def ppo_trajectory():
    """Full PPO v1 historical test-window trajectory (2019-2020)."""
    try:
        import ppo_backend
        return ppo_backend.get_trajectory()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"PPO backend unavailable: {e}")


@app.get("/api/ppo/live")
def ppo_live_trajectory():
    """PPO v1 trajectory on live yfinance data (2024-present).
    First call fetches ~500 days of prices and runs LightGBM + PPO — allow ~15 s."""
    try:
        import ppo_backend
        return ppo_backend.get_live_trajectory()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"PPO live backend unavailable: {e}")


# --- Static front end ------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
