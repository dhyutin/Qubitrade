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

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

# Timeframe -> number of trailing calendar days to show.
TIMEFRAMES = {
    "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "ALL": None,
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

    raw = yf.download(ticker, period="2y", interval="1d",
                      progress=False, auto_adjust=True)
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


@app.get("/api/series")
def series(
    ticker: str = Query(...),
    timeframe: str = Query("3M"),
    forecast_days: int = Query(10, ge=0, le=60),
):
    """Real prices (black) + prediction overlay (green) for one ticker."""
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Unknown timeframe {timeframe}")

    df = _fetch_full_history(ticker)

    days = TIMEFRAMES[timeframe]
    if days is not None:
        cutoff = df.index.max() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in this window")

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    closes = [round(float(c), 4) for c in df["close"].tolist()]

    pred = predictor.predict(ticker, dates, closes, forecast_days=forecast_days)

    return {
        "ticker": ticker,
        "sector": SECTOR_LABEL.get(TICKER_SECTOR[ticker], ""),
        "timeframe": timeframe,
        "real": {"dates": dates, "close": closes},
        "prediction": pred,
        "stats": _series_stats(dates, closes, pred),
    }


# --- Static front end ------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
