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
    }


# --- Static front end ------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
