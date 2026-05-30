# Qubitrade Dashboard

Interactive UI for the quantum alpha pipeline. Pick a **ticker** on top, a
**timeframe** on the bottom; the white line is the **real** price, the green
line is the **prediction**.

## Run

```bash
cd ui
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

## Wiring in real predictions

The green line comes from [`predictor.py`](predictor.py) → `predict(...)`,
which is currently a **stub** (a synthetic momentum path, flagged "PREDICTION:
STUB" in the UI). To make it real, replace the body of `predict()` with your
pipeline's output (e.g. `alpha_mu` predicted next-day returns) — keep the
return shape, and set `STUB = False`. The reconstruction
`price[t+1] = price[t] * (1 + r_t)` turns predicted returns into the price path.

## Layout

- `app.py` — FastAPI backend: `/api/tickers`, `/api/series`. Prices via yfinance (cached 30 min).
- `predictor.py` — the prediction plug-in point.
- `static/` — front end (Plotly.js, no build step).
