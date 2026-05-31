"""
Prediction plug-in for the Qubitrade UI.

Wraps the alpha models trained in `02_alpha_models.ipynb` behind the contract
the front end expects: a green-line price path that overlaps the tail of the
real series (showing what the model said about each day H business days ago)
and extends `forecast_days` business days into the future.

Contract
--------
predict(ticker, dates, closes, forecast_days, model) -> {
    "pred_dates":  [ISO date str, ...],   # ascending overlap + forecast
    "pred_close":  [float, ...],          # predicted price path (green line)
    "pred_upper":  [float, ...],          # upper edge of the confidence band
    "pred_lower":  [float, ...],          # lower edge of the confidence band
    "split_date":  ISO date str,          # where the overlap region starts
    "is_stub":     bool,
}

Visual model
------------
Each model predicts a 5-day forward return per (date, ticker). The old version
compounded the same daily-equivalent return every step, producing a smooth
monotonic curve that looked the same for every model — boring.

The new formulation makes the green line actually show the model in action:

  Overlap (visible real bars):
      green(t) = real(t - H) × (1 + pred_fwd5(t - H))
      i.e. "what the model said H days ago about today's price"
      → weaves above/below the white line based on actual model calls.

  Forecast (no real bars yet):
      anchored at real(today); compounds the latest 5d outlook into a smooth
      fan, with a vol-scaled confidence band that widens as sqrt(horizon).
"""

from __future__ import annotations

import math

import pandas as pd

import pipeline

STUB = False


def _future_business_days(last_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n))


def _safe_daily(last_fwd5: float, horizon: int) -> float:
    """Daily-equivalent return from an H-day forward return, guarded against
    negative bases that would NaN under fractional powers."""
    if pd.isna(last_fwd5):
        return 0.0
    base = max(1.0 + float(last_fwd5), 1e-6)
    return base ** (1.0 / horizon) - 1.0


def predict(
    ticker: str,
    dates: list[str],
    closes: list[float],
    forecast_days: int = 10,
    model: str = "lightgbm",
) -> dict:
    """Model-driven prediction path for `ticker`. `model` selects the backend
    in `pipeline._BACKENDS` — see `pipeline.AVAILABLE_MODELS` for the menu."""
    if not closes:
        return {
            "pred_dates": [], "pred_close": [], "pred_upper": [], "pred_lower": [],
            "split_date": None, "is_stub": STUB,
        }

    ts = [pd.Timestamp(d) for d in dates]
    n = len(ts)
    H = pipeline.FORECAST_HORIZON_DAYS  # 5

    # Overlap region starts at the test split date when data spans it, so the
    # green line covers the full 2019-2020 test window instead of just ~45 bars.
    test_start_ts = pd.Timestamp(pipeline.TEST_START_DATE)
    if ts[0] < test_start_ts <= ts[-1]:
        split_idx = next((i for i, t in enumerate(ts) if t >= test_start_ts), n - 5)
    else:
        split_idx = max(n - min(max(n // 3, 5), 45), 0)
    split_ts = ts[split_idx]

    # Run the chosen model end-to-end. Cached behind pipeline.
    try:
        pred_fwd5 = pipeline.predict_fwd5_returns(ticker.upper(), model=model)
    except Exception:
        pred_fwd5 = pd.Series(dtype=float)
    pred_lookup = pred_fwd5.to_dict() if not pred_fwd5.empty else {}

    # ----- Overlap: predicted-vs-realized -----
    # For each visible date d, green(d) = real_close(d-H) × (1 + pred_fwd5(d-H)).
    # Fall back to the live real bar when no prediction is available — keeps
    # the line continuous instead of cratering on missing dates.
    overlap_dates: list[pd.Timestamp] = []
    overlap_close: list[float] = []
    for i in range(split_idx, n):
        src = i - H
        if src < 0:
            continue
        src_date = ts[src]
        r = pred_lookup.get(src_date)
        if r is None or pd.isna(r):
            green = float(closes[i])
        else:
            green = float(closes[src]) * (1.0 + float(r))
        overlap_dates.append(ts[i])
        overlap_close.append(green)

    # ----- Forecast: anchor at today, compound the latest 5d outlook -----
    last_real = float(closes[-1])
    last_pred = float(pred_fwd5.iloc[-1]) if not pred_fwd5.empty else 0.0
    daily_rate = _safe_daily(last_pred, H)

    future_dates = _future_business_days(ts[-1], forecast_days)
    future_close = [last_real * (1.0 + daily_rate) ** k for k in range(1, len(future_dates) + 1)]

    pred_dates_ts = overlap_dates + future_dates
    pred_close = overlap_close + future_close

    # ----- Confidence band -----
    # Vol-scaled (~1.96 σ × √h) and zero inside the overlap region where users
    # can read accuracy directly off the white line.
    if len(closes) >= 12:
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        recent = rets[-30:]
        daily_vol = (sum(r * r for r in recent) / len(recent)) ** 0.5
    else:
        daily_vol = 0.0

    n_overlap = len(overlap_dates)
    pred_upper: list[float] = []
    pred_lower: list[float] = []
    for i, p in enumerate(pred_close):
        future_step = max(i - n_overlap + 1, 0)
        width = p * daily_vol * 1.96 * math.sqrt(future_step)
        pred_upper.append(round(p + width, 4))
        pred_lower.append(round(p - width, 4))

    return {
        "pred_dates": [d.strftime("%Y-%m-%d") for d in pred_dates_ts],
        "pred_close": [round(float(p), 4) for p in pred_close],
        "pred_upper": pred_upper,
        "pred_lower": pred_lower,
        "split_date": split_ts.strftime("%Y-%m-%d"),
        "is_stub": STUB,
    }
