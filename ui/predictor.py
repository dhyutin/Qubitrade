"""
Prediction plug-in for the Qubitrade UI.

Wraps the LightGBM alpha model (trained in `02_alpha_models.ipynb`) behind the
contract the front end expects: a green-line price path that overlaps the tail
of the real series and extends `forecast_days` business days into the future.

Contract
--------
predict(ticker, dates, closes, forecast_days) -> {
    "pred_dates": [ISO date str, ...],   # ascending; overlaps the tail of the
                                         # real series, then extends into the
                                         # future by `forecast_days`
    "pred_close": [float, ...],          # predicted price path (green line),
                                         # aligned 1:1 with pred_dates
    "split_date": ISO date str,          # where the green line is anchored
    "is_stub": bool,
}

The model predicts a 5-day forward return per (date, ticker). To draw a daily
green line we treat each prediction as the daily-equivalent return implied by
the most recent 5d outlook: r_daily = (1 + r_fwd5)^(1/5) - 1, then compound:
    price[0]   = real_close[split_date]
    price[t+1] = price[t] * (1 + r_daily[t])
"""

from __future__ import annotations

import math

import pandas as pd

import pipeline

STUB = False


def _fwd5_to_daily(r_fwd5: float) -> float:
    if pd.isna(r_fwd5):
        return 0.0
    # (1 + r)^(1/H) - 1, safe-guarded against tiny negatives that would NaN.
    base = max(1.0 + float(r_fwd5), 1e-6)
    return base ** (1.0 / pipeline.FORECAST_HORIZON_DAYS) - 1.0


def _future_business_days(last_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n))


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
        return {"pred_dates": [], "pred_close": [], "split_date": None, "is_stub": STUB}

    ts = [pd.Timestamp(d) for d in dates]
    n = len(closes)

    # Same overlap heuristic the old stub used — start the green line partway
    # through the visible window so users can see model vs. real before the
    # forecast region.
    overlap = min(max(n // 3, 5), 45)
    split_idx = max(n - overlap, 0)
    split_ts = ts[split_idx]
    anchor_price = closes[split_idx]

    # Run the chosen model end-to-end. Cached behind pipeline.
    try:
        pred_fwd5 = pipeline.predict_fwd5_returns(ticker.upper(), model=model)
    except Exception:
        pred_fwd5 = pd.Series(dtype=float)

    # Target dates the green line should land on after the anchor: each
    # overlap date except the split itself, plus the forward horizon.
    overlap_targets = ts[split_idx + 1:]
    future_targets = _future_business_days(ts[-1], forecast_days)

    if pred_fwd5.empty:
        overlap_returns = [0.0] * len(overlap_targets)
        last_daily = 0.0
    else:
        pred_daily = pred_fwd5.apply(_fwd5_to_daily)
        # Return at date t drives price from t -> t+1, so to land at overlap_targets
        # we source returns from the preceding dates (split_ts ... ts[-2]).
        overlap_sources = ts[split_idx:-1]
        overlap_returns = (
            pred_daily.reindex(overlap_sources, method="ffill").fillna(0.0).tolist()
        )
        last_daily = float(pred_daily.iloc[-1])

    future_returns = [last_daily] * len(future_targets)
    all_targets = list(overlap_targets) + list(future_targets)
    all_returns = overlap_returns + future_returns

    # Reconstruct the price path from anchor + per-step returns.
    pred_close = [anchor_price]
    for r in all_returns:
        if pd.isna(r) or math.isinf(r):
            r = 0.0
        pred_close.append(pred_close[-1] * (1.0 + float(r)))

    pred_dates_ts = [split_ts] + all_targets
    pred_dates = [d.strftime("%Y-%m-%d") for d in pred_dates_ts]

    return {
        "pred_dates": pred_dates,
        "pred_close": [round(float(p), 4) for p in pred_close],
        "split_date": split_ts.strftime("%Y-%m-%d"),
        "is_stub": STUB,
    }
