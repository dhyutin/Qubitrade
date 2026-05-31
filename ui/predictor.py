"""
Prediction plug-in for the Qubitrade UI.

THIS IS THE STUB. The UI calls `predict(...)` to draw the green line.
Right now it produces a *synthetic* forward path so the UI is fully wired and
demoable. To make the green line real, replace the body of `predict()` with
your pipeline's output (e.g. `alpha_mu` from Stage 2, or the QAOA/meta-labeled
forward returns) — the function signature and return shape are the contract the
front end depends on, so keep them the same.

Contract
--------
predict(ticker, dates, closes, forecast_days) -> {
    "pred_dates": [ISO date str, ...],   # ascending; overlaps the tail of the
                                         # real series, then extends into the
                                         # future by `forecast_days`
    "pred_close": [float, ...],          # predicted price path (the green line),
                                         # aligned 1:1 with pred_dates
    "pred_upper": [float, ...],          # upper edge of the confidence band
    "pred_lower": [float, ...],          # lower edge of the confidence band
    "split_date": ISO date str,          # where the prediction starts (last real
                                         # point the green line is anchored to)
    "is_stub": bool,
}

How a real model plugs in
--------------------------
Your `alpha_mu` is a predicted *next-day return* per (date, ticker). To turn a
series of predicted returns r_t into the green price path the UI wants:

    price[0]   = last_real_close_at_split
    price[t+1] = price[t] * (1 + r_t)

So all you have to do is supply `pred_returns` (one per step, overlap + future)
and the reconstruction below does the rest. Set STUB = False once it's real.
"""

from __future__ import annotations

import math
import pandas as pd

STUB = True


def _synthetic_returns(closes: list[float], n_steps: int) -> list[float]:
    """Believable placeholder predicted daily returns.

    Uses trailing momentum (mean of recent real returns) with mild
    amplification so the green line visibly diverges from black like a model
    that has taken a directional view. Deterministic — no randomness — so the
    demo looks identical every refresh.
    """
    if len(closes) < 12:
        drift = 0.0
    else:
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        recent = rets[-10:]
        drift = sum(recent) / len(recent)

    # Amplify the directional view a touch and let it decay over the horizon so
    # the forecast fans out smoothly rather than running away.
    out = []
    for t in range(n_steps):
        decay = math.exp(-t / max(n_steps, 1) * 1.5)
        out.append(drift * 1.6 * decay)
    return out


def _future_business_days(last_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    rng = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)
    return list(rng)


def predict(
    ticker: str,
    dates: list[str],
    closes: list[float],
    forecast_days: int = 10,
) -> dict:
    """Return the green prediction path for `ticker`.

    See module docstring for the contract. The stub anchors the prediction at a
    split point (~last third of the visible window, capped) so you can see the
    green line track and then diverge from black, then projects `forecast_days`
    into the future.
    """
    if not closes:
        return {
            "pred_dates": [], "pred_close": [], "pred_upper": [], "pred_lower": [],
            "split_date": None, "is_stub": STUB,
        }

    ts = [pd.Timestamp(d) for d in dates]
    n = len(closes)

    # Anchor: start the green line partway through the window so it overlaps the
    # real series (for visual comparison) before extending past the last point.
    overlap = min(max(n // 3, 5), 45)
    split_idx = max(n - overlap, 0)
    anchor_price = closes[split_idx]

    n_overlap = n - split_idx                 # steps that overlap real data
    n_steps = n_overlap + forecast_days       # total predicted steps

    # --- THE PLUG-IN POINT --------------------------------------------------
    # Replace `_synthetic_returns(...)` with your model's predicted daily
    # returns for these n_steps (overlap region + forecast horizon).
    pred_returns = _synthetic_returns(closes, n_steps)
    # ------------------------------------------------------------------------

    # Reconstruct price path from predicted returns.
    pred_close = [anchor_price]
    for r in pred_returns:
        pred_close.append(pred_close[-1] * (1.0 + r))
    pred_close = pred_close[: n_steps + 1]

    future_dates = _future_business_days(ts[-1], forecast_days)
    pred_dates = ts[split_idx:] + future_dates
    pred_dates = pred_dates[: len(pred_close)]

    # Confidence band: widens with the forecast horizon (~ vol * sqrt(t)),
    # zero inside the overlap region where we can still see the real price.
    if len(closes) >= 12:
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        daily_vol = (sum(r * r for r in rets[-30:]) / min(len(rets), 30)) ** 0.5
    else:
        daily_vol = 0.0
    pred_upper, pred_lower = [], []
    for i, p in enumerate(pred_close):
        future_step = max(i - n_overlap, 0)          # 0 until forecast starts
        width = p * daily_vol * 1.96 * math.sqrt(future_step)
        pred_upper.append(round(p + width, 4))
        pred_lower.append(round(p - width, 4))

    return {
        "pred_dates": [d.strftime("%Y-%m-%d") for d in pred_dates],
        "pred_close": [round(float(p), 4) for p in pred_close],
        "pred_upper": pred_upper,
        "pred_lower": pred_lower,
        "split_date": ts[split_idx].strftime("%Y-%m-%d"),
        "is_stub": STUB,
    }
