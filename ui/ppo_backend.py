"""PPO portfolio inference backend — PPO v1 (41-dim state, Sharpe 1.04).

PPO v1 uses AlphaPortfolioEnv: state = [lgb_alpha×20, vol_regime×1, weights×20] = 41-dim.
PPO v2 uses the richer 101-dim state but achieved a lower Sharpe (0.80 vs 1.04), so v1
is the one surfaced in the UI.

Trajectory is computed once (lazy) and cached in memory. Each step exposes:
  date        - ISO date string
  ppo_weights - per-ticker portfolio weights (L1-normalised)
  ew_weights  - equal-weight long-short LGB benchmark weights
  ppo_return  - realised 1-day portfolio return under PPO weights
  ew_return   - realised 1-day return under equal-weight LGB
  fwd_returns - per-ticker 1-day realised returns (lets JS simulate
                arbitrary user weights with no server round-trip)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

WORKSPACE  = Path(__file__).resolve().parent.parent / "workspace"
DATA_DIR   = WORKSPACE / "data"
MODELS_DIR = WORKSPACE / "models"

N_TICKERS = 20
TEST_START = pd.Timestamp("2019-06-17")
TC_COST    = 0.001

LIVE_START = "2024-01-01"   # start of live / out-of-sample window

_TRAJ:      dict | None = None
_LIVE_TRAJ: dict | None = None


class _AlphaPortfolioEnv(gym.Env):
    """Exact replica of AlphaPortfolioEnv (v1) from 05_ppo_portfolio.ipynb.
    State: [lgb_alpha×20, vol_regime×1, current_weights×20] = 41-dim.
    """
    metadata = {"render_modes": []}

    def __init__(self, dates, alpha_wide, fwd_1d, vol_regime):
        super().__init__()
        self.dates      = list(dates)
        self.alpha      = alpha_wide.loc[dates].fillna(0).values
        self.fwd        = fwd_1d.loc[dates].fillna(0).values
        self.vol_regime = vol_regime.reindex(dates).fillna(0).values
        self.N          = N_TICKERS
        self.tc_cost    = TC_COST
        self.eta        = 0.01
        obs_dim = self.N * 2 + 1   # alpha(20) + vol_regime(1) + weights(20)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.N,),  dtype=np.float32)
        self.reset()

    def _normalize_alpha(self, raw):
        std = raw.std() + 1e-8
        return np.clip(raw / std, -3, 3).astype(np.float32)

    def _get_obs(self):
        a = self._normalize_alpha(self.alpha[self.t])
        v = np.array([self.vol_regime[self.t]], dtype=np.float32)
        w = np.clip(self.current_weights, -1, 1).astype(np.float32)
        return np.concatenate([a, v, w])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.current_weights = np.zeros(self.N, dtype=np.float32)
        self.A = 0.0
        self.B = 1e-6
        self.episode_returns = []
        return self._get_obs(), {}

    def step(self, action):
        l1 = np.abs(action).sum() + 1e-8
        weights = action / l1
        ret = float(np.dot(weights, self.fwd[self.t]))
        self.episode_returns.append(ret)

        dA    = ret - self.A
        dB    = ret**2 - self.B
        denom = self.B - self.A**2
        dsr   = (self.B * dA - 0.5 * self.A * dB) / (denom**1.5 + 1e-8) if denom > 1e-8 else 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB

        turnover = float(np.abs(weights - self.current_weights).sum())
        reward   = float(np.clip(dsr - self.tc_cost * turnover, -1.0, 1.0))

        self.current_weights = weights
        self.t += 1
        done = self.t >= len(self.dates) - 1
        obs  = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {}


def _load_all():
    alpha_raw = pd.read_parquet(DATA_DIR / "alpha_mu.parquet")
    tickers   = sorted(alpha_raw["lightgbm"].unstack("ticker").columns.tolist())

    alpha_wide = alpha_raw["lightgbm"].unstack("ticker")[tickers]
    alpha_wide.index = pd.to_datetime(alpha_wide.index)

    meta = pd.read_parquet(DATA_DIR / "meta_features.parquet")
    meta.index = pd.to_datetime(meta.index)

    def _get_col(ticker, col):
        if (ticker, col) in meta.columns:
            return meta[(ticker, col)]
        flat = f"({ticker}, {col})"
        return meta[flat] if flat in meta.columns else pd.Series(0.0, index=meta.index)

    vol_regime = pd.DataFrame(
        {t: _get_col(t, "vol_regime") for t in tickers}
    ).mean(axis=1)

    fwd_1d = pd.read_parquet(DATA_DIR / "forward_returns_1d.parquet")[tickers]
    fwd_1d.index = pd.to_datetime(fwd_1d.index)

    return tickers, alpha_wide, vol_regime, fwd_1d


def _sharpe(r, ann=252):
    r = np.asarray(r)
    return float(r.mean() / (r.std() + 1e-9) * np.sqrt(ann))


def _run_ppo(env: _AlphaPortfolioEnv, ppo) -> list[dict]:
    """Step ppo deterministically through env, collecting per-step data."""
    obs, _ = env.reset()
    steps: list[dict] = []
    done = False
    while not done:
        action, _ = ppo.predict(obs, deterministic=True)
        l1      = np.abs(action).sum() + 1e-8
        weights = action / l1
        t       = env.t
        date    = env.dates[t]
        fwd_day = env.fwd[t]
        ppo_ret = float(np.dot(weights, fwd_day))

        rank = np.argsort(env.alpha[t])
        n    = max(1, N_TICKERS // 4)
        ew_w = np.zeros(N_TICKERS)
        ew_w[rank[-n:]] =  1.0 / n
        ew_w[rank[:n]]  = -1.0 / n
        ew_ret = float(np.dot(ew_w, fwd_day))

        steps.append({
            "date":        date.strftime("%Y-%m-%d"),
            "ppo_weights": [round(float(w), 4) for w in weights],
            "ew_weights":  [round(float(w), 4) for w in ew_w],
            "ppo_return":  round(ppo_ret, 6),
            "ew_return":   round(ew_ret, 6),
            "fwd_returns": [round(float(r), 6) for r in fwd_day],
        })
        obs, _, done, _, _ = env.step(action)
    return steps


def _build_result(tickers, steps, version, mode="historical") -> dict:
    ppo_r = np.array([s["ppo_return"] for s in steps])
    ew_r  = np.array([s["ew_return"]  for s in steps])
    return {
        "tickers": tickers,
        "steps":   steps,
        "version": version,
        "mode":    mode,
        "summary": {
            "ppo_sharpe":       round(_sharpe(ppo_r), 3),
            "ew_sharpe":        round(_sharpe(ew_r),  3),
            "ppo_total_return": round(float(np.prod(1 + ppo_r) - 1), 4),
            "ew_total_return":  round(float(np.prod(1 + ew_r)  - 1), 4),
            "n_steps":          len(steps),
            "test_start":       steps[0]["date"] if steps else None,
            "test_end":         steps[-1]["date"] if steps else None,
        },
    }


def get_trajectory() -> dict:
    """Compute (or return cached) full PPO v1 historical test-window trajectory."""
    global _TRAJ
    if _TRAJ is not None:
        return _TRAJ

    from stable_baselines3 import PPO

    tickers, alpha_wide, vol_regime, fwd_1d = _load_all()
    test_dates = alpha_wide.index[alpha_wide.index >= TEST_START].intersection(fwd_1d.index)

    env  = _AlphaPortfolioEnv(test_dates, alpha_wide, fwd_1d, vol_regime)
    ppo  = PPO.load(str(MODELS_DIR / "ppo_portfolio"))
    steps = _run_ppo(env, ppo)
    _TRAJ = _build_result(tickers, steps, "v1 (41-dim, Sharpe 1.04)", mode="historical")
    return _TRAJ


def get_live_trajectory() -> dict:
    """PPO v1 trajectory on live yfinance data (2024-present).

    Uses the same model file (ppo_portfolio.zip) but feeds it live signals:
      - LightGBM alpha from pipeline.get_lgb_model()
      - vol_regime proxy: cross-sectional mean vol z-score from live prices
      - fwd_returns: actual next-day returns from yfinance

    The last date in the window has no realized forward return (tomorrow is
    unknown), so it is dropped. Distribution shift vs 2019 training is expected
    but the signal ordering is consistent.
    """
    global _LIVE_TRAJ
    if _LIVE_TRAJ is not None:
        return _LIVE_TRAJ

    import yfinance as yf
    from stable_baselines3 import PPO
    import sys, importlib

    # Import pipeline from the same directory without circular dependency
    ui_dir = str(Path(__file__).parent)
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    import pipeline as _pl

    tickers = _pl.TICKERS  # same 20-ticker universe from signal_cols.json

    # --- Fetch live panel (2024-present) ---
    raw = yf.download(
        tickers, start=LIVE_START, interval="1d",
        progress=False, auto_adjust=True, group_by="column",
    )
    close = raw["Close"].reindex(columns=tickers).astype(float)
    high  = raw["High"].reindex(columns=tickers).astype(float)
    low   = raw["Low"].reindex(columns=tickers).astype(float)
    close.index = pd.to_datetime(close.index)
    high.index  = pd.to_datetime(high.index)
    low.index   = pd.to_datetime(low.index)

    # --- Compute 28 signals per ticker ---
    panels = _pl.compute_signals(close, high, low)

    # --- LightGBM alpha: batch across dates per ticker ---
    lgb = _pl.get_lgb_model()
    alpha_cols: dict[str, pd.Series] = {}
    for t in tickers:
        feats = panels[t].dropna()
        if feats.empty:
            alpha_cols[t] = pd.Series(dtype=float)
        else:
            preds = lgb.predict(feats.values)
            alpha_cols[t] = pd.Series(preds, index=feats.index)
    alpha_wide = pd.DataFrame(alpha_cols).reindex(columns=tickers)

    # --- Vol-regime proxy (cross-sectional mean vol z-score) ---
    ret     = close.pct_change()
    vol20   = ret.rolling(20).std()
    vol_mu  = vol20.rolling(60).mean()
    vol_std = vol20.rolling(60).std().replace(0, np.nan)
    vol_z   = ((vol20 - vol_mu) / vol_std).clip(-3, 3)
    vol_regime = vol_z.mean(axis=1).fillna(0)

    # --- 1-day forward returns (next day's actual return) ---
    fwd_1d = ret.shift(-1)   # unknown for last date → dropped below

    # --- Align dates: drop rows with any NaN in alpha or fwd ---
    valid = (
        alpha_wide.dropna(how="any").index
        .intersection(fwd_1d.dropna(how="any").index)
    )
    valid = valid.sort_values()
    if valid.empty:
        raise RuntimeError("No valid live dates after alignment")

    env   = _AlphaPortfolioEnv(valid, alpha_wide, fwd_1d, vol_regime)
    ppo   = PPO.load(str(MODELS_DIR / "ppo_portfolio"))
    steps = _run_ppo(env, ppo)
    _LIVE_TRAJ = _build_result(tickers, steps, "v1 (41-dim) on live data 2024–present", mode="live")
    return _LIVE_TRAJ
