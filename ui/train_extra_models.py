"""
One-off training for the classical baselines that the comparison CSV lists but
notebook 02 doesn't actually train: LinearReg, Ridge, RandomForest.

Same input as LightGBM (the 28 signals from notebook 01) and the same train/
test split (<= 2017-12-31). Saves pickles under workspace/models/ so the UI
backend can load them directly. Re-run after a fresh data pull.

    python ui/train_extra_models.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
DATA_DIR = WORKSPACE / "data"
MODELS_DIR = WORKSPACE / "models"

TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
TARGET = "fwd_return_5d"


def main() -> None:
    with open(MODELS_DIR / "signal_cols.json") as f:
        meta = json.load(f)
    signal_cols = meta["signal_cols"]

    master = pd.read_parquet(DATA_DIR / "master_dataset.parquet")
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    master = master.set_index(["timestamp", "ticker"]).dropna(subset=[TARGET])

    ts = master.index.get_level_values("timestamp")
    train = master[ts <= pd.Timestamp(TRAIN_END)]
    test = master[ts >= pd.Timestamp(TEST_START)]

    Xtr = train[signal_cols].astype(float).fillna(0).values
    ytr = train[TARGET].astype(float).values
    Xte = test[signal_cols].astype(float).fillna(0).values
    yte = test[TARGET].astype(float).values

    print(f"train rows={len(Xtr):,}  test rows={len(Xte):,}  features={Xtr.shape[1]}")

    models = {
        "linear_reg":    LinearRegression(),
        "ridge":         Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            n_jobs=-1, random_state=42,
        ),
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        ic = float(spearmanr(pred, yte)[0])
        out = MODELS_DIR / f"{name}_alpha.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
        print(f"  {name:14s} test IC={ic:+.4f}  -> {out.name}")


if __name__ == "__main__":
    main()
