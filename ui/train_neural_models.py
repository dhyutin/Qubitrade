"""
Train the 6 models from `02.2_neural_ensemble.ipynb` headlessly:
  MLP, Transformer (lookback=20), LSTM (5/21/63), Ridge meta-learner.

Saves to workspace/models/:
  nn_mlp.pt, nn_transformer.pt, nn_weekly.pt, nn_monthly.pt, nn_quarterly.pt
  nn_meta_learner.pkl, nn_scaler.pkl
  nn_features.json — feature schema + hidden dims so inference can rebuild the
  exact same nets without importing this file.

    python ui/train_neural_models.py
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
DATA_DIR = WORKSPACE / "data"
MODELS_DIR = WORKSPACE / "models"

TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
TARGET = "fwd_return_5d"

MLP_EPOCHS = 50
TRANSFORMER_EPOCHS = 40
SCALE_EPOCHS = 35
BATCH_SIZE = 256
LR = 1e-3
DROPOUT = 0.2
SEED = 42

TRANSFORMER_LOOKBACK = 20
SCALES = {"weekly": 5, "monthly": 21, "quarterly": 63}

torch.manual_seed(SEED)
np.random.seed(SEED)


def _device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---- Net definitions (identical to notebook 02.2) ----
class MLPForecaster(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        dim = n_in
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(DROPOUT)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TemporalTransformer(nn.Module):
    def __init__(self, n_features: int, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 512, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        T = x.size(1)
        x = self.input_proj(x) + self.pos_enc[:, :T, :]
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class ScaleLSTM(nn.Module):
    def __init__(self, n_features: int, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=DROPOUT if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(DROPOUT), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---- Training helpers ----
def train_mlp(X_tr, y_tr, device):
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = MLPForecaster(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    loss_fn = nn.HuberLoss(delta=0.01)

    for ep in range(MLP_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model.eval()


def train_seq(model_cls, n_features, X_tr, y_tr, epochs, device):
    model = model_cls(n_features).to(device)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR * 5, steps_per_epoch=len(loader), epochs=epochs,
    )
    loss_fn = nn.HuberLoss(delta=0.01)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model.eval()


def predict_seq(model, X, device, batch=BATCH_SIZE) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ---- Sequence builder (same as notebook) ----
def build_sequence_dataset(df, signal_cols, target, lookback, train_end, test_start, scaler):
    feat_arr = scaler.transform(df[signal_cols].fillna(df[signal_cols].median()))
    feat_df = pd.DataFrame(feat_arr, index=df.index, columns=signal_cols)

    Xs_tr, ys_tr, idx_tr = [], [], []
    Xs_te, ys_te, idx_te = [], [], []
    train_ts = pd.Timestamp(train_end)
    test_ts = pd.Timestamp(test_start)

    for ticker in df.index.get_level_values("ticker").unique():
        try:
            tf = feat_df.xs(ticker, level="ticker").sort_index()
            tt = df.xs(ticker, level="ticker")[target].sort_index().dropna()
        except KeyError:
            continue
        for i, date in enumerate(tt.index):
            if i < lookback:
                continue
            win = tf.index[tf.index <= date][-lookback:]
            if len(win) < lookback:
                continue
            seq = tf.loc[win].values.astype(np.float32)
            label = float(tt[date])
            if date <= train_ts:
                Xs_tr.append(seq); ys_tr.append(label); idx_tr.append((date, ticker))
            elif date >= test_ts:
                Xs_te.append(seq); ys_te.append(label); idx_te.append((date, ticker))
    return (
        np.array(Xs_tr), np.array(ys_tr),
        pd.MultiIndex.from_tuples(idx_tr, names=["timestamp", "ticker"]),
        np.array(Xs_te), np.array(ys_te),
        pd.MultiIndex.from_tuples(idx_te, names=["timestamp", "ticker"]),
    )


def main() -> None:
    device = _device()
    print(f"Device: {device}")

    # Load & split
    master = pd.read_parquet(DATA_DIR / "master_dataset.parquet")
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    master = master.set_index(["timestamp", "ticker"])
    exclude = {
        "sector", "price", "high", "low", "volume",
        "return_1d", "fwd_return_1d", "fwd_return_5d", "regime",
        "news_today", "news_yesterday", "market_news_today", "market_news_yesterday",
        "n_news_today", "n_market_news_today", "news_sent_today", "news_sent_yesterday",
    }
    signal_cols = [c for c in master.columns if c not in exclude]
    n_features = len(signal_cols)
    df = master.dropna(subset=[TARGET]).copy()
    ts = df.index.get_level_values("timestamp")
    X_train = df.loc[ts <= pd.Timestamp(TRAIN_END), signal_cols]
    y_train = df.loc[ts <= pd.Timestamp(TRAIN_END), TARGET]
    X_test = df.loc[ts >= pd.Timestamp(TEST_START), signal_cols]
    y_test = df.loc[ts >= pd.Timestamp(TEST_START), TARGET]

    train_medians = X_train.median()
    X_train_f = X_train.fillna(train_medians)
    X_test_f = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)
    print(f"train={X_train_s.shape}  test={X_test_s.shape}  features={n_features}")

    # ---- MLP ----
    t = time.time()
    print("Training MLP...")
    mlp = train_mlp(X_train_s, y_train, device)
    mlp_train = pd.Series(
        mlp(torch.tensor(X_train_s, dtype=torch.float32).to(device)).detach().cpu().numpy(),
        index=y_train.index, name="mlp",
    )
    mlp_test = pd.Series(
        mlp(torch.tensor(X_test_s, dtype=torch.float32).to(device)).detach().cpu().numpy(),
        index=y_test.index, name="mlp",
    )
    print(f"  done in {time.time()-t:.0f}s  test IC={spearmanr(mlp_test, y_test)[0]:+.4f}")

    # ---- Transformer ----
    t = time.time()
    print("Training Transformer...")
    Xs_tr, ys_tr, idx_tr, Xs_te, ys_te, idx_te = build_sequence_dataset(
        df, signal_cols, TARGET, TRANSFORMER_LOOKBACK, TRAIN_END, TEST_START, scaler,
    )
    trans = train_seq(TemporalTransformer, n_features, Xs_tr, ys_tr, TRANSFORMER_EPOCHS, device)
    trans_train = pd.Series(predict_seq(trans, Xs_tr, device), index=idx_tr, name="transformer")
    trans_test = pd.Series(predict_seq(trans, Xs_te, device), index=idx_te, name="transformer")
    print(f"  done in {time.time()-t:.0f}s  test IC={spearmanr(trans_test, y_test.reindex(idx_te))[0]:+.4f}")

    # ---- 3 LSTMs ----
    lstm_models = {}
    lstm_train_preds: dict[str, pd.Series] = {}
    lstm_test_preds: dict[str, pd.Series] = {}
    for name, lookback in SCALES.items():
        t = time.time()
        print(f"Training NN-{name} (lookback={lookback})...")
        sXs_tr, sys_tr, sidx_tr, sXs_te, sys_te, sidx_te = build_sequence_dataset(
            df, signal_cols, TARGET, lookback, TRAIN_END, TEST_START, scaler,
        )
        mdl = train_seq(ScaleLSTM, n_features, sXs_tr, sys_tr, SCALE_EPOCHS, device)
        lstm_models[name] = mdl
        lstm_train_preds[name] = pd.Series(predict_seq(mdl, sXs_tr, device), index=sidx_tr, name=f"nn_{name}")
        lstm_test_preds[name] = pd.Series(predict_seq(mdl, sXs_te, device), index=sidx_te, name=f"nn_{name}")
        print(f"  done in {time.time()-t:.0f}s  test IC={spearmanr(lstm_test_preds[name], y_test.reindex(sidx_te))[0]:+.4f}")

    # ---- Meta-learner ----
    print("Training Ridge meta-learner...")
    meta_train = pd.DataFrame({
        "mlp": mlp_train,
        "transformer": trans_train,
        **{f"nn_{k}": v for k, v in lstm_train_preds.items()},
    }).join(y_train.rename("y"), how="inner").dropna()
    feat_cols = [c for c in meta_train.columns if c != "y"]
    meta = Ridge(alpha=1.0, fit_intercept=False)
    meta.fit(meta_train[feat_cols].values, meta_train["y"].values)
    print("  weights:", dict(zip(feat_cols, np.round(meta.coef_, 4).tolist())))

    meta_test = pd.DataFrame({
        "mlp": mlp_test,
        "transformer": trans_test,
        **{f"nn_{k}": v for k, v in lstm_test_preds.items()},
    }).dropna()
    meta_pred = pd.Series(
        meta.predict(meta_test[feat_cols].values),
        index=meta_test.index, name="meta",
    )
    aligned_y = y_test.reindex(meta_pred.index)
    print(f"  meta test IC={spearmanr(meta_pred, aligned_y)[0]:+.4f}")

    # ---- Save ----
    torch.save(mlp.state_dict(), MODELS_DIR / "nn_mlp.pt")
    torch.save(trans.state_dict(), MODELS_DIR / "nn_transformer.pt")
    for name, mdl in lstm_models.items():
        torch.save(mdl.state_dict(), MODELS_DIR / f"nn_{name}.pt")
    with open(MODELS_DIR / "nn_meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(MODELS_DIR / "nn_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "nn_features.json", "w") as f:
        json.dump({
            "signal_cols": signal_cols,
            "transformer_lookback": TRANSFORMER_LOOKBACK,
            "scales": SCALES,
            "meta_features": feat_cols,
            "mlp_hidden": [256, 128, 64],
            "transformer_d_model": 64,
            "transformer_nhead": 4,
            "transformer_layers": 2,
            "lstm_hidden": 64,
            "lstm_layers": 2,
            "dropout": DROPOUT,
        }, f, indent=2)
    print(f"\nSaved 6 neural artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
