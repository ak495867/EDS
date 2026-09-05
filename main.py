import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr, norm
import matplotlib.pyplot as plt

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","BAC","XOM",
    "CVX","JNJ","PG","KO","WMT",
    "XLK","XLF","XLE","XLV","XLY",
    "EFA","EEM","FXI","EWJ","EWZ",
    "TLT","IEF","SHY","LQD","HYG",
    "GLD","SLV","USO","DBC","UNG",
    "BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOGE-USD",
    "UUP","FXE","FXY","FXB","FXA",
    "VNQ","O","SPG","PLD","AMT",
]

START = "2015-01-01"
END = "2025-01-01"
WINDOW = 20
HORIZON = 5
D_MODEL = 32
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
TRAIN_FRAC = 0.7
PURGE_DAYS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def download_all(tickers, start, end):
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or len(df) < 300:
                continue
            df = df[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
            out[t] = df
        except Exception:
            continue
    return out

def build_features(df, horizon):
    c = df["close"]
    v = df["volume"]
    r1 = c.pct_change(1)
    r5 = c.pct_change(5)
    r10 = c.pct_change(10)
    r20 = c.pct_change(20)
    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma5_ratio = ma5 / c
    ma10_ratio = ma10 / c
    ma20_ratio = ma20 / c
    ma5_ma20 = ma5 / ma20
    vol5 = r1.rolling(5).std()
    vol10 = r1.rolling(10).std()
    vol20 = r1.rolling(20).std()
    vol_mean20 = v.rolling(20).mean()
    vol_std20 = v.rolling(20).std()
    vol_zscore = (v - vol_mean20) / (vol_std20 + 1e-9)
    obv = (np.sign(c.diff()) * v).cumsum()
    obv_slope = obv.diff(5) / (v.rolling(5).mean() + 1e-9)
    low_20 = c.rolling(20).min()
    high_20 = c.rolling(20).max()
    stoch = (c - low_20) / (high_20 - low_20 + 1e-9)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rsi7 = 100 - 100 / (1 + up.rolling(7).mean() / (down.rolling(7).mean() + 1e-9))
    rsi14 = 100 - 100 / (1 + up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9))
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    feat = pd.concat([
        r1, r5, r10, r20,
        ma5_ratio, ma10_ratio, ma20_ratio, ma5_ma20,
        vol5, vol10, vol20,
        vol_zscore, obv_slope,
        stoch, rsi7, rsi14, bb_pos
    ], axis=1)
    feat.columns = [f"f{i}" for i in range(feat.shape[1])]
    fwd_ret = c.pct_change(horizon).shift(-horizon)
    full = pd.concat([feat, fwd_ret.rename("target")], axis=1)
    return full.dropna()

def make_windows(feat_df, window, ticker):
    cols = [f"f{i}" for i in range(feat_df.shape[1]-1)]
    X = feat_df[cols].values
    y = feat_df["target"].values
    r = feat_df["target"].values
    idx = feat_df.index
    Xs, ys, rs, ds, ts = [], [], [], [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
        rs.append(r[i])
        ds.append(idx[i])
        ts.append(ticker)
    if len(Xs) == 0:
        return None
    return np.array(Xs), np.array(ys), np.array(rs), np.array(ds), np.array(ts)

def assemble_panel(data, window, horizon):
    Xs, ys, rs, ds, ts = [], [], [], [], []
    for tkr, df in data.items():
        feat = build_features(df, horizon)
        if len(feat) < window + 30:
            continue
        packed = make_windows(feat, window, tkr)
        if packed is None:
            continue
        X, y, r, d, t = packed
        Xs.append(X); ys.append(y); rs.append(r); ds.append(d); ts.append(t)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    r = np.concatenate(rs, axis=0)
    d = np.concatenate(ds, axis=0)
    t = np.concatenate(ts, axis=0)
    order = np.argsort(d)
    return X[order], y[order], r[order], d[order], t[order]

def purged_split(X, y, r, d, t, train_frac, purge_days):
    unique_dates = np.sort(np.unique(d))
    cut = unique_dates[int(len(unique_dates) * train_frac)]
    purge_end = cut + np.timedelta64(purge_days, "D")
    train_mask = d <= cut
    test_mask = d > purge_end
    return (X[train_mask], y[train_mask], r[train_mask], d[train_mask], t[train_mask]), \
           (X[test_mask], y[test_mask], r[test_mask], d[test_mask], t[test_mask])

def fit_scaler(X_train):
    n, w, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, f))
    return scaler

def apply_scaler(X, scaler):
    n, w, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, w, f)

class WindowDataset(Dataset):
    def __init__(self, X, y, r):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.r = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.r[idx]

class Encoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, d_model))

    def forward(self, x):
        return self.net(x)

class SimplifiedEDS(nn.Module):
    def __init__(self, in_dim, d_model=32, ema_alpha=0.15):
        super().__init__()
        self.encoder = Encoder(in_dim, d_model)
        self.ema_alpha = ema_alpha
        self.head = nn.Sequential(nn.Linear(d_model * 3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        B, T, F = x.shape
        h = self.encoder(x)
        z_eq = h[:, 0, :]
        eqs = [z_eq]
        for t in range(1, T):
            z_eq = self.ema_alpha * h[:, t, :] + (1 - self.ema_alpha) * z_eq
            eqs.append(z_eq)
        z_eq_seq = torch.stack(eqs, dim=1)
        z_eq_last = z_eq_seq[:, -1, :]
        z_obs = h[:, -1, :]
        dz = z_obs - z_eq_last
        feat = torch.cat([z_obs, z_eq_last, dz], dim=-1)
        out = self.head(feat).squeeze(-1)
        return out

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class TransformerBaseline(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)

def train_regressor(model, loader, epochs, lr, name):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for X, y, r in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = mse(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        print(f"{name} epoch {ep+1}/{epochs} loss {total/len(loader.dataset):.6f}")
    return model

@torch.inference_mode()
def predict_torch(model, X):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    for i in range(0, len(Xt), BATCH_SIZE):
        batch = Xt[i:i+BATCH_SIZE]
        pred = model(batch)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)

def flatten_last(X):
    return X[:, -1, :]

def information_coefficient(scores, fwd_ret):
    mask = ~np.isnan(fwd_ret) & ~np.isnan(scores)
    if mask.sum() < 2:
        return 0.0
    ic, _ = spearmanr(scores[mask], fwd_ret[mask])
    return ic

def sharpe_ratio(daily_rets, periods=252):
    daily_rets = np.array(daily_rets)
    if daily_rets.std() == 0 or len(daily_rets) < 2:
        return 0.0
    return (daily_rets.mean() / daily_rets.std()) * np.sqrt(periods)

def backtest(dates, scores, fwd_ret, threshold=0.0):
    df = pd.DataFrame({"date": dates, "score": scores, "fwd_ret": fwd_ret})
    df["pos"] = np.sign(df["score"] - threshold)
    df["pnl"] = df["pos"] * df["fwd_ret"]
    daily = df.groupby("date")["pnl"].mean()
    curve = (1 + daily.fillna(0)).cumprod()
    return daily, curve

def diebold_mariano(err_a, err_b):
    d = err_a - err_b
    dbar = d.mean()
    n = len(d)
    var_d = d.var(ddof=1)
    if var_d == 0:
        return 0.0, 1.0
    dm_stat = dbar / np.sqrt(var_d / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def permutation_sharpe_test(daily_rets, n_perm=1000):
    observed = sharpe_ratio(daily_rets)
    rets = daily_rets.values
    perm_sharpes = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(rets)
        perm_sharpes.append(sharpe_ratio(shuffled))
    perm_sharpes = np.array(perm_sharpes)
    p_value = (np.sum(perm_sharpes >= observed) + 1) / (n_perm + 1)
    return observed, p_value

def plot_equity_curves(curves, path="equity_curves.png"):
    plt.figure(figsize=(10, 6))
    for name, curve in curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.legend()
    plt.title("Cumulative Strategy Performance by Model")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_bar_metric(metric_dict, title, ylabel, path):
    plt.figure(figsize=(8, 5))
    names = list(metric_dict.keys())
    vals = list(metric_dict.values())
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    data = download_all(TICKERS, START, END)
    X, y, r, d, t = assemble_panel(data, WINDOW, HORIZON)
    (Xtr, ytr, rtr, dtr, ttr), (Xte, yte, rte, dte, tte) = purged_split(X, y, r, d, t, TRAIN_FRAC, PURGE_DAYS)

    scaler = fit_scaler(Xtr)
    Xtr_s = apply_scaler(Xtr, scaler)
    Xte_s = apply_scaler(Xte, scaler)

    train_loader = DataLoader(
        WindowDataset(Xtr_s, ytr, rtr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    eds = SimplifiedEDS(in_dim=Xtr_s.shape[-1], d_model=D_MODEL)
    eds = train_regressor(eds, train_loader, EPOCHS, LR, "EDS")

    lstm = LSTMBaseline(in_dim=Xtr_s.shape[-1])
    lstm = train_regressor(lstm, train_loader, EPOCHS, LR, "LSTM")

    transformer = TransformerBaseline(in_dim=Xtr_s.shape[-1])
    transformer = train_regressor(transformer, train_loader, EPOCHS, LR, "Transformer")

    Xtr_flat = flatten_last(Xtr_s)
    Xte_flat = flatten_last(Xte_s)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr_flat, ytr)

    gbt = GradientBoostingRegressor(random_state=SEED)
    gbt.fit(Xtr_flat, ytr)

    scores = {
        "EDS": predict_torch(eds, Xte_s),
        "LSTM": predict_torch(lstm, Xte_s),
        "Transformer": predict_torch(transformer, Xte_s),
        "Ridge": ridge.predict(Xte_flat),
        "GradientBoosting": gbt.predict(Xte_flat),
    }

    ic_results = {}
    sharpe_results = {}
    curves = {}
    daily_returns = {}

    for name, s in scores.items():
        ic_results[name] = information_coefficient(s, rte)
        daily, curve = backtest(dte, s, rte)
        sharpe_results[name] = sharpe_ratio(daily)
        curves[name] = curve
        daily_returns[name] = daily

    print("Information Coefficients:", ic_results)
    print("Sharpe Ratios:", sharpe_results)

    eds_err = (scores["EDS"] - yte) ** 2
    for name in ["LSTM", "Transformer", "Ridge", "GradientBoosting"]:
        base_err = (scores[name] - yte) ** 2
        stat, pval = diebold_mariano(eds_err, base_err)
        print(f"DM test EDS vs {name}: stat={stat:.4f} p={pval:.4f}")

    for name, daily in daily_returns.items():
        obs, pval = permutation_sharpe_test(daily)
        print(f"Permutation test {name}: Sharpe={obs:.4f} p={pval:.4f}")

    plot_equity_curves(curves)
    plot_bar_metric(ic_results, "Information Coefficient by Model", "IC", "ic_by_model.png")
    plot_bar_metric(sharpe_results, "Sharpe Ratio by Model", "Sharpe", "sharpe_by_model.png")

if __name__ == "__main__":
    main()