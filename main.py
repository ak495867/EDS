import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

AMP_ENABLED = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

def get_tickers(n=150):
    tickers = []
    try:
        import io
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        import requests
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            tickers = df['Symbol'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
    except Exception:
        pass
    if len(tickers) < 50:
        fallback = [
            'AAPL','MSFT','GOOGL','AMZN','NVDA','META','BRK-B','JPM','V','PG','JNJ','UNH','HD','MA','DIS',
            'BAC','NFLX','ADBE','CRM','CSCO','PFE','TMO','ACN','ABT','NKE','LIN','CVX','WMT','MCD','TXN',
            'AMD','INTC','QCOM','IBM','GE','CAT','GS','MS','C','WFC','AXP','BLK','LMT','MMM','HON','UNP',
            'UPS','BA','FDX','RTX','LRCX','MU','PLD','AMT','CCI','EOG','COP','SLB','XOM','CVX','BP',
            'CL','KO','PEP','MDLZ','GIS','KHC','HSY','MCD','SBUX','DPZ','YUM','CMG','DRI','MGM','LVS',
            'WYNN','MAR','HLT','HST','RCL','CCL','NCLH','AAL','DAL','LUV','UAL','T','VZ','TMUS','CHTR',
            'CMCSA','FOXA','NWS','WBD','AMC','GME','BBY','HD','LOW','TJX','ROST','BURL','KSS','M',
            'PVH','RL','VFC','LEVI','COTY','ELF','CVS','WBA','HUM','UNH','CI','CNC','MOH','BHC','RIG',
            'NOV','FTI','HAL','SLB','APA','DVN','FANG','MRO','CVE','IMO','SU','CNQ','PBR','REP','ENI',
            'SN','TTE','BP','RDS-A'
        ]
        tickers = fallback
    extra = ['SPY','QQQ','IWM','DIA','TLT','GLD','SLV','USO','EFA','EEM']
    tickers = list(dict.fromkeys(tickers + extra))
    return tickers[:n]

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def compute_features_and_target_for_asset(df, horizon=5, seq_len=20):
    df = df.copy()
    df['ret'] = df['Close'].pct_change()
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volume_change'] = df['Volume'].pct_change()
    df['high_low'] = (df['High'] - df['Low']) / df['Close']
    df['close_open'] = (df['Close'] - df['Open']) / df['Open']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volatility'] = df['ret'].rolling(20).std()
    df['ret_5'] = df['Close'].pct_change(5)
    df['ret_10'] = df['Close'].pct_change(10)
    df['target'] = df['Close'].shift(-horizon) / df['Close'] - 1.0
    feature_cols = ['ret','log_ret','volume_change','high_low','close_open','rsi','volatility','ret_5','ret_10']
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return None, None, None
    for col in feature_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    return df[feature_cols].values.astype(np.float32), df['target'].values.astype(np.float32), df.index

def create_sequences(features, targets, seq_len, horizon):
    # Vectorized via sliding_window_view instead of a Python append loop.
    n = len(features) - horizon
    if n <= seq_len:
        return np.array([]), np.array([])
    windows = np.lib.stride_tricks.sliding_window_view(features[:n], seq_len, axis=0)
    # sliding_window_view puts the window axis last; move it to axis=1
    windows = np.moveaxis(windows, -1, 1)  # (num_windows, seq_len, n_features)
    y = targets[seq_len:n]
    finite_mask = np.all(np.isfinite(windows.reshape(windows.shape[0], -1)), axis=1) & np.isfinite(y)
    return windows[finite_mask].astype(np.float32), y[finite_mask].astype(np.float32)

# ---------------------------------------------------------------------------
# Data loading — batched/threaded yfinance download (the #1 real-world win)
# ---------------------------------------------------------------------------

def load_or_download_data(tickers, start='2010-01-01', end='2023-12-31', cache_dir='data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'asset_data_{start}_{end}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # yf.download accepts the whole ticker list at once and fans requests out
    # over its own thread pool — this replaces ~150 sequential HTTP round
    # trips with a handful of batched, concurrent ones.
    raw = yf.download(
        tickers, start=start, end=end, group_by='ticker',
        threads=True, progress=True, auto_adjust=False,
    )

    data = {}
    for ticker in tqdm(tickers, desc="Splitting per-ticker frames"):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].dropna(how='all')
            else:
                df = raw  # single-ticker edge case
            if len(df) > 500:
                data[ticker] = df
        except (KeyError, Exception):
            continue

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data

def prepare_panel_data(asset_data, horizon=5, seq_len=20):
    all_X, all_y, all_dates, all_tickers = [], [], [], []
    for ticker, df in tqdm(asset_data.items(), desc="Preprocessing assets"):
        result = compute_features_and_target_for_asset(df, horizon, seq_len)
        if result[0] is None:
            continue
        features, targets, dates = result
        if len(targets) > seq_len:
            X_seq, y_seq = create_sequences(features, targets, seq_len, horizon)
            if len(y_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_dates.append(dates[seq_len:len(dates) - horizon][:len(y_seq)])
                all_tickers.append(np.full(len(y_seq), ticker))
    if not all_X:
        return None, None, None, None
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    mask = np.all(np.isfinite(X_all.reshape(X_all.shape[0], -1)), axis=1) & np.isfinite(y_all)
    X_all = X_all[mask]
    y_all = y_all[mask]
    dates_all = np.concatenate(all_dates)[mask]
    tickers_all = np.concatenate(all_tickers)[mask]
    return X_all, y_all, dates_all, tickers_all

# ---------------------------------------------------------------------------
# Models (unchanged math — EDS_V2's per-step autograd.grad is inherent to the
# physics-style dynamics and can't be vectorized across time because z_eq at
# step t depends on z at step t-1. We speed it up with AMP + torch.compile
# instead of changing the model.)
# ---------------------------------------------------------------------------

class EDSModelV2(nn.Module):
    def __init__(self, input_dim, latent_dim=16, seq_len=20, horizon=5, gamma=0.15):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.gamma = gamma
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.impulse_rnn = nn.GRU(input_dim, latent_dim, batch_first=True)
        self.gru_eq = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.V = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.beta_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.mu_head = nn.Sequential(nn.Linear(latent_dim * 3 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.logvar_head = nn.Sequential(nn.Linear(latent_dim * 3 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = self.encoder(x)
        z_eq_seq, _ = self.gru_eq(h)
        impulse_seq, _ = self.impulse_rnn(x)
        z = torch.zeros(batch_size, self.latent_dim, device=x.device, requires_grad=True)
        v = torch.zeros(batch_size, self.latent_dim, device=x.device, requires_grad=True)
        z_seq, v_seq, z_eq_out, delta_seq, impulse_list, grad_V_list = [], [], [], [], [], []
        for t in range(seq_len):
            z_eq = z_eq_seq[:, t, :]
            delta = z - z_eq
            delta.requires_grad_(True)
            vol = x[:, t, 6].unsqueeze(-1)
            beta = self.beta_net(vol)
            z_eq = z_eq + beta * delta
            impulse = impulse_seq[:, t, :]
            V_vals = self.V(delta).squeeze(-1)
            V_total = V_vals.sum()
            grad_V = torch.autograd.grad(V_total, delta, create_graph=True)[0]
            acc = -self.gamma * v - grad_V + impulse
            v = v + acc
            z = z + v
            z_seq.append(z)
            v_seq.append(v)
            z_eq_out.append(z_eq)
            delta_seq.append(delta)
            impulse_list.append(impulse)
            grad_V_list.append(grad_V)
        z_seq = torch.stack(z_seq, dim=1)
        v_seq = torch.stack(v_seq, dim=1)
        z_eq_seq = torch.stack(z_eq_out, dim=1)
        delta_seq = torch.stack(delta_seq, dim=1)
        impulse_seq = torch.stack(impulse_list, dim=1)
        grad_V_seq = torch.stack(grad_V_list, dim=1)
        final_z = z_seq[:, -1, :]
        final_v = v_seq[:, -1, :]
        final_eq = z_eq_seq[:, -1, :]
        delta_final = final_z - final_eq
        V_final = self.V(delta_final)
        feat = torch.cat([final_z, final_v, final_eq, V_final], dim=1)
        mu = self.mu_head(feat).squeeze(-1)
        logvar = self.logvar_head(feat).squeeze(-1)
        return mu, logvar, z_seq, v_seq, z_eq_seq, delta_seq, impulse_seq, grad_V_seq

    def compute_loss(self, mu, logvar, target, z_seq, v_seq, z_eq_seq, delta_seq, impulse_seq, grad_V_seq, lambda1=0.1, lambda2=0.01):
        pred_loss = 0.5 * (logvar + (target - mu) ** 2 / torch.exp(logvar)).mean()
        acc_actual = v_seq[:, 1:, :] - v_seq[:, :-1, :]
        acc_pred = -self.gamma * v_seq[:, :-1, :] - grad_V_seq[:, :-1, :] + impulse_seq[:, :-1, :]
        dyn_loss = nn.functional.mse_loss(acc_actual, acc_pred)
        stab_loss = nn.functional.mse_loss(z_eq_seq[:, 1:, :], z_eq_seq[:, :-1, :])
        return pred_loss + lambda1 * dyn_loss + lambda2 * stab_loss

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.to(device)
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_enc[:, :seq_len, :]
        return self.fc(self.transformer(x)[:, -1, :]).squeeze(-1)

# ---------------------------------------------------------------------------
# GPU-resident manual batching — skips DataLoader/collate overhead entirely
# for data that comfortably fits in memory.
# ---------------------------------------------------------------------------

class GPUBatcher:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X.shape[0]

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.X.device) if self.shuffle else torch.arange(self.n, device=self.X.device)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i + self.batch_size]
            yield self.X[b], self.y[b]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

def train_eds(model, train_loader, val_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'EDS Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=AMP_ENABLED):
                mu, logvar, z_seq, v_seq, z_eq, delta_seq, impulse_seq, grad_V_seq = model(Xb)
                loss = model.compute_loss(mu, logvar, yb, z_seq, v_seq, z_eq, delta_seq, impulse_seq, grad_V_seq)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_loss = 0.0
        for Xb, yb in val_loader:
            mu, logvar, z_seq, v_seq, z_eq, delta_seq, impulse_seq, grad_V_seq = model(Xb)
            loss = model.compute_loss(mu, logvar, yb, z_seq, v_seq, z_eq, delta_seq, impulse_seq, grad_V_seq)
            val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        if val_loss < best_loss:
            best_loss = val_loss
        print(f"  EDS val_loss={val_loss:.6f}")
    return model

def train_simple(model, train_loader, val_loader, epochs=10, lr=0.001, name='model'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=AMP_ENABLED):
                loss = nn.functional.mse_loss(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += nn.functional.mse_loss(model(Xb), yb).item()
        val_loss /= max(len(val_loader), 1)
        if val_loss < best_loss:
            best_loss = val_loss
        print(f"  {name} val_loss={val_loss:.6f}")
    return model

def evaluate_model_batched(model, X_test, batch_size=8192):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            Xb = X_test[i:i + batch_size]
            preds.append(model(Xb).cpu().numpy())
    return np.concatenate(preds)

def evaluate_model_variance(model, X_test, batch_size=8192):
    model.eval()
    mus, logvars = [], []
    for i in range(0, X_test.shape[0], batch_size):
        Xb = X_test[i:i + batch_size]
        with torch.enable_grad():
            mu, logvar, *_ = model(Xb)
        mus.append(mu.detach().cpu().numpy())
        logvars.append(logvar.detach().cpu().numpy())
    return np.concatenate(mus), np.exp(np.concatenate(logvars))

def compute_strategy_returns_kelly(mu, sigma2, actual, top=0.1, lambda_reg=1e-3, tc=0.001):
    weights = mu / (sigma2 + lambda_reg)
    n = len(weights)
    ranks = np.argsort(weights)
    long_idx = ranks[-int(n * top):]
    short_idx = ranks[:int(n * top)]
    w = np.zeros(n)
    w[long_idx] = 1.0 / int(n * top)
    w[short_idx] = -1.0 / int(n * top)
    ret = w * actual
    cost = tc * np.sum(np.abs(w))
    ret -= cost / n
    return ret

def main():
    tickers = get_tickers(150)
    print(f"Attempting to load data for {len(tickers)} tickers")
    asset_data = load_or_download_data(tickers)
    print(f"Successfully loaded {len(asset_data)} assets")
    X_all, y_all, dates_all, tickers_all = prepare_panel_data(asset_data, horizon=5, seq_len=20)
    if X_all is None:
        print("No data available after preprocessing.")
        return
    print(f"Total samples: {len(X_all)}")
    date_series = pd.to_datetime(dates_all)
    unique_dates = np.sort(np.unique(date_series))
    train_dates_all = unique_dates[unique_dates < pd.Timestamp('2018-01-01')]
    test_dates = unique_dates[(unique_dates >= pd.Timestamp('2018-01-01')) & (unique_dates < pd.Timestamp('2021-01-01'))]
    if len(train_dates_all) == 0 or len(test_dates) == 0:
        print("Insufficient date range for split.")
        return

    # Real train/val split (last ~10% of training dates held out), instead of
    # validating on the training set itself.
    split_point = int(len(train_dates_all) * 0.9)
    train_dates = train_dates_all[:split_point]
    val_dates = train_dates_all[split_point:]

    train_mask = np.isin(date_series, train_dates)
    val_mask = np.isin(date_series, val_dates)
    test_mask = np.isin(date_series, test_dates)
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    if np.any(~np.isfinite(X_train_flat)):
        raise ValueError("Non-finite values found in training data.")
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Move everything to the GPU once; batch by indexing instead of DataLoader.
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    BATCH_SIZE = 2048  # bumped up from 512 — per-step ops are tiny, so small batches under-use the GPU

    models = {
        'EDS_V2': EDSModelV2(input_dim=X_train.shape[-1], latent_dim=16, seq_len=20, horizon=5, gamma=0.15),
        'LSTM': LSTMModel(input_dim=X_train.shape[-1]),
        'Transformer': TransformerModel(input_dim=X_train.shape[-1]),
    }

    if torch.cuda.is_available():
        for name in models:
            try:
                models[name] = torch.compile(models[name])
            except Exception:
                pass  # torch.compile unavailable/unsupported — fall back silently

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_loader = GPUBatcher(X_train_t, y_train_t, BATCH_SIZE, shuffle=True)
        val_loader = GPUBatcher(X_val_t, y_val_t, BATCH_SIZE, shuffle=False)
        if name == 'EDS_V2':
            model = train_eds(model, train_loader, val_loader, epochs=10, lr=0.001)
        else:
            model = train_simple(model, train_loader, val_loader, epochs=10, lr=0.001, name=name)
        trained[name] = model

    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, random_state=42,
        tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
        n_jobs=-1,
    )
    xgb_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)

    all_preds = {}
    for name, model in trained.items():
        if name == 'EDS_V2':
            mu, var = evaluate_model_variance(model, X_test_t, batch_size=8192)
            all_preds['EDS_V2_mu'] = mu
            all_preds['EDS_V2_var'] = var
        else:
            all_preds[name] = evaluate_model_batched(model, X_test_t, batch_size=8192)
    all_preds['Linear'] = lr_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    all_preds['XGBoost'] = xgb_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))

    ic_results = {}
    strat_ret = {}
    for name, preds in all_preds.items():
        if name.endswith('_mu') or name.endswith('_var'):
            continue
        ic_results[name] = spearmanr(preds, y_test)[0]
        strat_ret[name] = compute_strategy_returns_kelly(preds, np.ones_like(preds) * 0.01, y_test, top=0.1, lambda_reg=1e-3, tc=0.001)

    if 'EDS_V2_mu' in all_preds and 'EDS_V2_var' in all_preds:
        mu = all_preds['EDS_V2_mu']
        var = all_preds['EDS_V2_var']
        ic_results['EDS_V2'] = spearmanr(mu, y_test)[0]
        strat_ret['EDS_V2'] = compute_strategy_returns_kelly(mu, var, y_test, top=0.1, lambda_reg=1e-3, tc=0.001)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Long-Short Returns (with Kelly)')
    ax.legend()
    ax.grid(True)
    ax = axes[0, 1]
    ax.bar(list(ic_results.keys()), list(ic_results.values()))
    ax.set_title('Cross-Sectional IC (Spearman)')
    ax.grid(True)
    ax = axes[1, 0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Returns (Log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    ax = axes[1, 1]
    sharpe = {name: np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252) for name, ret in strat_ret.items()}
    ax.bar(list(sharpe.keys()), list(sharpe.values()))
    ax.set_title('Sharpe Ratio (Annualized)')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('eds_v2_eval.png')
    plt.show()

    print("IC Results:")
    for name, ic in ic_results.items():
        print(f"{name}: {ic:.4f}")
    print("\nSharpe Ratios (with Kelly & transaction costs):")
    for name, sr in sharpe.items():
        print(f"{name}: {sr:.4f}")

if __name__ == '__main__':
    main()