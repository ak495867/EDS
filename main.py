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
from torch.utils.data import DataLoader, TensorDataset
import requests
import os
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_tickers(n=200):
    tickers = []
    try:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(pd.StringIO(response.text))
            tickers = df['Symbol'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
    except:
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
    return df[feature_cols].values, df['target'].values, df.index

def create_sequences(features, targets, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(features) - horizon):
        seq = features[i-seq_len:i]
        if np.any(~np.isfinite(seq)):
            continue
        X.append(seq)
        y.append(targets[i])
    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_or_download_data(tickers, start='2010-01-01', end='2023-12-31', cache_dir='data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'asset_data_{start}_{end}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    data = {}
    for ticker in tqdm(tickers, desc="Downloading data"):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=15)
            if len(df) > 500:
                data[ticker] = df
        except:
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
                all_dates.append(dates[seq_len:len(dates)-horizon])
                all_tickers.append([ticker]*len(y_seq))
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

class EDSModel(nn.Module):
    def __init__(self, input_dim, latent_dim=16, seq_len=20, horizon=5, lambda_restore=0.5):
        super(EDSModel, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.lambda_restore = lambda_restore
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.gru_eq = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.impulse_net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = self.encoder(x)
        z_eq_seq, _ = self.gru_eq(h)
        z_obs = torch.zeros(batch_size, self.latent_dim, device=x.device)
        z_obs_seq, delta_z_seq, dz_pred_seq = [], [], []
        for t in range(seq_len):
            z_eq_t = z_eq_seq[:, t, :]
            delta_z = z_obs - z_eq_t
            impulse = self.impulse_net(x[:, t, :])
            dz = -self.lambda_restore * delta_z + impulse
            z_obs = z_obs + dz
            z_obs_seq.append(z_obs)
            delta_z_seq.append(delta_z)
            dz_pred_seq.append(dz)
        z_obs_seq = torch.stack(z_obs_seq, dim=1)
        delta_z_seq = torch.stack(delta_z_seq, dim=1)
        dz_pred_seq = torch.stack(dz_pred_seq, dim=1)
        pred_in = torch.cat([
            z_obs_seq[:, -1, :],
            z_eq_seq[:, -1, :],
            delta_z_seq[:, -1, :],
            dz_pred_seq[:, -1, :].norm(dim=1, keepdim=True)
        ], dim=1)
        return self.pred_head(pred_in).squeeze(-1), z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq

    def compute_loss(self, pred, target, z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq, lambda1=0.1, lambda2=0.01):
        pred_loss = nn.functional.mse_loss(pred, target)
        dz_actual = z_obs_seq[:, 1:, :] - z_obs_seq[:, :-1, :]
        dz_pred = dz_pred_seq[:, :-1, :]
        dynamics_loss = nn.functional.mse_loss(dz_actual, dz_pred)
        stability_loss = nn.functional.mse_loss(z_eq_seq[:, 1:, :], z_eq_seq[:, :-1, :])
        return pred_loss + lambda1 * dynamics_loss + lambda2 * stability_loss

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
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

def train_eds(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'EDS Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, z_obs, z_eq, delta, dz_pred = model(Xb)
            loss = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, z_obs, z_eq, delta, dz_pred = model(Xb)
                loss = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_lstm(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'LSTM Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += nn.functional.mse_loss(model(Xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_transformer(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'Transformer Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += nn.functional.mse_loss(model(Xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def evaluate_model(model, X_test, y_test, is_torch=True):
    if is_torch:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            if hasattr(model, 'compute_loss'):
                pred, _, _, _, _ = model(X_t)
            else:
                pred = model(X_t)
            return pred.cpu().numpy()
    else:
        return model.predict(X_test.reshape(X_test.shape[0], -1))

def compute_strategy_returns(preds, actual, top=0.1):
    n = len(preds)
    ranks = np.argsort(preds)
    long_idx = ranks[-int(n*top):]
    short_idx = ranks[:int(n*top)]
    ret = np.zeros(n)
    ret[long_idx] += actual[long_idx] / int(n*top)
    ret[short_idx] -= actual[short_idx] / int(n*top)
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
    train_dates = unique_dates[unique_dates < pd.Timestamp('2018-01-01')]
    test_dates = unique_dates[(unique_dates >= pd.Timestamp('2018-01-01')) & (unique_dates < pd.Timestamp('2021-01-01'))]
    if len(train_dates) == 0 or len(test_dates) == 0:
        print("Insufficient date range for split.")
        return

    train_mask = np.isin(date_series, train_dates)
    test_mask = np.isin(date_series, test_dates)
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    if np.any(~np.isfinite(X_train_flat)):
        raise ValueError("Non-finite values found in training data after preprocessing.")
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    models = {
        'EDS': EDSModel(input_dim=X_train.shape[-1], latent_dim=16, seq_len=20, horizon=5),
        'LSTM': LSTMModel(input_dim=X_train.shape[-1]),
        'Transformer': TransformerModel(input_dim=X_train.shape[-1])
    }
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
        val_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, pin_memory=True)
        if name == 'EDS':
            model = train_eds(model, train_loader, val_loader, epochs=20)
        elif name == 'LSTM':
            model = train_lstm(model, train_loader, val_loader, epochs=20)
        elif name == 'Transformer':
            model = train_transformer(model, train_loader, val_loader, epochs=20)
        trained[name] = model

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu')
    xgb_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)

    all_preds = {}
    for name, model in trained.items():
        all_preds[name] = evaluate_model(model, X_test_scaled, y_test, is_torch=True)
    all_preds['Linear'] = lr_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    all_preds['XGBoost'] = xgb_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))

    ic_results = {name: spearmanr(preds, y_test)[0] for name, preds in all_preds.items()}
    strat_ret = {name: compute_strategy_returns(preds, y_test) for name, preds in all_preds.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Long-Short Returns')
    ax.legend()
    ax.grid(True)

    ax = axes[0,1]
    ax.bar(list(ic_results.keys()), list(ic_results.values()))
    ax.set_title('Cross-Sectional IC (Spearman)')
    ax.grid(True)

    ax = axes[1,0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Returns (Log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    ax = axes[1,1]
    sharpe = {name: np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252) for name, ret in strat_ret.items()}
    ax.bar(list(sharpe.keys()), list(sharpe.values()))
    ax.set_title('Sharpe Ratio (Annualized)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('eds_multi_asset_eval.png')
    plt.show()

    print("IC Results:")
    for name, ic in ic_results.items():
        print(f"{name}: {ic:.4f}")
    print("\nSharpe Ratios:")
    for name, sr in sharpe.items():
        print(f"{name}: {sr:.4f}")

if __name__ == '__main__':
    main()import torch
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
from torch.utils.data import DataLoader, TensorDataset
import requests
import os
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_tickers(n=200):
    tickers = []
    try:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(pd.StringIO(response.text))
            tickers = df['Symbol'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
    except:
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
    return df[feature_cols].values, df['target'].values, df.index

def create_sequences(features, targets, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(features) - horizon):
        seq = features[i-seq_len:i]
        if np.any(~np.isfinite(seq)):
            continue
        X.append(seq)
        y.append(targets[i])
    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_or_download_data(tickers, start='2010-01-01', end='2023-12-31', cache_dir='data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'asset_data_{start}_{end}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    data = {}
    for ticker in tqdm(tickers, desc="Downloading data"):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=15)
            if len(df) > 500:
                data[ticker] = df
        except:
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
                all_dates.append(dates[seq_len:len(dates)-horizon])
                all_tickers.append([ticker]*len(y_seq))
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

class EDSModel(nn.Module):
    def __init__(self, input_dim, latent_dim=16, seq_len=20, horizon=5, lambda_restore=0.5):
        super(EDSModel, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.lambda_restore = lambda_restore
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.gru_eq = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.impulse_net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = self.encoder(x)
        z_eq_seq, _ = self.gru_eq(h)
        z_obs = torch.zeros(batch_size, self.latent_dim, device=x.device)
        z_obs_seq, delta_z_seq, dz_pred_seq = [], [], []
        for t in range(seq_len):
            z_eq_t = z_eq_seq[:, t, :]
            delta_z = z_obs - z_eq_t
            impulse = self.impulse_net(x[:, t, :])
            dz = -self.lambda_restore * delta_z + impulse
            z_obs = z_obs + dz
            z_obs_seq.append(z_obs)
            delta_z_seq.append(delta_z)
            dz_pred_seq.append(dz)
        z_obs_seq = torch.stack(z_obs_seq, dim=1)
        delta_z_seq = torch.stack(delta_z_seq, dim=1)
        dz_pred_seq = torch.stack(dz_pred_seq, dim=1)
        pred_in = torch.cat([
            z_obs_seq[:, -1, :],
            z_eq_seq[:, -1, :],
            delta_z_seq[:, -1, :],
            dz_pred_seq[:, -1, :].norm(dim=1, keepdim=True)
        ], dim=1)
        return self.pred_head(pred_in).squeeze(-1), z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq

    def compute_loss(self, pred, target, z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq, lambda1=0.1, lambda2=0.01):
        pred_loss = nn.functional.mse_loss(pred, target)
        dz_actual = z_obs_seq[:, 1:, :] - z_obs_seq[:, :-1, :]
        dz_pred = dz_pred_seq[:, :-1, :]
        dynamics_loss = nn.functional.mse_loss(dz_actual, dz_pred)
        stability_loss = nn.functional.mse_loss(z_eq_seq[:, 1:, :], z_eq_seq[:, :-1, :])
        return pred_loss + lambda1 * dynamics_loss + lambda2 * stability_loss

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
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

def train_eds(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'EDS Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, z_obs, z_eq, delta, dz_pred = model(Xb)
            loss = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, z_obs, z_eq, delta, dz_pred = model(Xb)
                loss = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_lstm(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'LSTM Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += nn.functional.mse_loss(model(Xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_transformer(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tqdm(train_loader, desc=f'Transformer Epoch {epoch+1}/{epochs}'):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += nn.functional.mse_loss(model(Xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def evaluate_model(model, X_test, y_test, is_torch=True):
    if is_torch:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            if hasattr(model, 'compute_loss'):
                pred, _, _, _, _ = model(X_t)
            else:
                pred = model(X_t)
            return pred.cpu().numpy()
    else:
        return model.predict(X_test.reshape(X_test.shape[0], -1))

def compute_strategy_returns(preds, actual, top=0.1):
    n = len(preds)
    ranks = np.argsort(preds)
    long_idx = ranks[-int(n*top):]
    short_idx = ranks[:int(n*top)]
    ret = np.zeros(n)
    ret[long_idx] += actual[long_idx] / int(n*top)
    ret[short_idx] -= actual[short_idx] / int(n*top)
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
    train_dates = unique_dates[unique_dates < pd.Timestamp('2018-01-01')]
    test_dates = unique_dates[(unique_dates >= pd.Timestamp('2018-01-01')) & (unique_dates < pd.Timestamp('2021-01-01'))]
    if len(train_dates) == 0 or len(test_dates) == 0:
        print("Insufficient date range for split.")
        return

    train_mask = np.isin(date_series, train_dates)
    test_mask = np.isin(date_series, test_dates)
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    if np.any(~np.isfinite(X_train_flat)):
        raise ValueError("Non-finite values found in training data after preprocessing.")
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    models = {
        'EDS': EDSModel(input_dim=X_train.shape[-1], latent_dim=16, seq_len=20, horizon=5),
        'LSTM': LSTMModel(input_dim=X_train.shape[-1]),
        'Transformer': TransformerModel(input_dim=X_train.shape[-1])
    }
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
        val_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, pin_memory=True)
        if name == 'EDS':
            model = train_eds(model, train_loader, val_loader, epochs=20)
        elif name == 'LSTM':
            model = train_lstm(model, train_loader, val_loader, epochs=20)
        elif name == 'Transformer':
            model = train_transformer(model, train_loader, val_loader, epochs=20)
        trained[name] = model

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu')
    xgb_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)

    all_preds = {}
    for name, model in trained.items():
        all_preds[name] = evaluate_model(model, X_test_scaled, y_test, is_torch=True)
    all_preds['Linear'] = lr_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    all_preds['XGBoost'] = xgb_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))

    ic_results = {name: spearmanr(preds, y_test)[0] for name, preds in all_preds.items()}
    strat_ret = {name: compute_strategy_returns(preds, y_test) for name, preds in all_preds.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Long-Short Returns')
    ax.legend()
    ax.grid(True)

    ax = axes[0,1]
    ax.bar(list(ic_results.keys()), list(ic_results.values()))
    ax.set_title('Cross-Sectional IC (Spearman)')
    ax.grid(True)

    ax = axes[1,0]
    for name, ret in strat_ret.items():
        ax.plot(np.cumsum(ret), label=name)
    ax.set_title('Cumulative Returns (Log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    ax = axes[1,1]
    sharpe = {name: np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252) for name, ret in strat_ret.items()}
    ax.bar(list(sharpe.keys()), list(sharpe.values()))
    ax.set_title('Sharpe Ratio (Annualized)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('eds_multi_asset_eval.png')
    plt.show()

    print("IC Results:")
    for name, ic in ic_results.items():
        print(f"{name}: {ic:.4f}")
    print("\nSharpe Ratios:")
    for name, sr in sharpe.items():
        print(f"{name}: {sr:.4f}")

if __name__ == '__main__':
    main()