import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

def get_tickers(n=500):
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(sp500_url)[0]
    tickers = sp500['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'SLV', 'USO', 'EFA', 'EEM']
    commodities = ['GC=F', 'CL=F', 'SI=F', 'HG=F', 'NG=F']
    bonds = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']
    tickers = tickers[:450] + etfs + commodities + bonds
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
    feature_cols = ['ret', 'log_ret', 'volume_change', 'high_low', 'close_open',
                    'rsi', 'volatility', 'ret_5', 'ret_10']
    df = df.dropna()
    return df[feature_cols].values, df['target'].values, df.index

def create_sequences(features, targets, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(features) - horizon):
        X.append(features[i-seq_len:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class ScalerWrapper:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    def fit(self, X):
        self.scaler.fit(X)
        self.fitted = True
    def transform(self, X):
        return self.scaler.transform(X)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class EDSModel(nn.Module):
    def __init__(self, input_dim, latent_dim=16, seq_len=20, horizon=5, lambda_restore=0.5):
        super(EDSModel, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.lambda_restore = lambda_restore
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.gru_eq = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.impulse_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        h = self.encoder(x)
        z_eq_seq, _ = self.gru_eq(h)
        z_obs = torch.zeros(batch_size, self.latent_dim, device=x.device)
        z_obs_seq = []
        delta_z_seq = []
        dz_pred_seq = []
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
        last_state = z_obs_seq[:, -1, :]
        last_eq = z_eq_seq[:, -1, :]
        last_delta = delta_z_seq[:, -1, :]
        last_dz = dz_pred_seq[:, -1, :]
        pred_in = torch.cat([last_state, last_eq, last_delta, last_dz.norm(dim=1, keepdim=True)], dim=1)
        pred = self.pred_head(pred_in).squeeze(-1)
        return pred, z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq

    def compute_loss(self, pred, target, z_obs_seq, z_eq_seq, delta_z_seq, dz_pred_seq, lambda1=0.1, lambda2=0.01):
        pred_loss = nn.functional.mse_loss(pred, target)
        dz_actual = z_obs_seq[:, 1:, :] - z_obs_seq[:, :-1, :]
        dz_pred = dz_pred_seq[:, :-1, :]
        dynamics_loss = nn.functional.mse_loss(dz_actual, dz_pred)
        stability_loss = nn.functional.mse_loss(z_eq_seq[:, 1:, :], z_eq_seq[:, :-1, :])
        total_loss = pred_loss + lambda1 * dynamics_loss + lambda2 * stability_loss
        return total_loss, pred_loss.item(), dynamics_loss.item(), stability_loss.item()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
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
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_enc[:, :seq_len, :]
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_eds(model, train_loader, val_loader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred, z_obs, z_eq, delta, dz_pred = model(Xb)
            loss, pl, dl, sl = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred, z_obs, z_eq, delta, dz_pred = model(Xb)
                loss, _, _, _ = model.compute_loss(pred, yb, z_obs, z_eq, delta, dz_pred)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_lstm(model, train_loader, val_loader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb)
                loss = nn.functional.mse_loss(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def train_transformer(model, train_loader, val_loader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb)
                loss = nn.functional.mse_loss(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def evaluate_model(model, X_test, y_test, is_torch=True):
    if is_torch:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            if hasattr(model, 'compute_loss'):
                pred, _, _, _, _ = model(X_t)
            else:
                pred = model(X_t)
            pred = pred.cpu().numpy()
    else:
        pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    return pred

def get_asset_data(tickers, start='2010-01-01', end='2023-12-31'):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 500:
                data[ticker] = df
        except:
            continue
    return data

def prepare_panel_data(asset_data, horizon=5, seq_len=20):
    all_X = []
    all_y = []
    all_dates = []
    all_tickers = []
    for ticker, df in asset_data.items():
        features, targets, dates = compute_features_and_target_for_asset(df, horizon, seq_len)
        if len(targets) > seq_len:
            X_seq, y_seq = create_sequences(features, targets, seq_len, horizon)
            if len(y_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
                all_dates.append(dates[seq_len:len(dates)-horizon])
                all_tickers.append([ticker]*len(y_seq))
    if len(all_X) == 0:
        return None, None, None, None
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    dates_all = np.concatenate(all_dates)
    tickers_all = np.concatenate(all_tickers)
    return X_all, y_all, dates_all, tickers_all

def cross_sectional_split_by_time(X, y, dates, tickers, train_start, train_end, test_start, test_end):
    train_mask = (dates >= train_start) & (dates <= train_end)
    test_mask = (dates >= test_start) & (dates <= test_end)
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]

def main():
    tickers = get_tickers(200)
    asset_data = get_asset_data(tickers, start='2010-01-01', end='2023-12-31')
    print(f"Loaded {len(asset_data)} assets")
    X_all, y_all, dates_all, tickers_all = prepare_panel_data(asset_data, horizon=5, seq_len=20)
    if X_all is None:
        print("No data available.")
        return

    date_series = pd.to_datetime(dates_all)
    unique_dates = np.sort(np.unique(date_series))
    train_dates = unique_dates[unique_dates < pd.Timestamp('2018-01-01')]
    test_dates = unique_dates[(unique_dates >= pd.Timestamp('2018-01-01')) & (unique_dates < pd.Timestamp('2021-01-01'))]
    if len(train_dates) == 0 or len(test_dates) == 0:
        print("Insufficient date range.")
        return

    train_mask = np.isin(date_series, train_dates)
    test_mask = np.isin(date_series, test_dates)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print("No training or testing samples.")
        return

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    X_train_seq = X_train_scaled
    X_test_seq = X_test_scaled
    y_train_seq = y_train
    y_test_seq = y_test

    models = {
        'EDS': EDSModel(input_dim=X_train.shape[-1], latent_dim=16, seq_len=20, horizon=5),
        'LSTM': LSTMModel(input_dim=X_train.shape[-1]),
        'Transformer': TransformerModel(input_dim=X_train.shape[-1])
    }
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        if name == 'EDS':
            model = train_eds(model, train_loader, val_loader, epochs=30)
        elif name == 'LSTM':
            model = train_lstm(model, train_loader, val_loader, epochs=30)
        elif name == 'Transformer':
            model = train_transformer(model, train_loader, val_loader, epochs=30)
        trained_models[name] = model

    lr_model = LinearRegression()
    lr_model.fit(X_train_seq.reshape(X_train_seq.shape[0], -1), y_train_seq)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    xgb_model.fit(X_train_seq.reshape(X_train_seq.shape[0], -1), y_train_seq)

    all_preds = {}
    for name, model in trained_models.items():
        preds = evaluate_model(model, X_test_seq, y_test_seq, is_torch=True)
        all_preds[name] = preds
    all_preds['Linear'] = lr_model.predict(X_test_seq.reshape(X_test_seq.shape[0], -1))
    all_preds['XGBoost'] = xgb_model.predict(X_test_seq.reshape(X_test_seq.shape[0], -1))

    ic_results = {}
    for name, preds in all_preds.items():
        ic = spearmanr(preds, y_test_seq)[0]
        ic_results[name] = ic

    def compute_strategy_returns(preds, actual, top=0.1):
        n = len(preds)
        ranks = np.argsort(preds)
        long_idx = ranks[-int(n*top):]
        short_idx = ranks[:int(n*top)]
        ret = np.zeros(n)
        ret[long_idx] += actual[long_idx] / int(n*top)
        ret[short_idx] -= actual[short_idx] / int(n*top)
        return ret

    strategy_returns = {}
    for name, preds in all_preds.items():
        ret = compute_strategy_returns(preds, y_test_seq)
        strategy_returns[name] = ret

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    for name, ret in strategy_returns.items():
        cum = np.cumsum(ret)
        ax.plot(np.arange(len(cum)), cum, label=name)
    ax.set_title('Cumulative Long-Short Strategy Returns (Test)')
    ax.legend()
    ax.grid(True)

    ax = axes[0,1]
    names = list(ic_results.keys())
    ics = list(ic_results.values())
    ax.bar(names, ics)
    ax.set_title('Cross-Sectional IC (Spearman)')
    ax.grid(True)

    ax = axes[1,0]
    for name, ret in strategy_returns.items():
        cum = np.cumsum(ret)
        ax.plot(np.arange(len(cum)), cum, label=name)
    ax.set_title('Cumulative Returns (Log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    ax = axes[1,1]
    sharpe = {}
    for name, ret in strategy_returns.items():
        sharpe[name] = np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252)
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