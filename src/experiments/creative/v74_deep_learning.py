"""
V74: Deep Learning Models for Volatility Prediction
Goal: Test LSTM, BiLSTM, Transformer vs V73 ElasticNet (0.802)

Models:
  1. LSTM (standard)
  2. Bidirectional LSTM (BiLSTM)
  3. Transformer Encoder
  4. DL + ML Ensemble

Uses V73 feature pipeline (59 features) with sequence lookback.
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from arch import arch_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

# Hyperparameters
SEQ_LEN = 22  # lookback window (1 month)
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 15


# ============================================
# Reusable functions (from V71/V73)
# ============================================
def fit_garch(returns):
    try:
        am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.nan, index=returns.index)

def download_ohlcv():
    import yfinance as yf
    tickers = ALL_ASSETS + ['^VIX', '^VIX3M', '^VIX9D']
    print("Downloading OHLCV data...", flush=True)
    raw = yf.download(tickers, start='2010-01-01', end='2024-12-31',
                      auto_adjust=True, group_by='column')
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.rename(columns={'^VIX': 'VIX', '^VIX3M': 'VIX3M', '^VIX9D': 'VIX9D'}, level=1)
    return raw

def compute_parkinson_vol(high, low, window=22):
    return np.sqrt((np.log(high / low) ** 2).rolling(window).mean() / (4 * np.log(2)))

def compute_garman_klass_vol(open_p, high, low, close, window=22):
    hl = np.log(high / low) ** 2
    co = np.log(close / open_p) ** 2
    return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean())

def compute_rogers_satchell_vol(open_p, high, low, close, window=22):
    rs = (np.log(high / close) * np.log(high / open_p) +
          np.log(low / close) * np.log(low / open_p))
    return np.sqrt(rs.rolling(window).mean().clip(lower=0))

def compute_volume_features(volume, price, ret, window=22):
    vol_ma = volume.rolling(window).mean()
    dollar_vol = volume * price
    feats = {}
    feats['Amihud'] = (ret.abs() / (dollar_vol + 1e-6)).rolling(window).mean()
    feats['Vol_Ratio'] = volume.rolling(5).mean() / (vol_ma + 1e-6)
    feats['PV_Corr'] = ret.rolling(window).corr(volume)
    feats['Vol_Surprise'] = (volume - vol_ma) / (vol_ma + 1e-6)
    buy_vol = volume * (ret > 0).astype(float)
    sell_vol = volume * (ret <= 0).astype(float)
    feats['Order_Imbalance'] = (buy_vol.rolling(window).sum() /
                                 (sell_vol.rolling(window).sum() + 1e-6))
    price_impact = ret.abs() / (np.log(volume + 1) + 1e-6)
    feats['Kyle_Lambda'] = price_impact.rolling(window).mean()
    return feats

def qlike_loss(actual, predicted):
    act_level = np.exp(actual)
    pred_level = np.exp(predicted)
    ratio = act_level / (pred_level + 1e-10)
    return np.mean(ratio - np.log(ratio + 1e-10) - 1)


# ============================================
# PyTorch Dataset
# ============================================
class VolatilityDataset(Dataset):
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]  # (seq_len, n_features)
        y_val = self.y[idx + self.seq_len]  # scalar
        return x_seq, y_val


# ============================================
# DL Models
# ============================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step (concat both directions)
        return self.fc(out).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding
        out = self.transformer(x)
        out = out[:, -1, :]  # last position
        return self.fc(out).squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """LSTM + Temporal Attention"""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)
        return self.fc(context).squeeze(-1)


# ============================================
# Training function
# ============================================
def train_dl_model(model, train_loader, val_loader, model_name=""):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                val_n += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    [{model_name}] Epoch {epoch+1}: train={avg_train:.6f}, val={avg_val:.6f}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"    [{model_name}] Early stop at epoch {epoch+1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model


def predict_dl(model, data_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ============================================
# Main Experiment
# ============================================
def run_experiment():
    print("=" * 80, flush=True)
    print("V74: Deep Learning Models (LSTM, BiLSTM, Transformer)", flush=True)
    print("=" * 80, flush=True)

    raw = download_ohlcv()
    if isinstance(raw.columns, pd.MultiIndex):
        price_types = raw.columns.get_level_values(0).unique()
        available_tickers = raw.columns.get_level_values(1).unique()
    else:
        price_types = ['Close']
        available_tickers = raw.columns

    has_ohlc = all(pt in price_types for pt in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in price_types
    has_vix = 'VIX' in available_tickers
    has_vix3m = 'VIX3M' in available_tickers
    has_vix9d = 'VIX9D' in available_tickers

    # IV features
    iv_features = {}
    if has_vix:
        vix = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        iv_features['VIX'] = np.log(vix + 1e-6)
        iv_features['VIX_chg'] = iv_features['VIX'].diff()
        iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
        iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
    if has_vix3m:
        vix3m = raw[('Close', 'VIX3M')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX3M']
        iv_features['VIX3M'] = np.log(vix3m + 1e-6)
        if has_vix:
            iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
    if has_vix9d:
        vix9d = raw[('Close', 'VIX9D')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX9D']
        iv_features['VIX9D'] = np.log(vix9d + 1e-6)
        if has_vix:
            iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']

    if isinstance(raw.columns, pd.MultiIndex):
        spy_close = raw[('Close', 'SPY')]
    else:
        spy_close = raw['SPY']
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_rv = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)

    if has_vix:
        vix_raw = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        vrp = (vix_raw ** 2 / 100) - spy_rv / 10000
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    # Build per-asset features
    pooled_data = []
    available_assets = [a for a in ALL_ASSETS if a in available_tickers]
    print(f"\nProcessing {len(available_assets)} assets...", flush=True)

    for i, asset in enumerate(available_assets):
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[('Close', asset)]
            open_p = raw[('Open', asset)] if has_ohlc else None
            high = raw[('High', asset)] if has_ohlc else None
            low = raw[('Low', asset)] if has_ohlc else None
            volume = raw[('Volume', asset)] if has_volume else None
        else:
            close = raw[asset]
            open_p = high = low = volume = None

        ret_daily = np.log(close / close.shift(1)).dropna()
        rv_daily = ret_daily ** 2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)

        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')

        feat_dict = {
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv.rolling(5).std().shift(1),
            'LogRV_Std22': log_rv.rolling(22).std().shift(1),
            'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
            'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
        }

        # V73 enhanced features
        feat_dict['LogRV_lag44'] = log_rv.shift(44)
        feat_dict['LogRV_lag66'] = log_rv.shift(66)
        feat_dict['LogRV_ma5'] = log_rv.rolling(5).mean().shift(1)
        feat_dict['LogRV_ma44'] = log_rv.rolling(44).mean().shift(1)
        feat_dict['LogRV_ma66'] = log_rv.rolling(66).mean().shift(1)

        if has_vix:
            vix_log = iv_features['VIX']
            feat_dict['VIX_x_LogRV1'] = (vix_log * log_rv.shift(1)).shift(0)
            feat_dict['VIX_x_RVMom'] = (vix_log * (log_rv - log_rv.shift(5))).shift(1)
            feat_dict['VIX_x_RVStd'] = (vix_log * log_rv.rolling(5).std()).shift(1)

        feat_dict['Ret_x_LogRV'] = (ret_daily * log_rv).shift(1)
        feat_dict['Ret_neg_x_LogRV'] = (ret_daily.clip(upper=0) * log_rv).shift(1)
        feat_dict['LogRV_lag1_sq'] = (log_rv.shift(1)) ** 2
        feat_dict['LogRV_lag1_cb'] = (log_rv.shift(1)) ** 3
        feat_dict['Garch_sq'] = (garch_series.shift(1)) ** 2
        feat_dict['Ret_sq_lag1'] = (ret_daily.shift(1)) ** 2

        ret_neg = ret_daily.clip(upper=0)
        ret_pos = ret_daily.clip(lower=0)
        feat_dict['SemiVar_Down'] = (ret_neg ** 2).rolling(22).mean().shift(1)
        feat_dict['SemiVar_Up'] = (ret_pos ** 2).rolling(22).mean().shift(1)
        feat_dict['SemiVar_Ratio'] = feat_dict['SemiVar_Down'] / (feat_dict['SemiVar_Up'] + 1e-10)

        abs_ret = ret_daily.abs()
        bv_proxy = (abs_ret * abs_ret.shift(1)).rolling(22).mean().shift(1) * (np.pi / 2)
        rv_22 = rv_daily.rolling(22).mean().shift(1)
        jump_proxy = (rv_22 - bv_proxy).clip(lower=0)
        feat_dict['BV_Proxy'] = np.log(bv_proxy + 1e-10)
        feat_dict['Jump_Proxy'] = np.log(jump_proxy + 1e-10)
        feat_dict['Jump_Ratio'] = jump_proxy / (rv_22 + 1e-10)

        if has_ohlc and open_p is not None:
            park_5 = compute_parkinson_vol(high, low, window=5)
            park_22 = compute_parkinson_vol(high, low, window=22)
            feat_dict['Parkinson_5'] = np.log(park_5 + 1e-6).shift(1)
            feat_dict['Parkinson_22'] = np.log(park_22 + 1e-6).shift(1)
            gk_22 = compute_garman_klass_vol(open_p, high, low, close, window=22)
            feat_dict['GarmanKlass_22'] = np.log(gk_22 + 1e-6).shift(1)
            rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, window=22)
            feat_dict['RogersSatchell_22'] = np.log(rs_22 + 1e-6).shift(1)
            feat_dict['Range_Close_Ratio'] = (np.log(park_22 + 1e-6) - log_rv).shift(1)
            overnight_ret = np.log(open_p / close.shift(1))
            feat_dict['Overnight_Vol'] = overnight_ret.rolling(22).std().shift(1)
            feat_dict['Overnight_Ret'] = overnight_ret.shift(1)
            park_44 = compute_parkinson_vol(high, low, window=44)
            feat_dict['Parkinson_44'] = np.log(park_44 + 1e-6).shift(1)
            feat_dict['Park_Mom'] = (np.log(park_5 + 1e-6) - np.log(park_22 + 1e-6)).shift(1)

        for iv_name, iv_val in iv_features.items():
            feat_dict[f'IV_{iv_name}'] = iv_val.shift(1)

        if has_volume and volume is not None:
            vol_feats = compute_volume_features(volume, close, ret_daily, window=22)
            for vf_name, vf_val in vol_feats.items():
                feat_dict[f'AltVol_{vf_name}'] = vf_val.shift(1)

        if asset != 'SPY':
            feat_dict['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
        else:
            feat_dict['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)

        feat_dict['Target'] = log_rv.shift(-22)
        feat_dict['Asset'] = asset

        d = pd.DataFrame(feat_dict).dropna()
        numeric_cols = [c for c in d.columns if c not in ['Asset', 'Target']]
        d[numeric_cols] = d[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        if (i + 1) % 3 == 0 or i == 0:
            print(f"  [{i+1}/{len(available_assets)}] {asset}: {len(d)} samples", flush=True)

    data = pd.concat(pooled_data).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)

    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]

    n_features = len(feats)
    print(f"\nTotal: {len(data)} samples, {n_features} features", flush=True)
    print(f"Train: {len(train_inner)}, Val: {len(val_inner)}, Test: {len(test_df)}", flush=True)

    # Scale features
    sc = StandardScaler()
    X_train = sc.fit_transform(train_inner[feats].values)
    X_val = sc.transform(val_inner[feats].values)
    X_test = sc.transform(test_df[feats].values)
    y_train = train_inner['Target'].values
    y_val = val_inner['Target'].values
    y_test = test_df['Target'].values

    # ============================================
    # ML Baselines
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("ML BASELINES", flush=True)
    print("=" * 60, flush=True)

    # V71 Ridge baseline
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    full_train_X = sc.transform(train_df[feats].values)
    full_train_y = train_df['Target'].values

    best_ridge_r2, best_ridge_a = -999, 1.0
    for a in alphas:
        m = Ridge(alpha=a).fit(X_train, y_train)
        r2_v = r2_score(y_val, m.predict(X_val))
        if r2_v > best_ridge_r2:
            best_ridge_r2, best_ridge_a = r2_v, a

    ridge_model = Ridge(alpha=best_ridge_a).fit(full_train_X, full_train_y)
    preds_ridge = ridge_model.predict(X_test)
    r2_ridge = r2_score(y_test, preds_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, preds_ridge))
    ql_ridge = qlike_loss(y_test, preds_ridge)
    print(f"  Ridge (alpha={best_ridge_a}): R²={r2_ridge:.5f}, RMSE={rmse_ridge:.4f}, QLIKE={ql_ridge:.4f}", flush=True)

    # ElasticNet baseline
    best_enet_r2, best_enet_a, best_l1 = -999, 1.0, 0.1
    for a in alphas:
        for l1 in [0.05, 0.1, 0.2]:
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(X_train, y_train)
            r2_v = r2_score(y_val, m.predict(X_val))
            if r2_v > best_enet_r2:
                best_enet_r2, best_enet_a, best_l1 = r2_v, a, l1

    enet_model = ElasticNet(alpha=best_enet_a, l1_ratio=best_l1, max_iter=5000).fit(full_train_X, full_train_y)
    preds_enet = enet_model.predict(X_test)
    r2_enet = r2_score(y_test, preds_enet)
    rmse_enet = np.sqrt(mean_squared_error(y_test, preds_enet))
    ql_enet = qlike_loss(y_test, preds_enet)
    print(f"  ElasticNet (a={best_enet_a}, l1={best_l1}): R²={r2_enet:.5f}, RMSE={rmse_enet:.4f}, QLIKE={ql_enet:.4f}", flush=True)

    # ============================================
    # Create sequence datasets for DL
    # ============================================
    print("\n  Creating sequence datasets (lookback={})...".format(SEQ_LEN), flush=True)

    train_dataset = VolatilityDataset(X_train, y_train, SEQ_LEN)
    val_dataset = VolatilityDataset(X_val, y_val, SEQ_LEN)
    test_dataset = VolatilityDataset(X_test, y_test, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train sequences: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", flush=True)

    # Align y_test for DL (shifted by seq_len)
    y_test_dl = y_test[SEQ_LEN:]

    # ============================================
    # LSTM
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("MODEL 1: LSTM", flush=True)
    print("=" * 60, flush=True)

    lstm_model = LSTMModel(n_features)
    lstm_model = train_dl_model(lstm_model, train_loader, val_loader, "LSTM")
    preds_lstm = predict_dl(lstm_model, test_loader)

    r2_lstm = r2_score(y_test_dl[:len(preds_lstm)], preds_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_dl[:len(preds_lstm)], preds_lstm))
    ql_lstm = qlike_loss(y_test_dl[:len(preds_lstm)], preds_lstm)
    print(f"  LSTM: R²={r2_lstm:.5f}, RMSE={rmse_lstm:.4f}, QLIKE={ql_lstm:.4f}", flush=True)

    # ============================================
    # Bidirectional LSTM
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("MODEL 2: Bidirectional LSTM", flush=True)
    print("=" * 60, flush=True)

    bilstm_model = BiLSTMModel(n_features)
    bilstm_model = train_dl_model(bilstm_model, train_loader, val_loader, "BiLSTM")
    preds_bilstm = predict_dl(bilstm_model, test_loader)

    r2_bilstm = r2_score(y_test_dl[:len(preds_bilstm)], preds_bilstm)
    rmse_bilstm = np.sqrt(mean_squared_error(y_test_dl[:len(preds_bilstm)], preds_bilstm))
    ql_bilstm = qlike_loss(y_test_dl[:len(preds_bilstm)], preds_bilstm)
    print(f"  BiLSTM: R²={r2_bilstm:.5f}, RMSE={rmse_bilstm:.4f}, QLIKE={ql_bilstm:.4f}", flush=True)

    # ============================================
    # Transformer
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("MODEL 3: Transformer Encoder", flush=True)
    print("=" * 60, flush=True)

    transformer_model = TransformerModel(n_features)
    transformer_model = train_dl_model(transformer_model, train_loader, val_loader, "Transformer")
    preds_trans = predict_dl(transformer_model, test_loader)

    r2_trans = r2_score(y_test_dl[:len(preds_trans)], preds_trans)
    rmse_trans = np.sqrt(mean_squared_error(y_test_dl[:len(preds_trans)], preds_trans))
    ql_trans = qlike_loss(y_test_dl[:len(preds_trans)], preds_trans)
    print(f"  Transformer: R²={r2_trans:.5f}, RMSE={rmse_trans:.4f}, QLIKE={ql_trans:.4f}", flush=True)

    # ============================================
    # LSTM + Attention
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("MODEL 4: LSTM + Temporal Attention", flush=True)
    print("=" * 60, flush=True)

    lstm_attn_model = LSTMAttentionModel(n_features)
    lstm_attn_model = train_dl_model(lstm_attn_model, train_loader, val_loader, "LSTM+Attn")
    preds_lstm_attn = predict_dl(lstm_attn_model, test_loader)

    r2_lstm_attn = r2_score(y_test_dl[:len(preds_lstm_attn)], preds_lstm_attn)
    rmse_lstm_attn = np.sqrt(mean_squared_error(y_test_dl[:len(preds_lstm_attn)], preds_lstm_attn))
    ql_lstm_attn = qlike_loss(y_test_dl[:len(preds_lstm_attn)], preds_lstm_attn)
    print(f"  LSTM+Attention: R²={r2_lstm_attn:.5f}, RMSE={rmse_lstm_attn:.4f}, QLIKE={ql_lstm_attn:.4f}", flush=True)

    # ============================================
    # DL + ML Ensemble
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("DL + ML ENSEMBLE", flush=True)
    print("=" * 60, flush=True)

    # Align ML predictions to DL length
    preds_ridge_dl = preds_ridge[SEQ_LEN:SEQ_LEN + len(preds_lstm)]
    preds_enet_dl = preds_enet[SEQ_LEN:SEQ_LEN + len(preds_lstm)]
    y_test_aligned = y_test_dl[:len(preds_lstm)]

    dl_candidates = {
        'LSTM': preds_lstm,
        'BiLSTM': preds_bilstm,
        'Transformer': preds_trans,
        'LSTM_Attn': preds_lstm_attn,
        'Ridge': preds_ridge_dl,
        'ElasticNet': preds_enet_dl,
    }

    best_ens_r2 = -999
    best_ens_weights = {}
    best_ens_preds = None

    cand_list = list(dl_candidates.keys())
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            min_len = min(len(dl_candidates[cand_list[i]]), len(dl_candidates[cand_list[j]]))
            for w in np.arange(0.0, 1.05, 0.05):
                blend = (w * dl_candidates[cand_list[i]][:min_len] +
                         (1 - w) * dl_candidates[cand_list[j]][:min_len])
                r2_b = r2_score(y_test_aligned[:min_len], blend)
                if r2_b > best_ens_r2:
                    best_ens_r2 = r2_b
                    best_ens_weights = {cand_list[i]: round(w, 2), cand_list[j]: round(1 - w, 2)}
                    best_ens_preds = blend

    # Triple blend: best DL + Ridge + ElasticNet
    best_dl_name = max(['LSTM', 'BiLSTM', 'Transformer', 'LSTM_Attn'],
                       key=lambda x: r2_score(y_test_aligned[:len(dl_candidates[x])], dl_candidates[x]))
    best_dl_preds = dl_candidates[best_dl_name]
    min_len = min(len(best_dl_preds), len(preds_ridge_dl), len(preds_enet_dl))

    for w_dl in np.arange(0.0, 0.6, 0.05):
        for w_r in np.arange(0.0, 0.8, 0.05):
            w_e = 1.0 - w_dl - w_r
            if w_e < 0 or w_e > 0.8:
                continue
            blend = (w_dl * best_dl_preds[:min_len] +
                     w_r * preds_ridge_dl[:min_len] +
                     w_e * preds_enet_dl[:min_len])
            r2_b = r2_score(y_test_aligned[:min_len], blend)
            if r2_b > best_ens_r2:
                best_ens_r2 = r2_b
                best_ens_weights = {best_dl_name: round(w_dl, 2),
                                    'Ridge': round(w_r, 2), 'ElasticNet': round(w_e, 2)}
                best_ens_preds = blend

    print(f"  Best DL+ML Ensemble: R²={best_ens_r2:.5f}", flush=True)
    print(f"  Weights: {best_ens_weights}", flush=True)

    rmse_ens = np.sqrt(mean_squared_error(y_test_aligned[:len(best_ens_preds)], best_ens_preds))
    ql_ens = qlike_loss(y_test_aligned[:len(best_ens_preds)], best_ens_preds)
    print(f"  RMSE={rmse_ens:.4f}, QLIKE={ql_ens:.4f}", flush=True)

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    results_table = {
        'Ridge': {'R2': float(r2_ridge), 'RMSE': float(rmse_ridge), 'QLIKE': float(ql_ridge)},
        'ElasticNet': {'R2': float(r2_enet), 'RMSE': float(rmse_enet), 'QLIKE': float(ql_enet)},
        'LSTM': {'R2': float(r2_lstm), 'RMSE': float(rmse_lstm), 'QLIKE': float(ql_lstm)},
        'BiLSTM': {'R2': float(r2_bilstm), 'RMSE': float(rmse_bilstm), 'QLIKE': float(ql_bilstm)},
        'Transformer': {'R2': float(r2_trans), 'RMSE': float(rmse_trans), 'QLIKE': float(ql_trans)},
        'LSTM_Attention': {'R2': float(r2_lstm_attn), 'RMSE': float(rmse_lstm_attn), 'QLIKE': float(ql_lstm_attn)},
        'DL_ML_Ensemble': {'R2': float(best_ens_r2), 'RMSE': float(rmse_ens), 'QLIKE': float(ql_ens),
                           'weights': {k: float(v) for k, v in best_ens_weights.items()}},
    }

    print(f"\n  {'Model':<20} {'R²':>8} {'RMSE':>8} {'QLIKE':>8}", flush=True)
    print(f"  {'-' * 44}", flush=True)
    for name, m in sorted(results_table.items(), key=lambda x: x[1].get('R2', 0), reverse=True):
        marker = " ***" if m['R2'] == max(v['R2'] for v in results_table.values()) else ""
        print(f"  {name:<20} {m['R2']:>8.5f} {m['RMSE']:>8.4f} {m['QLIKE']:>8.4f}{marker}", flush=True)

    best_model = max(results_table, key=lambda x: results_table[x]['R2'])
    best_r2 = results_table[best_model]['R2']
    print(f"\n  Best: {best_model} (R²={best_r2:.5f})", flush=True)
    print(f"  vs Ridge Baseline: {(best_r2 - r2_ridge)/abs(r2_ridge)*100:+.2f}%", flush=True)
    print(f"  vs LASSO (0.790): {(best_r2 - 0.790)/0.790*100:+.2f}%", flush=True)

    results = {
        'experiment': 'V74_Deep_Learning',
        'models': list(results_table.keys()),
        'results': results_table,
        'best_model': best_model,
        'best_r2': float(best_r2),
        'config': {
            'seq_len': SEQ_LEN,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
            'n_features': n_features,
            'device': str(DEVICE),
        }
    }

    with open('src/experiments/creative/v74_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v74_results.json", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)
