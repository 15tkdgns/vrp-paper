"""V50 LSTM + V71 37 features experiment
Reuses V71's OHLCV cache (v71_ohlcv_cache.pkl) for fast loading.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from arch import arch_model
import json, time, warnings
warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.001

# ============ V50 Architecture ============
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
    def forward(self, lstm_output):
        w = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return torch.sum(lstm_output * w.unsqueeze(-1), dim=1), w

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=1)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, _ = self.attention(out)
        return self.fc(self.dropout(ctx))

# ============ Helpers ============
def fit_garch(returns):
    try:
        am = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        v = res.conditional_volatility / 100
        return pd.Series(v.values.flatten(), index=returns.index)
    except:
        return returns.rolling(22).std()

def compute_parkinson_vol(high, low, window=22):
    hl = np.log(high / low)
    return np.sqrt((hl**2).rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_garman_klass_vol(open_p, high, low, close, window=22):
    hl = np.log(high / low)
    co = np.log(close / open_p)
    gk = 0.5 * hl**2 - (2*np.log(2) - 1) * co**2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0) * 252)

def compute_rogers_satchell_vol(open_p, high, low, close, window=22):
    rs = np.log(high/close)*np.log(high/open_p) + np.log(low/close)*np.log(low/open_p)
    return np.sqrt(rs.rolling(window).mean().clip(lower=0) * 252)

def compute_volume_features(volume, price, ret, window=22):
    dollar_vol = volume * price
    f = {}
    f['Amihud'] = (ret.abs() / (dollar_vol + 1e-10)).rolling(window).mean()
    f['Vol_Ratio'] = volume.rolling(5).mean() / (volume.rolling(window).mean() + 1e-10)
    f['PV_Corr'] = ret.rolling(window).corr(np.log(volume + 1))
    f['Vol_Surprise'] = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-10)
    pos_vol = volume.where(ret > 0, 0).rolling(window).sum()
    neg_vol = volume.where(ret <= 0, 0).rolling(window).sum()
    f['Order_Imbalance'] = (pos_vol - neg_vol) / (pos_vol + neg_vol + 1e-10)
    f['Kyle_Lambda'] = ret.abs().rolling(window).sum() / (volume.rolling(window).sum() + 1e-10) * 1e6
    return f

def qlike(actual, predicted):
    ratio = np.exp(actual - predicted)
    return np.mean(ratio - (actual - predicted) - 1)

# ============ Main ============
print("=" * 60)
print("V50 LSTM + V71 37 Features Experiment")
print("=" * 60)
t0 = time.time()

# Load V71 OHLCV cache
CACHE = 'src/data/v71_ohlcv_cache.pkl'
if os.path.exists(CACHE):
    print(f"Loading cache: {CACHE}")
    raw = pd.read_pickle(CACHE)
else:
    print("Cache not found, downloading...")
    import yfinance as yf
    tickers = ALL_ASSETS + ['^VIX', '^VIX3M', '^VIX9D']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    if isinstance(raw.columns, pd.MultiIndex):
        new_cols = [(pt, t.replace('^','')) for pt, t in raw.columns]
        raw.columns = pd.MultiIndex.from_tuples(new_cols)
    raw = raw.ffill()
    raw.to_pickle(CACHE)

print(f"Data shape: {raw.shape}")

# Check available data
price_types = raw.columns.get_level_values(0).unique()
tickers = raw.columns.get_level_values(1).unique()
has_ohlc = all(pt in price_types for pt in ['Open','High','Low','Close'])
has_volume = 'Volume' in price_types
has_vix = 'VIX' in tickers
has_vix3m = 'VIX3M' in tickers
has_vix9d = 'VIX9D' in tickers
print(f"OHLC:{has_ohlc} Vol:{has_volume} VIX:{has_vix} VIX3M:{has_vix3m} VIX9D:{has_vix9d}")

# IV Surface features (global)
iv_features = {}
if has_vix:
    vix = raw[('Close','VIX')]
    iv_features['VIX'] = np.log(vix + 1e-6)
    iv_features['VIX_chg'] = iv_features['VIX'].diff()
    iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
    iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
if has_vix3m:
    iv_features['VIX3M'] = np.log(raw[('Close','VIX3M')] + 1e-6)
    if has_vix:
        iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
if has_vix9d:
    iv_features['VIX9D'] = np.log(raw[('Close','VIX9D')] + 1e-6)
    if has_vix:
        iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']

# SPY reference
spy_close = raw[('Close','SPY')]
spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
spy_log_rv = np.log(spy_rv + 1e-6)

# VRP
if has_vix:
    vix_raw = raw[('Close','VIX')]
    vrp = (vix_raw**2 / 100) - spy_rv / 10000
    iv_features['VRP'] = vrp
    iv_features['VRP_ma22'] = vrp.rolling(22).mean()

# Build features per asset
pooled_data = []
for i, asset in enumerate(ALL_ASSETS):
    close = raw[('Close', asset)]
    open_p = raw[('Open', asset)] if has_ohlc else None
    high = raw[('High', asset)] if has_ohlc else None
    low = raw[('Low', asset)] if has_ohlc else None
    volume = raw[('Volume', asset)] if has_volume else None

    ret_daily = np.log(close / close.shift(1)).dropna()
    rv = (ret_daily**2).rolling(22).mean() * 252 * 10000
    log_rv = np.log(rv + 1e-6)

    garch_d = fit_garch(ret_daily)
    ret_w = ret_daily.resample('W').sum()
    garch_w = fit_garch(ret_w).reindex(ret_daily.index, method='ffill')

    feat = {
        'LogRV_lag1': log_rv.shift(1),
        'LogRV_lag5': log_rv.shift(5),
        'LogRV_lag10': log_rv.shift(10),
        'LogRV_lag22': log_rv.shift(22),
        'Garch_Daily': garch_d.shift(1),
        'Garch_Weekly': garch_w.shift(1),
        'LogRV_Std5': log_rv.rolling(5).std().shift(1),
        'LogRV_Std22': log_rv.rolling(22).std().shift(1),
        'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
        'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
        'SPY_LogRV': spy_log_rv.shift(1),
        'Ret_lag1': ret_daily.shift(1),
        'Ret_abs_lag1': ret_daily.abs().shift(1),
    }

    if has_ohlc and open_p is not None:
        park_5 = compute_parkinson_vol(high, low, 5)
        park_22 = compute_parkinson_vol(high, low, 22)
        gk_22 = compute_garman_klass_vol(open_p, high, low, close, 22)
        rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, 22)
        feat['Parkinson_5'] = np.log(park_5 + 1e-6).shift(1)
        feat['Parkinson_22'] = np.log(park_22 + 1e-6).shift(1)
        feat['GarmanKlass_22'] = np.log(gk_22 + 1e-6).shift(1)
        feat['RogersSatchell_22'] = np.log(rs_22 + 1e-6).shift(1)
        feat['Range_Close_Ratio'] = (np.log(park_22 + 1e-6) - log_rv).shift(1)
        overnight_ret = np.log(open_p / close.shift(1))
        feat['Overnight_Vol'] = overnight_ret.rolling(22).std().shift(1)
        feat['Overnight_Ret'] = overnight_ret.shift(1)

    for iv_name, iv_val in iv_features.items():
        feat[f'IV_{iv_name}'] = iv_val.shift(1)

    if has_volume and volume is not None:
        vf = compute_volume_features(volume, close, ret_daily, 22)
        for n, v in vf.items():
            feat[f'AltVol_{n}'] = v.shift(1)

    if asset != 'SPY':
        feat['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
    else:
        feat['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)

    feat['Target'] = log_rv.shift(-22)
    feat['Asset'] = asset
    cls = [k for k,v in ASSET_GROUPS.items() if asset in v][0]
    feat['Class'] = cls

    d = pd.DataFrame(feat).dropna()
    ncols = [c for c in d.columns if c not in ['Asset','Target','Class']]
    d[ncols] = d[ncols].replace([np.inf, -np.inf], np.nan).fillna(0)
    pooled_data.append(d)
    print(f"  [{i+1}/{len(ALL_ASSETS)}] {asset}: {len(d)} samples, {len(ncols)} features")

data = pd.concat(pooled_data).sort_index().reset_index(drop=True)
feats = [c for c in data.columns if c not in ['Target','Asset','Class']]
data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)

split_idx = int(len(data) * 0.8)
train_df = data.iloc[:split_idx]
test_df = data.iloc[split_idx:]

print(f"\nTotal: {len(data)} samples, {len(feats)} features")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Create sequences per asset
scaler = StandardScaler()
scaler.fit(train_df[feats])

def make_sequences(df, seq_len=22):
    X_list, y_list, assets_list = [], [], []
    for asset in ALL_ASSETS:
        adf = df[df['Asset'] == asset].sort_index()
        if len(adf) < seq_len + 1:
            continue
        X_sc = scaler.transform(adf[feats])
        y_vals = adf['Target'].values
        for i in range(seq_len, len(X_sc)):
            X_list.append(X_sc[i-seq_len:i])
            y_list.append(y_vals[i])
            assets_list.append(asset)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), assets_list

print("\nCreating sequences...")
X_train, y_train, _ = make_sequences(train_df)
X_test, y_test, test_assets = make_sequences(test_df)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

input_dim = X_train.shape[2]
print(f"Input dim: {input_dim}")
params = sum(p.numel() for p in DualAttentionLSTM(input_dim, HIDDEN_DIM).parameters())
print(f"Model parameters: {params:,}")

# Train
torch.manual_seed(42); np.random.seed(42)
model = DualAttentionLSTM(input_dim, HIDDEN_DIM, dropout=0.2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nTraining ({EPOCHS} epochs)...")
model.train()
for ep in range(EPOCHS):
    total_loss = 0
    for bx, by in dl:
        optimizer.zero_grad()
        out = model(bx).squeeze()
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(dl)
    scheduler.step(avg)
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1}/{EPOCHS}, Loss: {avg:.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X_test)).squeeze().numpy()

r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
ql = qlike(y_test, preds)

print(f"\n{'='*50}")
print(f"Pooled OOS (37 features, {input_dim} input dim)")
print(f"R²:    {r2:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"QLIKE: {ql:.4f}")

# Per-asset
results = {
    'pooled': {'R2': round(r2,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4), 'QLIKE': round(ql,4)},
    'per_asset': {},
    'model': 'V50_LSTM_37features',
    'input_dim': int(input_dim),
    'seq_len': SEQ_LEN,
    'params': int(params)
}

test_a = np.array(test_assets)
print(f"\n{'Asset':<6} {'Class':<10} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8}")
print("-" * 56)
asset_r2s = []
for asset in ALL_ASSETS:
    m = test_a == asset
    if m.sum() > 0:
        a, p = y_test[m], preds[m]
        ar2 = r2_score(a, p)
        armse = np.sqrt(mean_squared_error(a, p))
        amae = mean_absolute_error(a, p)
        aql = qlike(a, p)
        cls = [k for k,v in ASSET_GROUPS.items() if asset in v][0]
        print(f"{asset:<6} {cls:<10} {ar2:>8.4f} {armse:>8.4f} {amae:>8.4f} {aql:>8.4f}")
        results['per_asset'][asset] = {
            'class': cls, 'R2': round(ar2,4), 'RMSE': round(armse,4),
            'MAE': round(amae,4), 'QLIKE': round(aql,4), 'n': int(m.sum())
        }
        asset_r2s.append(ar2)

print(f"\nClass Mean R²:")
for cls in ['Equity','Bond','Commodity']:
    vals = [v['R2'] for v in results['per_asset'].values() if v['class']==cls]
    print(f"  {cls}: {np.mean(vals):.4f}")
    results[f'class_{cls.lower()}'] = round(float(np.mean(vals)),4)

results['median_r2'] = round(float(np.median(asset_r2s)),4)
results['mean_r2'] = round(float(np.mean(asset_r2s)),4)
print(f"\nMedian R²: {results['median_r2']}")
print(f"Mean R²:   {results['mean_r2']}")

print(f"\n=== V50(2feat) vs V50(37feat) ===")
print(f"Pooled R²: 0.651 -> {r2:.4f} ({r2-0.651:+.4f})")
print(f"Median R²: -0.312 -> {results['median_r2']} ({results['median_r2']+0.312:+.4f})")

print(f"\nTime: {time.time()-t0:.1f}s")
with open('src/experiments/creative/v50_37feat_results.json','w') as f:
    json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o,'item') else o)
print("Saved: v50_37feat_results.json")
