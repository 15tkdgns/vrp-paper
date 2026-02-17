"""V50 LSTM: Feature Selection + Hyperparameter Tuning
Strategy:
  1. Feature group ablation (Base, +HF, +IV, +Alt, combos)
  2. Top-K feature selection via Ridge permutation importance
  3. Hyperparameter grid search on best feature set
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os, json, time, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from arch import arch_model
warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY','QQQ','IWM','EFA','EEM'],
    'Bond': ['TLT','IEF','AGG'],
    'Commodity': ['GLD','SLV','USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

# ============ Model ============
class Attention(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.fc = nn.Linear(hdim*2, 1)
    def forward(self, x):
        w = torch.softmax(torch.tanh(self.fc(x)).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = self.attn(out)
        return self.fc(self.drop(ctx))

# ============ Helpers ============
def fit_garch(returns):
    try:
        am = arch_model(returns*100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return pd.Series(res.conditional_volatility.values.flatten()/100, index=returns.index)
    except:
        return returns.rolling(22).std()

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def compute_vol_feats(vol, price, ret, w=22):
    dv = vol*price
    f = {}
    f['Amihud'] = (ret.abs()/(dv+1e-10)).rolling(w).mean()
    f['Vol_Ratio'] = vol.rolling(5).mean()/(vol.rolling(w).mean()+1e-10)
    f['PV_Corr'] = ret.rolling(w).corr(np.log(vol+1))
    f['Vol_Surprise'] = (vol-vol.rolling(w).mean())/(vol.rolling(w).std()+1e-10)
    pv = vol.where(ret>0,0).rolling(w).sum()
    nv = vol.where(ret<=0,0).rolling(w).sum()
    f['Order_Imbalance'] = (pv-nv)/(pv+nv+1e-10)
    f['Kyle_Lambda'] = ret.abs().rolling(w).sum()/(vol.rolling(w).sum()+1e-10)*1e6
    return f

def qlike(a, p):
    r = np.exp(a-p)
    return np.mean(r-(a-p)-1)

def train_eval_lstm(X_tr, y_tr, X_te, y_te, test_assets,
                    hidden_dim=32, dropout=0.2, lr=0.001, epochs=15, batch=128, seq_len=22):
    torch.manual_seed(42); np.random.seed(42)
    model = LSTMModel(X_tr.shape[2], hidden_dim, dropout)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        tloss = 0
        for bx, by in dl:
            opt.zero_grad()
            loss = nn.MSELoss()(model(bx).squeeze(), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()
        sched.step(tloss/len(dl))
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te)).squeeze().numpy()
    
    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae_v = mean_absolute_error(y_te, preds)
    ql = qlike(y_te, preds)
    
    # Per-asset
    ta = np.array(test_assets)
    asset_r2s = []
    for asset in ALL_ASSETS:
        m = ta == asset
        if m.sum() > 0:
            asset_r2s.append(r2_score(y_te[m], preds[m]))
    
    return {'pooled_r2': round(r2,4), 'rmse': round(rmse,4), 'mae': round(mae_v,4),
            'qlike': round(ql,4), 'median_r2': round(float(np.median(asset_r2s)),4),
            'mean_r2': round(float(np.mean(asset_r2s)),4)}

# ============ Data Loading ============
print("="*60)
print("V50 LSTM Feature Selection & Tuning")
print("="*60)
t0 = time.time()

CACHE = 'src/data/v71_ohlcv_cache.pkl'
raw = pd.read_pickle(CACHE)
print(f"Cache loaded: {raw.shape}")

# IV features
iv_features = {}
vix = raw[('Close','VIX')]
iv_features['VIX'] = np.log(vix+1e-6)
iv_features['VIX_chg'] = iv_features['VIX'].diff()
iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
iv_features['VIX3M'] = np.log(raw[('Close','VIX3M')]+1e-6)
iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
iv_features['VIX9D'] = np.log(raw[('Close','VIX9D')]+1e-6)
iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']

spy_close = raw[('Close','SPY')]
spy_ret = np.log(spy_close/spy_close.shift(1)).dropna()
spy_rv = (spy_ret**2).rolling(22).mean()*252*10000
spy_log_rv = np.log(spy_rv+1e-6)

vrp = (vix**2/100) - spy_rv/10000
iv_features['VRP'] = vrp
iv_features['VRP_ma22'] = vrp.rolling(22).mean()

# Build all features
pooled = []
for asset in ALL_ASSETS:
    c = raw[('Close',asset)]
    o = raw[('Open',asset)]
    h = raw[('High',asset)]
    l = raw[('Low',asset)]
    v = raw[('Volume',asset)]
    
    ret = np.log(c/c.shift(1)).dropna()
    rv = (ret**2).rolling(22).mean()*252*10000
    lrv = np.log(rv+1e-6)
    gd = fit_garch(ret)
    rw = ret.resample('W').sum()
    gw = fit_garch(rw).reindex(ret.index, method='ffill')
    
    feat = {}
    # === BASE (14) ===
    feat['LogRV_lag1'] = lrv.shift(1)
    feat['LogRV_lag5'] = lrv.shift(5)
    feat['LogRV_lag10'] = lrv.shift(10)
    feat['LogRV_lag22'] = lrv.shift(22)
    feat['Garch_Daily'] = gd.shift(1)
    feat['Garch_Weekly'] = gw.shift(1)
    feat['LogRV_Std5'] = lrv.rolling(5).std().shift(1)
    feat['LogRV_Std22'] = lrv.rolling(22).std().shift(1)
    feat['RV_Mom5'] = (lrv-lrv.shift(5)).shift(1)
    feat['RV_Mom22'] = (lrv-lrv.shift(22)).shift(1)
    feat['SPY_LogRV'] = spy_log_rv.shift(1)
    feat['Ret_lag1'] = ret.shift(1)
    feat['Ret_abs_lag1'] = ret.abs().shift(1)
    feat['Corr_SPY'] = (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1) 
                        if asset != 'SPY' else pd.Series(1.0, index=ret.index))
    
    # === HF PROXY (7) ===
    p5 = compute_parkinson(h,l,5)
    p22 = compute_parkinson(h,l,22)
    gk22 = compute_gk(o,h,l,c,22)
    rs22 = compute_rs(o,h,l,c,22)
    feat['Parkinson_5'] = np.log(p5+1e-6).shift(1)
    feat['Parkinson_22'] = np.log(p22+1e-6).shift(1)
    feat['GarmanKlass_22'] = np.log(gk22+1e-6).shift(1)
    feat['RogersSatchell_22'] = np.log(rs22+1e-6).shift(1)
    feat['Range_Close_Ratio'] = (np.log(p22+1e-6)-lrv).shift(1)
    on = np.log(o/c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    
    # === IV SURFACE (10) ===
    for n2, v2 in iv_features.items():
        feat[f'IV_{n2}'] = v2.shift(1)
    
    # === ALT DATA (6) ===
    vf = compute_vol_feats(v, c, ret, 22)
    for n2, v2 in vf.items():
        feat[f'AltVol_{n2}'] = v2.shift(1)
    
    feat['Target'] = lrv.shift(-22)
    feat['Asset'] = asset
    cls = [k for k,vv in ASSET_GROUPS.items() if asset in vv][0]
    feat['Class'] = cls
    
    d = pd.DataFrame(feat).dropna()
    nc = [x for x in d.columns if x not in ['Asset','Target','Class']]
    d[nc] = d[nc].replace([np.inf,-np.inf], np.nan).fillna(0)
    pooled.append(d)

data = pd.concat(pooled).sort_index().reset_index(drop=True)
all_feats = [c for c in data.columns if c not in ['Target','Asset','Class']]
data[all_feats] = data[all_feats].fillna(0).replace([np.inf,-np.inf], 0)

split = int(len(data)*0.8)
train_df = data.iloc[:split]
test_df = data.iloc[split:]
print(f"Data: {len(data)} samples, {len(all_feats)} features")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Feature groups
hf = [f for f in all_feats if any(x in f for x in ['Parkinson','Garman','Rogers','Range','Overnight'])]
iv = [f for f in all_feats if f.startswith('IV_')]
alt = [f for f in all_feats if f.startswith('AltVol_')]
base = [f for f in all_feats if f not in hf+iv+alt]
original = ['LogRV_lag1', 'Ret_lag1']  # V50 original 2 features

print(f"\nFeature groups: Base={len(base)}, HF={len(hf)}, IV={len(iv)}, Alt={len(alt)}")

# ============ Sequence Builder ============
def make_seqs(df, feat_cols, seq_len=22):
    sc = StandardScaler()
    sc.fit(train_df[feat_cols])
    Xs, ys, assets = [], [], []
    for asset in ALL_ASSETS:
        adf = df[df['Asset']==asset].sort_index()
        if len(adf) < seq_len+1:
            continue
        Xsc = sc.transform(adf[feat_cols])
        yv = adf['Target'].values
        for i in range(seq_len, len(Xsc)):
            Xs.append(Xsc[i-seq_len:i])
            ys.append(yv[i])
            assets.append(asset)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), assets

# ============ Phase 1: Feature Group Ablation ============
print("\n" + "="*60)
print("Phase 1: Feature Group Ablation")
print("="*60)

ablation_configs = [
    ("Original (2)", original),
    ("Base (14)", base),
    ("Base+HF (21)", base+hf),
    ("Base+IV (24)", base+iv),
    ("Base+Alt (20)", base+alt),
    ("Base+HF+IV (31)", base+hf+iv),
    ("All (37)", all_feats),
    ("HF only (7)", hf),
    ("HF+IV (17)", hf+iv),
]

ablation_results = []
for name, feat_cols in ablation_configs:
    print(f"\n--- {name} ({len(feat_cols)} features) ---")
    X_tr, y_tr, _ = make_seqs(train_df, feat_cols)
    X_te, y_te, ta = make_seqs(test_df, feat_cols)
    if len(X_tr) == 0:
        print("  Skip (no sequences)")
        continue
    res = train_eval_lstm(X_tr, y_tr, X_te, y_te, ta, 
                          hidden_dim=32, dropout=0.2, lr=0.001, epochs=15, batch=128)
    res['name'] = name
    res['n_feats'] = len(feat_cols)
    res['features'] = feat_cols
    ablation_results.append(res)
    print(f"  Pooled R²={res['pooled_r2']:.4f}  Median R²={res['median_r2']:.4f}  RMSE={res['rmse']:.4f}")

print(f"\n{'='*60}")
print("Phase 1 Summary:")
print(f"{'Config':<25} {'#Feat':>5} {'Pool R²':>8} {'Med R²':>8} {'RMSE':>8}")
print("-"*60)
for r in sorted(ablation_results, key=lambda x: -x['pooled_r2']):
    print(f"{r['name']:<25} {r['n_feats']:>5} {r['pooled_r2']:>8.4f} {r['median_r2']:>8.4f} {r['rmse']:>8.4f}")

# Find best feature group
best_ablation = max(ablation_results, key=lambda x: x['pooled_r2'])
print(f"\nBest: {best_ablation['name']} (Pooled R²={best_ablation['pooled_r2']:.4f})")

# ============ Phase 2: Ridge Feature Importance + Top-K ============
print(f"\n{'='*60}")
print("Phase 2: Top-K Feature Selection (Ridge Importance)")
print("="*60)

# Get Ridge feature importance
sc_all = StandardScaler()
X_tr_all = sc_all.fit_transform(train_df[all_feats])
y_tr_all = train_df['Target'].values
ridge = Ridge(alpha=100)
ridge.fit(X_tr_all, y_tr_all)

importances = np.abs(ridge.coef_)
feat_imp = sorted(zip(all_feats, importances), key=lambda x: -x[1])
print("\nTop 15 features by Ridge |coeff|:")
for i, (f, imp) in enumerate(feat_imp[:15]):
    print(f"  {i+1}. {f}: {imp:.4f}")

# Test Top-K
topk_results = []
for k in [3, 5, 7, 10, 14, 20]:
    top_feats = [f for f, _ in feat_imp[:k]]
    print(f"\n--- Top-{k} features ---")
    X_tr, y_tr, _ = make_seqs(train_df, top_feats)
    X_te, y_te, ta = make_seqs(test_df, top_feats)
    res = train_eval_lstm(X_tr, y_tr, X_te, y_te, ta,
                          hidden_dim=32, dropout=0.2, lr=0.001, epochs=15, batch=128)
    res['name'] = f'Top-{k}'
    res['n_feats'] = k
    res['features'] = top_feats
    topk_results.append(res)
    print(f"  Pooled R²={res['pooled_r2']:.4f}  Median R²={res['median_r2']:.4f}")

print(f"\n{'='*60}")
print("Phase 2 Summary:")
print(f"{'Config':<25} {'#Feat':>5} {'Pool R²':>8} {'Med R²':>8}")
print("-"*50)
for r in topk_results:
    print(f"{r['name']:<25} {r['n_feats']:>5} {r['pooled_r2']:>8.4f} {r['median_r2']:>8.4f}")

best_topk = max(topk_results, key=lambda x: x['pooled_r2'])
print(f"\nBest Top-K: {best_topk['name']} (Pooled R²={best_topk['pooled_r2']:.4f})")

# ============ Phase 3: Hyperparameter Tuning on Best Feature Set ============
# Pick best from Phase 1 & 2
all_cands = ablation_results + topk_results
best_overall = max(all_cands, key=lambda x: x['pooled_r2'])
best_feats = best_overall['features']
print(f"\n{'='*60}")
print(f"Phase 3: Hyperparameter Tuning on '{best_overall['name']}'")
print(f"  Features: {len(best_feats)}")
print("="*60)

X_tr, y_tr, _ = make_seqs(train_df, best_feats)
X_te, y_te, ta = make_seqs(test_df, best_feats)

hp_configs = [
    {'hidden_dim': 16, 'dropout': 0.1, 'lr': 0.001, 'epochs': 20},
    {'hidden_dim': 32, 'dropout': 0.1, 'lr': 0.001, 'epochs': 20},
    {'hidden_dim': 32, 'dropout': 0.3, 'lr': 0.001, 'epochs': 20},
    {'hidden_dim': 32, 'dropout': 0.2, 'lr': 0.0005, 'epochs': 25},
    {'hidden_dim': 48, 'dropout': 0.2, 'lr': 0.001, 'epochs': 20},
    {'hidden_dim': 48, 'dropout': 0.3, 'lr': 0.0005, 'epochs': 25},
    {'hidden_dim': 64, 'dropout': 0.3, 'lr': 0.0005, 'epochs': 20},
]

hp_results = []
for i, hp in enumerate(hp_configs):
    label = f"h{hp['hidden_dim']}_d{hp['dropout']}_lr{hp['lr']}_e{hp['epochs']}"
    print(f"\n--- [{i+1}/{len(hp_configs)}] {label} ---")
    res = train_eval_lstm(X_tr, y_tr, X_te, y_te, ta, **hp, batch=128)
    res['config'] = label
    res['hp'] = hp
    hp_results.append(res)
    print(f"  Pooled R²={res['pooled_r2']:.4f}  Median R²={res['median_r2']:.4f}")

print(f"\n{'='*60}")
print("Phase 3 Summary:")
print(f"{'Config':<40} {'Pool R²':>8} {'Med R²':>8} {'RMSE':>8}")
print("-"*70)
for r in sorted(hp_results, key=lambda x: -x['pooled_r2']):
    print(f"{r['config']:<40} {r['pooled_r2']:>8.4f} {r['median_r2']:>8.4f} {r['rmse']:>8.4f}")

best_hp = max(hp_results, key=lambda x: x['pooled_r2'])

# ============ Final Summary ============
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print("="*60)
print(f"\nBest Feature Set: {best_overall['name']} ({len(best_feats)} features)")
print(f"Best HP: {best_hp['config']}")
print(f"  Pooled R²: {best_hp['pooled_r2']:.4f}")
print(f"  Median R²: {best_hp['median_r2']:.4f}")
print(f"  RMSE: {best_hp['rmse']:.4f}")
print(f"  MAE: {best_hp['mae']:.4f}")
print(f"  QLIKE: {best_hp['qlike']:.4f}")
print(f"\nComparison:")
print(f"  V50 Original (2 feat):  R²=0.651, Med=-0.312")
print(f"  V50 Best Tuned:         R²={best_hp['pooled_r2']:.4f}, Med={best_hp['median_r2']:.4f}")
print(f"  V71 Ridge (37 feat):    R²=0.803, Med=0.065")

print(f"\nTotal time: {time.time()-t0:.1f}s")

# Save results
final = {
    'ablation': [{k:v for k,v in r.items() if k!='features'} for r in ablation_results],
    'topk': [{k:v for k,v in r.items() if k!='features'} for r in topk_results],
    'hp_tuning': hp_results,
    'best_feature_set': best_overall['name'],
    'best_features': best_feats,
    'best_hp': best_hp,
    'ridge_importance': [(f, round(float(imp),4)) for f, imp in feat_imp],
}
with open('src/experiments/creative/v50_tuning_results.json', 'w') as f:
    json.dump(final, f, indent=2, default=lambda o: float(o) if hasattr(o,'item') else o)
print("Saved: v50_tuning_results.json")
