"""
Multi-seed experiment for BiLSTM-A and XGBoost (22d horizon only).
Goal: investigate whether different seeds can reproduce paper's
BiLSTM-A=0.5361 and XGBoost=0.7773 (vs current JSON: 0.5217, 0.7758).

Outputs: results/multiseed_bilstm_results.json
"""
import numpy as np
import pandas as pd
import json
import os
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

HORIZONS  = [22]
SEQ_LEN   = 22
TEST_SEEDS = [0, 1, 2, 3, 42]

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8

# Hyperparams from main_benchmark_v6_results.json (22d)
BEST_PARAMS = {
    'Ridge':   {'Equity': {'alpha': 1000}, 'Bond': {'alpha': 10}, 'Commodity': {'alpha': 500}},
    'XGBoost': {'Equity': {'max_depth': 3, 'learning_rate': 0.05},
                'Bond':   {'max_depth': 3, 'learning_rate': 0.03},
                'Commodity': {'max_depth': 3, 'learning_rate': 0.03}},
    'BiLSTM-A': {'Equity':    {'hidden': 64, 'dropout': 0.3},
                 'Bond':      {'hidden': 32, 'dropout': 0.1},
                 'Commodity': {'hidden': 32, 'dropout': 0.1}},
    'pw': 0.9,
}

# ── Device ───────────────────────────────────────────────────────────────────
def _check_cuda():
    if not torch.cuda.is_available(): return 'cpu'
    try:
        torch.tensor([1.0]).cuda() + torch.tensor([1.0]).cuda()
        return 'cuda'
    except Exception:
        return 'cpu'

DEVICE = _check_cuda()
print(f"Device: {DEVICE}")

# ── Data loading ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = os.path.join(_SCRIPT_DIR, 'data')

def _load_from_parquet():
    vix_df  = pd.read_parquet(os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames  = {}
    for asset in ALL_ASSETS + ['VIX']:
        p = os.path.join(_DATA_DIR, f'{asset}.parquet')
        if not os.path.exists(p): continue
        frames[asset] = pd.read_parquet(p)
    combined = pd.concat(frames.values(), axis=1)
    combined[('Close', 'VIX')]   = vix_df['Close']
    combined[('Close', 'VIX3M')] = vix_df['Close_3M']
    combined[('Close', 'VIX9D')] = vix_df['Close_9D']
    return combined

print("Loading data...", flush=True)
raw = pd.read_pickle(_PKL_PATH) if os.path.exists(_PKL_PATH) else _load_from_parquet()

# ── Feature engineering (same as run_main_benchmark_v6.py) ───────────────────
vix     = raw[('Close', 'VIX')]
spy_c   = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_ret**2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

iv_features = {
    'VIX':           np.log(vix + 1e-6),
    'VIX_chg':       np.log(vix + 1e-6).diff(),
    'VIX_ma5':       np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5':      np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M':         np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D':         np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope':np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp_val = (vix**2 / 100) - spy_rv / 10000
iv_features['VRP']      = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean() / (4*np.log(2))) * np.sqrt(252)
def compute_gk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)
def forward_rv(ret_sq, hz):
    cs = ret_sq.cumsum()
    return np.log((cs.shift(-hz)-cs)/hz*252+1e-12)

asset_frames = {}
for asset in ALL_ASSETS:
    c  = raw[('Close', asset)]; o = raw[('Open', asset)]
    h  = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
    ret = np.log(c / c.shift(1)).dropna(); ret_sq = ret**2
    rv  = ret_sq.rolling(22).mean()*252*10000; lrv = np.log(rv+1e-6)
    feat = {
        'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
        'LogRV_lag10':lrv.shift(10),'LogRV_lag22':lrv.shift(22),
        'LogRV_Std5': lrv.rolling(5).std().shift(1),
        'LogRV_Std22':lrv.rolling(22).std().shift(1),
        'RV_Mom5':    (lrv-lrv.shift(5)).shift(1),
        'RV_Mom22':   (lrv-lrv.shift(22)).shift(1),
        'SPY_LogRV':  spy_lrv.shift(1),
        'Ret_lag1':   ret.shift(1), 'Ret_abs_lag1':ret.abs().shift(1),
        'Corr_SPY':   (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                       if asset != 'SPY' else pd.Series(1.0, index=ret.index)),
    }
    p5=compute_parkinson(h,l,5); p22=compute_parkinson(h,l,22)
    gk22=compute_gk(o,h,l,c,22); rs22=compute_rs(o,h,l,c,22)
    feat.update({
        'Parkinson_5':       np.log(p5  +1e-6).shift(1),
        'Parkinson_22':      np.log(p22 +1e-6).shift(1),
        'GarmanKlass_22':    np.log(gk22+1e-6).shift(1),
        'RogersSatchell_22': np.log(rs22+1e-6).shift(1),
        'Range_Close_Ratio': (np.log(p22+1e-6)-lrv).shift(1),
    })
    on = np.log(o / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    for k, val in iv_features.items(): feat['IV_'+k] = val.shift(1)
    dv = v*c
    feat.update({
        'AltVol_Amihud':          (ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1),
        'AltVol_Vol_Ratio':       (v.rolling(5).mean()/(v.rolling(22).mean()+1e-10)).shift(1),
        'AltVol_PV_Corr':         ret.rolling(22).corr(np.log(v+1)).shift(1),
        'AltVol_Vol_Surprise':    ((v-v.rolling(22).mean())/(v.rolling(22).std()+1e-10)).shift(1),
    })
    pv=v.where(ret>0,0).rolling(22).sum(); nv=v.where(ret<=0,0).rolling(22).sum()
    feat['AltVol_Order_Imbalance']=((pv-nv)/(pv+nv+1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']    =(ret.abs().rolling(22).sum()/(v.rolling(22).sum()+1e-10)*1e6).shift(1)
    d = pd.DataFrame(feat); d['ret_sq']=ret_sq; d['Asset']=asset
    d['Class'] = next(cls for cls,assets in ASSET_GROUPS.items() if asset in assets)
    asset_frames[asset] = d

# ── Build 22d panel ───────────────────────────────────────────────────────────
print("\nBuilding 22d panel...", flush=True)
hz = 22
pooled = []
for asset in ALL_ASSETS:
    df = asset_frames[asset].copy()
    df['Target'] = forward_rv(df['ret_sq'], hz)
    df = df.drop(columns=['ret_sq']).dropna()
    pooled.append(df)
data  = pd.concat(pooled).sort_index().reset_index(drop=True)
feats = [c for c in data.columns if c not in ['Target','Asset','Class']]
assert len(feats)==35, f"Expected 35, got {len(feats)}"

split    = int(len(data)*OUTER_TRAIN_RATIO)
train_df = data.iloc[:split-hz].copy()
test_df  = data.iloc[split:].copy()
y_te     = test_df['Target'].values

sc   = StandardScaler().fit(train_df[feats])
X_tr = sc.transform(train_df[feats])
X_te = sc.transform(test_df[feats])

n_tr    = len(train_df)
v_split = int(n_tr*INNER_TRAIN_RATIO)
itr_df  = train_df.iloc[:v_split-hz].copy()
ival_df = train_df.iloc[v_split:].copy()

print(f"Train: {len(train_df)}  Test: {len(test_df)}")

def pooled_r2(y_true, y_pred):
    valid = ~np.isnan(y_pred)
    return float(r2_score(y_true[valid], y_pred[valid]))

# ── Ridge baseline (deterministic) ───────────────────────────────────────────
p_ridge = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class']==cls]; te_m=(test_df['Class']==cls).values
    alpha = BEST_PARAMS['Ridge'][cls]['alpha']
    m = Ridge(alpha=alpha).fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
    p_ridge[te_m] = m.predict(sc.transform(test_df.loc[te_m,feats]))
print(f"\nRidge 22d Pooled R²: {pooled_r2(y_te, p_ridge):.4f}  (benchmark: 0.8026)")

# ── XGBoost — multiple seeds ──────────────────────────────────────────────────
print("\n=== XGBoost seed sweep ===")
xgb_results = {}
for seed in TEST_SEEDS:
    p_xgb = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class']==cls]; te_m=(test_df['Class']==cls).values
        xp = BEST_PARAMS['XGBoost'][cls]
        m = XGBRegressor(n_estimators=200, random_state=seed, verbosity=0,
                         max_depth=xp['max_depth'], learning_rate=xp['learning_rate'],
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
                         min_child_weight=5, n_jobs=1, device=DEVICE, tree_method='hist')
        m.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
        p_xgb[te_m] = m.predict(sc.transform(test_df.loc[te_m,feats]))
    r2 = pooled_r2(y_te, p_xgb)
    xgb_results[seed] = round(r2, 4)
    print(f"  seed={seed:2d}  XGBoost R²={r2:.4f}")
    # WEns
    pw = BEST_PARAMS['pw']
    p_wens = pw*p_ridge + (1-pw)*p_xgb
    wr2 = pooled_r2(y_te, p_wens)
    print(f"           WEns  R²={wr2:.4f}  (pw={pw})")

# ── BiLSTM-A — multiple seeds ─────────────────────────────────────────────────
class BiLSTMAttn(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.fc   = nn.Linear(hidden*2, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out,_ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w*out).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(-1)

def build_sequences(df, X_scaled, y):
    Xs, ys, ids = [], [], []
    for asset in df['Asset'].unique():
        mask=df['Asset']==asset; Xa=X_scaled[mask]; ya=y[mask]; idx=df.index[mask]
        if len(Xa)<=SEQ_LEN: continue
        for i in range(SEQ_LEN, len(Xa)):
            Xs.append(Xa[i-SEQ_LEN:i]); ys.append(ya[i]); ids.append(idx[i])
    if not Xs: return np.empty((0,SEQ_LEN,X_scaled.shape[1])), np.array([]), []
    return np.array(Xs), np.array(ys), ids

def train_bilstm(X_tr_c, y_tr_c, X_te_c, tr_c, te_c, cfg, seed, epochs=20):
    Xtr_s,ytr_s,_ = build_sequences(tr_c, X_tr_c, y_tr_c)
    Xte_s,_,te_ids = build_sequences(te_c, X_te_c, np.zeros(len(te_c)))
    if len(Xtr_s)==0 or len(Xte_s)==0: return np.array([]), []
    torch.manual_seed(seed); np.random.seed(seed)
    m = BiLSTMAttn(X_tr_c.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout']).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=0.001); lf=nn.MSELoss()
    Xt=torch.FloatTensor(Xtr_s).to(DEVICE); yt=torch.FloatTensor(ytr_s).to(DEVICE)
    m.train()
    for _ in range(epochs):
        perm=np.random.permutation(len(Xt))
        for s in range(0,len(perm),64):
            b=perm[s:s+64]; loss=lf(m(Xt[b]),yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(torch.FloatTensor(Xte_s).to(DEVICE)).cpu().numpy()
    return preds, te_ids

print("\n=== BiLSTM-A seed sweep ===")
bilstm_results = {}
for seed in TEST_SEEDS:
    p_bl = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c=train_df[train_df['Class']==cls].reset_index(drop=True)
        te_c=test_df[test_df['Class']==cls].reset_index(drop=True)
        if len(tr_c)<SEQ_LEN*2 or len(te_c)==0: continue
        cfg = BEST_PARAMS['BiLSTM-A'][cls]
        Xtr_c=sc.transform(tr_c[feats]); Xte_c=sc.transform(te_c[feats])
        preds,te_ids=train_bilstm(Xtr_c,tr_c['Target'].values,Xte_c,tr_c,te_c,cfg,seed,epochs=20)
        if len(preds)==0: continue
        pos=[test_df[test_df['Class']==cls].reset_index(drop=True).index.get_loc(i) for i in te_ids]
        te_global_mask=(test_df['Class']==cls).values
        te_global_idx=np.where(te_global_mask)[0]
        for local_pos,pred_val in zip(pos,preds):
            if local_pos<len(te_global_idx):
                p_bl[te_global_idx[local_pos]] = pred_val
    r2 = pooled_r2(y_te, p_bl)
    bilstm_results[seed] = round(r2, 4)
    print(f"  seed={seed:2d}  BiLSTM-A R²={r2:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (22d Pooled R²)")
print("="*60)
print(f"Ridge (deterministic): {pooled_r2(y_te, p_ridge):.4f}  (paper: 0.8026)")
print(f"\nXGBoost by seed:  {xgb_results}  (paper: 0.7773, JSON: 0.7758)")
print(f"BiLSTM-A by seed: {bilstm_results}  (paper: 0.5361, JSON: 0.5217)")

out = {
    'Ridge': round(pooled_r2(y_te, p_ridge), 4),
    'XGBoost_by_seed': xgb_results,
    'BiLSTM_by_seed':  bilstm_results,
    'paper_claims': {'XGBoost': 0.7773, 'BiLSTM-A': 0.5361, 'WEns': 0.8044},
    'current_json':  {'XGBoost': 0.7758, 'BiLSTM-A': 0.5217, 'WEns': 0.8041},
}
os.makedirs(os.path.join(_SCRIPT_DIR,'results'), exist_ok=True)
with open(os.path.join(_SCRIPT_DIR,'results','multiseed_bilstm_results.json'),'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/multiseed_bilstm_results.json")
