"""
WEns Verification Script: Targets 0.803 Pooled R^2 at 22d Horizon
Parameters derived from V71 Championship Run:
- Ridge Alphas: Equity=100, Bond=10, Commodity=10
- XGBoost: n_est=100, max_depth=4, lr=0.05, reg_alpha=1.0, reg_lambda=2.0
- Blending: 0.7 * Ridge + 0.3 * XGBoost
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':   ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HZ = 22
ALPHAS_PERCLASS = {'Equity': 100.0, 'Bond': 10.0, 'Commodity': 10.0}

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

# ── Loading ─────────────────────────────────────────────────
print("Loading V71 cache...", flush=True)
raw = pd.read_pickle('/root/vrp/src/data/v71_ohlcv_cache.pkl')
vix = raw[('Close', 'VIX')]
spy_c = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

iv_features = {
    'VIX': np.log(vix + 1e-6),
    'VIX_chg': np.log(vix + 1e-6).diff(),
    'VIX_ma5': np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5': np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M': np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D': np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope': np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp_val = (vix ** 2 / 100) - spy_rv / 10000
iv_features['VRP'] = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

# ── Feature Discovery (from original compute_parkinson etc.) ──
def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)
def compute_gk(o,h,l,c,w=22):
    hl=np.log(h/l); co=np.log(c/o)
    return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def compute_rs(o,h,l,c,w=22):
    rs=np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

# ── Build Dataset ──────────────────────────────────────────
print("Building dataset for 22d horizon...", flush=True)
pooled = []
for asset in ALL_ASSETS:
    c=raw[('Close',asset)]; o=raw[('Open',asset)]
    h=raw[('High',asset)]; l=raw[('Low',asset)]; vol=raw[('Volume',asset)]
    ret=np.log(c/c.shift(1)).dropna()
    rv=(ret**2).rolling(22).mean()*252*10000; lrv=np.log(rv+1e-6)
    
    # 37 features construction
    feat = {
        'LogRV_lag1':lrv.shift(1),'LogRV_lag5':lrv.shift(5),
        'LogRV_lag10':lrv.shift(10),'LogRV_lag22':lrv.shift(22),
        'LogRV_Std5':lrv.rolling(5).std().shift(1),
        'LogRV_Std22':lrv.rolling(22).std().shift(1),
        'RV_Mom5':(lrv-lrv.shift(5)).shift(1),
        'RV_Mom22':(lrv-lrv.shift(22)).shift(1),
        'SPY_LogRV':spy_lrv.shift(1),
        'Ret_lag1':ret.shift(1),'Ret_abs_lag1':ret.abs().shift(1),
        'Corr_SPY':ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1) if asset!='SPY' else pd.Series(1.0,index=ret.index),
    }
    # HF Proxy
    p5=compute_parkinson(h,l,5); p22=compute_parkinson(h,l,22)
    gk22=compute_gk(o,h,l,c,22); rs22=compute_rs(o,h,l,c,22)
    feat['Parkinson_5'] = np.log(p5+1e-6).shift(1)
    feat['Parkinson_22'] = np.log(p22+1e-6).shift(1)
    feat['GarmanKlass_22'] = np.log(gk22+1e-6).shift(1)
    feat['RogersSatchell_22'] = np.log(rs22+1e-6).shift(1)
    feat['Range_Close_Ratio'] = (np.log(p22+1e-6)-lrv).shift(1)
    on=np.log(o/c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    # IV
    for n2,v2 in iv_features.items(): feat[f'IV_{n2}']=v2.shift(1)
    # Alt
    dv=vol*c
    feat['AltVol_Amihud']=(ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']=(vol.rolling(5).mean()/(vol.rolling(22).mean()+1e-10)).shift(1)
    feat['AltVol_PV_Corr']=ret.rolling(22).corr(np.log(vol+1)).shift(1)
    feat['AltVol_Vol_Surprise']=((vol-vol.rolling(22).mean())/(vol.rolling(22).std()+1e-10)).shift(1)
    pv=vol.where(ret>0,0).rolling(22).sum(); nv=vol.where(ret<=0,0).rolling(22).sum()
    feat['AltVol_Order_Imbalance']=((pv-nv)/(pv+nv+1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']=(ret.abs().rolling(22).sum()/(vol.rolling(22).sum()+1e-10)*1e6).shift(1)

    d = pd.DataFrame(feat)
    d['Target'] = forward_rv(ret**2, HZ)
    d['Asset'] = asset
    d['Class'] = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    pooled.append(d.dropna())

data = pd.concat(pooled).sort_index().reset_index(drop=True)
feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]

# ── Split & Scale ───────────────────────────────────────────
split = int(len(data) * 0.8)
train_df = data.iloc[:split - HZ]
test_df = data.iloc[split:]
y_te = test_df['Target'].values

sc = StandardScaler().fit(train_df[feats])

# ── Verification Run (Full Optimization) ─────────────────────
print("Running full optimization verification for 22d horizon...", flush=True)

# 1. Tuning Alphas per class (using 80:20 validation split within training)
val_split = int(len(train_df) * 0.8)
tr_inner = train_df.iloc[:val_split - HZ]
va_inner = train_df.iloc[val_split:]

best_alphas = {}
alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

for cls in ASSET_GROUPS:
    best_cls_r2, best_a = -999, 1.0
    tr_c = tr_inner[tr_inner['Class'] == cls]
    va_c = va_inner[va_inner['Class'] == cls]
    if len(tr_c) < 50 or len(va_c) < 20:
        best_alphas[cls] = 10.0; continue
    
    tr_sc = StandardScaler().fit(tr_c[feats])
    X_tr_c = tr_sc.transform(tr_c[feats])
    X_va_c = tr_sc.transform(va_c[feats])
    
    for a in alphas:
        m = Ridge(alpha=a).fit(X_tr_c, tr_c['Target'])
        r2 = r2_score(va_c['Target'], m.predict(X_va_c))
        if r2 > best_cls_r2:
            best_cls_r2, best_a = r2, a
    best_alphas[cls] = best_a

print(f"Optimal Alphas found: {best_alphas}")

# 2. Final Ridge & XGBoost training
p_ridge = np.full(len(test_df), np.nan)
p_xgb = np.full(len(test_df), np.nan)

for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_m = test_df['Class'] == cls
    if te_m.sum() == 0: continue
    
    # Ridge
    m_r = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_c[feats]), tr_c['Target'])
    p_ridge[te_m.values] = m_r.predict(sc.transform(test_df.loc[te_m, feats]))
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                       random_state=42, n_jobs=1)
    xgb.fit(tr_c[feats].values, tr_c['Target'].values)
    p_xgb[te_m.values] = xgb.predict(test_df.loc[te_m, feats].values)

# 3. Tuning Ensemble Weight (w * Ridge + (1-w) * XGB)
best_w, max_r2 = 0.7, -999
# We stay on training/validation for weight tuning? 
# Usually, WEns weight is tuned on the same validation as alphas.
va_preds_ridge = np.full(len(va_inner), np.nan)
va_preds_xgb = np.full(len(va_inner), np.nan)

for cls in ASSET_GROUPS:
    tr_c = tr_inner[tr_inner['Class'] == cls]
    va_m = va_inner['Class'] == cls
    if va_m.sum() == 0: continue
    
    m_r = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_c[feats]), tr_c['Target'])
    va_preds_ridge[va_m.values] = m_r.predict(sc.transform(va_inner.loc[va_m, feats]))
    
    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    xgb.fit(tr_c[feats].values, tr_c['Target'].values)
    va_preds_xgb[va_m.values] = xgb.predict(va_inner.loc[va_m, feats].values)

for w in np.arange(0, 1.01, 0.05):
    ens = w * va_preds_ridge + (1-w) * va_preds_xgb
    r2 = r2_score(va_inner['Target'], ens)
    if r2 > max_r2:
        max_r2, best_w = r2, w

print(f"Optimal Ensemble Weight found: {best_w:.2f}")

# 4. Final Prediction
p_wens = best_w * p_ridge + (1 - best_w) * p_xgb

# ── Results ─────────────────────────────────────────────────
r2_ridge = r2_score(y_te, p_ridge)
r2_xgb = r2_score(y_te, p_xgb)
r2_wens = r2_score(y_te, p_wens)

print("\n" + "="*40)
print("  VERIFICATION RESULTS (FULL OPT)")
print("="*40)
print(f"  Ridge Pooled R2:   {r2_ridge:.4f}")
print(f"  XGBoost Pooled R2: {r2_xgb:.4f}")
print(f"  Best Weight:       {best_w:.2f}")
print(f"  WEns Pooled R2:    {r2_wens:.4f} <-- TARGET: 0.803")
print("="*40)

if r2_wens >= 0.8025:
    print("\nSUCCESS: Result matches or exceeds the 0.803 benchmark!")
else:
    print("\nNOTE: Result is still slightly below 0.803.")
