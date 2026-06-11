"""
Extended MLP hyperparameter search (22d horizon).
Main benchmark used only 6 configs; this expands to 24 configs.
Goal: check whether poor MLP performance is due to narrow search.

Original grid: hidden ∈ {(64,),(64,32),(128,64)} × alpha ∈ {0.0001,0.01}  → 6 configs
Extended grid: hidden ∈ {(64,),(128,),(64,32),(128,64),(256,128),(128,64,32)}
               × alpha ∈ {0.0001,0.001,0.01,0.1}  → 24 configs
               + max_iter=1000 (vs 500)

Output: results/mlp_extended_results.json
"""
import numpy as np
import pandas as pd
import json
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

HZ = 22
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = os.path.join(_SCRIPT_DIR, 'data')

# ── Extended MLP grid ─────────────────────────────────────────────────────────
HIDDEN_OPTS = [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)]
ALPHA_OPTS  = [0.0001, 0.001, 0.01, 0.1]
MLP_GRID = [{'hidden_layer_sizes': h, 'alpha': a}
            for h in HIDDEN_OPTS for a in ALPHA_OPTS]

# Original grid for comparison
MLP_ORIG_GRID = [{'hidden_layer_sizes': h, 'alpha': a}
                 for h in [(64,), (64, 32), (128, 64)]
                 for a in [0.0001, 0.01]]

print(f"Extended grid: {len(MLP_GRID)} configs  |  Original: {len(MLP_ORIG_GRID)} configs")

def _load_from_parquet():
    vix_df = pd.read_parquet(os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames = {}
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
    return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)
def forward_rv(ret_sq, hz):
    cs = ret_sq.cumsum()
    return np.log((cs.shift(-hz)-cs)/hz*252+1e-12)

asset_frames = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]; o = raw[('Open', asset)]
    h = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
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

print("Building 22d panel...", flush=True)
pooled = []
for asset in ALL_ASSETS:
    df = asset_frames[asset].copy()
    df['Target'] = forward_rv(df['ret_sq'], HZ)
    df = df.drop(columns=['ret_sq']).dropna()
    pooled.append(df)
data  = pd.concat(pooled).sort_index().reset_index(drop=True)
feats = [c for c in data.columns if c not in ['Target','Asset','Class']]
assert len(feats)==35

split    = int(len(data)*OUTER_TRAIN_RATIO)
train_df = data.iloc[:split-HZ].copy()
test_df  = data.iloc[split:].copy()
y_te     = test_df['Target'].values

sc = StandardScaler().fit(train_df[feats])

n_tr    = len(train_df)
v_split = int(n_tr*INNER_TRAIN_RATIO)
itr_df  = train_df.iloc[:v_split-HZ].copy()
ival_df = train_df.iloc[v_split:].copy()

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

def pooled_r2(y_true, y_pred):
    valid = ~np.isnan(y_pred)
    if valid.sum() < 2: return float('nan')
    return float(r2_score(y_true[valid], y_pred[valid]))

def run_mlp_grid(grid, label, max_iter=1000):
    print(f"\n=== {label} ({len(grid)} configs) ===", flush=True)
    best_params = {}
    final_models = {}

    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class']==cls]
        ival_c = ival_df[ival_df['Class']==cls]
        tr_c   = train_df[train_df['Class']==cls]

        best_r2, best_cfg = -np.inf, grid[0]
        for cfg in grid:
            m = MLPRegressor(
                hidden_layer_sizes=cfg['hidden_layer_sizes'],
                alpha=cfg['alpha'],
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            )
            m.fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
            r2 = pooled_r2(ival_c['Target'].values, m.predict(sc.transform(ival_c[feats])))
            if r2 > best_r2: best_r2, best_cfg = r2, cfg

        best_params[cls] = best_cfg
        m_final = MLPRegressor(
            hidden_layer_sizes=best_cfg['hidden_layer_sizes'],
            alpha=best_cfg['alpha'],
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )
        m_final.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
        final_models[cls] = m_final
        print(f"  {cls}: best={best_cfg}  inner_R²={best_r2:.4f}", flush=True)

    preds = np.full(len(test_df), np.nan)
    for cls, m in final_models.items():
        te_m = (test_df['Class']==cls).values
        preds[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))

    r2 = pooled_r2(y_te, preds)
    print(f"  → Test Pooled R²={r2:.4f}", flush=True)
    return r2, best_params

# Also run Ridge as deterministic baseline
p_ridge = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    itr_c  = itr_df[itr_df['Class']==cls]
    ival_c = ival_df[ival_df['Class']==cls]
    tr_c   = train_df[train_df['Class']==cls]
    te_m   = (test_df['Class']==cls).values
    best_r2, best_a = -np.inf, 1000
    for a in [10, 50, 100, 500, 1000, 2000]:
        m = Ridge(alpha=a).fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
        r2 = pooled_r2(ival_c['Target'].values, m.predict(sc.transform(ival_c[feats])))
        if r2 > best_r2: best_r2, best_a = r2, a
    m = Ridge(alpha=best_a).fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
    p_ridge[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))
ridge_r2 = pooled_r2(y_te, p_ridge)
print(f"\nRidge (deterministic): {ridge_r2:.4f}  (paper: 0.8026)")

orig_r2, orig_params   = run_mlp_grid(MLP_ORIG_GRID, "MLP-Original (6 configs)", max_iter=500)
ext_r2,  ext_params    = run_mlp_grid(MLP_GRID,      "MLP-Extended (24 configs)", max_iter=1000)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Ridge                 : {ridge_r2:.4f}  (paper: 0.8026)")
print(f"MLP Original  (6 cfg) : {orig_r2:.4f}  (paper: 0.5644)")
print(f"MLP Extended (24 cfg) : {ext_r2:.4f}")
delta = ext_r2 - orig_r2
print(f"Δ (Extended − Original): {delta:+.4f}")
print()
if abs(delta) < 0.02:
    print("→ 탐색 범위 확장 효과 미미: MLP 열위는 탐색 부족이 아닌 모형 구조 한계")
else:
    print("→ 탐색 범위 확장으로 유의한 개선: 탐색 부족이 성능에 기여했을 가능성")

out = {
    'Ridge_R2':          round(ridge_r2, 4),
    'MLP_original_R2':   round(orig_r2,  4),
    'MLP_extended_R2':   round(ext_r2,   4),
    'delta':             round(ext_r2 - orig_r2, 4),
    'MLP_original_params': {k: str(v) for k, v in orig_params.items()},
    'MLP_extended_params': {k: str(v) for k, v in ext_params.items()},
    'paper_baseline': {'Ridge': 0.8026, 'MLP': 0.5644},
}
os.makedirs(os.path.join(_SCRIPT_DIR, 'results'), exist_ok=True)
with open(os.path.join(_SCRIPT_DIR, 'results', 'mlp_extended_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved: results/mlp_extended_results.json")
