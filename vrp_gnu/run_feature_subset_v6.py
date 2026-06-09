"""
Feature subset experiment — v6 pipeline (35 features, global scaler, Ridge)
Replicates Table 8 using the same pipeline as run_main_benchmark_v6.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HZ = 22

def cp(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean() / (4*np.log(2))) * np.sqrt(252)
def cgk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def crs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)
def forward_rv(ret_sq, hz):
    cs = ret_sq.cumsum()
    return np.log((cs.shift(-hz) - cs) / hz * 252 + 1e-12)

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = _os.path.join(_SCRIPT_DIR, 'data')

def _load_from_parquet():
    vix_df  = pd.read_parquet(_os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames  = {}
    for asset in ALL_ASSETS + ['VIX']:
        p = _os.path.join(_DATA_DIR, f'{asset}.parquet')
        if not _os.path.exists(p): continue
        frames[asset] = pd.read_parquet(p)
    combined = pd.concat(frames.values(), axis=1)
    combined[('Close', 'VIX')]   = vix_df['Close']
    combined[('Close', 'VIX3M')] = vix_df['Close_3M']
    combined[('Close', 'VIX9D')] = vix_df['Close_9D']
    return combined

# Load data (same as v6)
print("Loading data...", flush=True)
if _os.path.exists(_PKL_PATH):
    raw = pd.read_pickle(_PKL_PATH)
else:
    raw = _load_from_parquet()
vix     = raw[('Close', 'VIX')]
spy_c   = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_ret**2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

ivf = {
    'VIX':           np.log(vix + 1e-6),
    'VIX_chg':       np.log(vix + 1e-6).diff(),
    'VIX_ma5':       np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5':      np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M':         np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D':         np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope':np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp = (vix**2 / 100) - spy_rv / 10000
ivf['VRP'] = vrp; ivf['VRP_ma22'] = vrp.rolling(22).mean()

af = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]; o = raw[('Open', asset)]
    h = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
    ret = np.log(c / c.shift(1)).dropna(); rs2 = ret**2
    rv2 = rs2.rolling(22).mean() * 252 * 10000; lrv = np.log(rv2 + 1e-6)
    ft = {
        'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
        'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
        'LogRV_Std5': lrv.rolling(5).std().shift(1),
        'LogRV_Std22': lrv.rolling(22).std().shift(1),
        'RV_Mom5': (lrv - lrv.shift(5)).shift(1),
        'RV_Mom22': (lrv - lrv.shift(22)).shift(1),
        'SPY_LogRV': spy_lrv.shift(1),
        'Ret_lag1': ret.shift(1), 'Ret_abs_lag1': ret.abs().shift(1),
        'Corr_SPY': (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                     if asset != 'SPY' else pd.Series(1.0, index=ret.index)),
    }
    p5 = cp(h, l, 5); p22 = cp(h, l, 22)
    gk22 = cgk(o, h, l, c, 22); rs22 = crs(o, h, l, c, 22)
    ft.update({
        'Parkinson_5':       np.log(p5   + 1e-6).shift(1),
        'Parkinson_22':      np.log(p22  + 1e-6).shift(1),
        'GarmanKlass_22':    np.log(gk22 + 1e-6).shift(1),
        'RogersSatchell_22': np.log(rs22 + 1e-6).shift(1),
        'Range_Close_Ratio': (np.log(p22 + 1e-6) - lrv).shift(1),
    })
    on = np.log(o / c.shift(1))
    ft['Overnight_Vol'] = on.rolling(22).std().shift(1)
    ft['Overnight_Ret'] = on.shift(1)
    for k, val in ivf.items(): ft['IV_' + k] = val.shift(1)
    dv = v * c
    ft.update({
        'AltVol_Amihud':         (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1),
        'AltVol_Vol_Ratio':      (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1),
        'AltVol_PV_Corr':        ret.rolling(22).corr(np.log(v + 1)).shift(1),
        'AltVol_Vol_Surprise':   ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1),
    })
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    ft['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    ft['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
    d = pd.DataFrame(ft); d['ret_sq'] = rs2; d['Asset'] = asset
    d['Class'] = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    af[asset] = d

# Build full panel for 22d
pool = []
for asset in ALL_ASSETS:
    df = af[asset].copy(); df['Target'] = forward_rv(df['ret_sq'], HZ)
    df = df.drop(columns=['ret_sq']).dropna(); pool.append(df)
data = pd.concat(pool).sort_index().reset_index(drop=True)
ALL_FEATS = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
assert len(ALL_FEATS) == 35, f"Expected 35, got {len(ALL_FEATS)}"

sp = int(len(data) * 0.8)
tr = data.iloc[:sp - HZ].copy()
te = data.iloc[sp:].copy()
y_te = te['Target'].values

# Feature group definitions (35-feature basis, no GARCH)
SUBSETS = {
    'HAR (3)':            ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22'],
    'Range only (5)':     ['Parkinson_5', 'Parkinson_22', 'GarmanKlass_22',
                           'RogersSatchell_22', 'Range_Close_Ratio'],
    'GARCH proxy (2)':    ['LogRV_Std5', 'LogRV_Std22'],   # closest analog in v6
    'IV Surface (10)':    [f for f in ALL_FEATS if f.startswith('IV_')],
    'HAR + Range (8)':    ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22',
                           'Parkinson_5', 'Parkinson_22', 'GarmanKlass_22',
                           'RogersSatchell_22', 'Range_Close_Ratio'],
    'Full (35)':          ALL_FEATS,
}

# Load v6 Ridge best params for 22d
with open('results/main_benchmark_v6_results.json') as f: saved = json.load(f)
best_alpha = saved['22d']['Ridge']['best_params']  # {cls: {alpha: ...}}

def eval_subset(feat_list, label):
    fidx = [ALL_FEATS.index(f) for f in feat_list]
    sc = StandardScaler().fit(tr[ALL_FEATS].values)
    X_tr = sc.transform(tr[ALL_FEATS])[:, fidx]
    X_te = sc.transform(te[ALL_FEATS])[:, fidx]

    preds = np.full(len(te), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        alpha = best_alpha.get(cls, {}).get('alpha', 1000) \
                if isinstance(best_alpha, dict) and cls in best_alpha else 1000
        m = Ridge(alpha=alpha).fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        preds[tem] = m.predict(X_te[te['Class'].values == cls])

    valid = ~np.isnan(preds) & ~np.isnan(y_te)
    pooled = float(r2_score(y_te[valid], preds[valid]))
    median = float(np.median([r2_score(y_te[te['Asset'].values == a],
                                       preds[te['Asset'].values == a])
                               for a in ALL_ASSETS
                               if (te['Asset'].values == a).sum() > 2]))
    print(f"  {label:<22} n={len(feat_list):>2}  Pooled={pooled:.4f}  Median={median:.4f}")
    return {'n_features': len(feat_list), 'pooled_r2': round(pooled, 4), 'median_r2': round(median, 4)}

print("\n[Table 8] Feature Subset Experiment — 22d Ridge (v6 pipeline)\n")
results = {}
for label, feats in SUBSETS.items():
    results[label] = eval_subset(feats, label)

# Save to JSON for paper
import json as json2
with open('paper/csv/feature_subset_v6.json', 'w') as f:
    json2.dump(results, f, indent=2)
print("\nSaved: paper/csv/feature_subset_v6.json")
