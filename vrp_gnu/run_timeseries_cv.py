"""
5-fold time-series cross-validation for WEns, Ridge, XGBoost (22d horizon).
Goal: verify that 80/20 single-split result is not a lucky split.
Each fold uses training data up to fold boundary; test is next segment.
Purge gap of 22d applied at each boundary.
Output: results/timeseries_cv_results.json
"""
import numpy as np
import pandas as pd
import json
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

HZ = 22
N_FOLDS = 5
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
INNER_RATIO = 0.8

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = os.path.join(_SCRIPT_DIR, 'data')

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
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
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
assert len(feats)==35, f"Expected 35, got {len(feats)}"

# unique dates for fold splitting (panel has 11 rows per date)
dates = data.index.tolist()
n = len(data)
fold_size = n // (N_FOLDS + 1)  # walk-forward: always grow training set

RIDGE_GRID  = [10, 50, 100, 500, 1000, 2000]
XGB_GRID    = [{'max_depth':d,'learning_rate':lr} for d in [3,4] for lr in [0.03,0.05,0.1]]
WENS_PW     = [0.4,0.5,0.6,0.7,0.8,0.9]

def pooled_r2(y_true, y_pred):
    valid = ~np.isnan(y_pred)
    if valid.sum() < 2: return float('nan')
    return float(r2_score(y_true[valid], y_pred[valid]))

fold_results = []

for fold in range(N_FOLDS):
    # Walk-forward: train = rows 0..split-1, test = rows split+hz..split+fold_size-1
    split    = fold_size * (fold + 1)
    tr_end   = split - HZ
    te_start = split
    te_end   = min(split + fold_size, n)

    if tr_end < 100 or te_start >= n: continue

    train_df = data.iloc[:tr_end].copy()
    test_df  = data.iloc[te_start:te_end].copy()
    y_te     = test_df['Target'].values

    sc   = StandardScaler().fit(train_df[feats])

    # inner holdout (last 20% of train)
    v_split = int(len(train_df) * INNER_RATIO)
    itr_df  = train_df.iloc[:v_split - HZ].copy()
    ival_df = train_df.iloc[v_split:].copy()

    print(f"\nFold {fold+1}/{N_FOLDS}: train={len(train_df):,}  test={len(test_df):,}", flush=True)

    # ── Ridge ──
    p_ridge = np.full(len(test_df), np.nan)
    best_ridge_params = {}
    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class']==cls]
        ival_c = ival_df[ival_df['Class']==cls]
        tr_c   = train_df[train_df['Class']==cls]
        te_m   = (test_df['Class']==cls).values
        best_r2, best_a = -np.inf, 1000
        if len(itr_c) >= 10 and len(ival_c) >= 5:
            for a in RIDGE_GRID:
                m = Ridge(alpha=a).fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
                r2 = pooled_r2(ival_c['Target'].values, m.predict(sc.transform(ival_c[feats])))
                if r2 > best_r2: best_r2, best_a = r2, a
        best_ridge_params[cls] = best_a
        m = Ridge(alpha=best_a).fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
        p_ridge[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))

    ridge_r2 = pooled_r2(y_te, p_ridge)
    print(f"  Ridge R²={ridge_r2:.4f}  params={best_ridge_params}", flush=True)

    # ── XGBoost ──
    p_xgb = np.full(len(test_df), np.nan)
    best_xgb_params = {}
    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class']==cls]
        ival_c = ival_df[ival_df['Class']==cls]
        tr_c   = train_df[train_df['Class']==cls]
        te_m   = (test_df['Class']==cls).values
        best_r2, best_cfg = -np.inf, XGB_GRID[0]
        if len(itr_c) >= 10 and len(ival_c) >= 5:
            for cfg in XGB_GRID:
                m = XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                 max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, n_jobs=1)
                m.fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
                r2 = pooled_r2(ival_c['Target'].values, m.predict(sc.transform(ival_c[feats])))
                if r2 > best_r2: best_r2, best_cfg = r2, cfg
        best_xgb_params[cls] = best_cfg
        m = XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                         max_depth=best_cfg['max_depth'], learning_rate=best_cfg['learning_rate'],
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, n_jobs=1)
        m.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
        p_xgb[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))

    xgb_r2 = pooled_r2(y_te, p_xgb)
    print(f"  XGBoost R²={xgb_r2:.4f}", flush=True)

    # ── WEns (pw search on inner val) ──
    p_r_iv = np.full(len(ival_df), np.nan)
    p_x_iv = np.full(len(ival_df), np.nan)
    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class']==cls]
        ival_c = ival_df[ival_df['Class']==cls]
        iv_m   = (ival_df['Class']==cls).values
        if len(itr_c) < 5: continue
        mr = Ridge(alpha=best_ridge_params[cls]).fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
        mx = XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                          max_depth=best_xgb_params[cls]['max_depth'],
                          learning_rate=best_xgb_params[cls]['learning_rate'],
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5, n_jobs=1)
        mx.fit(sc.transform(itr_c[feats]), itr_c['Target'].values)
        p_r_iv[iv_m] = mr.predict(sc.transform(ival_df.loc[iv_m, feats]))
        p_x_iv[iv_m] = mx.predict(sc.transform(ival_df.loc[iv_m, feats]))

    y_iv = ival_df['Target'].values
    best_pw, best_pw_r2 = 0.8, -np.inf
    for pw in WENS_PW:
        valid = ~(np.isnan(p_r_iv)|np.isnan(p_x_iv))
        if valid.sum() < 2: continue
        r2 = pooled_r2(y_iv[valid], pw*p_r_iv[valid]+(1-pw)*p_x_iv[valid])
        if r2 > best_pw_r2: best_pw_r2, best_pw = r2, pw

    valid = ~(np.isnan(p_ridge)|np.isnan(p_xgb))
    p_wens = np.full(len(test_df), np.nan)
    p_wens[valid] = best_pw*p_ridge[valid]+(1-best_pw)*p_xgb[valid]
    wens_r2 = pooled_r2(y_te, p_wens)
    print(f"  WEns  R²={wens_r2:.4f}  pw={best_pw}", flush=True)

    fold_results.append({
        'fold': fold+1,
        'train_n': len(train_df),
        'test_n':  len(test_df),
        'Ridge_R2':   round(ridge_r2, 4),
        'XGBoost_R2': round(xgb_r2,   4),
        'WEns_R2':    round(wens_r2,   4),
        'best_pw':    best_pw,
    })

# Summary
print("\n" + "="*60)
print("5-Fold Walk-Forward CV Summary (22d)")
print("="*60)
print(f"{'Fold':<6} {'Train':>7} {'Test':>7} {'Ridge':>8} {'XGBoost':>9} {'WEns':>8} {'pw':>5}")
for r in fold_results:
    print(f"{r['fold']:<6} {r['train_n']:>7,} {r['test_n']:>7,} "
          f"{r['Ridge_R2']:>8.4f} {r['XGBoost_R2']:>9.4f} {r['WEns_R2']:>8.4f} {r['best_pw']:>5.1f}")

for model in ['Ridge_R2','XGBoost_R2','WEns_R2']:
    vals = [r[model] for r in fold_results if not np.isnan(r[model])]
    print(f"\n{model}: mean={np.mean(vals):.4f}  std={np.std(vals,ddof=1):.4f}  "
          f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
print(f"\n(Paper single-split: Ridge=0.8026  XGBoost=0.7758  WEns=0.8041)")

out = {
    'folds': fold_results,
    'summary': {
        m: {
            'mean': round(float(np.mean([r[m] for r in fold_results])),4),
            'std':  round(float(np.std([r[m] for r in fold_results],ddof=1)),4),
            'min':  round(float(np.min([r[m] for r in fold_results])),4),
            'max':  round(float(np.max([r[m] for r in fold_results])),4),
        }
        for m in ['Ridge_R2','XGBoost_R2','WEns_R2']
    },
    'paper_single_split': {'Ridge':0.8026,'XGBoost':0.7758,'WEns':0.8041},
}
os.makedirs(os.path.join(_SCRIPT_DIR,'results'), exist_ok=True)
with open(os.path.join(_SCRIPT_DIR,'results','timeseries_cv_results.json'),'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/timeseries_cv_results.json")
