"""
Extended DM Tests: WEns vs CatBoost across all horizons
========================================================
Reconstructs WEns (Ridge + XGBoost, stored best_params from v6) and
CatBoost (stored best_params from v6) predictions, then runs
Diebold-Mariano tests across all 8 horizons.

Motivation: run_baseline_comparison.py skipped CatBoost in DM_RECONSTRUCT
because it is slow. This script specifically addresses that gap, providing
the key WEns vs CatBoost statistical comparison requested by reviewers.

Outputs:
  results/dm_extended_results.json
  paper/csv/dm_extended_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from scipy.stats import t as scipy_t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed.")

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE      = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = _os.path.join(_SCRIPT_DIR, 'data')
_RES_DIR    = _os.path.join(_SCRIPT_DIR, 'results')
_CSV_DIR    = _os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv')

V6_JSON = _os.path.join(_RES_DIR, 'main_benchmark_v6_results.json')
with open(V6_JSON) as f:
    v6_results = json.load(f)

# ── Feature-engineering helpers (identical to v6) ────────────────────────────
def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h / l) ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h / l); co = np.log(c / o)
    return np.sqrt((0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(w).mean().clip(0) * 252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.rolling(w).mean().clip(0) * 252)

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

# ── Evaluation ────────────────────────────────────────────────────────────────
def pooled_r2(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2: return float('nan')
    return float(r2_score(y_true[valid], y_pred[valid]))

def pooled_rmse(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2: return float('nan')
    return float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))

# ── DM test (Harvey 1997, identical to baseline_comparison) ──────────────────
def dm_test(y_true, p1, p2, h=1):
    """Positive DM → p1 has larger MSE → p2 is better than p1."""
    e1 = (y_true - p1) ** 2
    e2 = (y_true - p2) ** 2
    d  = e1 - e2
    valid = ~(np.isnan(e1) | np.isnan(e2))
    d = d[valid]
    if len(d) < 10:
        return np.nan, np.nan, False
    d_bar = np.mean(d)
    n     = len(d)

    def autocov(x, k):
        x_mean = np.mean(x)
        return np.sum((x[:n-k] - x_mean) * (x[k:] - x_mean)) / n

    bw    = max(1, h)
    var_d = np.var(d)
    for k in range(1, bw):
        var_d += 2 * autocov(d, k)
    var_d = max(var_d, 1e-12)

    harvey_adj = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat    = d_bar / np.sqrt(var_d / n) * harvey_adj
    p_val      = float(2 * (1 - scipy_t.cdf(np.abs(dm_stat), df=n - 1)))
    better_p2  = bool(dm_stat > 0 and p_val < 0.05)
    return float(dm_stat), p_val, better_p2

# ── Model factory ─────────────────────────────────────────────────────────────
def make_ridge(cfg):
    return Ridge(alpha=cfg['alpha'])

def make_xgb(cfg):
    return XGBRegressor(
        n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=5, random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
        device='cuda', tree_method='hist')

def make_catboost(cfg):
    try:
        return CatBoostRegressor(
            iterations=500, depth=cfg['depth'], learning_rate=cfg['learning_rate'],
            l2_leaf_reg=cfg['l2_leaf_reg'], random_seed=RANDOM_STATE,
            verbose=0, loss_function='RMSE', task_type='GPU')
    except Exception:
        return CatBoostRegressor(
            iterations=500, depth=cfg['depth'], learning_rate=cfg['learning_rate'],
            l2_leaf_reg=cfg['l2_leaf_reg'], random_seed=RANDOM_STATE,
            verbose=0, loss_function='RMSE')

def make_lgbm(cfg):
    return LGBMRegressor(
        n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
        num_leaves=cfg['num_leaves'], subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, min_child_samples=5,
        random_state=RANDOM_STATE, n_jobs=1, verbose=-1)

def fit_predict_perclass(make_fn, bp, train_df, test_df, sc, feats):
    """Fit per-class models using stored best_params, predict on test set."""
    preds = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        cfg = bp.get(cls)
        if not cfg:
            continue
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 10 or te_m.sum() == 0:
            continue
        try:
            m = make_fn(cfg)
            m.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
            preds[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))
        except Exception as e:
            print(f"    WARNING: {cls} fit failed: {e}")
    return preds

# ── Data loading ──────────────────────────────────────────────────────────────
print("=" * 70)
print("Extended DM Tests: WEns vs CatBoost/LightGBM across all horizons")
print("=" * 70)
print("\nLoading data & building 35 features...", flush=True)

def _load_from_parquet():
    vix_df = pd.read_parquet(_os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames = {}
    for asset in ALL_ASSETS:
        p = _os.path.join(_DATA_DIR, f'{asset}.parquet')
        if _os.path.exists(p):
            frames[asset] = pd.read_parquet(p)
    combined = pd.concat(frames.values(), axis=1)
    combined[('Close', 'VIX')]   = vix_df['Close']
    combined[('Close', 'VIX3M')] = vix_df['Close_3M']
    combined[('Close', 'VIX9D')] = vix_df['Close_9D']
    return combined

try:
    raw = pd.read_pickle(_PKL_PATH)
except Exception:
    raw = _load_from_parquet()

vix     = raw[('Close', 'VIX')]
spy_c   = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
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
vrp_val = (vix ** 2 / 100) - spy_rv / 10000
iv_features['VRP']      = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

asset_frames = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]; o = raw[('Open', asset)]
    h = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
    ret    = np.log(c / c.shift(1)).dropna()
    ret_sq = ret ** 2
    rv     = ret_sq.rolling(22).mean() * 252 * 10000
    lrv    = np.log(rv + 1e-6)
    feat = {
        'LogRV_lag1':  lrv.shift(1),  'LogRV_lag5':  lrv.shift(5),
        'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
        'LogRV_Std5':  lrv.rolling(5).std().shift(1),
        'LogRV_Std22': lrv.rolling(22).std().shift(1),
        'RV_Mom5':     (lrv - lrv.shift(5)).shift(1),
        'RV_Mom22':    (lrv - lrv.shift(22)).shift(1),
        'SPY_LogRV':   spy_lrv.shift(1),
        'Ret_lag1':    ret.shift(1),
        'Ret_abs_lag1': ret.abs().shift(1),
        'Corr_SPY':    (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                        if asset != 'SPY' else pd.Series(1.0, index=ret.index)),
    }
    p5  = compute_parkinson(h, l, 5);  p22 = compute_parkinson(h, l, 22)
    gk22 = compute_gk(o, h, l, c, 22); rs22 = compute_rs(o, h, l, c, 22)
    feat['Parkinson_5']       = np.log(p5  + 1e-6).shift(1)
    feat['Parkinson_22']      = np.log(p22 + 1e-6).shift(1)
    feat['GarmanKlass_22']    = np.log(gk22 + 1e-6).shift(1)
    feat['RogersSatchell_22'] = np.log(rs22 + 1e-6).shift(1)
    feat['Range_Close_Ratio'] = (np.log(p22 + 1e-6) - lrv).shift(1)
    on = np.log(o / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    for k, val in iv_features.items():
        feat[f'IV_{k}'] = val.shift(1)
    dv = v * c
    feat['AltVol_Amihud']          = (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']       = (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1)
    feat['AltVol_PV_Corr']         = ret.rolling(22).corr(np.log(v + 1)).shift(1)
    feat['AltVol_Vol_Surprise']    = ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1)
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    feat['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']     = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset']  = asset
    d['Class']  = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    asset_frames[asset] = d

print(f"Assets: {len(asset_frames)}")

# ── Main Loop ─────────────────────────────────────────────────────────────────
results    = {}
dm_records = []

for hz in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*70}")
    hz_key = f'{hz}d'

    pooled = []
    for asset in ALL_ASSETS:
        df = asset_frames[asset].copy()
        df['Target'] = forward_rv(df['ret_sq'], hz)
        df = df.drop(columns=['ret_sq']).dropna()
        pooled.append(df)
    data  = pd.concat(pooled).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    assert len(feats) == 35, f"Expected 35 features, got {len(feats)}"

    split    = int(len(data) * OUTER_TRAIN_RATIO)
    train_df = data.iloc[:split - hz].copy()
    test_df  = data.iloc[split:].copy()
    y_te     = test_df['Target'].values

    sc = StandardScaler().fit(train_df[feats])

    v6_hz = v6_results.get(hz_key, {})
    hz_preds = {}
    hz_res   = {}

    # ── WEns: Ridge + XGBoost with stored pw ──────────────────────────────────
    wens_bp = v6_hz.get('WEns', {}).get('best_params', {})
    pw      = wens_bp.get('pw', 0.7)
    ridge_bp = wens_bp.get('Ridge', {})
    xgb_bp   = wens_bp.get('XGBoost', {})

    p_ridge = fit_predict_perclass(make_ridge, ridge_bp, train_df, test_df, sc, feats)
    p_xgb   = fit_predict_perclass(make_xgb,   xgb_bp,   train_df, test_df, sc, feats)

    valid = ~(np.isnan(p_ridge) | np.isnan(p_xgb))
    p_wens = np.full(len(test_df), np.nan)
    p_wens[valid] = pw * p_ridge[valid] + (1 - pw) * p_xgb[valid]
    hz_preds['WEns']   = p_wens
    hz_preds['Ridge']  = p_ridge
    hz_preds['XGBoost'] = p_xgb
    hz_res['WEns']    = {'Pooled_R2': round(pooled_r2(y_te, p_wens), 4),
                         'RMSE':      round(pooled_rmse(y_te, p_wens), 4), 'pw': pw}
    hz_res['Ridge']   = {'Pooled_R2': round(pooled_r2(y_te, p_ridge), 4)}
    hz_res['XGBoost'] = {'Pooled_R2': round(pooled_r2(y_te, p_xgb), 4)}
    print(f"  [WEns]    pw={pw:.1f}  → Pooled_R2={hz_res['WEns']['Pooled_R2']:.4f}")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    if HAS_CATBOOST:
        cb_bp = v6_hz.get('CatBoost', {}).get('best_params', {})
        p_cb  = fit_predict_perclass(make_catboost, cb_bp, train_df, test_df, sc, feats)
        hz_preds['CatBoost'] = p_cb
        hz_res['CatBoost']   = {'Pooled_R2': round(pooled_r2(y_te, p_cb), 4),
                                 'RMSE':      round(pooled_rmse(y_te, p_cb), 4)}
        print(f"  [CatBoost] → Pooled_R2={hz_res['CatBoost']['Pooled_R2']:.4f}")
    else:
        print("  [CatBoost] skipped (not installed)")
        hz_preds['CatBoost'] = np.full(len(test_df), np.nan)
        hz_res['CatBoost']   = v6_hz.get('CatBoost', {})

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if HAS_LGBM:
        lgbm_bp = v6_hz.get('LightGBM', {}).get('best_params', {})
        p_lgbm  = fit_predict_perclass(make_lgbm, lgbm_bp, train_df, test_df, sc, feats)
        hz_preds['LightGBM'] = p_lgbm
        hz_res['LightGBM']   = {'Pooled_R2': round(pooled_r2(y_te, p_lgbm), 4),
                                  'RMSE':      round(pooled_rmse(y_te, p_lgbm), 4)}
        print(f"  [LightGBM] → Pooled_R2={hz_res['LightGBM']['Pooled_R2']:.4f}")

    # ── HAR-3 (per-class Ridge, alpha=1, HAR features) ────────────────────────
    har_idx = [feats.index(f) for f in HAR_FEATS]
    p_har   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        m = Ridge(alpha=1.0).fit(sc.transform(tr_c[feats])[:, har_idx],
                                  tr_c['Target'].values)
        p_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats])[:, har_idx])
    hz_preds['HAR-3'] = p_har
    hz_res['HAR-3']   = {'Pooled_R2': round(pooled_r2(y_te, p_har), 4),
                          'RMSE':      round(pooled_rmse(y_te, p_har), 4)}
    print(f"  [HAR-3]    → Pooled_R2={hz_res['HAR-3']['Pooled_R2']:.4f}")

    results[hz_key] = hz_res

    # ── DM tests: WEns as reference vs all others ─────────────────────────────
    print(f"\n  DM Tests  WEns vs ... (h={hz}, Harvey-corrected, BW={hz}):")
    p_ref = hz_preds['WEns']
    challengers = [m for m in ['CatBoost', 'LightGBM', 'Ridge', 'XGBoost', 'HAR-3']
                   if not np.isnan(hz_preds.get(m, np.array([np.nan]))).all()]
    for mname in challengers:
        p_m = hz_preds[mname]
        dm_stat, p_val, ref_worse = dm_test(y_te, p_ref, p_m, h=hz)
        sig = '***' if (p_val is not None and p_val < 0.05) else ''
        print(f"    WEns vs {mname:<10}: DM={dm_stat:+.3f}  p={p_val:.3f} {sig}")
        dm_records.append({
            'Horizon':        hz_key,
            'Reference':      'WEns',
            'Challenger':     mname,
            'DM_stat':        round(dm_stat, 4) if not np.isnan(dm_stat) else None,
            'p_value':        round(p_val, 4)   if not np.isnan(p_val)   else None,
            'WEns_better_5pct': bool(dm_stat < 0 and p_val < 0.05),
            'WEns_R2':        hz_res['WEns']['Pooled_R2'],
            'Chal_R2':        hz_res.get(mname, {}).get('Pooled_R2'),
        })

    # CatBoost as reference vs WEns (reverse direction)
    if HAS_CATBOOST and not np.isnan(hz_preds['CatBoost']).all():
        p_cb_ref = hz_preds['CatBoost']
        dm_stat, p_val, cb_worse = dm_test(y_te, p_cb_ref, p_wens, h=hz)
        dm_records.append({
            'Horizon':        hz_key,
            'Reference':      'CatBoost',
            'Challenger':     'WEns',
            'DM_stat':        round(dm_stat, 4) if not np.isnan(dm_stat) else None,
            'p_value':        round(p_val, 4)   if not np.isnan(p_val)   else None,
            'WEns_better_5pct': bool(dm_stat > 0 and p_val < 0.05),
            'WEns_R2':        hz_res['WEns']['Pooled_R2'],
            'Chal_R2':        hz_res['CatBoost']['Pooled_R2'],
        })

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = _os.path.join(_RES_DIR, 'dm_extended_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

df_dm = pd.DataFrame(dm_records)
out_csv = _os.path.join(_CSV_DIR, 'dm_extended_results.csv')
df_dm.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# Summary table
print("\n" + "=" * 70)
print("Pooled R² Summary")
print("=" * 70)
header = f"{'Horizon':>8} {'WEns':>8} {'CatBoost':>10} {'LightGBM':>10} {'Ridge':>8} {'XGBoost':>9} {'HAR-3':>8}"
print(header)
for hz_key, res in results.items():
    row = (f"{hz_key:>8}"
           f"  {res.get('WEns',    {}).get('Pooled_R2', float('nan')):>8.4f}"
           f"  {res.get('CatBoost',{}).get('Pooled_R2', float('nan')):>8.4f}"
           f"  {res.get('LightGBM',{}).get('Pooled_R2', float('nan')):>8.4f}"
           f"  {res.get('Ridge',   {}).get('Pooled_R2', float('nan')):>8.4f}"
           f"  {res.get('XGBoost', {}).get('Pooled_R2', float('nan')):>8.4f}"
           f"  {res.get('HAR-3',   {}).get('Pooled_R2', float('nan')):>8.4f}")
    print(row)

print("\nDM Tests — WEns vs CatBoost (p<0.05 = WEns significantly better):")
df_wens_cb = df_dm[(df_dm['Reference'] == 'WEns') & (df_dm['Challenger'] == 'CatBoost')]
for _, row in df_wens_cb.iterrows():
    sig = '***' if row.get('WEns_better_5pct') else ''
    dm  = row['DM_stat'] if row['DM_stat'] is not None else float('nan')
    pv  = row['p_value']  if row['p_value']  is not None else float('nan')
    print(f"  {row['Horizon']:>5}: DM={dm:+.3f}  p={pv:.3f} {sig}"
          f"  (WEns={row['WEns_R2']:.4f}, CatBoost={row['Chal_R2']:.4f})")
