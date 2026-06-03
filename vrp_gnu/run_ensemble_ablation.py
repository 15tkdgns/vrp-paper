"""
Ensemble Ablation: Ridge+XGBoost vs Ridge+CatBoost vs XGBoost+CatBoost
=======================================================================
Tests whether WEns (Ridge + XGBoost) is the best two-model ensemble, or
whether replacing XGBoost with CatBoost (or Ridge with CatBoost) yields
better out-of-sample performance.

Design:
  - Individual model best_params loaded from main_benchmark_v6_results.json
  - Per-horizon pw search on inner holdout for each pair
  - DM tests between ensemble variants at each horizon
  - No retuning of base model hyperparameters (uses stored best_params)

Outputs:
  results/ensemble_ablation_results.json
  paper/csv/ensemble_ablation_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from scipy.stats import t as scipy_t
from itertools import combinations
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
PW_GRID           = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = _os.path.join(_SCRIPT_DIR, 'data')
_RES_DIR    = _os.path.join(_SCRIPT_DIR, 'results')
_CSV_DIR    = _os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv')

V6_JSON = _os.path.join(_RES_DIR, 'main_benchmark_v6_results.json')
with open(V6_JSON) as f:
    v6_results = json.load(f)

# ── Feature-engineering helpers ───────────────────────────────────────────────
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

# ── DM test ───────────────────────────────────────────────────────────────────
def dm_test(y_true, p1, p2, h=1):
    """Positive DM → p1 has larger MSE → p2 better."""
    e1 = (y_true - p1) ** 2
    e2 = (y_true - p2) ** 2
    d  = e1 - e2
    valid = ~(np.isnan(e1) | np.isnan(e2))
    d = d[valid]
    if len(d) < 10:
        return np.nan, np.nan, False
    d_bar = np.mean(d); n = len(d)

    def autocov(x, k):
        xm = np.mean(x)
        return np.sum((x[:n-k] - xm) * (x[k:] - xm)) / n

    bw = max(1, h)
    var_d = np.var(d)
    for k in range(1, bw):
        var_d += 2 * autocov(d, k)
    var_d = max(var_d, 1e-12)
    harvey_adj = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat    = d_bar / np.sqrt(var_d / n) * harvey_adj
    p_val      = float(2 * (1 - scipy_t.cdf(np.abs(dm_stat), df=n - 1)))
    return float(dm_stat), p_val, bool(dm_stat > 0 and p_val < 0.05)

# ── Model constructors ────────────────────────────────────────────────────────
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

MODEL_CONSTRUCTORS = {
    'Ridge':    make_ridge,
    'XGBoost':  make_xgb,
    'CatBoost': make_catboost,
    'LightGBM': make_lgbm,
}

def fit_predict_perclass(name, bp, train_df, test_df, sc, feats):
    make_fn = MODEL_CONSTRUCTORS[name]
    preds   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        cfg = bp.get(cls)
        if not cfg: continue
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 10 or te_m.sum() == 0: continue
        try:
            m = make_fn(cfg)
            m.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
            preds[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats]))
        except Exception as e:
            print(f"    WARNING: {name} {cls} failed: {e}")
    return preds

def find_best_pw(p1, p2, y_val, pw_grid=PW_GRID):
    """Find pw that maximises inner-val R² for pw*p1 + (1-pw)*p2."""
    best_r2, best_pw = -np.inf, pw_grid[0]
    for pw in pw_grid:
        valid = ~(np.isnan(p1) | np.isnan(p2))
        if valid.sum() < 5: continue
        blend = pw * p1[valid] + (1 - pw) * p2[valid]
        r2 = float(r2_score(y_val[valid], blend))
        if r2 > best_r2:
            best_r2, best_pw = r2, pw
    return best_pw, best_r2

def blend(p1, p2, pw):
    valid = ~(np.isnan(p1) | np.isnan(p2))
    out   = np.full(len(p1), np.nan)
    out[valid] = pw * p1[valid] + (1 - pw) * p2[valid]
    return out

# ── Data loading ──────────────────────────────────────────────────────────────
print("=" * 70)
print("Ensemble Ablation: Ridge+XGBoost vs Ridge+CatBoost vs ...")
print("=" * 70)
print("\nLoading data...", flush=True)

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
all_results = {}
dm_records  = []

# Determine available ensembles
BASE_MODELS = ['Ridge', 'XGBoost']
if HAS_CATBOOST: BASE_MODELS.append('CatBoost')
if HAS_LGBM:     BASE_MODELS.append('LightGBM')

ENSEMBLE_PAIRS = list(combinations(BASE_MODELS, 2))
print(f"Ensemble pairs: {ENSEMBLE_PAIRS}")

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
    assert len(feats) == 35

    split    = int(len(data) * OUTER_TRAIN_RATIO)
    train_df = data.iloc[:split - hz].copy()
    test_df  = data.iloc[split:].copy()
    y_te     = test_df['Target'].values

    n_tr    = len(train_df)
    v_split = int(n_tr * INNER_TRAIN_RATIO)
    itr_df  = train_df.iloc[:v_split - hz].copy()
    ival_df = train_df.iloc[v_split:].copy()
    y_ival  = ival_df['Target'].values

    sc = StandardScaler().fit(train_df[feats])

    v6_hz = v6_results.get(hz_key, {})

    # Fit/predict individual base models on inner-train → inner-val and on full-train → test
    indiv_ival = {}   # inner-val predictions for pw search
    indiv_test = {}   # test predictions

    for mname in BASE_MODELS:
        if mname == 'CatBoost' and not HAS_CATBOOST: continue
        if mname == 'LightGBM' and not HAS_LGBM:     continue

        bp = v6_hz.get(mname, {}).get('best_params', {})
        if not bp:
            print(f"    {mname}: no stored best_params, skipping")
            continue

        # Fit on inner-train, predict on inner-val
        indiv_ival[mname] = fit_predict_perclass(mname, bp, itr_df, ival_df, sc, feats)
        # Fit on full train, predict on test
        indiv_test[mname] = fit_predict_perclass(mname, bp, train_df, test_df, sc, feats)

        r2_te = pooled_r2(y_te, indiv_test[mname])
        print(f"    [{mname:<10}]  test_R²={r2_te:.4f}")

    hz_res = {}
    ensemble_preds = {}

    for (m1, m2) in ENSEMBLE_PAIRS:
        if m1 not in indiv_test or m2 not in indiv_test:
            continue

        label = f'{m1}+{m2}'
        p1_ival = indiv_ival[m1]; p2_ival = indiv_ival[m2]
        p1_test = indiv_test[m1]; p2_test = indiv_test[m2]

        # pw search on inner val (pw = weight for m1)
        best_pw, best_ival_r2 = find_best_pw(p1_ival, p2_ival, y_ival)

        p_ens = blend(p1_test, p2_test, best_pw)
        r2_te = pooled_r2(y_te, p_ens)
        rm_te = pooled_rmse(y_te, p_ens)

        ensemble_preds[label] = p_ens
        hz_res[label] = {
            'Pooled_R2': round(r2_te, 4),
            'RMSE':      round(rm_te, 4),
            'best_pw':   best_pw,
            'inner_R2':  round(best_ival_r2, 4),
        }
        print(f"    [{label:<22}]  pw={best_pw:.1f}  inner_R²={best_ival_r2:.4f}  test_R²={r2_te:.4f}")

    # DM tests between ensemble pairs
    labels = list(ensemble_preds.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            p1, p2 = ensemble_preds[l1], ensemble_preds[l2]
            dm_stat, p_val, l2_better = dm_test(y_te, p1, p2, h=hz)
            sig = '***' if (p_val is not None and p_val < 0.05) else ''
            print(f"    DM {l1} vs {l2}: stat={dm_stat:+.3f}  p={p_val:.3f} {sig}")
            dm_records.append({
                'Horizon':   hz_key,
                'Model_A':   l1,
                'Model_B':   l2,
                'DM_stat':   round(dm_stat, 4) if not np.isnan(dm_stat) else None,
                'p_value':   round(p_val, 4)   if not np.isnan(p_val)   else None,
                'B_better_5pct': l2_better,
                'R2_A':      hz_res.get(l1, {}).get('Pooled_R2'),
                'R2_B':      hz_res.get(l2, {}).get('Pooled_R2'),
            })

    all_results[hz_key] = hz_res

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = _os.path.join(_RES_DIR, 'ensemble_ablation_results.json')
with open(out_json, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_json}")

# Pivot table
rows = []
for hz_key, hz_data in all_results.items():
    for label, res in hz_data.items():
        rows.append({'Horizon': hz_key, 'Ensemble': label, **res})
df_out = pd.DataFrame(rows)
out_csv = _os.path.join(_CSV_DIR, 'ensemble_ablation_results.csv')
df_out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

df_dm = pd.DataFrame(dm_records)
out_dm = _os.path.join(_CSV_DIR, 'ensemble_ablation_dm.csv')
df_dm.to_csv(out_dm, index=False)
print(f"Saved: {out_dm}")

# Summary pivot
print("\n" + "=" * 70)
print("Pooled R² by Ensemble & Horizon")
print("=" * 70)
if not df_out.empty:
    pivot = df_out.pivot_table(index='Horizon', columns='Ensemble', values='Pooled_R2')
    pivot = pivot.reindex([f'{h}d' for h in HORIZONS])
    print(pivot.round(4).to_string())
