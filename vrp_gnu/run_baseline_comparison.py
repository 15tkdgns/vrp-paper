"""
Baseline Comparison: Naive RV, IV-only, GARCH(1,1) + Diebold-Mariano tests
=============================================================================
Implements paper-methodology baselines for fair comparison against v6 benchmark:

  Naive RV    — Corsi (2009) / Audrino & Knaus (2016): h-day trailing RV
  IV-only     — Carr & Wu (2009) / Bekaert & Hoerova (2014): Ridge on 10 IV feats
  GARCH(1,1)  — Hansen et al. (2022): per-asset GARCH with h-step mean-var formula

DM tests (Harvey 1997 correction, Newey-West bandwidth=h):
  HAR-3 vs each model  (HAR-3 as econometric baseline)
  WEns  vs each model  (WEns as best ML model)

Uses IDENTICAL data splits and feature set as run_main_benchmark_v6.py.
V6 model predictions are reconstructed from stored best_params in JSON.

Outputs:
  results/baseline_comparison_results.json
  paper/csv/baseline_comparison_performance.csv
  paper/csv/dm_test_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from scipy.stats import t as scipy_t
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("WARNING: arch not installed; GARCH will use rolling-std fallback.")

warnings.filterwarnings('ignore')

# ── Config (identical to v6) ─────────────────────────────────────────────────
HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE      = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8    # last 20% of outer-train → inner val
RIDGE_ALPHA_GRID  = [10, 50, 100, 500, 1000, 2000]   # same as v6 Ridge

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
IV_FEATS   = ['IV_VIX', 'IV_VIX_chg', 'IV_VIX_ma5', 'IV_VIX_std5',
              'IV_VIX3M', 'IV_VIX_TermSlope', 'IV_VIX9D', 'IV_VIX_ShortSlope',
              'IV_VRP', 'IV_VRP_ma22']

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = _os.path.join(_SCRIPT_DIR, 'data')
_RES_DIR    = _os.path.join(_SCRIPT_DIR, 'results')
_CSV_DIR    = _os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv')

V6_JSON = _os.path.join(_RES_DIR, 'main_benchmark_v6_results.json')
with open(V6_JSON) as f:
    v6_results = json.load(f)

# ── Feature-Engineering Helpers (identical to v6) ────────────────────────────
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

# ── Evaluation ───────────────────────────────────────────────────────────────
def calc_stats(df_eval, y_true, y_pred):
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()
    if len(y_t) != len(y_p):
        return {}
    valid = ~np.isnan(y_p)
    if valid.sum() < 2:
        return {}
    y_t, y_p = y_t[valid], y_p[valid]
    df_v = df_eval.iloc[valid].reset_index(drop=True)
    pooled = float(r2_score(y_t, y_p))
    rmse   = float(np.sqrt(mean_squared_error(y_t, y_p)))
    per_asset = []
    for a in df_v['Asset'].unique():
        m = (df_v['Asset'] == a).values
        if m.sum() < 2: continue
        per_asset.append(float(r2_score(y_t[m], y_p[m])))
    if not per_asset:
        return {'Pooled_R2': round(pooled, 4), 'RMSE': round(rmse, 4)}
    return {
        'Pooled_R2': round(pooled, 4),
        'Median_R2': round(float(np.median(per_asset)), 4),
        'Mean_R2':   round(float(np.mean(per_asset)), 4),
        'RMSE':      round(rmse, 4),
    }

# ── Diebold-Mariano Test (Harvey 1997 small-sample correction) ───────────────
def dm_test(y_true, p1, p2, h=1):
    """
    DM test: H0: p1 and p2 have equal MSE.
    Positive DM stat → p1 has larger MSE → p2 is better.
    Uses Newey-West autocorrelation correction with bandwidth = h.
    Harvey et al. (1997) small-sample correction applied.
    Returns (dm_stat, p_value, is_p2_better_at_5pct).
    """
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

    # Newey-West variance estimator with bandwidth = h
    bw = max(1, h)
    var_d = np.var(d)
    for k in range(1, bw):
        var_d += 2 * autocov(d, k)
    var_d = max(var_d, 1e-12)   # numerical floor

    # Harvey et al. (1997) small-sample correction
    harvey_adj  = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat     = d_bar / np.sqrt(var_d / n) * harvey_adj
    p_val       = float(2 * (1 - scipy_t.cdf(np.abs(dm_stat), df=n - 1)))
    better_p2   = bool(dm_stat > 0 and p_val < 0.05)

    return float(dm_stat), p_val, better_p2

# ── GARCH (1,1) helpers ───────────────────────────────────────────────────────
def fit_garch_params(train_ret_raw):
    """
    Fit GARCH(1,1) on training returns.
    Returns (omega, alpha, beta, sigma2_final) in original-return² units.
    """
    r = train_ret_raw.values if hasattr(train_ret_raw, 'values') else np.asarray(train_ret_raw)
    if not HAS_ARCH:
        # Rolling-std fallback: constant omega, no persistence
        vol  = np.std(r)
        return vol**2 * (1 - 0.85), 0.05, 0.90, vol**2

    try:
        am  = arch_model(r * 100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        # Parameters in %-squared units; scale back to r² units
        scale = 100 ** 2
        omega = float(res.params['omega']) / scale
        alpha = float(res.params['alpha[1]'])
        beta  = float(res.params['beta[1]'])
        # Final conditional variance (res.conditional_volatility is % std-dev)
        sigma2_last = float(res.conditional_volatility[-1]) ** 2 / scale
        return omega, alpha, beta, sigma2_last
    except Exception:
        vol = np.std(r)
        return vol**2 * (1 - 0.85), 0.05, 0.90, vol**2


def garch_hstep_mean_var(omega, alpha, beta, sigma2_t1, h):
    """
    Expected mean variance over h steps ahead, given 1-step-ahead variance sigma2_t1.
    Formula: long_run + (sigma2_t1 - long_run) * (1 - persistence^h) / (h*(1-persistence))
    """
    persistence = alpha + beta
    if abs(1.0 - persistence) < 1e-6:
        return sigma2_t1  # integrated GARCH edge case
    long_run = omega / (1.0 - persistence)
    decay    = (1.0 - persistence ** h) / (h * (1.0 - persistence))
    return long_run + (sigma2_t1 - long_run) * decay


def compute_garch_preds_asset(train_ret, test_ret, hz):
    """
    Rolling GARCH(1,1) predictions for one asset's test set.
    At test day i: predict using sigma2 updated with r_{test[i]} (end-of-day info),
    then forecast h-step mean variance for days i+1 ... i+h.
    Returns log(mean_var_h * 252 + 1e-12) aligned to test rows.
    """
    omega, alpha, beta, sigma2 = fit_garch_params(train_ret)
    preds = []
    for r_t in (test_ret.values if hasattr(test_ret, 'values') else np.asarray(test_ret)):
        # Update sigma2 with today's realized return (available at day close)
        sigma2 = omega + alpha * r_t**2 + beta * sigma2
        mean_var_h = garch_hstep_mean_var(omega, alpha, beta, sigma2, hz)
        preds.append(np.log(max(mean_var_h * 252, 1e-12)))
    return np.array(preds)


# ── Model Factory (subset, no BiLSTM-A) ─────────────────────────────────────
def make_model(name, cfg):
    if name == 'Ridge':
        return Ridge(alpha=cfg['alpha'])
    elif name == 'LASSO':
        return Lasso(alpha=cfg['alpha'], max_iter=5000)
    elif name == 'ENet':
        return ElasticNet(alpha=cfg['alpha'], l1_ratio=cfg['l1_ratio'], max_iter=5000)
    elif name == 'RF':
        return RandomForestRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], max_features='sqrt',
            min_samples_leaf=cfg['min_samples_leaf'], n_jobs=-1, random_state=RANDOM_STATE)
    elif name == 'XGBoost':
        return XGBRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            min_child_weight=5, random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
            tree_method='hist')
    elif name == 'LightGBM' and HAS_LGBM:
        return LGBMRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
            num_leaves=cfg['num_leaves'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, min_child_samples=5,
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    elif name == 'MLP':
        return MLPRegressor(
            hidden_layer_sizes=cfg['hidden_layer_sizes'], alpha=cfg['alpha'],
            max_iter=500, early_stopping=True, n_iter_no_change=20,
            random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")


def predict_perclass(final_models, test_df, sc, feats):
    preds = np.full(len(test_df), np.nan)
    for cls, m in final_models.items():
        if m is None: continue
        mask = (test_df['Class'] == cls).values
        if mask.sum() == 0: continue
        preds[mask] = m.predict(sc.transform(test_df.loc[mask, feats]))
    return preds


# ── Data Loading (identical to v6) ───────────────────────────────────────────
print("=" * 70)
print("Baseline Comparison: NaiveRV / IV-only / GARCH(1,1) + DM tests")
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
vix   = raw[('Close', 'VIX')]
spy_c = raw[('Close', 'SPY')]
spy_r = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_r**2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

iv_features = {
    'VIX':            np.log(vix + 1e-6),
    'VIX_chg':        np.log(vix + 1e-6).diff(),
    'VIX_ma5':        np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5':       np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M':          np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope':  np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D':          np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope': np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp_val = (vix**2 / 100) - spy_rv / 10000
iv_features['VRP']      = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

asset_frames = {}
asset_ret    = {}   # raw daily log-returns for GARCH
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
        'Corr_SPY':    (ret.rolling(22).corr(spy_r.reindex(ret.index)).shift(1)
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
    feat['AltVol_Amihud']           = (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']        = (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1)
    feat['AltVol_PV_Corr']          = ret.rolling(22).corr(np.log(v + 1)).shift(1)
    feat['AltVol_Vol_Surprise']     = ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1)
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    feat['AltVol_Order_Imbalance']  = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']      = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)

    d = pd.DataFrame(feat)
    d['ret_sq']   = ret_sq       # keep for Naive RV target alignment
    d['ret_raw']  = ret          # keep for GARCH
    d['Asset']    = asset
    d['Class']    = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    asset_frames[asset] = d
    asset_ret[asset]    = ret    # daily log-returns

print(f"Assets: {len(asset_frames)}, IV features: {len(IV_FEATS)}")

# ── Main Loop ─────────────────────────────────────────────────────────────────
all_results  = {}   # horizon → model → metrics
all_preds    = {}   # horizon → model → np.array (aligned to test_df)
dm_results   = []   # list of dicts for DM test CSV

MODELS_V6 = ['HAR-3', 'Ridge', 'LASSO', 'ENet', 'RF', 'XGBoost', 'WEns']
if HAS_LGBM:
    MODELS_V6_EXTRA = ['LightGBM', 'MLP']
else:
    MODELS_V6_EXTRA = ['MLP']
MODELS_NEW = ['NaiveRV', 'RepeatRV', 'IV-only', 'GARCH']
ALL_MODEL_NAMES = MODELS_V6 + MODELS_V6_EXTRA + MODELS_NEW

# Only reconstruct fast models for DM tests (skip RF/LightGBM/MLP which are slow)
# RF/LightGBM/MLP metrics come from v6 JSON; DM tests not reported for them
DM_RECONSTRUCT = ['Ridge', 'LASSO', 'ENet', 'XGBoost']  # fast linear + tree


for hz in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*70}")
    hz_key = f'{hz}d'

    # ── Build pooled dataset (same as v6) ──
    pooled = []
    for asset in ALL_ASSETS:
        df = asset_frames[asset].copy()
        df['Target']  = forward_rv(df['ret_sq'], hz)
        # Naive RV: h-day trailing mean ret_sq (past), log-annualised — horizon-matched
        df['NaiveRV_pred'] = np.log(df['ret_sq'].rolling(hz).mean().shift(1) * 252 + 1e-6)
        # Repeat RV: fixed 22d trailing mean, applied to all horizons — horizon-agnostic
        df['RepeatRV_pred'] = np.log(df['ret_sq'].rolling(22).mean().shift(1) * 252 + 1e-6)
        df = df.drop(columns=['ret_sq', 'ret_raw']).dropna()
        pooled.append(df)

    data  = pd.concat(pooled).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns
             if c not in ['Target', 'Asset', 'Class', 'NaiveRV_pred', 'RepeatRV_pred']]
    assert len(feats) == 35, f"Expected 35 features, got {len(feats)}"

    # ── Outer split with purge (identical to v6) ──
    split    = int(len(data) * OUTER_TRAIN_RATIO)
    train_df = data.iloc[:split - hz].copy()
    test_df  = data.iloc[split:].copy()
    y_te     = test_df['Target'].values

    sc   = StandardScaler().fit(train_df[feats])
    X_tr = sc.transform(train_df[feats])
    X_te = sc.transform(test_df[feats])
    y_tr = train_df['Target'].values

    # Inner holdout (for IV-only tuning)
    n_tr    = len(train_df)
    v_split = int(n_tr * INNER_TRAIN_RATIO)
    itr_df  = train_df.iloc[:v_split - hz].copy()
    ival_df = train_df.iloc[v_split:].copy()

    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    hz_res   = {}
    hz_preds = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Naive RV
    # ──────────────────────────────────────────────────────────────────────────
    p_naive = test_df['NaiveRV_pred'].values
    hz_preds['NaiveRV'] = p_naive
    hz_res['NaiveRV']   = calc_stats(test_df, y_te, p_naive)
    print(f"  [NaiveRV]  → Pooled_R2={hz_res['NaiveRV'].get('Pooled_R2', float('nan')):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1b. Repeat RV (fixed 22d window, horizon-agnostic)
    # ──────────────────────────────────────────────────────────────────────────
    p_repeat = test_df['RepeatRV_pred'].values
    hz_preds['RepeatRV'] = p_repeat
    hz_res['RepeatRV']   = calc_stats(test_df, y_te, p_repeat)
    print(f"  [RepeatRV] → Pooled_R2={hz_res['RepeatRV'].get('Pooled_R2', float('nan')):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. IV-only  (Ridge on 10 IV features, per-class inner-holdout tuning)
    # ──────────────────────────────────────────────────────────────────────────
    iv_idx = [feats.index(f) for f in IV_FEATS]
    iv_best_params  = {}
    iv_final_models = {}

    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class'] == cls]
        ival_c = ival_df[ival_df['Class'] == cls]
        tr_c   = train_df[train_df['Class'] == cls]

        if len(itr_c) < 30 or len(ival_c) < 10:
            iv_best_params[cls] = {'alpha': 100}
        else:
            X_itr_c  = sc.transform(itr_c[feats])[:, iv_idx]
            X_ival_c = sc.transform(ival_c[feats])[:, iv_idx]
            best_r2, best_a = -np.inf, 100
            for a in RIDGE_ALPHA_GRID:
                m = Ridge(alpha=a).fit(X_itr_c, itr_c['Target'].values)
                r2 = float(r2_score(ival_c['Target'].values, m.predict(X_ival_c)))
                if r2 > best_r2:
                    best_r2, best_a = r2, a
            iv_best_params[cls] = {'alpha': best_a}

        X_tr_c = sc.transform(tr_c[feats])[:, iv_idx]
        m_final = Ridge(alpha=iv_best_params[cls]['alpha']).fit(X_tr_c, tr_c['Target'].values)
        iv_final_models[cls] = m_final

    p_iv = np.full(len(test_df), np.nan)
    for cls, m in iv_final_models.items():
        mask = (test_df['Class'] == cls).values
        if mask.sum() == 0: continue
        p_iv[mask] = m.predict(sc.transform(test_df.loc[mask, feats])[:, iv_idx])

    hz_preds['IV-only'] = p_iv
    hz_res['IV-only']   = calc_stats(test_df, y_te, p_iv)
    hz_res['IV-only']['best_params'] = iv_best_params
    print(f"  [IV-only]  params={iv_best_params}  "
          f"→ Pooled_R2={hz_res['IV-only'].get('Pooled_R2', float('nan')):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. GARCH(1,1)  per-asset, h-step mean-variance formula
    # ──────────────────────────────────────────────────────────────────────────
    p_garch = np.full(len(test_df), np.nan)

    for asset in ALL_ASSETS:
        # Identify train / test rows for this asset in pooled data
        asset_mask_tr = (train_df['Asset'] == asset)
        asset_mask_te = (test_df['Asset'] == asset)
        if asset_mask_tr.sum() < 50 or asset_mask_te.sum() == 0:
            continue

        # Get raw daily returns via asset_ret (aligned by date to asset_frames)
        # We need returns for exactly the dates in train and test splits.
        # Use asset_frames dates that survived dropna → match via positional index.
        # Simpler: recompute from ret series for this asset using date alignment.
        ret_series = asset_ret[asset]  # full history daily returns

        # Dates present in train / test for this asset
        # (data was sorted by date-index before reset_index → use original Asset rows)
        # We need the ORIGINAL date index for GARCH alignment.
        # Rebuild asset-specific df with dates to extract ret values.
        asset_data = asset_frames[asset].copy()
        asset_data['Target'] = forward_rv(asset_data['ret_sq'], hz)
        asset_data = asset_data.dropna()  # same dates as pooled (before concat)

        n_full      = len(asset_data)
        split_asset = int(n_full * OUTER_TRAIN_RATIO)
        train_dates = asset_data.index[:split_asset - hz]
        test_dates  = asset_data.index[split_asset:]

        train_ret_asset = ret_series.reindex(train_dates).dropna()
        test_ret_asset  = ret_series.reindex(test_dates).dropna()
        if len(train_ret_asset) < 50:
            continue

        garch_preds_asset = compute_garch_preds_asset(train_ret_asset, test_ret_asset, hz)

        # Map back: find positions in test_df for this asset's test_dates
        common_dates = test_ret_asset.index.intersection(test_dates)
        for i, d in enumerate(test_ret_asset.index):
            rows = (test_df['Asset'] == asset)
            # Match by finding asset rows in test_df order
        # Simpler: test_df has reset_index, so match by Asset + sequential order
        te_rows = np.where(asset_mask_te.values)[0]
        n_match = min(len(te_rows), len(garch_preds_asset))
        p_garch[te_rows[:n_match]] = garch_preds_asset[:n_match]

    hz_preds['GARCH'] = p_garch
    hz_res['GARCH']   = calc_stats(test_df, y_te, p_garch)
    print(f"  [GARCH]    → Pooled_R2={hz_res['GARCH'].get('Pooled_R2', float('nan')):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Reconstruct V6 model predictions using stored best_params
    # ──────────────────────────────────────────────────────────────────────────
    v6_hz = v6_results.get(hz_key, {})

    # HAR-3 (fixed, per-class, alpha=1.0)
    har_idx = [feats.index(f) for f in HAR_FEATS]
    p_har   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        m = Ridge(alpha=1.0).fit(sc.transform(tr_c[feats])[:, har_idx], tr_c['Target'].values)
        p_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats])[:, har_idx])
    hz_preds['HAR-3'] = p_har
    hz_res['HAR-3']   = v6_hz.get('HAR-3', calc_stats(test_df, y_te, p_har))
    print(f"  [HAR-3]    → Pooled_R2={hz_res['HAR-3'].get('Pooled_R2', float('nan')):.4f}")

    # Load v6 metrics for slow models directly from JSON (no refit needed)
    for mname in (['RF', 'BiLSTM-A'] + (['LightGBM'] if HAS_LGBM else []) + ['MLP']):
        hz_res[mname]   = v6_hz.get(mname, {})
        hz_preds[mname] = np.full(len(test_df), np.nan)  # no DM test for these

    # Reconstruct fast models for DM tests using stored best_params
    stored_best = {}
    for mname in DM_RECONSTRUCT:
        v6_m = v6_hz.get(mname, {})
        bp   = v6_m.get('best_params', None)
        if bp is None:
            hz_preds[mname] = np.full(len(test_df), np.nan)
            hz_res[mname]   = v6_m
            continue

        final_models = {}
        for cls in ASSET_GROUPS:
            tr_c = train_df[train_df['Class'] == cls]
            if len(tr_c) < 10:
                final_models[cls] = None; continue
            cfg = bp.get(cls, {})
            if not cfg:
                final_models[cls] = None; continue
            # Convert list values back to tuples for MLP hidden_layer_sizes
            if mname == 'MLP' and 'hidden_layer_sizes' in cfg:
                cfg = dict(cfg)
                cfg['hidden_layer_sizes'] = tuple(cfg['hidden_layer_sizes'])
            try:
                m = make_model(mname, cfg)
                m.fit(sc.transform(tr_c[feats]), tr_c['Target'].values)
                final_models[cls] = m
            except Exception as e:
                print(f"    WARNING: {mname} {cls} refit failed: {e}")
                final_models[cls] = None

        preds = predict_perclass(final_models, test_df, sc, feats)
        hz_preds[mname]  = preds
        hz_res[mname]    = v6_hz.get(mname, calc_stats(test_df, y_te, preds))
        stored_best[mname] = bp
        print(f"  [{mname:<10}]  → Pooled_R2={hz_res[mname].get('Pooled_R2', float('nan')):.4f}")

    # WEns: use stored pw and per-class Ridge + XGBoost models
    wens_v6 = v6_hz.get('WEns', {})
    wens_bp = wens_v6.get('best_params', {})
    if wens_bp and 'Ridge' in hz_preds and 'XGBoost' in hz_preds:
        pw = wens_bp.get('pw', 0.7)
        p_r, p_x = hz_preds['Ridge'], hz_preds['XGBoost']
        valid = ~(np.isnan(p_r) | np.isnan(p_x))
        p_wens = np.full(len(test_df), np.nan)
        p_wens[valid] = pw * p_r[valid] + (1 - pw) * p_x[valid]
        hz_preds['WEns'] = p_wens
        hz_res['WEns']   = wens_v6  # use stored metrics (same model)
        print(f"  [WEns]     pw={pw}  → Pooled_R2={hz_res['WEns'].get('Pooled_R2', float('nan')):.4f}")
    else:
        hz_preds['WEns'] = np.full(len(test_df), np.nan)
        hz_res['WEns']   = wens_v6

    all_results[hz_key]  = hz_res
    all_preds[hz_key]    = hz_preds

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Diebold-Mariano Tests
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n  DM Tests (horizon={hz}d, Harvey-corrected, BW={hz}):")
    reference_models = ['HAR-3', 'WEns']
    # Only test models that have actual predictions (not NaN-filled placeholders)
    DM_TEST_MODELS = ['HAR-3', 'Ridge', 'LASSO', 'ENet', 'XGBoost', 'WEns',
                      'NaiveRV', 'RepeatRV', 'IV-only', 'GARCH']

    for ref in reference_models:
        p_ref = hz_preds.get(ref)
        if p_ref is None or np.isnan(p_ref).all():
            continue
        for mname in DM_TEST_MODELS:
            if mname == ref: continue
            p_m = hz_preds.get(mname)
            if p_m is None or np.isnan(p_m).all():
                continue
            dm_stat, p_val, ref_worse = dm_test(y_te, p_ref, p_m, h=hz)
            dm_results.append({
                'Horizon':       hz_key,
                'Reference':     ref,
                'Challenger':    mname,
                'DM_stat':       round(dm_stat, 4) if not np.isnan(dm_stat) else None,
                'p_value':       round(p_val, 4) if not np.isnan(p_val) else None,
                'Ref_worse_5pct': ref_worse,   # True → challenger significantly better
                'Ref_Pooled_R2': hz_res.get(ref, {}).get('Pooled_R2'),
                'Chal_Pooled_R2': hz_res.get(mname, {}).get('Pooled_R2'),
            })
            sig = '***' if (p_val is not None and p_val < 0.05) else ''
            print(f"    {ref} vs {mname:<12}: DM={dm_stat:+.3f}  p={p_val:.3f} {sig}")

    # Horizon summary table
    print(f"\n  {'Model':<14} {'Pooled_R2':>10} {'Median_R2':>10} {'Mean_R2':>10} {'RMSE':>8}")
    print(f"  {'-'*56}")
    for name in ALL_MODEL_NAMES:
        res = hz_res.get(name, {})
        pr  = res.get('Pooled_R2',  float('nan'))
        mr  = res.get('Median_R2',  float('nan'))
        mnr = res.get('Mean_R2',    float('nan'))
        rm  = res.get('RMSE',       float('nan'))
        print(f"  {name:<14} {pr:>10.4f} {mr:>10.4f} {mnr:>10.4f} {rm:>8.4f}")


# ── Save Outputs ──────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

# JSON
out_json = _os.path.join(_RES_DIR, 'baseline_comparison_results.json')
# Serialize: convert numpy types
def _safe(v):
    if isinstance(v, (np.floating, np.integer)): return float(v)
    if isinstance(v, dict): return {k: _safe(vv) for k, vv in v.items()}
    return v

with open(out_json, 'w') as f:
    json.dump({hz: {m: _safe(res) for m, res in hd.items()}
               for hz, hd in all_results.items()}, f, indent=2)
print(f"\nSaved: {out_json}")

# Performance CSV
rows = []
for hz_key, hz_data in all_results.items():
    for model_name, res in hz_data.items():
        rows.append({
            'Model':     model_name,
            'Horizon':   hz_key,
            'Pooled_R2': res.get('Pooled_R2'),
            'Median_R2': res.get('Median_R2'),
            'Mean_R2':   res.get('Mean_R2'),
            'RMSE':      res.get('RMSE'),
        })
df_perf = pd.DataFrame(rows)
out_perf = _os.path.join(_CSV_DIR, 'baseline_comparison_performance.csv')
df_perf.to_csv(out_perf, index=False)
print(f"Saved: {out_perf}")

# DM Test CSV
df_dm = pd.DataFrame(dm_results)
out_dm = _os.path.join(_CSV_DIR, 'dm_test_results.csv')
df_dm.to_csv(out_dm, index=False)
print(f"Saved: {out_dm}")

# Pooled_R2 pivot
pivot = df_perf.pivot_table(index='Horizon', columns='Model', values='Pooled_R2')
pivot = pivot.reindex([f'{h}d' for h in HORIZONS])
print("\nPooled_R2 Summary:")
print(pivot.round(4).to_string())

# DM summary: WEns vs each model at 22d
print("\nDM Tests — WEns vs each model @ 22d (p<0.05 = significant):")
dm22 = df_dm[(df_dm['Horizon'] == '22d') & (df_dm['Reference'] == 'WEns')]
for _, row in dm22.iterrows():
    sig = '***' if row['Ref_worse_5pct'] else ('  ' if not row['Ref_worse_5pct'] else '')
    print(f"  WEns vs {row['Challenger']:<12}: DM={row['DM_stat']:+.3f}  p={row['p_value']:.3f} {sig}")
