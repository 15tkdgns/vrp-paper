"""
DM Test: WEns vs Random Forest across all horizons
====================================================
Reconstructs WEns (Ridge + XGBoost) and RF predictions using stored
best_params from main_benchmark_v6_results.json, then runs Harvey(1997)-
corrected Diebold-Mariano tests across all 8 horizons.

Motivation: The paper claims that at 1d the RF(0.1177)–WEns(0.1176) gap
is "not statistically distinguishable". This script provides the DM stat.

Outputs:
  results/dm_wens_vs_rf_results.json
  paper/csv/dm_wens_vs_rf_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from scipy.stats import t as scipy_t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE      = 42
OUTER_TRAIN_RATIO = 0.8

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = os.path.join(_SCRIPT_DIR, 'data')
_RES_DIR    = os.path.join(_SCRIPT_DIR, 'results')
_CSV_DIR    = os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv')

V6_JSON = os.path.join(_RES_DIR, 'main_benchmark_v6_results.json')
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

# ── DM test (Harvey 1997, same as run_dm_extended.py) ─────────────────────────
def dm_test(y_true, p1, p2, h=1):
    """Positive DM → p1 has larger MSE → p2 is better than p1."""
    e1 = (y_true - p1) ** 2
    e2 = (y_true - p2) ** 2
    d  = e1 - e2
    valid = ~(np.isnan(e1) | np.isnan(e2))
    d = d[valid]
    if len(d) < 10:
        return np.nan, np.nan
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
    return float(dm_stat), p_val

# ── Model factories ───────────────────────────────────────────────────────────
def make_ridge(cfg):
    return Ridge(alpha=cfg['alpha'])

def make_xgb(cfg):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=5, random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
        device='cuda', tree_method='hist')

def make_rf(cfg):
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=cfg['max_depth'],
        min_samples_leaf=cfg['min_samples_leaf'],
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1)

def fit_predict_perclass(make_fn, bp, train_df, test_df, sc, feats):
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
print("DM Test: WEns vs Random Forest across all horizons")
print("=" * 70)
print("\nLoading data & building 35 features...", flush=True)

def _load_from_parquet():
    vix_df = pd.read_parquet(os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames = {}
    for asset in ALL_ASSETS:
        p = os.path.join(_DATA_DIR, f'{asset}.parquet')
        if os.path.exists(p):
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
records = []

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

    # ── WEns: Ridge + XGBoost ─────────────────────────────────────────────────
    wens_bp  = v6_hz.get('WEns', {}).get('best_params', {})
    pw       = wens_bp.get('pw', 0.7)
    ridge_bp = wens_bp.get('Ridge', {})
    xgb_bp   = wens_bp.get('XGBoost', {})

    p_ridge = fit_predict_perclass(make_ridge, ridge_bp, train_df, test_df, sc, feats)
    p_xgb   = fit_predict_perclass(make_xgb,   xgb_bp,   train_df, test_df, sc, feats)

    valid  = ~(np.isnan(p_ridge) | np.isnan(p_xgb))
    p_wens = np.full(len(test_df), np.nan)
    p_wens[valid] = pw * p_ridge[valid] + (1 - pw) * p_xgb[valid]

    wens_r2 = pooled_r2(y_te, p_wens)
    print(f"  [WEns]  pw={pw:.1f}  → Pooled_R2={wens_r2:.4f}")

    # ── RF ────────────────────────────────────────────────────────────────────
    rf_bp = v6_hz.get('RF', {}).get('best_params', {})
    print(f"  [RF]    fitting per-class...", flush=True)
    p_rf  = fit_predict_perclass(make_rf, rf_bp, train_df, test_df, sc, feats)
    rf_r2 = pooled_r2(y_te, p_rf)
    print(f"  [RF]    → Pooled_R2={rf_r2:.4f}")

    # ── DM test: WEns (ref=p1) vs RF (challenger=p2) ─────────────────────────
    # Convention: negative DM → WEns has smaller MSE → WEns is better
    dm_stat, p_val = dm_test(y_te, p_wens, p_rf, h=hz)
    sig = '***' if (not np.isnan(p_val) and p_val < 0.05) else ''
    print(f"\n  DM Test WEns vs RF (h={hz}, Harvey-corrected, BW={hz}):")
    print(f"    DM={dm_stat:+.3f}  p={p_val:.3f} {sig}")
    print(f"    (negative DM = WEns has smaller MSE = WEns better)")

    records.append({
        'Horizon':      hz_key,
        'WEns_R2':      round(wens_r2, 4),
        'RF_R2':        round(rf_r2, 4),
        'pw':           pw,
        'DM_stat':      round(dm_stat, 4) if not np.isnan(dm_stat) else None,
        'p_value':      round(p_val, 4)   if not np.isnan(p_val)   else None,
        'WEns_better':  bool(dm_stat < 0 and p_val < 0.05),
        'RF_better':    bool(dm_stat > 0 and p_val < 0.05),
    })

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = os.path.join(_RES_DIR, 'dm_wens_vs_rf_results.json')
with open(out_json, 'w') as f:
    json.dump(records, f, indent=2)
print(f"\nSaved: {out_json}")

df_out = pd.DataFrame(records)
out_csv = os.path.join(_CSV_DIR, 'dm_wens_vs_rf_results.csv')
df_out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Summary: WEns vs RF DM Tests")
print("=" * 70)
print(f"{'Horizon':>8} {'WEns_R2':>9} {'RF_R2':>8} {'DM_stat':>9} {'p_value':>8} {'Sig':>4} {'Better':>8}")
for r in records:
    dm  = r['DM_stat']  if r['DM_stat']  is not None else float('nan')
    pv  = r['p_value']  if r['p_value']  is not None else float('nan')
    sig = '***' if (not np.isnan(pv) and pv < 0.05) else ''
    better = 'WEns' if r['WEns_better'] else ('RF' if r['RF_better'] else 'n.s.')
    print(f"  {r['Horizon']:>6} {r['WEns_R2']:>9.4f} {r['RF_R2']:>8.4f}"
          f" {dm:>+9.3f} {pv:>8.3f} {sig:>4} {better:>8}")
