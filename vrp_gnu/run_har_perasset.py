"""
HAR-3 Per-Asset vs DLinear: Isolating Architecture vs Training-Mode
====================================================================
DLinear (Pooled R² ≈ -17 to -21) performs catastrophically worse than
pooled HAR-3 (≈ 0.76 at 22d). Two hypotheses:
  (A) Per-asset training is the problem — DLinear has too few samples per asset
  (B) DLinear architecture (MA decomp + linear) is the problem

This script isolates hypothesis (A) by fitting HAR-3 per-asset (same
training mode as DLinear), then compares:
  - HAR-3 pooled (per-class, all assets in class share one model)
  - HAR-3 per-asset (one model per asset, only that asset's data)
  - DLinear per-asset (loaded from dlinear_benchmark_results.json)

If HAR-3 per-asset ≈ HAR-3 pooled → per-asset mode is fine → DLinear
architecture is the bottleneck (hypothesis B confirmed).
If HAR-3 per-asset << HAR-3 pooled → per-asset training degrades HAR-3
too → DLinear failure is partly explainable by per-asset data sparsity.

Outputs:
  results/har_perasset_results.json
  paper/csv/har_perasset_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE      = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8
RIDGE_ALPHA_GRID  = [10, 50, 100, 500, 1000, 2000]

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

# Load DLinear results for comparison
DLINEAR_JSON = _os.path.join(_RES_DIR, 'dlinear_benchmark_results.json')
try:
    with open(DLINEAR_JSON) as f:
        dlinear_results = json.load(f)
    print(f"Loaded DLinear results: {DLINEAR_JSON}")
except Exception:
    dlinear_results = {}
    print(f"WARNING: DLinear results not found at {DLINEAR_JSON}")

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

def per_asset_r2(y_true, y_pred, assets_arr):
    result = {}
    for a in np.unique(assets_arr):
        m = assets_arr == a
        if m.sum() < 2: continue
        result[a] = round(float(r2_score(y_true[m], y_pred[m])), 4)
    return result

# ── Data loading ──────────────────────────────────────────────────────────────
print("=" * 70)
print("HAR-3 Per-Asset vs DLinear: Architecture vs Training-Mode Analysis")
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

for hz in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*70}")
    hz_key = f'{hz}d'

    # Build pooled dataset (full 35 features for consistency)
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

    sc = StandardScaler().fit(train_df[feats])

    har_idx = [feats.index(f) for f in HAR_FEATS]

    # ── Model A: HAR-3 pooled (per-class) — same as v6 ───────────────────────
    p_har_pooled = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        m = Ridge(alpha=1.0).fit(sc.transform(tr_c[feats])[:, har_idx],
                                  tr_c['Target'].values)
        p_har_pooled[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats])[:, har_idx])

    r2_pool = pooled_r2(y_te, p_har_pooled)
    print(f"  [HAR-3 pooled]    Pooled_R²={r2_pool:.4f}")

    # ── Model B: HAR-3 per-asset — inner holdout alpha tuning per asset ────────
    p_har_asset = np.full(len(test_df), np.nan)
    asset_r2_har = {}

    for asset in ALL_ASSETS:
        te_m  = (test_df['Asset'] == asset).values
        itr_a = itr_df[itr_df['Asset'] == asset]
        ival_a = ival_df[ival_df['Asset'] == asset]
        tr_a  = train_df[train_df['Asset'] == asset]

        if te_m.sum() == 0 or len(tr_a) < 10:
            continue

        # Inner alpha search
        best_alpha, best_r2_in = RIDGE_ALPHA_GRID[0], -np.inf
        if len(itr_a) >= 5 and len(ival_a) >= 5:
            for alpha in RIDGE_ALPHA_GRID:
                Xitr  = sc.transform(itr_a[feats])[:, har_idx]
                Xival = sc.transform(ival_a[feats])[:, har_idx]
                m = Ridge(alpha=alpha).fit(Xitr, itr_a['Target'].values)
                r2 = float(r2_score(ival_a['Target'].values, m.predict(Xival)))
                if r2 > best_r2_in:
                    best_r2_in, best_alpha = r2, alpha

        # Fit on full train for this asset
        Xtr_a = sc.transform(tr_a[feats])[:, har_idx]
        Xte_a = sc.transform(test_df.loc[te_m, feats])[:, har_idx]
        m_final = Ridge(alpha=best_alpha).fit(Xtr_a, tr_a['Target'].values)
        p_har_asset[te_m] = m_final.predict(Xte_a)

        a_r2 = pooled_r2(y_te[te_m], p_har_asset[te_m])
        asset_r2_har[asset] = round(a_r2, 4)
        print(f"    {asset}: alpha={best_alpha}  test_R²={a_r2:.4f}")

    r2_asset = pooled_r2(y_te, p_har_asset)
    median_r2_asset = float(np.median(list(asset_r2_har.values()))) if asset_r2_har else float('nan')
    print(f"  [HAR-3 per-asset] Pooled_R²={r2_asset:.4f}  Median_R²={median_r2_asset:.4f}")

    # ── DLinear from JSON ──────────────────────────────────────────────────────
    dlinear_hz = dlinear_results.get(hz_key, {})
    dlinear_r2 = dlinear_hz.get('Pooled_R2', float('nan'))
    dlinear_med = dlinear_hz.get('Median_R2', float('nan'))
    print(f"  [DLinear per-asset] Pooled_R²={dlinear_r2:.4f}  Median_R²={dlinear_med:.4f}  (from JSON)")

    # ── Gap analysis ──────────────────────────────────────────────────────────
    gap_pool_vs_perasset  = round(r2_pool  - r2_asset, 4)
    gap_har_vs_dlinear    = round(r2_asset - dlinear_r2, 4)
    gap_pool_vs_dlinear   = round(r2_pool  - dlinear_r2, 4)

    print(f"\n  Gap analysis (22d benchmark):")
    print(f"    HAR-3 pooled − HAR-3 per-asset  = {gap_pool_vs_perasset:+.4f}  (pooling benefit)")
    print(f"    HAR-3 per-asset − DLinear        = {gap_har_vs_dlinear:+.4f}  (architecture effect)")
    print(f"    HAR-3 pooled − DLinear           = {gap_pool_vs_dlinear:+.4f}  (total deficit)")

    all_results[hz_key] = {
        'HAR3_pooled':   {'Pooled_R2': round(r2_pool, 4)},
        'HAR3_perasset': {'Pooled_R2': round(r2_asset, 4), 'Median_R2': round(median_r2_asset, 4),
                          'per_asset': asset_r2_har},
        'DLinear':       {'Pooled_R2': round(dlinear_r2, 4), 'Median_R2': round(dlinear_med, 4),
                          'per_asset': dlinear_hz.get('per_asset', {})},
        'gaps': {
            'pooling_benefit':    gap_pool_vs_perasset,
            'architecture_effect': gap_har_vs_dlinear,
            'total_deficit':      gap_pool_vs_dlinear,
        }
    }

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = _os.path.join(_RES_DIR, 'har_perasset_results.json')
with open(out_json, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_json}")

rows = []
for hz_key, res in all_results.items():
    for model_key, label in [('HAR3_pooled', 'HAR-3 Pooled'),
                              ('HAR3_perasset', 'HAR-3 Per-Asset'),
                              ('DLinear', 'DLinear')]:
        r = res.get(model_key, {})
        rows.append({
            'Horizon':  hz_key,
            'Model':    label,
            'Pooled_R2': r.get('Pooled_R2'),
            'Median_R2': r.get('Median_R2'),
        })
    g = res.get('gaps', {})
    rows.append({'Horizon': hz_key, 'Model': 'Gap_pooling',      'Pooled_R2': g.get('pooling_benefit')})
    rows.append({'Horizon': hz_key, 'Model': 'Gap_architecture',  'Pooled_R2': g.get('architecture_effect')})
    rows.append({'Horizon': hz_key, 'Model': 'Gap_total',         'Pooled_R2': g.get('total_deficit')})

df_out = pd.DataFrame(rows)
out_csv = _os.path.join(_CSV_DIR, 'har_perasset_results.csv')
df_out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# Summary
print("\n" + "=" * 70)
print("Gap Analysis: HAR-3 Pooled vs HAR-3 Per-Asset vs DLinear")
print(f"{'Horizon':>8}  {'HAR-3 Pool':>11}  {'HAR-3 Asset':>12}  {'DLinear':>9}"
      f"  {'Pool-Asset':>11}  {'Asset-DLin':>11}")
print("-" * 75)
for hz_key, res in all_results.items():
    p  = res['HAR3_pooled']['Pooled_R2']
    a  = res['HAR3_perasset']['Pooled_R2']
    d  = res['DLinear']['Pooled_R2']
    g  = res['gaps']
    print(f"{hz_key:>8}  {p:>11.4f}  {a:>12.4f}  {d:>9.4f}"
          f"  {g['pooling_benefit']:>+11.4f}  {g['architecture_effect']:>+11.4f}")
