"""
Feature analysis v6 pipeline:
1. Group removal experiment (WEns, 22d) — Table 9
2. Kendall's W across horizons
3. Multi-horizon feature importance (top features per horizon)
"""
import json, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

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

def kendall_w(rankings):
    """rankings: (n_raters, n_items) array"""
    m, n = rankings.shape
    rank_sums = rankings.sum(axis=0)
    mean_rank = rank_sums.mean()
    S = ((rank_sums - mean_rank)**2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    # chi-squared test
    chi2 = m * (n - 1) * W
    p = 1 - stats.chi2.cdf(chi2, df=n-1)
    return float(W), float(p)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
raw     = pd.read_pickle('src/data/v71_ohlcv_cache.pkl')
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
        'AltVol_Amihud':       (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1),
        'AltVol_Vol_Ratio':    (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1),
        'AltVol_PV_Corr':      ret.rolling(22).corr(np.log(v + 1)).shift(1),
        'AltVol_Vol_Surprise': ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1),
    })
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    ft['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    ft['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
    d = pd.DataFrame(ft); d['ret_sq'] = rs2; d['Asset'] = asset
    d['Class'] = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    af[asset] = d

with open('results/main_benchmark_v6_results.json') as f: saved = json.load(f)
print("Data loaded.", flush=True)

# ── Helper: build panel ──────────────────────────────────────────────────────
def build_panel(hz):
    pool = []
    for asset in ALL_ASSETS:
        df = af[asset].copy(); df['Target'] = forward_rv(df['ret_sq'], hz)
        df = df.drop(columns=['ret_sq']).dropna(); pool.append(df)
    data = pd.concat(pool).sort_index().reset_index(drop=True)
    fs = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    sp = int(len(data) * 0.8)
    tr = data.iloc[:sp - hz].copy()
    te = data.iloc[sp:].copy()
    return tr, te, fs

def fit_wens(tr, te, fs, hz):
    bpr = saved[f'{hz}d']['Ridge']['best_params']
    bpw = saved[f'{hz}d']['WEns']['best_params']
    pw  = bpw.get('pw', 0.8)
    sc  = StandardScaler().fit(tr[fs].values)
    X_tr = sc.transform(tr[fs]); X_te = sc.transform(te[fs])

    pr2 = np.full(len(te), np.nan)
    px  = np.full(len(te), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        alpha = bpr.get(cls, {}).get('alpha', 1000)
        m = Ridge(alpha=alpha).fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        pr2[tem] = m.predict(X_te[te['Class'].values == cls])

        xp = bpw.get('XGBoost', {}).get(cls, {'max_depth': 3, 'learning_rate': 0.03})
        mx = XGBRegressor(n_estimators=300, random_state=42, verbosity=0,
                          max_depth=xp.get('max_depth', 3), learning_rate=xp.get('learning_rate', 0.03))
        mx.fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        px[tem] = mx.predict(X_te[te['Class'].values == cls])

    return pw * pr2 + (1 - pw) * px, sc, X_tr, X_te

def pooled_r2(te, preds):
    valid = ~np.isnan(preds) & ~np.isnan(te['Target'].values)
    return float(r2_score(te['Target'].values[valid], preds[valid]))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Group Removal Experiment (WEns, 22d) — Table 9
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. Group Removal Experiment (WEns, 22d)")
print("="*60)

tr22, te22, fs22 = build_panel(22)
full_preds, sc22, X_tr22, X_te22 = fit_wens(tr22, te22, fs22, 22)
full_r2 = pooled_r2(te22, full_preds)
print(f"Full WEns 22d: {full_r2:.4f}")

GROUPS = {
    'IV Surface':  [f for f in fs22 if f.startswith('IV_')],
    'HF Proxy':    ['Parkinson_5', 'Parkinson_22', 'GarmanKlass_22',
                    'RogersSatchell_22', 'Range_Close_Ratio', 'Overnight_Vol', 'Overnight_Ret'],
    'Alternative': ['AltVol_Amihud', 'AltVol_Vol_Ratio', 'AltVol_PV_Corr',
                    'AltVol_Vol_Surprise', 'AltVol_Order_Imbalance', 'AltVol_Kyle_Lambda'],
}

removal_results = {}
bpr22 = saved['22d']['Ridge']['best_params']
bpw22 = saved['22d']['WEns']['best_params']
pw22  = bpw22.get('pw', 0.8)

for grp_name, grp_feats in GROUPS.items():
    reduced_fs = [f for f in fs22 if f not in grp_feats]
    ridx = [fs22.index(f) for f in reduced_fs]
    sc_r = StandardScaler().fit(tr22[fs22].values)
    Xtr_r = sc_r.transform(tr22[fs22])[:, ridx]
    Xte_r = sc_r.transform(te22[fs22])[:, ridx]

    pr2_r = np.full(len(te22), np.nan)
    px_r  = np.full(len(te22), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr22[tr22['Class'] == cls]; tem = (te22['Class'] == cls).values
        if len(trc) < 5: continue
        alpha = bpr22.get(cls, {}).get('alpha', 1000)
        m = Ridge(alpha=alpha).fit(Xtr_r[tr22['Class'].values == cls], trc['Target'].values)
        pr2_r[tem] = m.predict(Xte_r[te22['Class'].values == cls])
        xp = bpw22.get('XGBoost', {}).get(cls, {'max_depth': 3, 'learning_rate': 0.03})
        mx = XGBRegressor(n_estimators=300, random_state=42, verbosity=0,
                          max_depth=xp.get('max_depth', 3), learning_rate=xp.get('learning_rate', 0.03))
        mx.fit(Xtr_r[tr22['Class'].values == cls], trc['Target'].values)
        px_r[tem] = mx.predict(Xte_r[te22['Class'].values == cls])

    preds_r = pw22 * pr2_r + (1 - pw22) * px_r
    r2_r = pooled_r2(te22, preds_r)
    delta = full_r2 - r2_r
    print(f"  Remove {grp_name:<14} ({len(grp_feats)} feats): {r2_r:.4f}  delta={delta:+.4f}")
    removal_results[grp_name] = {
        'n_removed': len(grp_feats),
        'full_r2': round(full_r2, 4),
        'reduced_r2': round(r2_r, 4),
        'delta': round(delta, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-horizon Feature Importance + Kendall's W
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. Multi-horizon Feature Importance (Ridge permutation) + Kendall's W")
print("="*60)

hz_importance = {}
kendall_results = {}

for hz in HORIZONS:
    print(f"\nHorizon: {hz}d", flush=True)
    tr, te, fs = build_panel(hz)
    bpr = saved[f'{hz}d']['Ridge']['best_params']
    sc  = StandardScaler().fit(tr[fs].values)
    X_tr = sc.transform(tr[fs]); X_te = sc.transform(te[fs])
    y_te = te['Target'].values

    # Per-class Ridge fit
    ridge_coefs = np.zeros(len(fs))
    xgb_gain    = np.zeros(len(fs))
    perm_imp    = np.zeros(len(fs))

    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        alpha = bpr.get(cls, {}).get('alpha', 1000)
        m = Ridge(alpha=alpha).fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        ridge_coefs += np.abs(m.coef_)

        mx = XGBRegressor(n_estimators=300, random_state=42, verbosity=0,
                          max_depth=3, learning_rate=0.03)
        mx.fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        xgb_gain += mx.feature_importances_

        # Permutation importance (simplified: shuffle each feature)
        base_r2 = r2_score(y_te[tem], m.predict(X_te[te['Class'].values == cls]))
        for fi in range(len(fs)):
            Xte_perm = X_te[te['Class'].values == cls].copy()
            np.random.seed(42); np.random.shuffle(Xte_perm[:, fi])
            perm_r2 = r2_score(y_te[tem], m.predict(Xte_perm))
            perm_imp[fi] += max(0, base_r2 - perm_r2)

    # Normalize each method to sum=1 (importance share)
    def normalize(scores):
        s = scores.sum()
        return scores / s if s > 0 else scores

    n_ridge = normalize(ridge_coefs)
    n_xgb   = normalize(xgb_gain)
    n_perm  = normalize(perm_imp)

    # Consensus score = mean of 3 normalized scores (higher = more important)
    consensus_score = (n_ridge + n_xgb + n_perm) / 3

    # Also compute ranks for Kendall's W
    def to_rank(scores):
        order = np.argsort(-scores)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, len(scores)+1)
        return ranks

    r_ridge = to_rank(ridge_coefs)
    r_xgb   = to_rank(xgb_gain)
    r_perm  = to_rank(perm_imp)

    # Kendall's W across 3 methods
    rankings = np.stack([r_ridge, r_xgb, r_perm])  # (3, n_feats)
    W, p = kendall_w(rankings)
    kendall_results[f'{hz}d'] = {'W': round(W, 4), 'p': round(p, 6)}
    print(f"  Kendall W={W:.4f}  p={p:.4f}")

    # Consensus rank (mean of 3) — keep for backward compatibility
    consensus_rank = (r_ridge + r_xgb + r_perm) / 3
    top10_idx = np.argsort(consensus_rank)[:10]
    top10 = [(fs[i], round(float(consensus_rank[i]), 3)) for i in top10_idx]
    hz_importance[f'{hz}d'] = top10

    # Save full importance scores for all features (as percentage * 100)
    hz_importance[f'{hz}d_scores'] = {
        fs[i]: round(float(consensus_score[i]) * 100, 4) for i in range(len(fs))
    }
    print(f"  Top-5: {[t[0] for t in top10[:5]]}")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
output = {
    'group_removal': removal_results,
    'kendall_w': kendall_results,
    'hz_importance': hz_importance,
}
with open('paper/csv/feature_analysis_v6.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: paper/csv/feature_analysis_v6.json")

# Summary
print("\n=== Group Removal Summary (Table 9) ===")
print(f"{'Group':<16} {'Full R2':>8} {'w/o':>8} {'Delta':>8}")
for g, v in removal_results.items():
    print(f"{g:<16} {v['full_r2']:>8.4f} {v['reduced_r2']:>8.4f} {v['delta']:>+8.4f}")

print("\n=== Kendall's W by Horizon ===")
print(f"{'Horizon':<8} {'W':>8} {'p':>10}")
for hz, v in kendall_results.items():
    print(f"{hz:<8} {v['W']:>8.4f} {v['p']:>10.6f}")
