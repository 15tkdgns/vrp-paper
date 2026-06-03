"""
V71 Multi-Method Feature Importance Analysis
=============================================
8 different feature importance methods for cross-validation:

1. Ridge Coefficient Magnitude (standardized |β| × std(X))
2. Permutation Importance (Ridge, model-agnostic)
3. XGBoost Gain (split-based, captures nonlinearity)
4. SHAP LinearExplainer (Ridge, exact Shapley values)
5. SHAP TreeExplainer (XGBoost, exact tree SHAP)
6. Mutual Information Regression (nonparametric, model-free)
7. Spearman Rank Correlation (univariate, monotonic association)
8. RFE with Ridge (Recursive Feature Elimination)

Results saved as JSON with per-method rankings and cross-method consensus.
"""

import numpy as np
import pandas as pd
import json, time, os, pickle, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression, RFE
from scipy import stats

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
OHLCV_CACHE = 'src/data/v71_ohlcv_cache.pkl'


# =====================================================================
# Dataset builder (reuse from data_builder)
# =====================================================================
def fit_garch(returns):
    try:
        from arch import arch_model
        ret = returns * 100
        am = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def get_asset_class(a):
    for c, aa in ASSET_GROUPS.items():
        if a in aa: return c
    return 'Unknown'

def build_dataset():
    raw = pd.read_pickle(OHLCV_CACHE).ffill()
    avail = raw.columns.get_level_values(1).unique()

    spy_close = raw[('Close','SPY')]
    spy_ret = np.log(spy_close/spy_close.shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean()*252*10000
    spy_lrv = np.log(spy_rv+1e-6)

    vix = raw[('Close','VIX')]
    iv = {
        'VIX': np.log(vix+1e-6),
        'VIX_chg': np.log(vix+1e-6).diff(),
        'VIX_ma5': np.log(vix+1e-6).rolling(5).mean(),
        'VIX_std5': np.log(vix+1e-6).rolling(5).std(),
        'VIX3M': np.log(raw[('Close','VIX3M')]+1e-6),
        'VIX_TermSlope': np.log(vix+1e-6) - np.log(raw[('Close','VIX3M')]+1e-6),
        'VIX9D': np.log(raw[('Close','VIX9D')]+1e-6),
        'VIX_ShortSlope': np.log(raw[('Close','VIX9D')]+1e-6) - np.log(vix+1e-6),
        'VRP': (vix**2/100) - spy_rv/10000,
        'VRP_ma22': ((vix**2/100) - spy_rv/10000).rolling(22).mean(),
    }

    pooled = []
    for asset in [a for a in ALL_ASSETS if a in avail]:
        c=raw[('Close',asset)]; o=raw[('Open',asset)]
        h=raw[('High',asset)]; l=raw[('Low',asset)]; vol=raw[('Volume',asset)]
        ret=np.log(c/c.shift(1)).dropna()
        rv=(ret**2).rolling(22).mean()*252*10000; lrv=np.log(rv+1e-6)
        gd=pd.Series(fit_garch(ret), index=ret.index)
        rw=ret.resample('W').sum()
        gw=pd.Series(fit_garch(rw), index=rw.index).reindex(ret.index, method='ffill')
        p5=compute_parkinson(h,l,5);p22=compute_parkinson(h,l,22)
        gk22=compute_gk(o,h,l,c,22);rs22=compute_rs(o,h,l,c,22)
        on=np.log(o/c.shift(1))
        dv=vol*c
        pv=vol.where(ret>0,0).rolling(22).sum()
        nv=vol.where(ret<=0,0).rolling(22).sum()

        feat={
            'LogRV_lag1':lrv.shift(1),'LogRV_lag5':lrv.shift(5),
            'LogRV_lag10':lrv.shift(10),'LogRV_lag22':lrv.shift(22),
            'Garch_Daily':gd.shift(1),'Garch_Weekly':gw.shift(1),
            'LogRV_Std5':lrv.rolling(5).std().shift(1),
            'LogRV_Std22':lrv.rolling(22).std().shift(1),
            'RV_Mom5':(lrv-lrv.shift(5)).shift(1),
            'RV_Mom22':(lrv-lrv.shift(22)).shift(1),
            'SPY_LogRV':spy_lrv.shift(1),
            'Ret_lag1':ret.shift(1),'Ret_abs_lag1':ret.abs().shift(1),
            'Corr_SPY':ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                       if asset!='SPY' else pd.Series(1.0, index=ret.index),
            'Parkinson_5':np.log(p5+1e-6).shift(1),
            'Parkinson_22':np.log(p22+1e-6).shift(1),
            'GarmanKlass_22':np.log(gk22+1e-6).shift(1),
            'RogersSatchell_22':np.log(rs22+1e-6).shift(1),
            'Range_Close_Ratio':(np.log(p22+1e-6)-lrv).shift(1),
            'Overnight_Vol':on.rolling(22).std().shift(1),
            'Overnight_Ret':on.shift(1),
            'AltVol_Amihud':(ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1),
            'AltVol_Vol_Ratio':(vol.rolling(5).mean()/(vol.rolling(22).mean()+1e-10)).shift(1),
            'AltVol_PV_Corr':ret.rolling(22).corr(np.log(vol+1)).shift(1),
            'AltVol_Vol_Surprise':((vol-vol.rolling(22).mean())/(vol.rolling(22).std()+1e-10)).shift(1),
            'AltVol_Order_Imbalance':((pv-nv)/(pv+nv+1e-10)).shift(1),
            'AltVol_Kyle_Lambda':(ret.abs().rolling(22).sum()/(vol.rolling(22).sum()+1e-10)*1e6).shift(1),
            'Target':lrv.shift(-22),'Asset':asset,'Class':get_asset_class(asset),
        }
        for n2,v2 in iv.items(): feat[f'IV_{n2}']=v2.shift(1)
        d=pd.DataFrame(feat)
        d=d.dropna(subset=['Target'])
        nc=[x for x in d.columns if x not in ['Asset','Class','Target']]
        d[nc]=d[nc].replace([np.inf,-np.inf],np.nan).fillna(0)
        pooled.append(d)

    data=pd.concat(pooled).sort_index().reset_index(drop=True)
    feats=[c for c in data.columns if c not in ['Target','Asset','Class']]
    data[feats]=data[feats].fillna(0).replace([np.inf,-np.inf],0)
    return data, feats


# =====================================================================
# Method 1: Ridge Coefficient Magnitude
# =====================================================================
def method_ridge_coef(train_df, test_df, feats):
    print("\n[1/8] Ridge Coefficient Magnitude...", flush=True)
    sc = StandardScaler().fit(train_df[feats])
    results = {}

    for cls in ASSET_GROUPS:
        tr = train_df[train_df['Class']==cls]
        if len(tr) < 100: continue
        m = Ridge(alpha=10.0).fit(sc.transform(tr[feats]), tr['Target'])
        # Standardized coef: already standardized input, so |coef| = importance
        for i, f in enumerate(feats):
            if f not in results: results[f] = []
            results[f].append(abs(m.coef_[i]))

    importance = {f: float(np.mean(v)) for f, v in results.items()}
    total = sum(importance.values())
    importance = {f: v/total for f, v in importance.items()}  # normalize

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 2: Permutation Importance (Ridge)
# =====================================================================
def method_permutation(train_df, test_df, feats):
    print("\n[2/8] Permutation Importance (Ridge)...", flush=True)
    sc = StandardScaler().fit(train_df[feats])

    # Train per-class models
    models = {}
    best_alphas = {}
    val_split = int(len(train_df)*0.8)
    tr_inner = train_df.iloc[:val_split-22]
    va_inner = train_df.iloc[val_split:]

    for cls in ASSET_GROUPS:
        tr = tr_inner[tr_inner['Class']==cls]
        va = va_inner[va_inner['Class']==cls]
        if len(tr)<100 or len(va)<30: best_alphas[cls]=1.0; continue
        best_r2, best_a = -999, 1.0
        for a in [0.1,1,10,100,1000]:
            m = Ridge(alpha=a).fit(sc.transform(tr[feats]), tr['Target'])
            r2 = r2_score(va['Target'], m.predict(sc.transform(va[feats])))
            if r2>best_r2: best_r2,best_a=r2,a
        best_alphas[cls] = best_a

    for cls in ASSET_GROUPS:
        tr = train_df[train_df['Class']==cls]
        if len(tr)<100: continue
        models[cls] = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr[feats]), tr['Target'])

    # Baseline prediction
    preds_base = np.full(len(test_df), np.nan)
    for cls, m in models.items():
        idx = test_df['Class']==cls
        if idx.sum()==0: continue
        preds_base[idx.values] = m.predict(sc.transform(test_df.loc[idx, feats]))
    r2_base = r2_score(test_df['Target'].values, preds_base)

    importance = {}
    np.random.seed(42)
    for fi, f in enumerate(feats):
        drops = []
        for rep in range(5):
            X_perm = test_df[feats].copy()
            X_perm[f] = np.random.permutation(X_perm[f].values)
            preds_p = np.full(len(test_df), np.nan)
            for cls, m in models.items():
                idx = test_df['Class']==cls
                if idx.sum()==0: continue
                preds_p[idx.values] = m.predict(sc.transform(X_perm.loc[idx]))
            drops.append(r2_base - r2_score(test_df['Target'].values, preds_p))
        importance[f] = float(np.mean(drops))

    # Normalize (clamp negatives to 0)
    importance = {f: max(v, 0) for f, v in importance.items()}
    total = sum(importance.values()) or 1
    importance = {f: v/total for f, v in importance.items()}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 3: XGBoost Gain
# =====================================================================
def method_xgboost_gain(train_df, test_df, feats):
    print("\n[3/8] XGBoost Feature Importance (Gain)...", flush=True)
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  SKIP: xgboost not installed", flush=True)
        return {}

    importance_acc = {f: 0.0 for f in feats}
    count = 0

    for cls in ASSET_GROUPS:
        tr = train_df[train_df['Class']==cls]
        te = test_df[test_df['Class']==cls]
        if len(tr)<100 or len(te)<30: continue

        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0
        )
        xgb.fit(tr[feats].values, tr['Target'].values,
                eval_set=[(te[feats].values, te['Target'].values)],
                verbose=False)

        gains = xgb.feature_importances_  # gain by default
        for i, f in enumerate(feats):
            importance_acc[f] += gains[i]
        count += 1

    if count > 0:
        importance = {f: v/count for f, v in importance_acc.items()}
        total = sum(importance.values()) or 1
        importance = {f: v/total for f, v in importance.items()}
    else:
        importance = {}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 4: SHAP LinearExplainer (Ridge)
# =====================================================================
def method_shap_linear(train_df, test_df, feats):
    print("\n[4/8] SHAP LinearExplainer (Ridge)...", flush=True)
    try:
        import shap
    except ImportError:
        print("  SKIP: shap not installed", flush=True)
        return {}

    sc = StandardScaler().fit(train_df[feats])
    importance_acc = {f: 0.0 for f in feats}
    count = 0

    for cls in ASSET_GROUPS:
        tr = train_df[train_df['Class']==cls]
        te = test_df[test_df['Class']==cls]
        if len(tr)<100 or len(te)<30: continue

        m = Ridge(alpha=10.0).fit(sc.transform(tr[feats]), tr['Target'])

        # Sample test set for speed
        te_sample = te.sample(min(500, len(te)), random_state=42)
        X_test_sc = sc.transform(te_sample[feats])

        explainer = shap.LinearExplainer(m, sc.transform(tr[feats]))
        shap_values = explainer.shap_values(X_test_sc)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        for i, f in enumerate(feats):
            importance_acc[f] += mean_abs_shap[i]
        count += 1

    if count > 0:
        importance = {f: v/count for f, v in importance_acc.items()}
        total = sum(importance.values()) or 1
        importance = {f: v/total for f, v in importance.items()}
    else:
        importance = {}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 5: SHAP TreeExplainer (XGBoost)
# =====================================================================
def method_shap_tree(train_df, test_df, feats):
    print("\n[5/8] SHAP TreeExplainer (XGBoost)...", flush=True)
    try:
        import shap
        from xgboost import XGBRegressor
    except ImportError:
        print("  SKIP: shap or xgboost not installed", flush=True)
        return {}

    importance_acc = {f: 0.0 for f in feats}
    count = 0

    for cls in ASSET_GROUPS:
        tr = train_df[train_df['Class']==cls]
        te = test_df[test_df['Class']==cls]
        if len(tr)<100 or len(te)<30: continue

        xgb = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        xgb.fit(tr[feats].values, tr['Target'].values, verbose=False)

        te_sample = te.sample(min(500, len(te)), random_state=42)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(te_sample[feats].values)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        for i, f in enumerate(feats):
            importance_acc[f] += mean_abs_shap[i]
        count += 1

    if count > 0:
        importance = {f: v/count for f, v in importance_acc.items()}
        total = sum(importance.values()) or 1
        importance = {f: v/total for f, v in importance.items()}
    else:
        importance = {}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 6: Mutual Information Regression
# =====================================================================
def method_mutual_info(train_df, feats):
    print("\n[6/8] Mutual Information Regression...", flush=True)

    # Sample for speed
    sample = train_df.sample(min(10000, len(train_df)), random_state=42)
    X = sample[feats].values
    y = sample['Target'].values

    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    total = mi.sum() or 1
    importance = {f: float(mi[i]/total) for i, f in enumerate(feats)}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 7: Spearman Rank Correlation
# =====================================================================
def method_spearman(train_df, feats):
    print("\n[7/8] Spearman Rank Correlation...", flush=True)

    sample = train_df.sample(min(20000, len(train_df)), random_state=42)
    y = sample['Target'].values

    importance = {}
    for f in feats:
        rho, _ = stats.spearmanr(sample[f].values, y)
        importance[f] = abs(rho) if not np.isnan(rho) else 0.0

    total = sum(importance.values()) or 1
    importance = {f: v/total for f, v in importance.items()}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Method 8: RFE (Recursive Feature Elimination) with Ridge
# =====================================================================
def method_rfe(train_df, feats):
    print("\n[8/8] RFE with Ridge...", flush=True)

    # Sample for speed
    sample = train_df.sample(min(15000, len(train_df)), random_state=42)
    sc = StandardScaler().fit(sample[feats])
    X = sc.transform(sample[feats])
    y = sample['Target'].values

    m = Ridge(alpha=10.0)
    rfe = RFE(m, n_features_to_select=1, step=1)
    rfe.fit(X, y)

    # ranking_: 1 = best, higher = eliminated earlier
    n_feats = len(feats)
    importance = {}
    for i, f in enumerate(feats):
        # Invert ranking so rank 1 gets highest score
        importance[f] = (n_feats - rfe.ranking_[i] + 1) / n_feats

    total = sum(importance.values()) or 1
    importance = {f: v/total for f, v in importance.items()}

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {f:<28} {v:.4f}", flush=True)
    return importance


# =====================================================================
# Consensus Analysis
# =====================================================================
def compute_consensus(all_results, feats):
    """Compute consensus ranking across methods using mean rank."""
    print("\n" + "="*70)
    print("CONSENSUS FEATURE IMPORTANCE (Mean Rank)")
    print("="*70, flush=True)

    methods = [m for m in all_results if all_results[m]]
    n_feats = len(feats)

    # Per-method ranking
    rankings = {}
    for method in methods:
        imp = all_results[method]
        sorted_feats = sorted(imp.keys(), key=lambda f: imp[f], reverse=True)
        for rank, f in enumerate(sorted_feats, 1):
            if f not in rankings:
                rankings[f] = {}
            rankings[f][method] = rank

    # Mean rank
    consensus = {}
    for f in feats:
        ranks = [rankings.get(f, {}).get(m, n_feats) for m in methods]
        consensus[f] = {
            'mean_rank': float(np.mean(ranks)),
            'std_rank': float(np.std(ranks)),
            'min_rank': int(np.min(ranks)),
            'max_rank': int(np.max(ranks)),
            'per_method': {m: rankings.get(f, {}).get(m, n_feats) for m in methods},
        }

    sorted_consensus = sorted(consensus.items(), key=lambda x: x[1]['mean_rank'])

    # Print table
    method_short = {
        'ridge_coef': 'Coef',
        'permutation': 'Perm',
        'xgboost_gain': 'XGB',
        'shap_linear': 'SHAP-L',
        'shap_tree': 'SHAP-T',
        'mutual_info': 'MI',
        'spearman': 'Spear',
        'rfe': 'RFE',
    }

    header = f"{'Rank':>4} {'Feature':<28} {'Mean':>5} {'Std':>5}"
    for m in methods:
        header += f" {method_short.get(m, m[:5]):>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for rank, (f, c) in enumerate(sorted_consensus, 1):
        line = f"{rank:>4} {f:<28} {c['mean_rank']:>5.1f} {c['std_rank']:>5.1f}"
        for m in methods:
            r = c['per_method'].get(m, n_feats)
            line += f" {r:>6}"
        print(line, flush=True)

    # Top-10 consensus features
    top10 = [f for f, _ in sorted_consensus[:10]]
    print(f"\nTop-10 Consensus Features: {top10}", flush=True)

    # Kendall's tau between methods
    print(f"\nInter-method Agreement (Kendall's tau):", flush=True)
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            common = set(all_results[m1].keys()) & set(all_results[m2].keys())
            if len(common) < 5: continue
            ranks1 = [rankings[f].get(m1, n_feats) for f in common]
            ranks2 = [rankings[f].get(m2, n_feats) for f in common]
            tau, p = stats.kendalltau(ranks1, ranks2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {method_short.get(m1,'?'):>6} vs {method_short.get(m2,'?'):<6}: tau={tau:.3f} (p={p:.4f}) {sig}", flush=True)

    return dict(sorted_consensus)


# =====================================================================
# Main
# =====================================================================
def main():
    print("="*70)
    print("V71 Multi-Method Feature Importance Analysis")
    print("="*70, flush=True)
    t0 = time.time()

    data, feats = build_dataset()
    print(f"Dataset: {len(data)} samples, {len(feats)} features", flush=True)

    purge = 22
    split = int(len(data)*0.8)
    train_df = data.iloc[:split-purge]
    test_df = data.iloc[split:]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    all_results = {}

    # Method 1: Ridge Coefficient
    try:
        all_results['ridge_coef'] = method_ridge_coef(train_df, test_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['ridge_coef'] = {}

    # Method 2: Permutation Importance
    try:
        all_results['permutation'] = method_permutation(train_df, test_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['permutation'] = {}

    # Method 3: XGBoost Gain
    try:
        all_results['xgboost_gain'] = method_xgboost_gain(train_df, test_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['xgboost_gain'] = {}

    # Method 4: SHAP Linear
    try:
        all_results['shap_linear'] = method_shap_linear(train_df, test_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['shap_linear'] = {}

    # Method 5: SHAP Tree
    try:
        all_results['shap_tree'] = method_shap_tree(train_df, test_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['shap_tree'] = {}

    # Method 6: Mutual Information
    try:
        all_results['mutual_info'] = method_mutual_info(train_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['mutual_info'] = {}

    # Method 7: Spearman
    try:
        all_results['spearman'] = method_spearman(train_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['spearman'] = {}

    # Method 8: RFE
    try:
        all_results['rfe'] = method_rfe(train_df, feats)
    except Exception as e:
        print(f"  FAILED: {e}"); all_results['rfe'] = {}

    # Consensus
    consensus = compute_consensus(all_results, feats)

    # Save
    output = {
        'methods': {m: {f: float(v) for f, v in imp.items()} for m, imp in all_results.items() if imp},
        'consensus': {f: v for f, v in consensus.items()},
        'n_features': len(feats),
        'feature_list': feats,
    }
    out_path = 'src/experiments/creative/v71_feature_importance_multi_results.json'
    with open(out_path, 'w') as fp:
        json.dump(output, fp, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
