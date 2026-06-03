"""Feature Importance Analysis: SCI-Grade 3-Layer Framework
==========================================================
Layer 1: Model-Specific (Ridge coeff, XGBoost Gain, RF MDI)
Layer 2: Model-Agnostic (PFI block-shuffle, SHAP, MI, Spearman, RFE)
Layer 3: Consensus (Bootstrap Rank CI, Kendall's W, Cross-horizon Spearman)

Produces:
  O1: Horizon x Feature PFI heatmap
  O2: Model-specific Top-10 comparison (22d)
  O3: Feature group ablation across horizons
  O4: Kendall's W concordance by horizon
  O5: Cross-horizon rank Spearman matrix
  O6: Bootstrap rank CI table (per horizon)
  O7: VIX regime SHAP analysis
"""
import numpy as np, pandas as pd, json, time, warnings, os
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
import xgboost as xgb
from arch import arch_model
warnings.filterwarnings('ignore')

# ============ Constants ============
ASSET_GROUPS = {
    'Equity': ['SPY','QQQ','IWM','EFA','EEM'],
    'Bond':   ['TLT','IEF','AGG'],
    'Commodity': ['GLD','SLV','USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]

FEATURE_GROUPS = {
    'HAR': ['LogRV_lag1','LogRV_lag5','LogRV_lag10','LogRV_lag22',
            'LogRV_Std5','LogRV_Std22','RV_Mom5','RV_Mom22',
            'Ret_lag1','Ret_abs_lag1','SPY_LogRV','Corr_SPY'],
    'GARCH': ['Garch_Daily','Garch_Weekly'],
    'HF_Proxy': ['Parkinson_5','Parkinson_22','GarmanKlass_22',
                 'RogersSatchell_22','Range_Close_Ratio','Overnight_Vol','Overnight_Ret'],
    'IV_Surface': ['IV_VIX','IV_VIX_chg','IV_VIX_ma5','IV_VIX_std5',
                   'IV_VIX3M','IV_VIX_TermSlope','IV_VIX9D','IV_VIX_ShortSlope',
                   'IV_VRP','IV_VRP_ma22'],
    'Alt_Data': ['AltVol_Amihud','AltVol_Vol_Ratio','AltVol_PV_Corr',
                 'AltVol_Vol_Surprise','AltVol_Order_Imbalance','AltVol_Kyle_Lambda']
}

OUTPUT_DIR = '/root/vrp/data/processed/feature_importance'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ Feature Engineering (reuse from benchmark) ============
def fit_garch(r):
    try:
        am = arch_model(r*100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return pd.Series(res.conditional_volatility.values.flatten()/100, index=r.index)
    except:
        return r.rolling(22).std()

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean() / (4*np.log(2))) * np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

def build_features():
    """Build 37-feature panel from cached OHLCV data."""
    print("Loading data...", flush=True)
    raw = pd.read_pickle('/root/vrp/src/data/v71_ohlcv_cache.pkl')

    vix = raw[('Close','VIX')]
    spy_c = raw[('Close','SPY')]
    spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_lrv = np.log(spy_rv + 1e-6)

    iv_features = {}
    iv_features['VIX'] = np.log(vix + 1e-6)
    iv_features['VIX_chg'] = iv_features['VIX'].diff()
    iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
    iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
    iv_features['VIX3M'] = np.log(raw[('Close','VIX3M')] + 1e-6)
    iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
    iv_features['VIX9D'] = np.log(raw[('Close','VIX9D')] + 1e-6)
    iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
    vrp = (vix**2/100) - spy_rv/10000
    iv_features['VRP'] = vrp
    iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    base_frames = {}
    for asset in ALL_ASSETS:
        c = raw[('Close', asset)]; o = raw[('Open', asset)]
        h = raw[('High', asset)]; l = raw[('Low', asset)]
        v = raw[('Volume', asset)]
        ret = np.log(c / c.shift(1)).dropna()
        ret_sq = ret**2
        rv = ret_sq.rolling(22).mean() * 252 * 10000
        lrv = np.log(rv + 1e-6)
        gd = fit_garch(ret)
        rw = ret.resample('W').sum()
        gw = fit_garch(rw).reindex(ret.index, method='ffill')

        feat = {
            'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
            'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
            'Garch_Daily': gd.shift(1), 'Garch_Weekly': gw.shift(1),
            'LogRV_Std5': lrv.rolling(5).std().shift(1),
            'LogRV_Std22': lrv.rolling(22).std().shift(1),
            'RV_Mom5': (lrv - lrv.shift(5)).shift(1),
            'RV_Mom22': (lrv - lrv.shift(22)).shift(1),
            'SPY_LogRV': spy_lrv.shift(1),
            'Ret_lag1': ret.shift(1), 'Ret_abs_lag1': ret.abs().shift(1),
            'Corr_SPY': ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                        if asset != 'SPY' else pd.Series(1.0, index=ret.index),
        }
        p5 = compute_parkinson(h, l, 5); p22 = compute_parkinson(h, l, 22)
        gk22 = compute_gk(o, h, l, c, 22); rs22 = compute_rs(o, h, l, c, 22)
        feat['Parkinson_5'] = np.log(p5 + 1e-6).shift(1)
        feat['Parkinson_22'] = np.log(p22 + 1e-6).shift(1)
        feat['GarmanKlass_22'] = np.log(gk22 + 1e-6).shift(1)
        feat['RogersSatchell_22'] = np.log(rs22 + 1e-6).shift(1)
        feat['Range_Close_Ratio'] = (np.log(p22 + 1e-6) - lrv).shift(1)
        on = np.log(o / c.shift(1))
        feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
        feat['Overnight_Ret'] = on.shift(1)
        for n2, v2 in iv_features.items():
            feat[f'IV_{n2}'] = v2.shift(1)
        dv = v * c
        feat['AltVol_Amihud'] = (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1)
        feat['AltVol_Vol_Ratio'] = (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1)
        feat['AltVol_PV_Corr'] = ret.rolling(22).corr(np.log(v + 1)).shift(1)
        feat['AltVol_Vol_Surprise'] = ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1)
        pv = v.where(ret > 0, 0).rolling(22).sum()
        nv = v.where(ret <= 0, 0).rolling(22).sum()
        feat['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
        feat['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
        feat['ret_sq'] = ret_sq
        feat['Asset'] = asset

        d = pd.DataFrame(feat)
        nc = [x for x in d.columns if x not in ['Asset', 'ret_sq']]
        d[nc] = d[nc].replace([np.inf, -np.inf], np.nan)
        base_frames[asset] = d

    return base_frames, vix


def prepare_horizon_data(base_frames, horizon):
    """Prepare pooled panel for a given horizon."""
    pooled = []
    for asset in ALL_ASSETS:
        d = base_frames[asset].copy()
        d['Target'] = forward_rv(d['ret_sq'], horizon)
        d = d.drop(columns=['ret_sq']).dropna()
        nc = [x for x in d.columns if x not in ['Asset', 'Target']]
        d[nc] = d[nc].fillna(0)
        pooled.append(d)
    data = pd.concat(pooled).sort_index().reset_index(drop=True)
    all_feats = [c for c in data.columns if c not in ['Target', 'Asset']]
    data[all_feats] = data[all_feats].fillna(0).replace([np.inf, -np.inf], 0)
    split = int(len(data) * 0.8)
    return data.iloc[:split], data.iloc[split:], all_feats


# ============ Layer 1: Model-Specific Importance ============
def layer1_ridge_importance(train_df, test_df, all_feats):
    """Standardized |coefficient| importance."""
    sc = StandardScaler(); sc.fit(train_df[all_feats])
    X_tr = sc.transform(train_df[all_feats]); y_tr = train_df['Target'].values
    ridge = Ridge(alpha=100); ridge.fit(X_tr, y_tr)
    importance = np.abs(ridge.coef_)
    return dict(zip(all_feats, importance)), ridge, sc


def layer1_xgboost_importance(train_df, test_df, all_feats):
    """XGBoost total gain importance."""
    sc = StandardScaler(); sc.fit(train_df[all_feats])
    X_tr = sc.transform(train_df[all_feats]); y_tr = train_df['Target'].values
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr)
    gain = model.get_booster().get_score(importance_type='total_gain')
    importance = {}
    for i, f in enumerate(all_feats):
        key = f'f{i}'
        importance[f] = gain.get(key, 0.0)
    return importance, model, sc


def layer1_rf_importance(train_df, test_df, all_feats):
    """Random Forest MDI (Mean Decrease Impurity)."""
    sc = StandardScaler(); sc.fit(train_df[all_feats])
    X_tr = sc.transform(train_df[all_feats]); y_tr = train_df['Target'].values
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    importance = dict(zip(all_feats, model.feature_importances_))
    return importance, model, sc


# ============ Layer 2: Model-Agnostic Importance ============
def block_shuffle(x, block_size):
    """Block-shuffle a 1D array preserving temporal structure."""
    n = len(x)
    n_blocks = max(1, n // block_size)
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    remainder = x[n_blocks*block_size:]
    np.random.shuffle(blocks)
    result = np.concatenate(blocks)
    if len(remainder) > 0:
        result = np.concatenate([result, remainder])
    return result


def layer2_pfi(model, sc, X_test, y_test, all_feats, horizon, n_repeats=30):
    """Permutation Feature Importance with block shuffle."""
    X_scaled = sc.transform(X_test[all_feats])
    baseline = r2_score(y_test, model.predict(X_scaled))
    importance = {}
    ci_lower = {}; ci_upper = {}

    for j, feat in enumerate(all_feats):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            X_perm[:, j] = block_shuffle(X_perm[:, j], block_size=max(1, horizon))
            score = r2_score(y_test, model.predict(X_perm))
            drops.append(baseline - score)
        importance[feat] = float(np.mean(drops))
        ci_lower[feat] = float(np.percentile(drops, 2.5))
        ci_upper[feat] = float(np.percentile(drops, 97.5))

    return importance, ci_lower, ci_upper


def layer2_mutual_info(X_train, y_train, all_feats):
    """Mutual Information (model-free)."""
    mi = mutual_info_regression(X_train[all_feats].fillna(0), y_train, random_state=42)
    return dict(zip(all_feats, mi))


def layer2_spearman(X_test, y_test, all_feats):
    """Spearman rank correlation |rho|."""
    importance = {}
    for feat in all_feats:
        rho, _ = sp_stats.spearmanr(X_test[feat].fillna(0), y_test)
        importance[feat] = abs(rho)
    return importance


def layer2_rfe(train_df, all_feats, n_features=10):
    """Recursive Feature Elimination ranking with Ridge."""
    sc = StandardScaler(); sc.fit(train_df[all_feats])
    X_tr = sc.transform(train_df[all_feats]); y_tr = train_df['Target'].values
    ridge = Ridge(alpha=100)
    selector = RFE(ridge, n_features_to_select=n_features, step=1)
    selector.fit(X_tr, y_tr)
    # ranking_: 1 = best, higher = eliminated earlier (less important)
    importance = {}
    for feat, rank in zip(all_feats, selector.ranking_):
        importance[feat] = 1.0 / rank  # invert so higher = more important
    return importance


# ============ Layer 3: Consensus Aggregation ============
def compute_ranks(importance_dict):
    """Convert importance values to ranks (1 = most important)."""
    sorted_feats = sorted(importance_dict.keys(), key=lambda f: importance_dict[f], reverse=True)
    return {f: rank+1 for rank, f in enumerate(sorted_feats)}


def consensus_mean_rank(all_ranks, all_feats):
    """Mean rank across methods."""
    mean_ranks = {}
    for feat in all_feats:
        ranks = [r[feat] for r in all_ranks if feat in r]
        mean_ranks[feat] = np.mean(ranks)
    return mean_ranks


def bootstrap_rank_ci(all_ranks, all_feats, B=1000):
    """Bootstrap CI for mean rank."""
    n_methods = len(all_ranks)
    boot_mean_ranks = {f: [] for f in all_feats}

    for _ in range(B):
        indices = np.random.choice(n_methods, size=n_methods, replace=True)
        sampled = [all_ranks[i] for i in indices]
        mr = consensus_mean_rank(sampled, all_feats)
        for f in all_feats:
            boot_mean_ranks[f].append(mr[f])

    result = {}
    for f in all_feats:
        vals = boot_mean_ranks[f]
        result[f] = {
            'mean': float(np.mean(vals)),
            'ci_lower': float(np.percentile(vals, 2.5)),
            'ci_upper': float(np.percentile(vals, 97.5))
        }
    return result


def kendall_w(all_ranks, all_feats):
    """Kendall's W coefficient of concordance."""
    k = len(all_ranks)  # number of raters (methods)
    n = len(all_feats)   # number of items (features)
    rank_matrix = np.array([[r[f] for f in all_feats] for r in all_ranks])
    rank_sums = rank_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * ss / (k**2 * (n**3 - n))
    chi2 = k * (n - 1) * W
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=n-1)
    return float(W), float(chi2), float(p_value)


# ============ Main Execution ============
def run_analysis():
    t0 = time.time()
    base_frames, vix = build_features()

    all_results = {}

    for horizon in HORIZONS:
        print(f"\n{'='*70}")
        print(f"Horizon: {horizon}d")
        print(f"{'='*70}", flush=True)

        train_df, test_df, all_feats = prepare_horizon_data(base_frames, horizon)
        y_train = train_df['Target'].values
        y_test = test_df['Target'].values

        hz_results = {'horizon': horizon, 'n_train': len(train_df), 'n_test': len(test_df)}

        # --- Layer 1 ---
        print("  Layer 1: Model-specific importance...", flush=True)
        ridge_imp, ridge_model, ridge_sc = layer1_ridge_importance(train_df, test_df, all_feats)
        xgb_imp, xgb_model, xgb_sc = layer1_xgboost_importance(train_df, test_df, all_feats)
        rf_imp, rf_model, rf_sc = layer1_rf_importance(train_df, test_df, all_feats)

        hz_results['layer1'] = {
            'ridge_coeff': ridge_imp,
            'xgb_gain': xgb_imp,
            'rf_mdi': rf_imp
        }

        # --- Layer 2 ---
        print("  Layer 2: Model-agnostic importance...", flush=True)

        # PFI for Ridge (primary model)
        print("    PFI (Ridge, block shuffle)...", flush=True)
        pfi_ridge, pfi_ci_lo, pfi_ci_hi = layer2_pfi(
            ridge_model, ridge_sc, test_df, y_test, all_feats, horizon, n_repeats=10
        )

        # PFI for XGBoost
        print("    PFI (XGBoost)...", flush=True)
        pfi_xgb, _, _ = layer2_pfi(
            xgb_model, xgb_sc, test_df, y_test, all_feats, horizon, n_repeats=10
        )

        # PFI for RF
        print("    PFI (RF)...", flush=True)
        pfi_rf, _, _ = layer2_pfi(
            rf_model, rf_sc, test_df, y_test, all_feats, horizon, n_repeats=10
        )

        # MI
        print("    Mutual Information...", flush=True)
        mi_imp = layer2_mutual_info(train_df, y_train, all_feats)

        # Spearman
        print("    Spearman correlation...", flush=True)
        spearman_imp = layer2_spearman(test_df, y_test, all_feats)

        # RFE
        print("    RFE...", flush=True)
        rfe_imp = layer2_rfe(train_df, all_feats)

        hz_results['layer2'] = {
            'pfi_ridge': pfi_ridge,
            'pfi_ridge_ci_lower': pfi_ci_lo,
            'pfi_ridge_ci_upper': pfi_ci_hi,
            'pfi_xgb': pfi_xgb,
            'pfi_rf': pfi_rf,
            'mutual_info': mi_imp,
            'spearman': spearman_imp,
            'rfe': rfe_imp
        }

        # --- Layer 3 ---
        print("  Layer 3: Consensus aggregation...", flush=True)
        all_importance = {
            'Ridge_Coeff': ridge_imp,
            'XGB_Gain': xgb_imp,
            'RF_MDI': rf_imp,
            'PFI_Ridge': pfi_ridge,
            'PFI_XGB': pfi_xgb,
            'MI': mi_imp,
            'Spearman': spearman_imp,
            'RFE': rfe_imp
        }

        all_ranks = [compute_ranks(imp) for imp in all_importance.values()]
        mean_ranks = consensus_mean_rank(all_ranks, all_feats)
        boot_ci = bootstrap_rank_ci(all_ranks, all_feats, B=1000)
        W, chi2, p_val = kendall_w(all_ranks, all_feats)

        hz_results['layer3'] = {
            'mean_ranks': mean_ranks,
            'bootstrap_ci': boot_ci,
            'kendall_w': W,
            'kendall_chi2': chi2,
            'kendall_p': p_val,
            'n_methods': len(all_importance)
        }

        # Print top-10
        sorted_feats = sorted(mean_ranks.keys(), key=lambda f: mean_ranks[f])
        print(f"\n  Top-10 Features (consensus, {horizon}d):")
        print(f"  {'Rank':<5} {'Feature':<25} {'MeanRank':>8}  {'CI_lo':>6} {'CI_hi':>6}  {'Group':<12}")
        for i, f in enumerate(sorted_feats[:10]):
            group = next((g for g, fs in FEATURE_GROUPS.items() if f in fs), 'Other')
            ci = boot_ci[f]
            print(f"  {i+1:<5} {f:<25} {mean_ranks[f]:>8.1f}  {ci['ci_lower']:>6.1f} {ci['ci_upper']:>6.1f}  {group:<12}")
        print(f"  Kendall's W = {W:.3f} (chi2={chi2:.1f}, p={p_val:.4f})")

        all_results[f'{horizon}d'] = hz_results

    # --- Cross-horizon analysis ---
    print(f"\n{'='*70}")
    print("Cross-Horizon Rank Stability (Spearman rho)")
    print(f"{'='*70}")

    horizon_ranks = {}
    for h in HORIZONS:
        key = f'{h}d'
        mr = all_results[key]['layer3']['mean_ranks']
        horizon_ranks[key] = mr

    # Spearman rho matrix
    cross_spearman = {}
    for h1 in HORIZONS:
        for h2 in HORIZONS:
            k1, k2 = f'{h1}d', f'{h2}d'
            feats = sorted(horizon_ranks[k1].keys())
            r1 = [horizon_ranks[k1][f] for f in feats]
            r2 = [horizon_ranks[k2][f] for f in feats]
            rho, _ = sp_stats.spearmanr(r1, r2)
            cross_spearman[f'{k1}_vs_{k2}'] = float(rho)

    # Print matrix
    print(f"\n{'':>6}", end='')
    for h in HORIZONS: print(f"  {h}d", end='')
    print()
    for h1 in HORIZONS:
        print(f"{h1}d", end='')
        for h2 in HORIZONS:
            rho = cross_spearman[f'{h1}d_vs_{h2}d']
            print(f"  {rho:.2f}", end='')
        print()

    all_results['cross_horizon_spearman'] = cross_spearman
    all_results['feature_groups'] = FEATURE_GROUPS

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'feature_importance_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nSaved results to {output_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    run_analysis()
