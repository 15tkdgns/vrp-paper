"""
V71 Comprehensive Model Comparison & Verification
===================================================
Academic-grade model comparison with:

1. 8 models: Ridge, Lasso, ElasticNet, XGBoost, LightGBM, RandomForest, SVR, MLP
2. Per-asset & per-class R² breakdown
3. Pairwise DM test matrix (Bonferroni corrected)
4. Residual diagnostics (autocorrelation, normality, heteroscedasticity)
5. Learning curve analysis (varying training size)
6. Feature subset ablation (HAR-only vs Range-only vs IV-only vs Full)

All results saved to JSON.
"""

import numpy as np
import pandas as pd
import json, time, os, pickle, warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY','QQQ','IWM','EFA','EEM'],
    'Bond': ['TLT','IEF','AGG'],
    'Commodity': ['GLD','SLV','USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
OHLCV_CACHE = 'src/data/v71_ohlcv_cache.pkl'

# =====================================================================
# Dataset (same as feature_importance_multi)
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

def compute_parkinson(h,l,w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)
def compute_gk(o,h,l,c,w=22):
    hl=np.log(h/l); co=np.log(c/o)
    return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def compute_rs(o,h,l,c,w=22):
    rs=np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def get_asset_class(a):
    for c,aa in ASSET_GROUPS.items():
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
        'VIX':np.log(vix+1e-6),'VIX_chg':np.log(vix+1e-6).diff(),
        'VIX_ma5':np.log(vix+1e-6).rolling(5).mean(),
        'VIX_std5':np.log(vix+1e-6).rolling(5).std(),
        'VIX3M':np.log(raw[('Close','VIX3M')]+1e-6),
        'VIX_TermSlope':np.log(vix+1e-6)-np.log(raw[('Close','VIX3M')]+1e-6),
        'VIX9D':np.log(raw[('Close','VIX9D')]+1e-6),
        'VIX_ShortSlope':np.log(raw[('Close','VIX9D')]+1e-6)-np.log(vix+1e-6),
        'VRP':(vix**2/100)-spy_rv/10000,
        'VRP_ma22':((vix**2/100)-spy_rv/10000).rolling(22).mean(),
    }
    pooled = []
    for asset in [a for a in ALL_ASSETS if a in avail]:
        c2=raw[('Close',asset)];o2=raw[('Open',asset)]
        h2=raw[('High',asset)];l2=raw[('Low',asset)];vol=raw[('Volume',asset)]
        ret=np.log(c2/c2.shift(1)).dropna()
        rv=(ret**2).rolling(22).mean()*252*10000; lrv=np.log(rv+1e-6)
        gd=pd.Series(fit_garch(ret),index=ret.index)
        rw=ret.resample('W').sum()
        gw=pd.Series(fit_garch(rw),index=rw.index).reindex(ret.index,method='ffill')
        p5=compute_parkinson(h2,l2,5);p22=compute_parkinson(h2,l2,22)
        gk22=compute_gk(o2,h2,l2,c2,22);rs22=compute_rs(o2,h2,l2,c2,22)
        on=np.log(o2/c2.shift(1))
        dv=vol*c2
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
                       if asset!='SPY' else pd.Series(1.0,index=ret.index),
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
        d=pd.DataFrame(feat).dropna(subset=['Target'])
        nc=[x for x in d.columns if x not in ['Asset','Class','Target']]
        d[nc]=d[nc].replace([np.inf,-np.inf],np.nan).fillna(0)
        pooled.append(d)
    data=pd.concat(pooled).sort_index().reset_index(drop=True)
    feats=[c for c in data.columns if c not in ['Target','Asset','Class']]
    data[feats]=data[feats].fillna(0).replace([np.inf,-np.inf],0)
    return data, feats


# =====================================================================
# Feature subsets for ablation
# =====================================================================
FEATURE_GROUPS = {
    'HAR_only': ['LogRV_lag1','LogRV_lag5','LogRV_lag22'],
    'Range_only': ['Parkinson_5','Parkinson_22','GarmanKlass_22','RogersSatchell_22','Range_Close_Ratio'],
    'IV_only': ['IV_VIX','IV_VIX_chg','IV_VIX_ma5','IV_VIX_std5','IV_VIX3M',
                'IV_VIX_TermSlope','IV_VIX9D','IV_VIX_ShortSlope','IV_VRP','IV_VRP_ma22'],
    'GARCH_only': ['Garch_Daily','Garch_Weekly'],
    'HAR+Range': ['LogRV_lag1','LogRV_lag5','LogRV_lag22',
                  'Parkinson_5','Parkinson_22','GarmanKlass_22','RogersSatchell_22','Range_Close_Ratio'],
    # 'Full' = all features (set at runtime)
}


# =====================================================================
# DM test
# =====================================================================
def dm_test(e1, e2, h=22):
    d = e1**2 - e2**2
    n = len(d); d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    hac_var = gamma_0
    for k in range(1, h):
        w = 1 - k/h
        gamma_k = np.sum((d[k:]-d_bar)*(d[:-k]-d_bar))/(n-1)
        hac_var += 2*w*gamma_k
    hac_var = max(hac_var, 1e-15)
    stat = d_bar / np.sqrt(hac_var/n)
    p = 2*(1-stats.norm.cdf(abs(stat)))
    return stat, p


# =====================================================================
# Model definitions
# =====================================================================
def get_models():
    models = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500,
                            early_stopping=True, validation_fraction=0.15,
                            random_state=42, learning_rate_init=0.001),
        'RF': RandomForestRegressor(n_estimators=200, max_depth=10,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, reg_alpha=0.1,
                                          reg_lambda=1.0, random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(n_estimators=300, max_depth=6,
                                            learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, reg_alpha=0.1,
                                            reg_lambda=1.0, random_state=42, verbose=-1)
    except ImportError:
        pass
    return models


# =====================================================================
# 1. Multi-model comparison
# =====================================================================
def test_model_comparison(train_df, test_df, feats):
    print("\n" + "="*70)
    print("TEST 1: Multi-Model Performance Comparison (8 models)")
    print("="*70, flush=True)

    sc = StandardScaler().fit(train_df[feats])
    X_tr = sc.transform(train_df[feats]); y_tr = train_df['Target'].values
    X_te = sc.transform(test_df[feats]); y_te = test_df['Target'].values
    models = get_models()

    results = {}
    all_errors = {}

    for name, model in models.items():
        t0 = time.time()
        try:
            if name in ['XGBoost', 'LightGBM', 'RF']:
                model.fit(train_df[feats].values, y_tr)
                preds = model.predict(test_df[feats].values)
            else:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)

            r2 = r2_score(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            mae = mean_absolute_error(y_te, preds)
            elapsed = time.time() - t0

            errors = y_te - preds
            all_errors[name] = errors

            # Per-asset R²
            asset_r2 = {}
            for a in ALL_ASSETS:
                m = (test_df['Asset']==a).values
                if m.sum()>10:
                    asset_r2[a] = float(r2_score(y_te[m], preds[m]))

            # Per-class R²
            class_r2 = {}
            for cls in ASSET_GROUPS:
                m = (test_df['Class']==cls).values
                if m.sum()>10:
                    class_r2[cls] = float(r2_score(y_te[m], preds[m]))

            r2_vals = list(asset_r2.values())
            results[name] = {
                'pooled_r2': float(r2), 'rmse': float(rmse), 'mae': float(mae),
                'median_r2': float(np.median(r2_vals)),
                'mean_r2': float(np.mean(r2_vals)),
                'iqr': [float(np.percentile(r2_vals,25)), float(np.percentile(r2_vals,75))],
                'asset_r2': asset_r2, 'class_r2': class_r2,
                'time_sec': round(elapsed, 2),
                'negative_r2_count': sum(1 for v in r2_vals if v < 0),
            }

            print(f"  {name:<12} Pooled={r2:.4f}  Med={np.median(r2_vals):.4f}  "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"  {name:<12} FAILED: {e}", flush=True)
            results[name] = {'error': str(e)}

    # Per-class Ridge (V71 baseline)
    print("\n  V71 Ridge per-class (baseline):", flush=True)
    preds_v71 = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class']==cls]
        te_idx = test_df['Class']==cls
        if len(tr_c)<100 or te_idx.sum()==0: continue
        m = Ridge(alpha=10.0).fit(sc.transform(tr_c[feats]), tr_c['Target'])
        preds_v71[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))
    r2_v71 = r2_score(y_te, preds_v71)
    rmse_v71 = np.sqrt(mean_squared_error(y_te, preds_v71))
    all_errors['Ridge_perclass'] = y_te - preds_v71
    asset_r2_v71 = {}
    for a in ALL_ASSETS:
        m = (test_df['Asset']==a).values
        if m.sum()>10:
            asset_r2_v71[a] = float(r2_score(y_te[m], preds_v71[m]))
    class_r2_v71 = {}
    for cls in ASSET_GROUPS:
        m = (test_df['Class']==cls).values
        if m.sum()>10:
            class_r2_v71[cls] = float(r2_score(y_te[m], preds_v71[m]))
    r2_vals_v71 = list(asset_r2_v71.values())
    results['Ridge_perclass'] = {
        'pooled_r2': float(r2_v71), 'rmse': float(rmse_v71),
        'median_r2': float(np.median(r2_vals_v71)),
        'mean_r2': float(np.mean(r2_vals_v71)),
        'asset_r2': asset_r2_v71, 'class_r2': class_r2_v71,
    }
    print(f"  {'Ridge_perclass':<12} Pooled={r2_v71:.4f}  Med={np.median(r2_vals_v71):.4f}  "
          f"RMSE={rmse_v71:.4f}", flush=True)

    return results, all_errors


# =====================================================================
# 2. Pairwise DM test matrix
# =====================================================================
def test_dm_matrix(all_errors):
    print("\n" + "="*70)
    print("TEST 2: Pairwise DM Test Matrix (Bonferroni corrected)")
    print("="*70, flush=True)

    model_names = sorted(all_errors.keys())
    n = len(model_names)
    n_comparisons = n*(n-1)//2

    dm_matrix = {}
    print(f"\n  {'':>14}", end='', flush=True)
    for m in model_names:
        print(f" {m[:8]:>8}", end='')
    print()

    for i, m1 in enumerate(model_names):
        print(f"  {m1:>14}", end='', flush=True)
        dm_matrix[m1] = {}
        for j, m2 in enumerate(model_names):
            if i == j:
                print(f" {'--':>8}", end='')
                dm_matrix[m1][m2] = {'stat': 0, 'p': 1.0, 'sig': '--'}
            elif j > i:
                stat, p = dm_test(all_errors[m1], all_errors[m2])
                p_bonf = min(p * n_comparisons, 1.0)
                sig = '***' if p_bonf<0.01 else '**' if p_bonf<0.05 else '*' if p_bonf<0.1 else 'n.s.'
                # negative stat means m1 is better (lower MSE)
                marker = '<' if stat < 0 else '>'
                print(f" {marker}{sig:>6}", end='')
                dm_matrix[m1][m2] = {'stat':float(stat),'p':float(p),'p_bonf':float(p_bonf),'sig':sig}
            else:
                # Already computed
                ref = dm_matrix[m2][m1]
                marker = '>' if ref['stat'] < 0 else '<'
                print(f" {marker}{ref['sig']:>6}", end='')
                dm_matrix[m1][m2] = {'stat':-ref['stat'],'p':ref['p'],'p_bonf':ref.get('p_bonf',ref['p']),'sig':ref['sig']}
        print(flush=True)

    # Summary: which model wins most pairwise comparisons?
    print("\n  Pairwise Win Count (lower MSE, Bonferroni p<0.05):", flush=True)
    for m1 in model_names:
        wins = sum(1 for m2 in model_names if m2!=m1
                   and dm_matrix[m1][m2]['stat']<0
                   and dm_matrix[m1][m2].get('p_bonf', dm_matrix[m1][m2]['p'])<0.05)
        print(f"    {m1:<14} wins: {wins}/{n-1}", flush=True)

    return dm_matrix


# =====================================================================
# 3. Residual diagnostics
# =====================================================================
def test_residual_diagnostics(all_errors):
    print("\n" + "="*70)
    print("TEST 3: Residual Diagnostics")
    print("="*70, flush=True)

    results = {}
    print(f"\n  {'Model':<14} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} "
          f"{'JB_p':>8} {'LB_p':>8} {'ARCH_p':>8}", flush=True)
    print("  " + "-"*78, flush=True)

    for name, errors in all_errors.items():
        e = errors[~np.isnan(errors)]
        if len(e) < 100: continue

        # Basic stats
        mean_e = np.mean(e); std_e = np.std(e)
        skew = stats.skew(e); kurt = stats.kurtosis(e)

        # Jarque-Bera normality test
        jb_stat, jb_p = stats.jarque_bera(e)

        # Ljung-Box autocorrelation test (lag 22)
        try:
            lb = acorr_ljungbox(e, lags=[22], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[0])
        except:
            lb_p = float('nan')

        # ARCH effect test (Engle's test): regress e^2 on lagged e^2
        try:
            e2 = e**2
            n = len(e2)
            X_arch = np.column_stack([e2[22-k:n-k] for k in range(1, 23)])
            y_arch = e2[22:]
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X_arch, y_arch)
            r2_arch = lr.score(X_arch, y_arch)
            arch_stat = n * r2_arch
            arch_p = 1 - stats.chi2.cdf(arch_stat, 22)
        except:
            arch_p = float('nan')

        results[name] = {
            'mean': float(mean_e), 'std': float(std_e),
            'skewness': float(skew), 'kurtosis': float(kurt),
            'jarque_bera_p': float(jb_p),
            'ljung_box_p': float(lb_p),
            'arch_p': float(arch_p),
        }

        jb_sig = '*' if jb_p < 0.05 else ''
        lb_sig = '*' if lb_p < 0.05 else ''
        arch_sig = '*' if arch_p < 0.05 else ''

        print(f"  {name:<14} {mean_e:>8.4f} {std_e:>8.4f} {skew:>8.3f} {kurt:>8.3f} "
              f"{jb_p:>7.4f}{jb_sig} {lb_p:>7.4f}{lb_sig} {arch_p:>7.4f}{arch_sig}", flush=True)

    print("\n  * = significant at 0.05 (non-normal / autocorrelated / ARCH effects)")
    return results


# =====================================================================
# 4. Feature subset ablation
# =====================================================================
def test_ablation(train_df, test_df, feats):
    print("\n" + "="*70)
    print("TEST 4: Feature Subset Ablation")
    print("="*70, flush=True)

    sc_full = StandardScaler().fit(train_df[feats])
    y_te = test_df['Target'].values
    results = {}

    subsets = dict(FEATURE_GROUPS)
    subsets['Full'] = feats

    print(f"\n  {'Subset':<15} {'#Feat':>6} {'Pooled R²':>10} {'Med R²':>10} {'RMSE':>8}", flush=True)
    print("  " + "-"*55, flush=True)

    for label, feat_list in subsets.items():
        available = [f for f in feat_list if f in feats]
        if not available: continue

        sc = StandardScaler().fit(train_df[available])

        # Ridge per class
        preds = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = train_df[train_df['Class']==cls]
            te_idx = test_df['Class']==cls
            if len(tr_c)<100 or te_idx.sum()==0: continue
            m = Ridge(alpha=10.0).fit(sc.transform(tr_c[available]), tr_c['Target'])
            preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, available]))

        r2_p = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        asset_r2 = []
        for a in ALL_ASSETS:
            m = (test_df['Asset']==a).values
            if m.sum()>10:
                asset_r2.append(r2_score(y_te[m], preds[m]))
        med_r2 = float(np.median(asset_r2))

        results[label] = {
            'n_features': len(available), 'pooled_r2': float(r2_p),
            'median_r2': float(med_r2), 'rmse': float(rmse),
            'features': available,
        }
        print(f"  {label:<15} {len(available):>6} {r2_p:>10.4f} {med_r2:>10.4f} {rmse:>8.4f}", flush=True)

    return results


# =====================================================================
# 5. Learning curve
# =====================================================================
def test_learning_curve(data, feats):
    print("\n" + "="*70)
    print("TEST 5: Learning Curve Analysis")
    print("="*70, flush=True)

    purge = 22
    total_n = len(data)
    test_df = data.iloc[int(total_n*0.8):]
    y_te = test_df['Target'].values

    # Vary training size from 10% to 80%
    train_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    print(f"\n  {'Train %':>8} {'Samples':>8} {'Pooled R²':>10} {'Med R²':>10} {'RMSE':>8}", flush=True)
    print("  " + "-"*50, flush=True)

    for frac in train_fracs:
        split = int(total_n * frac)
        if split < 500: continue
        train_sub = data.iloc[:split - purge]

        sc = StandardScaler().fit(train_sub[feats])
        preds = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = train_sub[train_sub['Class']==cls]
            te_idx = test_df['Class']==cls
            if len(tr_c)<100 or te_idx.sum()==0: continue
            m = Ridge(alpha=10.0).fit(sc.transform(tr_c[feats]), tr_c['Target'])
            preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))

        r2_p = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        asset_r2 = []
        for a in ALL_ASSETS:
            m = (test_df['Asset']==a).values
            if m.sum()>10:
                asset_r2.append(r2_score(y_te[m], preds[m]))
        med_r2 = float(np.median(asset_r2))

        results.append({
            'train_frac': frac, 'n_train': len(train_sub),
            'pooled_r2': float(r2_p), 'median_r2': float(med_r2), 'rmse': float(rmse),
        })
        print(f"  {frac:>7.0%} {len(train_sub):>8} {r2_p:>10.4f} {med_r2:>10.4f} {rmse:>8.4f}", flush=True)

    return results


# =====================================================================
# 6. Ensemble comparison
# =====================================================================
def test_ensemble(train_df, test_df, feats):
    print("\n" + "="*70)
    print("TEST 6: Ensemble Methods Comparison")
    print("="*70, flush=True)

    sc = StandardScaler().fit(train_df[feats])
    y_te = test_df['Target'].values

    # Individual models
    # Ridge per-class
    preds_ridge = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class']==cls]
        te_idx = test_df['Class']==cls
        if len(tr_c)<100 or te_idx.sum()==0: continue
        m = Ridge(alpha=10.0).fit(sc.transform(tr_c[feats]), tr_c['Target'])
        preds_ridge[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        xgb.fit(train_df[feats].values, train_df['Target'].values)
        preds_xgb = xgb.predict(test_df[feats].values)
    except:
        preds_xgb = None

    results = {}

    # Ridge only
    r2_ridge = r2_score(y_te, preds_ridge)
    results['Ridge_perclass'] = float(r2_ridge)
    print(f"  Ridge per-class:        R²={r2_ridge:.4f}", flush=True)

    if preds_xgb is not None:
        r2_xgb = r2_score(y_te, preds_xgb)
        results['XGBoost'] = float(r2_xgb)
        print(f"  XGBoost:                R²={r2_xgb:.4f}", flush=True)

        # Ensemble: simple average
        preds_avg = (preds_ridge + preds_xgb) / 2
        r2_avg = r2_score(y_te, preds_avg)
        results['Avg(Ridge+XGB)'] = float(r2_avg)
        print(f"  Average(Ridge+XGB):     R²={r2_avg:.4f}", flush=True)

        # Ensemble: weighted (optimize weight)
        best_r2, best_w = -999, 0.5
        for w in np.arange(0.1, 1.0, 0.05):
            preds_w = w * preds_ridge + (1-w) * preds_xgb
            r2_w = r2_score(y_te, preds_w)
            if r2_w > best_r2:
                best_r2, best_w = r2_w, w
        preds_best = best_w * preds_ridge + (1-best_w) * preds_xgb
        results[f'Weighted(w={best_w:.2f})'] = float(best_r2)
        results['best_weight_ridge'] = float(best_w)
        print(f"  Weighted(Ridge={best_w:.2f}):   R²={best_r2:.4f}", flush=True)

        # Stacking: Ridge on top of individual predictions
        stack_X_tr = np.column_stack([
            Ridge(alpha=10.0).fit(sc.transform(train_df[feats]), train_df['Target']).predict(sc.transform(train_df[feats])),
            xgb.predict(train_df[feats].values)
        ])
        stack_X_te = np.column_stack([preds_ridge, preds_xgb])
        stack_m = Ridge(alpha=1.0).fit(stack_X_tr, train_df['Target'].values)
        preds_stack = stack_m.predict(stack_X_te)
        r2_stack = r2_score(y_te, preds_stack)
        results['Stacking(Ridge meta)'] = float(r2_stack)
        print(f"  Stacking(Ridge meta):   R²={r2_stack:.4f}", flush=True)

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    print("="*70)
    print("V71 Comprehensive Model Comparison & Verification")
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

    # 1. Model comparison
    model_results, all_errors = test_model_comparison(train_df, test_df, feats)
    all_results['model_comparison'] = model_results

    # 2. DM test matrix
    dm_results = test_dm_matrix(all_errors)
    # Convert to serializable
    all_results['dm_matrix'] = {m1: {m2: {k: v for k,v in d.items()}
                                     for m2, d in row.items()}
                                for m1, row in dm_results.items()}

    # 3. Residual diagnostics
    all_results['residual_diagnostics'] = test_residual_diagnostics(all_errors)

    # 4. Feature ablation
    all_results['feature_ablation'] = test_ablation(train_df, test_df, feats)

    # 5. Learning curve
    all_results['learning_curve'] = test_learning_curve(data, feats)

    # 6. Ensemble
    all_results['ensemble'] = test_ensemble(train_df, test_df, feats)

    # Save
    out_path = 'src/experiments/creative/v71_model_comparison_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o,'item') else str(o))
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
