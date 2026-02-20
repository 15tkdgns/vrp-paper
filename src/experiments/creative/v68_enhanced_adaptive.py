"""
V68: Enhanced Asset-Adaptive Model
Goal: Beat V36 (R²=0.755) for 22-day prediction

Improvements over V36:
1. Multi-Horizon GARCH features (from V35)
2. Cross-Asset features (VIX proxy, correlation)
3. Per-asset alpha optimization (from V67 idea)
4. More lag features (lag10, rolling std)
5. XGBoost comparison
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from arch import arch_model
import os
import json
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# Asset Classification  
ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

def fit_garch(returns):
    try:
        ret_scaled = returns * 100
        am = arch_model(ret_scaled, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def run_experiment():
    print("="*80, flush=True)
    print("V68: Enhanced Asset-Adaptive Model", flush=True)
    print("="*80, flush=True)
    
    # Data Loading
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading cache: {CACHE_PATH}...", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    else:
        print("Downloading...", flush=True)
        raw = yf.download(ALL_ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # ============================================
    # Cross-Asset Features: VIX proxy from SPY
    # ============================================
    spy_ret = np.log(raw['SPY'] / raw['SPY'].shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    vix_proxy = np.log(spy_rv + 1e-6)
    
    pooled_data = []
    print(f"Processing {len(ALL_ASSETS)} assets with enhanced features...", flush=True)
    
    for i, asset in enumerate(ALL_ASSETS):
        if asset not in raw.columns:
            continue
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        
        # Base: Realized Volatility
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH feature
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        
        # Weekly GARCH
        ret_weekly = ret_daily.resample('W').sum()
        garch_w_raw = fit_garch(ret_weekly)
        garch_w = pd.Series(garch_w_raw, index=ret_weekly.index).reindex(ret_daily.index, method='ffill')
        
        # Rolling Std of LogRV (volatility of volatility)
        log_rv_std5 = log_rv.rolling(5).std()
        log_rv_std22 = log_rv.rolling(22).std()
        
        # Momentum features
        rv_momentum = log_rv - log_rv.shift(5)  # 5-day momentum
        rv_momentum22 = log_rv - log_rv.shift(22)  # 22-day momentum
        
        # Cross-asset correlation with SPY
        if asset != 'SPY':
            cross_corr = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index))
        else:
            cross_corr = pd.Series(1.0, index=ret_daily.index)
        
        d = pd.DataFrame({
            # HAR features (core)
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            
            # GARCH features
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            
            # Vol-of-Vol features
            'LogRV_Std5': log_rv_std5.shift(1),
            'LogRV_Std22': log_rv_std22.shift(1),
            
            # Momentum features
            'RV_Mom5': rv_momentum.shift(1),
            'RV_Mom22': rv_momentum22.shift(1),
            
            # Cross-Asset features
            'VIX_Proxy': vix_proxy.shift(1),
            'Cross_Corr': cross_corr.shift(1),
            
            # Return features
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
            
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        
        # Asset Class
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        print(f"  [{i+1}/{len(ALL_ASSETS)}] {asset}: {len(d)} samples", flush=True)
    
    data = pd.concat(pooled_data).sort_index()
    print(f"\nTotal samples: {len(data)}", flush=True)
    
    # Train/Test Split
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    print(f"Features ({len(feats)}): {feats}", flush=True)
    
    # ============================================
    # Strategy 1: V36 Baseline (alpha=1.0)
    # ============================================
    print("\n--- Strategy 1: V36 Baseline (Class-Specific, alpha=1.0) ---", flush=True)
    sc = StandardScaler()
    sc.fit(train_df[feats])
    
    preds_v36 = pd.Series(index=test_df.index, dtype=float)
    for cls in ASSET_GROUPS.keys():
        train_cls = train_df[train_df['Class'] == cls]
        test_mask = test_df['Class'] == cls
        if len(train_cls) < 100 or test_mask.sum() == 0:
            continue
        X_tr = sc.transform(train_cls[feats])
        X_te = sc.transform(test_df.loc[test_mask, feats])
        m = Ridge(alpha=1.0).fit(X_tr, train_cls['Target'])
        preds_v36[test_mask] = m.predict(X_te)
    
    r2_v36 = r2_score(test_df['Target'], preds_v36)
    print(f"  V36 Baseline R²: {r2_v36:.5f}", flush=True)
    
    # ============================================
    # Strategy 2: Per-Class Alpha Optimization
    # ============================================
    print("\n--- Strategy 2: Per-Class Alpha Optimization ---", flush=True)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    
    # Use validation split from training set
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]
    
    best_alphas = {}
    for cls in ASSET_GROUPS.keys():
        best_r2, best_a = -999, 1.0
        train_cls = train_inner[train_inner['Class'] == cls]
        val_cls = val_inner[val_inner['Class'] == cls]
        if len(train_cls) < 100 or len(val_cls) < 50:
            best_alphas[cls] = 1.0
            continue
        X_tr = sc.transform(train_cls[feats])
        X_val = sc.transform(val_cls[feats])
        for a in alphas:
            m = Ridge(alpha=a).fit(X_tr, train_cls['Target'])
            r2 = r2_score(val_cls['Target'], m.predict(X_val))
            if r2 > best_r2:
                best_r2, best_a = r2, a
        best_alphas[cls] = best_a
        print(f"  {cls}: best_alpha={best_a}, val_R²={best_r2:.5f}", flush=True)
    
    # Retrain with best alphas on full training set
    preds_opt = pd.Series(index=test_df.index, dtype=float)
    for cls in ASSET_GROUPS.keys():
        train_cls = train_df[train_df['Class'] == cls]
        test_mask = test_df['Class'] == cls
        if len(train_cls) < 100 or test_mask.sum() == 0:
            continue
        X_tr = sc.transform(train_cls[feats])
        X_te = sc.transform(test_df.loc[test_mask, feats])
        m = Ridge(alpha=best_alphas[cls]).fit(X_tr, train_cls['Target'])
        preds_opt[test_mask] = m.predict(X_te)
    
    r2_opt = r2_score(test_df['Target'], preds_opt)
    print(f"  Optimized Alpha R²: {r2_opt:.5f}", flush=True)
    
    # ============================================
    # Strategy 3: Per-Asset Models
    # ============================================
    print("\n--- Strategy 3: Per-Asset Models ---", flush=True)
    preds_asset = pd.Series(index=test_df.index, dtype=float)
    for asset in ALL_ASSETS:
        train_a = train_df[train_df['Asset'] == asset]
        test_mask = test_df['Asset'] == asset
        if len(train_a) < 100 or test_mask.sum() == 0:
            continue
        X_tr = sc.transform(train_a[feats])
        X_te = sc.transform(test_df.loc[test_mask, feats])
        # Quick alpha search
        best_r2_a, best_alpha_a = -999, 1.0
        val_s = int(len(train_a) * 0.8)
        tr_inner = train_a.iloc[:val_s]
        va_inner = train_a.iloc[val_s:]
        if len(va_inner) > 30:
            X_tr_i = sc.transform(tr_inner[feats])
            X_va_i = sc.transform(va_inner[feats])
            for a in [0.1, 1.0, 10.0, 100.0]:
                m = Ridge(alpha=a).fit(X_tr_i, tr_inner['Target'])
                r2_a = r2_score(va_inner['Target'], m.predict(X_va_i))
                if r2_a > best_r2_a:
                    best_r2_a, best_alpha_a = r2_a, a
        m = Ridge(alpha=best_alpha_a).fit(X_tr, train_a['Target'])
        preds_asset[test_mask] = m.predict(X_te)
    
    # Handle missing
    missing = preds_asset.isna()
    if missing.sum() > 0:
        preds_asset[missing] = preds_opt[missing]
    r2_asset = r2_score(test_df['Target'], preds_asset)
    print(f"  Per-Asset R²: {r2_asset:.5f}", flush=True)
    
    # ============================================
    # Strategy 4: XGBoost (if available)
    # ============================================
    try:
        from xgboost import XGBRegressor
        print("\n--- Strategy 4: XGBoost Per-Class ---", flush=True)
        preds_xgb = pd.Series(index=test_df.index, dtype=float)
        for cls in ASSET_GROUPS.keys():
            train_cls = train_df[train_df['Class'] == cls]
            test_mask = test_df['Class'] == cls
            if len(train_cls) < 100 or test_mask.sum() == 0:
                continue
            X_tr = sc.transform(train_cls[feats])
            X_te = sc.transform(test_df.loc[test_mask, feats])
            m = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbosity=0
            ).fit(X_tr, train_cls['Target'])
            preds_xgb[test_mask] = m.predict(X_te)
        r2_xgb = r2_score(test_df['Target'], preds_xgb)
        print(f"  XGBoost Per-Class R²: {r2_xgb:.5f}", flush=True)
    except ImportError:
        print("\n--- Strategy 4: XGBoost not available, skipping ---", flush=True)
        r2_xgb = None
    
    # ============================================
    # Strategy 5: Ensemble (Average of best)
    # ============================================
    print("\n--- Strategy 5: Ensemble ---", flush=True)
    ensemble_preds = preds_opt.copy()
    n_models = 1
    if r2_xgb is not None and r2_xgb > 0.5:
        ensemble_preds = ensemble_preds + preds_xgb
        n_models += 1
    if r2_asset > 0.5:
        ensemble_preds = ensemble_preds + preds_asset
        n_models += 1
    ensemble_preds = ensemble_preds / n_models
    r2_ensemble = r2_score(test_df['Target'], ensemble_preds)
    print(f"  Ensemble R² ({n_models} models): {r2_ensemble:.5f}", flush=True)
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*80, flush=True)
    print("SUMMARY", flush=True)
    print("="*80, flush=True)
    results = {
        'V36_baseline': r2_v36,
        'alpha_optimized': r2_opt,
        'per_asset': r2_asset,
        'ensemble': r2_ensemble,
        'best_alphas': best_alphas
    }
    if r2_xgb is not None:
        results['xgboost'] = r2_xgb
    
    all_r2 = {
        'V36 Baseline (enhanced feat)': r2_v36,
        'Alpha Optimized': r2_opt,
        'Per-Asset': r2_asset,
        'Ensemble': r2_ensemble
    }
    if r2_xgb is not None:
        all_r2['XGBoost'] = r2_xgb
    
    for name, r2 in sorted(all_r2.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if r2 == max(all_r2.values()) else ""
        print(f"  {name}: R²={r2:.5f}{marker}", flush=True)
    
    best_name = max(all_r2, key=all_r2.get)
    best_r2 = max(all_r2.values())
    print(f"\nBest: {best_name} (R²={best_r2:.5f})", flush=True)
    print(f"Original V36: 0.755", flush=True)
    print(f"Improvement: {(best_r2 - 0.755):.5f} ({(best_r2 - 0.755)/0.755*100:.1f}%)", flush=True)
    
    results['best_name'] = best_name
    results['best_r2'] = best_r2
    
    with open('src/experiments/creative/v68_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v68_results.json", flush=True)

if __name__ == "__main__":
    run_experiment()
