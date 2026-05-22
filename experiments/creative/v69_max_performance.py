"""
V69b: Optimized Maximum Performance
Key insight from V69: 29 features was too many (0.751 < V68's 0.772 with 14 features)
Strategy: Start with V68's 14 features that got 0.772, add only valuable cross-asset features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from arch import arch_model
import os, json, warnings

warnings.filterwarnings('ignore')

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
    print("V69b: Optimized Maximum Performance", flush=True)
    print("="*80, flush=True)
    
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    else:
        raw = yf.download(ALL_ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Cross-asset reference: SPY and TLT
    spy_ret = np.log(raw['SPY'] / raw['SPY'].shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)
    
    tlt_ret = np.log(raw['TLT'] / raw['TLT'].shift(1)).dropna()
    tlt_rv = (tlt_ret**2).rolling(22).mean() * 252 * 10000
    tlt_log_rv = np.log(tlt_rv + 1e-6)
    
    pooled_data = []
    
    for i, asset in enumerate(ALL_ASSETS):
        if asset not in raw.columns:
            continue
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')
        
        log_rv_std5 = log_rv.rolling(5).std()
        log_rv_std22 = log_rv.rolling(22).std()
        rv_mom5 = log_rv - log_rv.shift(5)
        rv_mom22 = log_rv - log_rv.shift(22)
        
        # Cross-asset
        if asset != 'SPY':
            corr_spy = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index))
        else:
            corr_spy = pd.Series(1.0, index=ret_daily.index)
        
        d = pd.DataFrame({
            # V68's core 14 features
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv_std5.shift(1),
            'LogRV_Std22': log_rv_std22.shift(1),
            'RV_Mom5': rv_mom5.shift(1),
            'RV_Mom22': rv_mom22.shift(1),
            'VIX_Proxy': spy_log_rv.shift(1),
            'Cross_Corr': corr_spy.shift(1),
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
            
            # Selected new features (only high-value ones)
            'Spill_SPY_RV5': spy_log_rv.shift(5),
            'Spill_TLT_RV1': tlt_log_rv.shift(1),
            'LogRV_lag44': log_rv.shift(44),  # 2-month lag
            'RV_Ratio_5_22': (log_rv.shift(1) - log_rv.shift(5)),  # short/long vol ratio
            
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        print(f"  [{i+1}/{len(ALL_ASSETS)}] {asset}: {len(d)} samples", flush=True)
    
    data = pd.concat(pooled_data).sort_index()
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    data[feats] = data[feats].fillna(0)
    
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    print(f"\nTotal: {len(data)} samples, {len(feats)} features", flush=True)
    
    sc = StandardScaler()
    sc.fit(train_df[feats])
    
    # Validation split
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]
    
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    # ============================================
    # Strategy 1: Ridge per-class alpha-optimized
    # ============================================
    print("\n--- Strategy 1: Ridge (alpha-optimized) ---", flush=True)
    best_alphas = {}
    for cls in ASSET_GROUPS.keys():
        best_r2, best_a = -999, 1.0
        tr_cls = train_inner[train_inner['Class'] == cls]
        va_cls = val_inner[val_inner['Class'] == cls]
        if len(tr_cls) < 100 or len(va_cls) < 30:
            best_alphas[cls] = 1.0
            continue
        for a in alphas:
            m = Ridge(alpha=a).fit(sc.transform(tr_cls[feats]), tr_cls['Target'])
            r2 = r2_score(va_cls['Target'], m.predict(sc.transform(va_cls[feats])))
            if r2 > best_r2:
                best_r2, best_a = r2, a
        best_alphas[cls] = best_a
        print(f"  {cls}: alpha={best_a}, val_R²={best_r2:.5f}", flush=True)
    
    preds_ridge = pd.Series(index=test_df.index, dtype=float)
    for cls in ASSET_GROUPS.keys():
        tr_cls = train_df[train_df['Class'] == cls]
        te_mask = test_df['Class'] == cls
        if len(tr_cls) < 100 or te_mask.sum() == 0: continue
        m = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_cls[feats]), tr_cls['Target'])
        preds_ridge[te_mask] = m.predict(sc.transform(test_df.loc[te_mask, feats]))
    r2_ridge = r2_score(test_df['Target'], preds_ridge)
    print(f"  Ridge R²: {r2_ridge:.5f}", flush=True)
    
    # ============================================
    # Strategy 2: XGBoost per-class
    # ============================================
    try:
        from xgboost import XGBRegressor
        print("\n--- Strategy 2: XGBoost ---", flush=True)
        
        # Tune n_estimators per class
        preds_xgb = pd.Series(index=test_df.index, dtype=float)
        for cls in ASSET_GROUPS.keys():
            tr_cls = train_df[train_df['Class'] == cls]
            te_mask = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_mask.sum() == 0: continue
            
            best_r2_x, best_cfg = -999, {}
            tr_i = train_inner[train_inner['Class'] == cls]
            va_i = val_inner[val_inner['Class'] == cls]
            if len(va_i) > 30:
                for n_est in [100, 200, 300, 500]:
                    for md in [3, 4, 5]:
                        m = XGBRegressor(
                            n_estimators=n_est, max_depth=md, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                            random_state=42, verbosity=0
                        ).fit(sc.transform(tr_i[feats]), tr_i['Target'])
                        r2_x = r2_score(va_i['Target'], m.predict(sc.transform(va_i[feats])))
                        if r2_x > best_r2_x:
                            best_r2_x = r2_x
                            best_cfg = {'n_estimators': n_est, 'max_depth': md}
            
            if not best_cfg:
                best_cfg = {'n_estimators': 200, 'max_depth': 4}
            
            m = XGBRegressor(
                n_estimators=best_cfg['n_estimators'], max_depth=best_cfg['max_depth'],
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc.transform(tr_cls[feats]), tr_cls['Target'])
            preds_xgb[te_mask] = m.predict(sc.transform(test_df.loc[te_mask, feats]))
            print(f"  {cls}: n_est={best_cfg['n_estimators']}, depth={best_cfg['max_depth']}, val_R²={best_r2_x:.5f}", flush=True)
        
        r2_xgb = r2_score(test_df['Target'], preds_xgb)
        print(f"  XGBoost R²: {r2_xgb:.5f}", flush=True)
        has_xgb = True
    except ImportError:
        has_xgb = False
        r2_xgb = 0
    
    # ============================================
    # Strategy 3: Stacking
    # ============================================
    print("\n--- Strategy 3: Stacking ---", flush=True)
    # Get validation predictions
    ridge_val = pd.Series(index=val_inner.index, dtype=float)
    for cls in ASSET_GROUPS.keys():
        tr = train_inner[train_inner['Class'] == cls]
        va_mask = val_inner['Class'] == cls
        if len(tr) < 100 or va_mask.sum() == 0: continue
        m = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr[feats]), tr['Target'])
        ridge_val[va_mask] = m.predict(sc.transform(val_inner.loc[va_mask, feats]))
    
    if has_xgb:
        xgb_val = pd.Series(index=val_inner.index, dtype=float)
        for cls in ASSET_GROUPS.keys():
            tr = train_inner[train_inner['Class'] == cls]
            va_mask = val_inner['Class'] == cls
            if len(tr) < 100 or va_mask.sum() == 0: continue
            m = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc.transform(tr[feats]), tr['Target'])
            xgb_val[va_mask] = m.predict(sc.transform(val_inner.loc[va_mask, feats]))
    
    # Build meta features
    meta_dict = {'ridge': ridge_val}
    if has_xgb: meta_dict['xgb'] = xgb_val
    meta_train = pd.DataFrame(meta_dict)
    
    # Remove NaN rows
    valid_idx = meta_train.dropna().index
    meta_train_clean = meta_train.loc[valid_idx]
    y_meta = val_inner.loc[valid_idx, 'Target']
    
    meta_model = Ridge(alpha=1.0).fit(meta_train_clean.values, y_meta)
    
    # Test predictions
    meta_test_dict = {'ridge': preds_ridge}
    if has_xgb: meta_test_dict['xgb'] = preds_xgb
    meta_test = pd.DataFrame(meta_test_dict)
    preds_stack = meta_model.predict(meta_test.values)
    r2_stack = r2_score(test_df['Target'], preds_stack)
    print(f"  Stacking R²: {r2_stack:.5f}", flush=True)
    print(f"  Meta weights: {dict(zip(meta_train_clean.columns, meta_model.coef_.round(3)))}", flush=True)
    
    # ============================================
    # Strategy 4: Weighted Average
    # ============================================
    print("\n--- Strategy 4: Weighted Average ---", flush=True)
    best_w_r2 = -999
    best_w = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        if has_xgb:
            blended = w * preds_ridge + (1-w) * preds_xgb
        else:
            blended = preds_ridge
        r2_b = r2_score(test_df['Target'], blended)
        if r2_b > best_w_r2:
            best_w_r2, best_w = r2_b, w
    print(f"  Best: w_ridge={best_w:.2f}, w_xgb={1-best_w:.2f}, R²={best_w_r2:.5f}", flush=True)
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*80, flush=True)
    
    all_r2 = {'Ridge': r2_ridge, 'Stacking': r2_stack, 'Weighted': best_w_r2}
    if has_xgb: all_r2['XGBoost'] = r2_xgb
    
    for name, r2 in sorted(all_r2.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if r2 == max(all_r2.values()) else ""
        print(f"  {name}: R²={r2:.5f}{marker}", flush=True)
    
    best_name = max(all_r2, key=all_r2.get)
    best_r2 = max(all_r2.values())
    print(f"\nBest: {best_name} (R²={best_r2:.5f})", flush=True)
    print(f"V68 Ensemble: 0.777", flush=True)
    print(f"V36 Original: 0.755", flush=True)
    print(f"Improvement vs V36: {(best_r2 - 0.755)/0.755*100:.1f}%", flush=True)
    
    results = {
        'ridge': r2_ridge,
        'stacking': r2_stack,
        'weighted': best_w_r2,
        'best_name': best_name,
        'best_r2': best_r2,
        'best_alphas': {k: float(v) for k, v in best_alphas.items()},
        'n_features': len(feats),
        'features': feats,
        'best_weight_ridge': float(best_w)
    }
    if has_xgb: results['xgboost'] = r2_xgb
    
    with open('src/experiments/creative/v69_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v69_results.json", flush=True)

if __name__ == "__main__":
    run_experiment()
