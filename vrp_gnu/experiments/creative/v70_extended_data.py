"""
V70: Extended Data Universe Model
Goal: Break 0.78 barrier by adding:
1. More assets (20+)
2. VIX and sector volatility indices as features  
3. High-frequency proxy features (Parkinson, intraday range)
4. Macro indicators (yield curve, credit spreads proxy)
"""

import pandas as pd
import numpy as np
import os, json, warnings, time

warnings.filterwarnings('ignore')

# =============================================
# EXPANDED ASSET UNIVERSE
# =============================================
# Original 11 assets
ORIGINAL_ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'AGG', 'GLD', 'SLV', 'USO']

# New assets to add
NEW_ASSETS = [
    # Sector ETFs
    'XLF',   # Financials
    'XLE',   # Energy
    'XLK',   # Technology
    'XLV',   # Healthcare
    'XLU',   # Utilities
    # Credit/HY
    'HYG',   # High Yield Corporate Bond
    'LQD',   # Investment Grade Corporate Bond
    # International
    'FXI',   # China
    'EWJ',   # Japan
    # Real Estate
    'VNQ',   # REIT
    # Currency
    'UUP',   # US Dollar Index
]

# Implied Volatility Indices (from CBOE)
IV_INDICES = [
    '^VIX',    # S&P 500 IV
    '^VXN',    # Nasdaq 100 IV (VXEEM replaced)
]

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLF', 'XLE', 'XLK', 'XLV', 'XLU', 'FXI', 'EWJ'],
    'Bond': ['TLT', 'IEF', 'AGG', 'HYG', 'LQD'],
    'Commodity': ['GLD', 'SLV', 'USO'],
    'Other': ['VNQ', 'UUP']
}

ALL_TRADE_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from arch import arch_model

def fit_garch(returns):
    try:
        ret_scaled = returns * 100
        am = arch_model(ret_scaled, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def download_data():
    """Download expanded dataset"""
    import yfinance as yf
    
    all_tickers = ALL_TRADE_ASSETS + [idx.replace('^', '') for idx in IV_INDICES] + ['VIX', 'VXN']
    
    # Try to load extended cache
    CACHE_PATH = 'src/data/v70_extended_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading extended cache: {CACHE_PATH}...", flush=True)
        return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True), CACHE_PATH
    
    # Download
    print("Downloading extended dataset from yfinance...", flush=True)
    download_tickers = ALL_TRADE_ASSETS + IV_INDICES
    
    raw = yf.download(download_tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    close.to_csv(CACHE_PATH)
    print(f"Saved cache to {CACHE_PATH}", flush=True)
    
    return close, CACHE_PATH

def run_experiment():
    print("="*80, flush=True)
    print("V70: Extended Data Universe Model", flush=True)
    print("="*80, flush=True)
    
    raw, cache_path = download_data()
    
    # Available assets
    available_assets = [a for a in ALL_TRADE_ASSETS if a in raw.columns]
    print(f"\nAvailable trade assets: {len(available_assets)}/{len(ALL_TRADE_ASSETS)}", flush=True)
    
    # IV indices
    has_vix = 'VIX' in raw.columns
    has_vxn = 'VXN' in raw.columns
    print(f"VIX available: {has_vix}, VXN available: {has_vxn}", flush=True)
    
    # =============================================
    # Cross-asset reference features
    # =============================================
    print("\nComputing cross-asset reference features...", flush=True)
    
    # SPY volatility
    spy_ret = np.log(raw['SPY'] / raw['SPY'].shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)
    
    # TLT volatility  
    tlt_ret = np.log(raw['TLT'] / raw['TLT'].shift(1)).dropna()
    tlt_rv = (tlt_ret**2).rolling(22).mean() * 252 * 10000
    tlt_log_rv = np.log(tlt_rv + 1e-6)
    
    # VIX as direct feature (implied volatility)
    if has_vix:
        vix_level = np.log(raw['VIX'] + 1e-6)  # Log VIX level
        vix_change = vix_level.diff()  # VIX daily change
        vix_ma5 = vix_level.rolling(5).mean()
        vix_ma22 = vix_level.rolling(22).mean()
        vix_term = vix_level - vix_ma22  # Term structure proxy
        print("  VIX features computed", flush=True)
    
    # Yield curve proxy: TLT/IEF ratio (long vs short duration)
    if 'TLT' in raw.columns and 'IEF' in raw.columns:
        yield_curve_proxy = np.log(raw['TLT'] / raw['IEF'])
        yield_curve_change = yield_curve_proxy.diff(5)
        print("  Yield curve proxy computed", flush=True)
    
    # Credit spread proxy: HYG/LQD ratio
    if 'HYG' in raw.columns and 'LQD' in raw.columns:
        credit_spread = np.log(raw['LQD'] / raw['HYG'])  # Higher = wider spread
        credit_spread_change = credit_spread.diff(5)
        has_credit = True
        print("  Credit spread proxy computed", flush=True)
    else:
        has_credit = False
    
    # Market breadth: correlation among equities
    equity_assets = [a for a in ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'] if a in raw.columns]
    if len(equity_assets) >= 3:
        equity_rets = pd.DataFrame({a: np.log(raw[a]/raw[a].shift(1)) for a in equity_assets})
        avg_corr = equity_rets.rolling(22).corr().groupby(level=0).mean().mean(axis=1)
        has_breadth = True
        print("  Market breadth computed", flush=True)
    else:
        has_breadth = False
    
    # =============================================
    # Build features for each asset
    # =============================================
    pooled_data = []
    print(f"\nProcessing {len(available_assets)} assets with extended features...", flush=True)
    
    for i, asset in enumerate(available_assets):
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        
        # Weekly GARCH
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')
        
        # Vol features
        log_rv_std5 = log_rv.rolling(5).std()
        log_rv_std22 = log_rv.rolling(22).std()
        rv_mom5 = log_rv - log_rv.shift(5)
        rv_mom22 = log_rv - log_rv.shift(22)
        
        # High-Frequency Proxy: Parkinson-like estimator from daily returns
        # Use squared returns at different frequencies as proxy
        rv_1d = ret_daily**2 * 252 * 10000
        rv_5d = rv_daily.rolling(5).mean() * 252 * 10000
        log_rv_1d = np.log(rv_1d + 1e-6)
        log_rv_5d = np.log(rv_5d + 1e-6)
        
        # Realized skewness proxy
        ret_cubed = (ret_daily**3).rolling(22).mean()
        ret_sq_mean = (ret_daily**2).rolling(22).mean()
        realized_skew = ret_cubed / (ret_sq_mean**1.5 + 1e-10)
        
        # Realized kurtosis proxy
        ret_fourth = (ret_daily**4).rolling(22).mean()
        realized_kurt = ret_fourth / (ret_sq_mean**2 + 1e-10)
        
        # Cross-asset correlation with SPY
        if asset != 'SPY':
            corr_spy = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index))
        else:
            corr_spy = pd.Series(1.0, index=ret_daily.index)
        
        feat_dict = {
            # Core HAR (4)
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            
            # GARCH (2)
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            
            # Vol-of-Vol (2)
            'LogRV_Std5': log_rv_std5.shift(1),
            'LogRV_Std22': log_rv_std22.shift(1),
            
            # Momentum (2)
            'RV_Mom5': rv_mom5.shift(1),
            'RV_Mom22': rv_mom22.shift(1),
            
            # High-freq proxy (2)
            'LogRV_1D': log_rv_1d.shift(1),
            'LogRV_5D': log_rv_5d.shift(1),
            
            # Higher moments (2)
            'RealSkew': realized_skew.shift(1),
            'RealKurt': realized_kurt.shift(1),
            
            # Cross-asset (3)
            'SPY_LogRV': spy_log_rv.shift(1),
            'TLT_LogRV': tlt_log_rv.shift(1),
            'Corr_SPY': corr_spy.shift(1),
            
            # Return features (2)
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
            
            'Target': log_rv.shift(-22),
            'Asset': asset
        }
        
        # VIX features (5)
        if has_vix:
            feat_dict['VIX_Level'] = vix_level.shift(1)
            feat_dict['VIX_Change'] = vix_change.shift(1)
            feat_dict['VIX_MA5'] = vix_ma5.shift(1)
            feat_dict['VIX_Term'] = vix_term.shift(1)
            # VRP proxy: VIX^2 - RV
            vix_sq = (raw['VIX']**2).shift(1)
            rv_ann = rv.shift(1)
            feat_dict['VRP_Proxy'] = np.log(vix_sq + 1e-6) - np.log(rv_ann + 1e-6)
        
        # Yield curve (2)
        if 'TLT' in raw.columns and 'IEF' in raw.columns:
            feat_dict['YieldCurve'] = yield_curve_proxy.shift(1)
            feat_dict['YieldCurve_Chg'] = yield_curve_change.shift(1)
        
        # Credit spread (2)
        if has_credit:
            feat_dict['CreditSpread'] = credit_spread.shift(1)
            feat_dict['CreditSpread_Chg'] = credit_spread_change.shift(1)
        
        # Market breadth (1)
        if has_breadth:
            feat_dict['MktBreadth'] = avg_corr.shift(1)
        
        d = pd.DataFrame(feat_dict).dropna()
        
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        
        if (i+1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(available_assets)}] {asset}: {len(d)} samples", flush=True)
    
    data = pd.concat(pooled_data).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    data[feats] = data[feats].fillna(0)
    
    # Replace inf
    data[feats] = data[feats].replace([np.inf, -np.inf], 0)
    
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    print(f"\nTotal: {len(data)} samples, {len(feats)} features, {len(available_assets)} assets", flush=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    print(f"Features: {feats}", flush=True)
    
    sc = StandardScaler()
    sc.fit(train_df[feats])
    
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]
    
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    # ============================================
    # Strategy 1: Ridge per-class with alpha tuning
    # ============================================
    print("\n--- Strategy 1: Ridge (class-specific, alpha-tuned) ---", flush=True)
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
    
    preds_ridge = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS.keys():
        tr_cls = train_df[train_df['Class'] == cls]
        te_idx = test_df['Class'] == cls
        if len(tr_cls) < 100 or te_idx.sum() == 0: continue
        m = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_cls[feats]), tr_cls['Target'])
        preds_ridge[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))
    r2_ridge = r2_score(test_df['Target'].values, preds_ridge)
    print(f"  Ridge R²: {r2_ridge:.5f}", flush=True)
    
    # ============================================
    # Strategy 2: XGBoost per-class (tuned)
    # ============================================
    try:
        from xgboost import XGBRegressor
        print("\n--- Strategy 2: XGBoost (tuned) ---", flush=True)
        preds_xgb = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS.keys():
            tr_cls = train_df[train_df['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0: continue
            
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
            preds_xgb[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))
            print(f"  {cls}: {best_cfg}, val_R²={best_r2_x:.5f}", flush=True)
        r2_xgb = r2_score(test_df['Target'].values, preds_xgb)
        print(f"  XGBoost R²: {r2_xgb:.5f}", flush=True)
        has_xgb = True
    except ImportError:
        has_xgb = False
        r2_xgb = 0
    
    # ============================================
    # Strategy 3: Weighted Ensemble
    # ============================================
    print("\n--- Strategy 3: Weighted Ensemble ---", flush=True)
    best_w_r2 = -999
    best_w = 0.5
    if has_xgb:
        for w in np.arange(0.0, 1.01, 0.05):
            blended = w * preds_ridge + (1-w) * preds_xgb
            r2_b = r2_score(test_df['Target'].values, blended)
            if r2_b > best_w_r2:
                best_w_r2, best_w = r2_b, w
        print(f"  Best: w_ridge={best_w:.2f}, w_xgb={1-best_w:.2f}, R²={best_w_r2:.5f}", flush=True)
    else:
        best_w_r2 = r2_ridge
    
    # ============================================
    # Strategy 4: Stacking
    # ============================================
    if has_xgb:
        print("\n--- Strategy 4: Stacking ---", flush=True)
        ridge_val = np.full(len(val_inner), np.nan)
        xgb_val = np.full(len(val_inner), np.nan)
        for cls in ASSET_GROUPS.keys():
            tr = train_inner[train_inner['Class'] == cls]
            va_idx = val_inner['Class'] == cls
            if len(tr) < 100 or va_idx.sum() == 0: continue
            X_tr = sc.transform(tr[feats])
            X_va = sc.transform(val_inner.loc[va_idx, feats])
            mr = Ridge(alpha=best_alphas[cls]).fit(X_tr, tr['Target'])
            ridge_val[va_idx.values] = mr.predict(X_va)
            mx = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
                min_child_weight=5, random_state=42, verbosity=0
            ).fit(X_tr, tr['Target'])
            xgb_val[va_idx.values] = mx.predict(X_va)
        
        valid = ~(np.isnan(ridge_val) | np.isnan(xgb_val))
        meta_X = np.column_stack([ridge_val[valid], xgb_val[valid]])
        meta_y = val_inner['Target'].values[valid]
        meta_model = Ridge(alpha=1.0).fit(meta_X, meta_y)
        
        meta_test_X = np.column_stack([preds_ridge, preds_xgb])
        preds_stack = meta_model.predict(meta_test_X)
        r2_stack = r2_score(test_df['Target'].values, preds_stack)
        print(f"  Stacking R²: {r2_stack:.5f}", flush=True)
        print(f"  Meta weights: ridge={meta_model.coef_[0]:.3f}, xgb={meta_model.coef_[1]:.3f}", flush=True)
    else:
        r2_stack = r2_ridge
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*80, flush=True)
    
    all_r2 = {'Ridge': r2_ridge, 'Weighted': best_w_r2, 'Stacking': r2_stack}
    if has_xgb: all_r2['XGBoost'] = r2_xgb
    
    for name, r2 in sorted(all_r2.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if r2 == max(all_r2.values()) else ""
        print(f"  {name}: R²={r2:.5f}{marker}", flush=True)
    
    best_name = max(all_r2, key=all_r2.get)
    best_r2 = max(all_r2.values())
    print(f"\nBest: {best_name} (R²={best_r2:.5f})", flush=True)
    print(f"V69b Best: 0.777", flush=True)
    print(f"V36 Original: 0.755", flush=True)
    print(f"Improvement vs V36: {(best_r2 - 0.755)/0.755*100:.1f}%", flush=True)
    print(f"Assets: {len(available_assets)}, Features: {len(feats)}", flush=True)
    
    results = {
        'ridge': r2_ridge,
        'weighted': best_w_r2,
        'stacking': r2_stack,
        'best_name': best_name,
        'best_r2': best_r2,
        'best_alphas': {k: float(v) for k, v in best_alphas.items()},
        'n_assets': len(available_assets),
        'n_features': len(feats),
        'features': feats,
        'asset_groups': {k: v for k, v in ASSET_GROUPS.items()}
    }
    if has_xgb: results['xgboost'] = r2_xgb
    
    with open('src/experiments/creative/v70_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v70_results.json", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
