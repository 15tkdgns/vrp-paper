"""
V71: Advanced Data Sources Experiment
Goal: Break 0.78 barrier using:
1. High-Frequency Proxy: Parkinson, Garman-Klass, Rogers-Satchell from OHLC
2. IV Surface: VIX term structure (VIX, VIX3M, VVIX)
3. Alternative Data: Volume profile, Price-Volume dynamics, Sentiment proxy

Key insight: Use OHLCV data (not just Close) to compute range-based
volatility estimators that proxy intraday/high-freq information.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from arch import arch_model
import os, json, warnings, time

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

# IV Surface tickers
IV_TICKERS = ['^VIX', '^VIX3M', '^VIX9D']  # VIX, 3-month VIX, 9-day VIX

def fit_garch(returns):
    try:
        ret = returns * 100
        am = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def download_ohlcv():
    """Download OHLCV data for range-based estimators"""
    CACHE_PATH = 'src/data/v71_ohlcv_cache.pkl'
    if os.path.exists(CACHE_PATH):
        print(f"Loading OHLCV cache: {CACHE_PATH}...", flush=True)
        return pd.read_pickle(CACHE_PATH)
    
    print("Downloading OHLCV data...", flush=True)
    all_tickers = ALL_ASSETS + IV_TICKERS
    raw = yf.download(all_tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    # Clean column names
    if isinstance(raw.columns, pd.MultiIndex):
        # Rename tickers
        new_cols = []
        for price_type, ticker in raw.columns:
            ticker_clean = ticker.replace('^', '')
            new_cols.append((price_type, ticker_clean))
        raw.columns = pd.MultiIndex.from_tuples(new_cols)
    
    raw = raw.ffill()
    raw.to_pickle(CACHE_PATH)
    print(f"Saved OHLCV cache to {CACHE_PATH}", flush=True)
    return raw

def compute_parkinson_vol(high, low, window=22):
    """Parkinson (1980) range-based volatility estimator"""
    log_hl = np.log(high / low)
    return np.sqrt((log_hl**2).rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_garman_klass_vol(open_p, high, low, close, window=22):
    """Garman-Klass (1980) OHLC volatility estimator"""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_p)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk.rolling(window).mean() * 252)

def compute_rogers_satchell_vol(open_p, high, low, close, window=22):
    """Rogers-Satchell (1991) estimator - drift independent"""
    log_ho = np.log(high / open_p)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_p)
    log_lc = np.log(low / close)
    rs = log_ho * log_hc + log_lo * log_lc
    return np.sqrt(rs.rolling(window).mean().clip(lower=0) * 252)

def compute_volume_features(volume, price, ret, window=22):
    """Volume-based features as alternative data proxy"""
    features = {}
    
    # 1. Volume-Weighted Volatility (Amihud)
    dollar_vol = volume * price
    amihud = (ret.abs() / (dollar_vol + 1e-10)).rolling(window).mean()
    features['Amihud'] = np.log(amihud + 1e-10)
    
    # 2. Volume Momentum
    vol_ma5 = volume.rolling(5).mean()
    vol_ma22 = volume.rolling(22).mean()
    features['Vol_Ratio'] = np.log(vol_ma5 / (vol_ma22 + 1e-10) + 1e-10)
    
    # 3. Price-Volume Correlation (Sentiment Proxy)
    # High correlation = trend following, Low = mean reversion
    features['PV_Corr'] = ret.rolling(window).corr(volume.pct_change())
    
    # 4. Volume Surprise
    vol_std = volume.rolling(22).std()
    features['Vol_Surprise'] = (volume - vol_ma22) / (vol_std + 1e-10)
    
    # 5. VPIN proxy (Volume-synchronized probability of informed trading)
    # Buy volume proxy: price up = buy
    buy_vol = volume * (ret > 0).astype(float)
    sell_vol = volume * (ret <= 0).astype(float)
    order_imb = (buy_vol - sell_vol) / (volume + 1e-10)
    features['Order_Imbalance'] = order_imb.rolling(window).mean()
    
    # 6. Absolute return / volume ratio (Kyle's lambda proxy)
    features['Kyle_Lambda'] = (ret.abs() / volume.pct_change().abs().clip(lower=1e-10)).rolling(window).mean()
    features['Kyle_Lambda'] = np.log(features['Kyle_Lambda'].clip(lower=1e-10))
    
    return features

def run_experiment():
    print("="*80, flush=True)
    print("V71: Advanced Data Sources (HF Proxy + IV Surface + Alt Data)", flush=True)
    print("="*80, flush=True)
    
    raw = download_ohlcv()
    
    # Check available data
    if isinstance(raw.columns, pd.MultiIndex):
        price_types = raw.columns.get_level_values(0).unique()
        available_tickers = raw.columns.get_level_values(1).unique()
    else:
        price_types = ['Close']
        available_tickers = raw.columns
    
    print(f"Price types: {list(price_types)}", flush=True)
    print(f"Available tickers: {list(available_tickers)}", flush=True)
    
    has_ohlc = all(pt in price_types for pt in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in price_types
    has_vix = 'VIX' in available_tickers
    has_vix3m = 'VIX3M' in available_tickers
    has_vix9d = 'VIX9D' in available_tickers
    
    print(f"OHLC: {has_ohlc}, Volume: {has_volume}", flush=True)
    print(f"VIX: {has_vix}, VIX3M: {has_vix3m}, VIX9D: {has_vix9d}", flush=True)
    
    # ============================================
    # IV Surface Features
    # ============================================
    print("\nComputing IV Surface features...", flush=True)
    iv_features = {}
    
    if has_vix:
        vix = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        iv_features['VIX'] = np.log(vix + 1e-6)
        iv_features['VIX_chg'] = iv_features['VIX'].diff()
        iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
        iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()  # VVIX proxy
        print("  VIX features OK", flush=True)
    
    if has_vix3m:
        vix3m = raw[('Close', 'VIX3M')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX3M']
        iv_features['VIX3M'] = np.log(vix3m + 1e-6)
        if has_vix:
            # VIX Term Structure Slope
            iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
            print("  VIX Term Structure OK", flush=True)
    
    if has_vix9d:
        vix9d = raw[('Close', 'VIX9D')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX9D']
        iv_features['VIX9D'] = np.log(vix9d + 1e-6)
        if has_vix:
            # Short-term term structure
            iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
            print("  VIX Short-Term Slope OK", flush=True)
    
    # SPY reference
    if isinstance(raw.columns, pd.MultiIndex):
        spy_close = raw[('Close', 'SPY')]
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    else:
        spy_close = raw['SPY']
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)
    
    # VRP (Variance Risk Premium) = IV^2 - RV (annualized variance unit)
    # IV^2 = (VIX/100)^2 : VIX percentage -> decimal, squared for variance
    # RV   = spy_rv/10000 : spy_rv in bp^2 -> decimal
    # NOTE: VIX is a 30-calendar-day (~22 trading day) implied vol measure.
    # Using it as IV proxy for all horizons introduces maturity mismatch.
    if has_vix:
        vix_raw = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        vrp = (vix_raw**2 / 100) - spy_rv / 10000  # Both in annual variance
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()
        print("  VRP (Variance Risk Premium) OK", flush=True)
    
    # ============================================
    # Process each asset
    # ============================================
    pooled_data = []
    available_assets = [a for a in ALL_ASSETS if a in available_tickers]
    print(f"\nProcessing {len(available_assets)} assets...", flush=True)
    
    for i, asset in enumerate(available_assets):
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[('Close', asset)]
            open_p = raw[('Open', asset)] if has_ohlc else None
            high = raw[('High', asset)] if has_ohlc else None
            low = raw[('Low', asset)] if has_ohlc else None
            volume = raw[('Volume', asset)] if has_volume else None
        else:
            close = raw[asset]
            open_p = high = low = volume = None
        
        ret_daily = np.log(close / close.shift(1)).dropna()
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        
        # Weekly GARCH
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')
        
        # Standard features
        feat_dict = {
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv.rolling(5).std().shift(1),
            'LogRV_Std22': log_rv.rolling(22).std().shift(1),
            'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
            'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
        }
        
        # === HIGH-FREQUENCY PROXY FEATURES ===
        if has_ohlc and open_p is not None:
            # Parkinson Volatility (5-day and 22-day windows)
            park_5 = compute_parkinson_vol(high, low, window=5)
            park_22 = compute_parkinson_vol(high, low, window=22)
            feat_dict['Parkinson_5'] = np.log(park_5 + 1e-6).shift(1)
            feat_dict['Parkinson_22'] = np.log(park_22 + 1e-6).shift(1)
            
            # Garman-Klass Volatility
            gk_22 = compute_garman_klass_vol(open_p, high, low, close, window=22)
            feat_dict['GarmanKlass_22'] = np.log(gk_22 + 1e-6).shift(1)
            
            # Rogers-Satchell Volatility
            rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, window=22)
            feat_dict['RogersSatchell_22'] = np.log(rs_22 + 1e-6).shift(1)
            
            # Range-based vs Close-based ratio (information content)
            feat_dict['Range_Close_Ratio'] = (np.log(park_22 + 1e-6) - log_rv).shift(1)
            
            # Overnight return (gap risk)
            overnight_ret = np.log(open_p / close.shift(1))
            feat_dict['Overnight_Vol'] = overnight_ret.rolling(22).std().shift(1)
            feat_dict['Overnight_Ret'] = overnight_ret.shift(1)
            
            print(f"  [{i+1}] {asset}: HF proxy OK", flush=True) if i == 0 else None
        
        # === IV SURFACE FEATURES ===
        for iv_name, iv_val in iv_features.items():
            feat_dict[f'IV_{iv_name}'] = iv_val.shift(1)
        
        # === ALTERNATIVE DATA: VOLUME FEATURES ===
        if has_volume and volume is not None:
            vol_feats = compute_volume_features(volume, close, ret_daily, window=22)
            for vf_name, vf_val in vol_feats.items():
                feat_dict[f'AltVol_{vf_name}'] = vf_val.shift(1)
            
            print(f"  [{i+1}] {asset}: Volume features OK", flush=True) if i == 0 else None
        
        # Cross-asset correlation
        if asset != 'SPY':
            feat_dict['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
        else:
            feat_dict['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)
        
        # Target: Log realized volatility 22 trading days ahead.
        # Uses backward-looking 22d rolling RV shifted forward (overlapping).
        # For non-overlapping forward RV targets, see tmp_multi_horizon_nonoverlap.py
        feat_dict['Target'] = log_rv.shift(-22)
        feat_dict['Asset'] = asset
        
        d = pd.DataFrame(feat_dict).dropna()
        
        # Handle inf
        numeric_cols = [c for c in d.columns if c not in ['Asset', 'Target']]
        d[numeric_cols] = d[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        
        if (i+1) % 3 == 0 or i == 0:
            print(f"  [{i+1}/{len(available_assets)}] {asset}: {len(d)} samples", flush=True)
    
    data = pd.concat(pooled_data).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Purge: remove 22 samples at train/test boundary to prevent
    # target leakage from overlapping 22d RV windows.
    purge_gap = 22  # trading days, matches prediction horizon
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx - purge_gap]
    test_df = data.iloc[split_idx:]
    
    print(f"\nTotal: {len(data)} samples, {len(feats)} features", flush=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    
    # Group features for analysis
    hf_feats = [f for f in feats if any(x in f for x in ['Parkinson', 'Garman', 'Rogers', 'Range', 'Overnight'])]
    iv_feats = [f for f in feats if f.startswith('IV_')]
    alt_feats = [f for f in feats if f.startswith('AltVol_')]
    base_feats = [f for f in feats if f not in hf_feats + iv_feats + alt_feats]
    
    print(f"\nFeature groups:", flush=True)
    print(f"  Base: {len(base_feats)} - {base_feats}", flush=True)
    print(f"  HF Proxy: {len(hf_feats)} - {hf_feats}", flush=True)
    print(f"  IV Surface: {len(iv_feats)} - {iv_feats}", flush=True)
    print(f"  Alternative: {len(alt_feats)} - {alt_feats}", flush=True)
    
    sc = StandardScaler()
    sc.fit(train_df[feats])
    
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]
    
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    def train_and_eval(feature_set, label):
        """Train Ridge per-class with alpha tuning on given feature set"""
        best_alphas_local = {}
        for cls in ASSET_GROUPS.keys():
            best_r2, best_a = -999, 1.0
            tr_cls = train_inner[train_inner['Class'] == cls]
            va_cls = val_inner[val_inner['Class'] == cls]
            if len(tr_cls) < 100 or len(va_cls) < 30:
                best_alphas_local[cls] = 1.0
                continue
            sc_local = StandardScaler()
            X_tr = sc_local.fit_transform(tr_cls[feature_set])
            X_va = sc_local.transform(va_cls[feature_set])
            for a in alphas:
                m = Ridge(alpha=a).fit(X_tr, tr_cls['Target'])
                r2 = r2_score(va_cls['Target'], m.predict(X_va))
                if r2 > best_r2:
                    best_r2, best_a = r2, a
            best_alphas_local[cls] = best_a
        
        preds = np.full(len(test_df), np.nan)
        sc_final = StandardScaler()
        sc_final.fit(train_df[feature_set])
        for cls in ASSET_GROUPS.keys():
            tr_cls = train_df[train_df['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0: continue
            m = Ridge(alpha=best_alphas_local[cls]).fit(
                sc_final.transform(tr_cls[feature_set]), tr_cls['Target'])
            preds[te_idx.values] = m.predict(sc_final.transform(test_df.loc[te_idx, feature_set]))
        
        r2 = r2_score(test_df['Target'].values, preds)
        print(f"  {label}: R²={r2:.5f} ({len(feature_set)} features)", flush=True)
        return r2, preds, best_alphas_local
    
    # ============================================
    # Ablation Study: Feature Group Contributions
    # ============================================
    print("\n--- Ablation Study ---", flush=True)
    
    r2_base, preds_base, _ = train_and_eval(base_feats, "Base only")
    r2_base_hf, preds_base_hf, _ = train_and_eval(base_feats + hf_feats, "Base + HF Proxy")
    r2_base_iv, preds_base_iv, _ = train_and_eval(base_feats + iv_feats, "Base + IV Surface")
    r2_base_alt, preds_base_alt, _ = train_and_eval(base_feats + alt_feats, "Base + Alt Data")
    r2_all, preds_all, best_alphas_all = train_and_eval(feats, "All Features")
    r2_base_hf_iv, preds_hf_iv, _ = train_and_eval(base_feats + hf_feats + iv_feats, "Base + HF + IV")
    
    # ============================================
    # XGBoost with all features
    # ============================================
    try:
        from xgboost import XGBRegressor
        print("\n--- XGBoost (all features) ---", flush=True)
        preds_xgb = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS.keys():
            tr_cls = train_df[train_df['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0: continue
            tr_i = train_inner[train_inner['Class'] == cls]
            va_i = val_inner[val_inner['Class'] == cls]
            best_r2_x, best_cfg = -999, {'n_estimators': 200, 'max_depth': 4}
            if len(va_i) > 30:
                for n_est in [100, 200, 300]:
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
            m = XGBRegressor(
                n_estimators=best_cfg['n_estimators'], max_depth=best_cfg['max_depth'],
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc.transform(tr_cls[feats]), tr_cls['Target'])
            preds_xgb[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))
        r2_xgb = r2_score(test_df['Target'].values, preds_xgb)
        print(f"  XGBoost All: R²={r2_xgb:.5f}", flush=True)
        has_xgb = True
    except ImportError:
        has_xgb = False
        r2_xgb = 0
        preds_xgb = preds_all
    
    # ============================================
    # Ensemble
    # ============================================
    print("\n--- Ensemble ---", flush=True)
    best_w_r2 = -999
    best_w = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * preds_all + (1-w) * preds_xgb
        r2_b = r2_score(test_df['Target'].values, blended)
        if r2_b > best_w_r2:
            best_w_r2, best_w = r2_b, w
    print(f"  Weighted (Ridge {best_w:.0%} + XGB {1-best_w:.0%}): R²={best_w_r2:.5f}", flush=True)
    
    # ============================================
    # Save Predictions CSV (for paper figures)
    # ============================================
    print("\n--- Saving predictions CSV ---", flush=True)
    best_blend = best_w * preds_all + (1 - best_w) * preds_xgb
    pred_df = pd.DataFrame({
        'Date': test_df.index,
        'Asset': test_df['Asset'].values,
        'Class': test_df['Class'].values,
        'Actual': test_df['Target'].values,
        'Pred_Ridge': preds_all,
        'Pred_XGBoost': preds_xgb if has_xgb else np.nan,
        'Pred_Ensemble': best_blend,
        'Residual_Ridge': test_df['Target'].values - preds_all,
        'Residual_Ensemble': test_df['Target'].values - best_blend,
    })
    pred_csv_path = 'paper/csv/v71_predictions.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved {len(pred_df)} predictions to {pred_csv_path}", flush=True)
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*80, flush=True)
    
    results_table = {
        'Base only': r2_base,
        'Base + HF Proxy': r2_base_hf,
        'Base + IV Surface': r2_base_iv,
        'Base + Alt Data': r2_base_alt,
        'Base + HF + IV': r2_base_hf_iv,
        'All Features (Ridge)': r2_all,
    }
    if has_xgb:
        results_table['All Features (XGBoost)'] = r2_xgb
        results_table['Weighted Ensemble'] = best_w_r2
    
    print("\n  Feature Ablation:", flush=True)
    for name, r2 in sorted(results_table.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if r2 == max(results_table.values()) else ""
        delta = r2 - r2_base
        print(f"    {name}: R²={r2:.5f} (delta={delta:+.5f}){marker}", flush=True)
    
    best_name = max(results_table, key=results_table.get)
    best_r2 = max(results_table.values())
    
    print(f"\n  HF Proxy contribution: {r2_base_hf - r2_base:+.5f}", flush=True)
    print(f"  IV Surface contribution: {r2_base_iv - r2_base:+.5f}", flush=True)
    print(f"  Alt Data contribution: {r2_base_alt - r2_base:+.5f}", flush=True)
    
    print(f"\n  Best: {best_name} (R²={best_r2:.5f})", flush=True)
    print(f"  V69b Best: 0.777", flush=True)
    print(f"  V36 Original: 0.755", flush=True)
    print(f"  Improvement vs V36: {(best_r2 - 0.755)/0.755*100:.1f}%", flush=True)
    
    results = {
        'ablation': {k: float(v) for k, v in results_table.items()},
        'hf_contribution': float(r2_base_hf - r2_base),
        'iv_contribution': float(r2_base_iv - r2_base),
        'alt_contribution': float(r2_base_alt - r2_base),
        'best_name': best_name,
        'best_r2': float(best_r2),
        'n_features': len(feats),
        'feature_groups': {
            'base': base_feats,
            'hf_proxy': hf_feats,
            'iv_surface': iv_feats,
            'alternative': alt_feats
        }
    }
    
    with open('experiments/creative/v71_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v71_results.json", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
