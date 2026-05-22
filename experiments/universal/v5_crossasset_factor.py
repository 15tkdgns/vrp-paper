"""
V5 Cross-Asset Factor Experiment
=================================
Two improvements to feature engineering:
1. Unified Log-RV: All cross-asset RV features as log-RV (same scale as basic features)
2. Cross-Asset Factors: Summarize by asset class (Equity, Rate, Commodity)

Compare:
- V5 Original: SPY_RV, TLT_RV, GLD_RV (raw log-RV)
- V5 Factor: EquityFactor, RateFactor, CommodityFactor (averaged log-RV)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Asset groups for factor construction
EQUITY_ASSETS = ['SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM']
RATE_ASSETS = ['TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'AGG']
COMMODITY_ASSETS = ['GLD', 'SLV', 'USO', 'DBC']

ALL_ASSETS = list(set(EQUITY_ASSETS + RATE_ASSETS + COMMODITY_ASSETS))

def compute_log_rv(prices, window=22):
    """Compute log-RV"""
    ret = np.log(prices / prices.shift(1))
    rv = ret.rolling(window).std() * np.sqrt(252) * 100
    return np.log(rv + 1e-6)

def run_factor_experiment():
    print("="*70)
    print("V5 Cross-Asset Factor Experiment")
    print("="*70)
    
    # 1. Download Data
    tickers = ALL_ASSETS + ['^VIX', '^VIX3M', '^SKEW']
    print(f"Downloading {len(tickers)} tickers...")
    
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # 2. Compute Log-RV for all assets
    print("\nComputing Log-RV for all assets...")
    log_rv_all = pd.DataFrame(index=close.index)
    
    for asset in ALL_ASSETS:
        if asset in close.columns:
            log_rv_all[asset] = compute_log_rv(close[asset])
    
    # 3. Create Cross-Asset Factors (Average Log-RV by class)
    print("Creating Cross-Asset Factors...")
    
    equity_cols = [a for a in EQUITY_ASSETS if a in log_rv_all.columns]
    rate_cols = [a for a in RATE_ASSETS if a in log_rv_all.columns]
    commodity_cols = [a for a in COMMODITY_ASSETS if a in log_rv_all.columns]
    
    log_rv_all['EquityFactor'] = log_rv_all[equity_cols].mean(axis=1)
    log_rv_all['RateFactor'] = log_rv_all[rate_cols].mean(axis=1)
    log_rv_all['CommodityFactor'] = log_rv_all[commodity_cols].mean(axis=1)
    
    print(f"  Equity assets: {len(equity_cols)} → EquityFactor")
    print(f"  Rate assets: {len(rate_cols)} → RateFactor")
    print(f"  Commodity assets: {len(commodity_cols)} → CommodityFactor")
    
    # 4. Prepare pooled dataset for each target asset
    pooled_original = []
    pooled_factor = []
    
    # Feature sets
    basic_features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom']
    global_features = ['LogVIX', 'VIX_Term', 'SKEW']
    
    # Original: Individual cross-asset log-RV
    cross_original = ['LogRV_SPY', 'LogRV_TLT', 'LogRV_GLD']
    
    # Factor: Aggregated factors
    cross_factor = ['EquityFactor', 'RateFactor', 'CommodityFactor']
    
    features_original = basic_features + global_features + cross_original
    features_factor = basic_features + global_features + cross_factor
    
    for asset in ALL_ASSETS:
        if asset not in close.columns:
            continue
            
        df = pd.DataFrame(index=close.index)
        
        # Basic features (from target asset)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        
        # Global features
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1) if 'VIX' in close.columns else 0
        df['VIX_Term'] = (close.get('VIX3M', close['VIX']) - close['VIX']).shift(1) if 'VIX' in close.columns else 0
        df['SKEW'] = (close['SKEW'] / 100 - 1).shift(1) if 'SKEW' in close.columns else 0
        
        # Original: Individual cross-asset log-RV
        df['LogRV_SPY'] = log_rv_all['SPY'].shift(1) if 'SPY' in log_rv_all.columns else 0
        df['LogRV_TLT'] = log_rv_all['TLT'].shift(1) if 'TLT' in log_rv_all.columns else 0
        df['LogRV_GLD'] = log_rv_all['GLD'].shift(1) if 'GLD' in log_rv_all.columns else 0
        
        # Factor: Aggregated cross-asset factors
        df['EquityFactor'] = log_rv_all['EquityFactor'].shift(1)
        df['RateFactor'] = log_rv_all['RateFactor'].shift(1)
        df['CommodityFactor'] = log_rv_all['CommodityFactor'].shift(1)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset'] = asset
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) > 200:
            pooled_original.append(df[basic_features + global_features + cross_original + ['Target', 'Asset']])
            pooled_factor.append(df[basic_features + global_features + cross_factor + ['Target', 'Asset']])
    
    df_original = pd.concat(pooled_original).reset_index(drop=True)
    df_factor = pd.concat(pooled_factor).reset_index(drop=True)
    
    print(f"\nTotal samples: {len(df_original)}")
    
    # 5. Normalize and Split
    def train_and_evaluate(df, features, name):
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[features] = scaler.fit_transform(df[features])
        df_scaled['Target_Scaled'] = StandardScaler().fit_transform(df[['Target']])
        
        split = int(len(df_scaled) * 0.8)
        train = df_scaled.iloc[:split]
        test = df_scaled.iloc[split:]
        
        X_train = train[features].values
        y_train = train['Target_Scaled'].values
        X_test = test[features].values
        y_test = test['Target_Scaled'].values
        
        # Ensemble
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Huber': HuberRegressor(epsilon=1.35),
            'RF': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
            'GBM': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        }
        
        predictions = []
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions.append(model.predict(X_test))
        
        ensemble_pred = np.mean(predictions, axis=0)
        overall_r2 = r2_score(y_test, ensemble_pred)
        
        # Per-asset
        test = test.copy()
        test['Pred'] = ensemble_pred
        
        asset_r2s = []
        for asset in ALL_ASSETS:
            mask = test['Asset'] == asset
            if mask.sum() < 50:
                continue
            r2 = r2_score(test.loc[mask, 'Target_Scaled'], test.loc[mask, 'Pred'])
            asset_r2s.append(r2)
        
        avg_asset_r2 = np.mean(asset_r2s) if asset_r2s else 0
        
        return overall_r2, avg_asset_r2
    
    # 6. Run experiments
    print("\n" + "="*70)
    print("EXPERIMENT 1: Original (LogRV_SPY, LogRV_TLT, LogRV_GLD)")
    print("="*70)
    r2_orig, avg_orig = train_and_evaluate(df_original, features_original, "Original")
    print(f"  Overall R²: {r2_orig:.4f}")
    print(f"  Avg Asset R²: {avg_orig:.4f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Factor (EquityFactor, RateFactor, CommodityFactor)")
    print("="*70)
    r2_factor, avg_factor = train_and_evaluate(df_factor, features_factor, "Factor")
    print(f"  Overall R²: {r2_factor:.4f}")
    print(f"  Avg Asset R²: {avg_factor:.4f}")
    
    # 7. Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Approach':<40} | {'Overall R²':>12} | {'Avg Asset R²':>12}")
    print("-"*70)
    print(f"{'Original (SPY/TLT/GLD Log-RV)':<40} | {r2_orig:>12.4f} | {avg_orig:>12.4f}")
    print(f"{'Factor (Equity/Rate/Commodity)':<40} | {r2_factor:>12.4f} | {avg_factor:>12.4f}")
    
    improvement = avg_factor - avg_orig
    print(f"\nFactor Improvement: {improvement:+.4f}")
    
    if improvement > 0:
        print("✅ Factor approach is BETTER - Cross-asset factors provide cleaner signal")
    else:
        print("⚠️ Original approach is better - Individual assets have more predictive info")
    
    # 8. Save results
    results = {
        'original': {'overall_r2': r2_orig, 'avg_asset_r2': avg_orig},
        'factor': {'overall_r2': r2_factor, 'avg_asset_r2': avg_factor},
        'improvement': improvement,
        'features_original': features_original,
        'features_factor': features_factor
    }
    
    out_path = 'experiments/07_v2_methodology/results/v5_crossasset_factor.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_factor_experiment()
