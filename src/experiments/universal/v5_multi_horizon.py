"""
V5 Multi-Horizon Prediction
============================
Test volatility prediction across multiple time horizons:
- 1 day
- 5 days (1 week)
- 22 days (1 month)
- 66 days (3 months)
- 132 days (6 months)
- 252 days (12 months)

Using: Ensemble of Ridge/Huber/RF/GBM with extended features
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'EFA', 'EEM',
    'TLT', 'IEF', 'TIP', 'LQD', 'AGG',
    'GLD', 'SLV', 'USO', 'DBC',
]

HORIZONS = {
    '1d': 1,
    '5d': 5,
    '1m': 22,
    '3m': 66,
    '6m': 132,
    '12m': 252,
}

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def run_multi_horizon():
    print("="*70)
    print("V5 Multi-Horizon Prediction Experiment")
    print("="*70)
    
    # 1. Download Data
    tickers = ASSETS + ['^VIX', '^VIX3M', '^SKEW', '^TNX']
    print(f"Downloading {len(tickers)} tickers...")
    
    raw = yf.download(tickers, start='2008-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    valid_assets = [a for a in ASSETS if a in close.columns]
    print(f"Valid assets: {len(valid_assets)}")
    
    # Cross-asset RVs
    cross_rvs = {}
    for ca in ['SPY', 'TLT', 'GLD']:
        if ca in close.columns:
            ret = np.log(close[ca] / close[ca].shift(1))
            cross_rvs[ca] = np.log(ret.rolling(22).std() * np.sqrt(252) * 100 + 1e-6)
    
    # Results storage
    all_results = {}
    
    for horizon_name, horizon_days in HORIZONS.items():
        print(f"\n{'='*50}")
        print(f"HORIZON: {horizon_name} ({horizon_days} days)")
        print(f"{'='*50}")
        
        pooled_data = []
        
        basic_features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom']
        global_features = ['LogVIX', 'VIX_Term', 'SKEW']
        cross_features = ['SPY_RV', 'TLT_RV', 'GLD_RV']
        all_features = basic_features + global_features + cross_features
        
        for asset in valid_assets:
            df = pd.DataFrame(index=close.index)
            
            ret = np.log(close[asset] / close[asset].shift(1))
            rv = ret.rolling(22).std() * np.sqrt(252) * 100
            
            # Features
            df['LogRV'] = np.log(rv + 1e-6)
            df['LogRV_lag1'] = df['LogRV'].shift(1)
            df['LogRV_lag5'] = df['LogRV'].shift(5)
            df['LogRV_lag22'] = df['LogRV'].shift(22)
            df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
            
            df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1) if 'VIX' in close.columns else 0
            df['VIX_Term'] = (close.get('VIX3M', close['VIX']) - close['VIX']).shift(1) if 'VIX' in close.columns else 0
            df['SKEW'] = (close['SKEW'] / 100 - 1).shift(1) if 'SKEW' in close.columns else 0
            
            df['SPY_RV'] = cross_rvs.get('SPY', pd.Series(0, index=df.index)).shift(1)
            df['TLT_RV'] = cross_rvs.get('TLT', pd.Series(0, index=df.index)).shift(1)
            df['GLD_RV'] = cross_rvs.get('GLD', pd.Series(0, index=df.index)).shift(1)
            
            # Target: Future RV at horizon
            future_rv = ret.rolling(horizon_days).std() * np.sqrt(252) * 100
            df['Target'] = np.log(future_rv.shift(-horizon_days) + 1e-6)
            
            df['Asset'] = asset
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) > 200:
                pooled_data.append(df)
        
        if not pooled_data:
            continue
            
        full_df = pd.concat(pooled_data).reset_index(drop=True)
        
        # Normalize
        scaler = StandardScaler()
        full_df[all_features] = scaler.fit_transform(full_df[all_features])
        full_df['Target_Scaled'] = StandardScaler().fit_transform(full_df[['Target']])
        
        # Split
        split = int(len(full_df) * 0.8)
        train_df = full_df.iloc[:split]
        test_df = full_df.iloc[split:]
        
        X_train = train_df[all_features].values
        y_train = train_df['Target_Scaled'].values
        X_test = test_df[all_features].values
        y_test = test_df['Target_Scaled'].values
        
        # Train ensemble
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Huber': HuberRegressor(epsilon=1.35),
            'RF': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
            'GBM': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        }
        
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)
        
        # Ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Metrics
        overall_r2 = r2_score(y_test, ensemble_pred)
        overall_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"  Samples: Train={len(train_df)}, Test={len(test_df)}")
        print(f"  Overall R²: {overall_r2:.4f}")
        print(f"  Overall MAE: {overall_mae:.4f}")
        
        # Per-asset
        test_df_copy = test_df.copy()
        test_df_copy['Pred'] = ensemble_pred
        
        asset_r2s = []
        for asset in valid_assets:
            mask = test_df_copy['Asset'] == asset
            if mask.sum() < 50:
                continue
            r2 = r2_score(test_df_copy.loc[mask, 'Target_Scaled'], test_df_copy.loc[mask, 'Pred'])
            asset_r2s.append(r2)
        
        avg_asset_r2 = np.mean(asset_r2s) if asset_r2s else 0
        print(f"  Avg Per-Asset R²: {avg_asset_r2:.4f}")
        
        all_results[horizon_name] = {
            'horizon_days': horizon_days,
            'overall_r2': overall_r2,
            'overall_mae': overall_mae,
            'avg_asset_r2': avg_asset_r2,
            'n_train': len(train_df),
            'n_test': len(test_df)
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Multi-Horizon Performance")
    print("="*70)
    print(f"{'Horizon':<10} | {'Days':>6} | {'Overall R²':>12} | {'Avg Asset R²':>12}")
    print("-"*55)
    
    for horizon_name, res in all_results.items():
        print(f"{horizon_name:<10} | {res['horizon_days']:>6} | {res['overall_r2']:>12.4f} | {res['avg_asset_r2']:>12.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    horizons = list(all_results.keys())
    overall_r2s = [all_results[h]['overall_r2'] for h in horizons]
    avg_asset_r2s = [all_results[h]['avg_asset_r2'] for h in horizons]
    
    axes[0].bar(horizons, overall_r2s, color='steelblue')
    axes[0].set_title('Overall R² by Horizon')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('R²')
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[1].bar(horizons, avg_asset_r2s, color='darkorange')
    axes[1].set_title('Avg Per-Asset R² by Horizon')
    axes[1].set_xlabel('Horizon')
    axes[1].set_ylabel('R²')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v5_multi_horizon.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    
    # Save results
    out_path = 'experiments/07_v2_methodology/results/v5_multi_horizon.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {out_path}")
    
    # Best horizon
    best = max(all_results.items(), key=lambda x: x[1]['avg_asset_r2'])
    print(f"\n★ Best Horizon: {best[0]} (Avg Asset R² = {best[1]['avg_asset_r2']:.4f})")

if __name__ == "__main__":
    run_multi_horizon()
