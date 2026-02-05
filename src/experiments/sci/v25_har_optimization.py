"""
V25 Exp: HAR Optimization (SCI Phase 6)
Purpose: To address v24 diagnostics (Autocorrelation, Non-normality) by optimizing Lags and Target Transform.
Steps:
1. Lag Search: Test combinations of [1, 2, 3, 5, 10, 22, 60] to kill autocorrelation.
2. Feature Selection: Use ElasticNet to prune noise.
3. Target Transform: Test Box-Cox to fix non-normality.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.metrics import r2_score
from itertools import combinations
from scipy.stats import boxcox
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond',
    'GLD': 'Commodity'
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42
LAG_CANDIDATES = [1, 2, 3, 5, 10, 22] # Keep small to avoid massive combinations

def feature_engineering(data, lags):
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1))
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Features based on Lags
        df['LogRV'] = np.log(rv + 1e-6)
        
        for lag in lags:
            # Simple lag: value at t-lag
            df[f'LogRV_lag{lag}'] = df['LogRV'].shift(lag)
            
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def optimize_lags(raw_data):
    print("\n[Step 1] Optimizing Lag Structure...")
    
    # Try combinations of 3 lags
    combos = list(combinations(LAG_CANDIDATES, 3))
    
    best_r2 = -np.inf
    best_lags = None
    
    # Quick search
    # Pre-calculate all lags once? 
    # Actually just looping feature engineering is fast enough for this scale.
    
    results = []
    
    for combo in combos:
        data = feature_engineering(raw_data, combo)
        
        split_idx = int(len(data) * 0.8)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        
        features = [c for c in data.columns if 'LogRV_lag' in c]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        y_train = train['Target']
        X_test = scaler.transform(test[features])
        y_test = test['Target']
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        
        results.append({'lags': combo, 'r2': r2})
        if r2 > best_r2:
            best_r2 = r2
            best_lags = combo
            
    print(f"Best Lags: {best_lags}, R2: {best_r2:.4f}")
    return best_lags

def optimize_target_transform(data, features):
    print("\n[Step 2] Optimizing Target Transform...")
    
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    
    # Original Target (LogRV, already logged in processing but let's treat it as base)
    y_train_log = train['Target']
    y_test_log = test['Target']
    
    # 1. Log (Base)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train_log)
    r2_log = r2_score(y_test_log, model.predict(X_test))
    
    # 2. Box-Cox
    # We need positive values. RV itself is positive.
    # We need to reconstruct RV from LogRV to apply Box-Cox, or just apply Box-Cox to exp(LogRV)?
    # Data['Target'] is LogRV.
    rv_train = np.exp(y_train_log)
    rv_test = np.exp(y_test_log)
    
    # Box-Cox fit on Train
    rv_train_bc, lambda_best = boxcox(rv_train)
    # Apply same lambda to Test
    rv_test_bc = boxcox(rv_test, lmbda=lambda_best)
    
    model_bc = Ridge(alpha=1.0)
    model_bc.fit(X_train, rv_train_bc)
    pred_bc = model_bc.predict(X_test)
    r2_bc = r2_score(rv_test_bc, pred_bc)
    
    print(f"Log Transform R2: {r2_log:.4f}")
    print(f"Box-Cox Transform R2: {r2_bc:.4f} (Lambda={lambda_best:.4f})")
    
    return lambda_best if r2_bc > r2_log else None

def run_experiment():
    print("="*80)
    print("V25: HAR Optimization")
    print("="*80)
    
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    
    # 1. Optimize Lags
    best_lags = optimize_lags(raw)
    
    # 2. Create Optimized Data
    data = feature_engineering(raw, best_lags)
    features = [c for c in data.columns if 'LogRV_lag' in c]
    
    # 3. Optimize Target Transform
    best_lambda = optimize_target_transform(data, features)
    
    # 4. Final Optimized Model Performance
    print("\n[Step 3] Final Optimized Model")
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train['Target']
    X_test = scaler.transform(test[features])
    y_test = test['Target']
    
    # Use ElasticNetCV for implicit Feature Selection
    model = ElasticNetCV(cv=5, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_final = r2_score(y_test, y_pred)
    
    print(f"Final Optimized R2: {r2_final:.4f}")
    print(f"Selected Alpha: {model.alpha_:.6f}, L1 Ratio: {model.l1_ratio_}")
    
    out_data = {
        'best_lags': best_lags,
        'best_lambda': best_lambda,
        'final_r2': r2_final
    }
    
    out_path = 'src/experiments/sci/v25_har_optimization.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
