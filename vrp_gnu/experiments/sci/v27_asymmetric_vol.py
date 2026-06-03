"""
V27 Exp: Asymmetric Volatility (Semivariance) (SCI Phase 7)
Purpose: To capture the "Leverage Effect" (Bad Vol persists longer/stronger than Good Vol).
Method:
1. Decompose Daily Returns into ret_pos (>0) and ret_neg (<0).
2. Calculate RV_pos (Good Vol) and RV_neg (Bad Vol).
3. Use log(RV_pos) and log(RV_neg) as separate features in HAR.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity'
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42

def feature_engineering(data):
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
        
        # Decompose Returns
        ret_pos = ret.clip(lower=0)
        ret_neg = ret.clip(upper=0)
        
        # Calculate Realized Semivariance (Approximation using daily data)
        # Ideally we need intraday, but daily squared returns separated is proxies.
        rv_daily = ret**2
        rv_pos_daily = ret_pos**2
        rv_neg_daily = ret_neg**2
        
        # Rolling Windows
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        rv_pos = rv_pos_daily.rolling(22).mean() * 252 * 10000
        rv_neg = rv_neg_daily.rolling(22).mean() * 252 * 10000
        
        # HAR Features (Standard)
        df['LogRV'] = np.log(rv + 1e-6)
        
        # Asymmetric Features
        df['LogRV_pos'] = np.log(rv_pos + 1e-6)
        df['LogRV_neg'] = np.log(rv_neg + 1e-6)
        
        # Create Lags for Asymmetric Features
        for lag in [1, 5, 22]:
            df[f'LogRV_pos_lag{lag}'] = df['LogRV_pos'].shift(lag)
            df[f'LogRV_neg_lag{lag}'] = df['LogRV_neg'].shift(lag)
            # Also keep standard lags for comparison
            df[f'LogRV_lag{lag}'] = df['LogRV'].shift(lag)
            
        # Target (Standard)
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V27: Asymmetric Volatility (Good vs Bad)")
    print("="*80)
    
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    y_train = train['Target']
    y_test = test['Target']
    
    # Model 1: Standard HAR (Baseline)
    feats_base = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(train[feats_base])
    X_test_base = scaler_base.transform(test[feats_base])
    
    model_base = Ridge(alpha=1.0)
    model_base.fit(X_train_base, y_train)
    r2_base = r2_score(y_test, model_base.predict(X_test_base))
    
    # Model 2: Asymmetric HAR (Target: Standard RV, Input: Good/Bad RV)
    feats_asym = []
    for lag in [1, 5, 22]:
        feats_asym.append(f'LogRV_pos_lag{lag}')
        feats_asym.append(f'LogRV_neg_lag{lag}')
        
    scaler_asym = StandardScaler()
    X_train_asym = scaler_asym.fit_transform(train[feats_asym])
    X_test_asym = scaler_asym.transform(test[feats_asym])
    
    model_asym = Ridge(alpha=1.0)
    model_asym.fit(X_train_asym, y_train)
    r2_asym = r2_score(y_test, model_asym.predict(X_test_asym))
    
    # Model 3: Hybrid (Standard + Asymmetric)
    feats_hybrid = feats_base + feats_asym
    
    scaler_hybrid = StandardScaler()
    X_train_hybrid = scaler_hybrid.fit_transform(train[feats_hybrid])
    X_test_hybrid = scaler_hybrid.transform(test[feats_hybrid])
    
    model_hybrid = Ridge(alpha=1.0)
    model_hybrid.fit(X_train_hybrid, y_train)
    r2_hybrid = r2_score(y_test, model_hybrid.predict(X_test_hybrid))
    
    print(f"\nBaseline R2: {r2_base:.4f}")
    print(f"Asymmetric R2: {r2_asym:.4f}")
    print(f"Hybrid R2:     {r2_hybrid:.4f}")
    
    best_r2_new = max(r2_asym, r2_hybrid)
    improv = (best_r2_new - r2_base) / abs(r2_base) * 100
    print(f"Improvement: {improv:.2f}%")
    
    # Evaluate Coefs of Hybrid
    print("\n[Analysis] Coefficients (Hybrid)")
    coefs = pd.Series(model_hybrid.coef_, index=feats_hybrid)
    print(coefs.sort_values(key=abs, ascending=False))
    
    out_data = {
        'baseline_r2': r2_base,
        'asymmetric_r2': r2_asym,
        'hybrid_r2': r2_hybrid,
        'improvement': improv,
        'coefficients': coefs.to_dict()
    }
    
    out_path = 'src/experiments/sci/v27_asymmetric_vol.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
