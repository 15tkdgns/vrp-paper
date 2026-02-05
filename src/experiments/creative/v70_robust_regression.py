import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
import os
import json
import warnings

warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def run_experiment():
    print("="*80)
    print("V70: Robust Regression (Huber Loss) Experiment")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    pooled_data = []
    
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        pooled_data.append(d)
        
    data = pd.concat(pooled_data).sort_index()
    
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[feats])
    X_test = sc.transform(test_df[feats])
    y_train = train_df['Target']
    y_test = test_df['Target']
    
    # 1. Baseline (Ridge / MSE)
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    pred_ridge = model_ridge.predict(X_test)
    r2_ridge = r2_score(y_test, pred_ridge)
    
    # 2. Robust (Huber)
    # epsilon=1.35 is standard (95% efficiency for normal errors)
    model_huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    model_huber.fit(X_train, y_train)
    pred_huber = model_huber.predict(X_test)
    r2_huber = r2_score(y_test, pred_huber)
    
    print("\n[Results]")
    print(f"Ridge (MSE) R2: {r2_ridge:.5f}")
    print(f"Huber (Robust) R2: {r2_huber:.5f}")
    print(f"Difference: {r2_huber - r2_ridge:.5f}")
    
    res = {
        'Ridge_R2': r2_ridge,
        'Huber_R2': r2_huber,
        'Difference': r2_huber - r2_ridge
    }
    
    with open('src/experiments/creative/v70_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_experiment()
