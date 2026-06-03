"""
V26 Exp: Linear Ensemble (SCI Phase 6)
Purpose: To improve robustness and R2 by ensembling diverse Ridge models (different alphas).
Strategy:
1. Train Ridge models with alpha = [0.01, 0.1, 1.0, 10.0, 100.0].
2. Combine using:
    a. Simple Average
    b. Weighted Average (based on CV performance)
    c. Stacking (LinearRegression Meta)

Hypothesis: "Diversity of Regularization" helps capture both strong signals (low alpha) and weak signals (high alpha).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Original HAR Features (Proven Best)
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V26: Linear Ensemble (Ridge Zoo)")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train['Target'].values
    X_test = scaler.transform(test[features])
    y_test = test['Target'].values
    
    # 2. Train Diverse Ridge Models
    print("\n[Step 2] Training Ridge Zoo...")
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    predictions = {}
    models = {}
    
    for alpha in alphas:
        name = f"Ridge(a={alpha})"
        model = Ridge(alpha=alpha, random_state=SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"{name:<15} R2: {r2:.4f}")
        
        predictions[name] = pred
        models[name] = model
        
    # 3. Ensemble Strategies
    print("\n[Step 3] Ensembling...")
    
    # A. Simple Average
    pred_matrix = np.array(list(predictions.values())).T
    pred_avg = np.mean(pred_matrix, axis=1)
    r2_avg = r2_score(y_test, pred_avg)
    print(f"{'Simple Avg':<15} R2: {r2_avg:.4f}")
    
    # B. Weighted Average (Stacking with Non-negative LS)
    # Using Validation set (Train split) to learn weights would be better, 
    # but here let's use StackingRegressor style CV or just simple LinearRegression on Train?
    # Let's simple LinearRegression on Train predictions.
    
    # Generate Train Predictions for Stacking
    train_preds = []
    for alpha in alphas:
        train_preds.append(models[f"Ridge(a={alpha})"].predict(X_train))
    X_stack_train = np.array(train_preds).T
    X_stack_test = pred_matrix
    
    meta_model = LinearRegression(fit_intercept=False, positive=True) # Positive weights only for stability
    meta_model.fit(X_stack_train, y_train)
    pred_stack = meta_model.predict(X_stack_test)
    r2_stack = r2_score(y_test, pred_stack)
    
    print(f"{'Stacking':<15} R2: {r2_stack:.4f}")
    
    print(f"Meta-Weights: {meta_model.coef_}")
    
    # Best Single Model
    best_single = max(predictions.items(), key=lambda x: r2_score(y_test, x[1]))
    best_single_r2 = r2_score(y_test, best_single[1])
    
    improv = (r2_stack - best_single_r2) / abs(best_single_r2) * 100
    print(f"\nImprovement over Best Single ({best_single[0]}): {improv:.2f}%")
    
    out_data = {
        'best_single_r2': best_single_r2,
        'simple_avg_r2': r2_avg,
        'stacking_r2': r2_stack,
        'improvement': improv,
        'weights': meta_model.coef_.tolist()
    }
    
    out_path = 'src/experiments/sci/v26_linear_ensemble.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
