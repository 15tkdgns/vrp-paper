"""
V20 Exp: Ensemble Model (Stacking & Voting) (SCI Phase 4.1)
Purpose: To test if combining models yields better performance than the single best model (HAR-Ridge).
Methods:
1. VotingRegressor: Simple average of predictions.
2. StackingRegressor: Use a meta-model (Linear) to combine base model predictions.

Base Models:
- HAR-Ridge (Linear Trend)
- GradientBoosting (Non-linear Residuals)
- SVR (Alternative Non-linear)

Hypothesis: Ensemble might reduce variance, but given high noise, simple HAR-Ridge might still win or tie.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
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
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Features
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

def diebold_mariano_test_simple(y_true, y_pred1, y_pred2):
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def run_experiment():
    print("="*80)
    print("V20: Ensemble Model (Stacking & Voting)")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    # Time Series Split (80/20)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df['Target']
    X_test = scaler.transform(test_df[features])
    y_test = test_df['Target']
    
    # 2. Define Base Models
    estimators = [
        ('har', Ridge(alpha=1.0, random_state=SEED)),
        ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ]
    
    # 3. Define Ensembles
    # A. Voting (Average)
    voting_model = VotingRegressor(estimators=estimators)
    
    # B. Stacking (Meta-Learner = Ridge)
    # Note: Stacking uses cross-validation predictions for training prediction
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5
    )
    
    models = {
        'HAR-Ridge (Base)': estimators[0][1],
        'Voting (Avg)': voting_model,
        'Stacking (Meta)': stacking_model
    }
    
    results = {}
    predictions = {}
    
    print(f"\n[Step 2] Training & Evaluating Ensembles (Train: {len(X_train)}, Test: {len(X_test)})")
    
    # Train Base Model (HAR) separately for comparison
    models['HAR-Ridge (Base)'].fit(X_train, y_train)
    
    # Train Ensembles (Voting & Stacking)
    # SVR in ensemble might take time, so we might need to subset if slow, but let's try full.
    # To speed up Stacking CV, we limit data if needed. 25k samples is boundary.
    # Current sample size ~25k. It should be fine.
    
    print(f"{'Model':<20} {'R2':<10} {'RMSE':<10} {'Status'}")
    print("-" * 55)
    
    for name, model in models.items():
        if name != 'HAR-Ridge (Base)': # HAR already fitted
             model.fit(X_train, y_train)
             
        pred = model.predict(X_test)
        predictions[name] = pred
        
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        print(f"{name:<20} {r2:6.4f}     {rmse:6.4f}     Done")
        results[name] = {'R2': r2, 'RMSE': rmse}

    # 3. Statistical Significance (vs HAR-Ridge)
    print("\n[Step 3] Diebold-Mariano Test (vs HAR-Ridge)")
    dm_results = {}
    base_pred = predictions['HAR-Ridge (Base)']
    
    print(f"{'Challenger':<20} {'DM Stat':<10} {'p-value':<10} {'Result'}")
    print("-" * 60)
    
    for name in models.keys():
        if name == 'HAR-Ridge (Base)': continue
        
        stat, p_val = diebold_mariano_test_simple(y_test, base_pred, predictions[name])
        
        if p_val < 0.05:
            sig = "Significant"
            winner = name if stat > 0 else "HAR-Ridge"
        else:
            sig = "Not Sig"
            winner = "Tie"
            
        print(f"{name:<20} {stat:6.2f}     {p_val:6.4f}     {winner} ({sig})")
        dm_results[name] = {'stat': stat, 'p_value': p_val, 'winner': winner}

    # Save
    out_data = {
        'performance': results,
        'dm_tests': dm_results
    }
    
    out_path = 'src/experiments/sci/v20_ensemble_model.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
