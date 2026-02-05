"""
V21 Exp: Hybrid Residual Learning (SCI Phase 5)
Purpose: To boost performance by modeling the "Non-linear Residuals" of the HAR-Ridge model.
Architecture:
    1. Base Model (Linear): HAR-Ridge -> Predicts Trend.
    2. Residual Model (Non-linear): GBM -> Predicts (Actual - Trend).
    3. Final Prediction = Base Pred + Residual Pred.

Hypothesis: Linear model captures the main autocorrelation, while GBM captures non-linear shocks/regime shifts retained in residuals.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
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
    print("V21: Hybrid Residual Learning (Linear + Boosting)")
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
    y_train = train_df['Target'].values
    X_test = scaler.transform(test_df[features])
    y_test = test_df['Target'].values
    
    # 2. Train Stage 1: Base Linear Model (HAR-Ridge)
    print("\n[Step 2] Stage 1: Base Linear Model (HAR-Ridge)")
    model_base = Ridge(alpha=1.0, random_state=SEED)
    model_base.fit(X_train, y_train)
    
    pred_train_base = model_base.predict(X_train)
    pred_test_base = model_base.predict(X_test)
    
    r2_base = r2_score(y_test, pred_test_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, pred_test_base))
    print(f"Base Model R2: {r2_base:.4f}, RMSE: {rmse_base:.4f}")
    
    # 3. Train Stage 2: Residual Model (GBM)
    print("\n[Step 3] Stage 2: Residual Learning (GBM)")
    # Calculate Residuals
    residuals_train = y_train - pred_train_base
    
    # Model for Residuals
    # Using small depth to avoid overfitting noise
    model_resid = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=SEED
    )
    model_resid.fit(X_train, residuals_train)
    
    # Predict Residuals on Test
    pred_test_resid = model_resid.predict(X_test)
    
    # 4. Final Combination
    print("\n[Step 4] Combining Predictions")
    pred_test_final = pred_test_base + pred_test_resid
    
    r2_final = r2_score(y_test, pred_test_final)
    rmse_final = np.sqrt(mean_squared_error(y_test, pred_test_final))
    
    print(f"Hybrid Model R2: {r2_final:.4f}, RMSE: {rmse_final:.4f}")
    
    # 5. Evaluation
    print("\n[Step 5] Comparison & DM Test")
    
    improv_r2 = (r2_final - r2_base) / np.abs(r2_base) * 100
    print(f"Improvement in R2: {improv_r2:.2f}%")
    
    stat, p_val = diebold_mariano_test_simple(y_test, pred_test_base, pred_test_final)
    
    if p_val < 0.05:
        sig = "Significant"
        winner = "Hybrid" if stat > 0 else "Base"
    else:
        sig = "Not Sig"
        winner = "Tie"
        
    print(f"DM Stat: {stat:.2f}, p-value: {p_val:.4f} -> Result: {winner} ({sig})")
    
    # Feature Importance of Residual Model
    # Does GBM use certain lags more for residuals?
    print("\n[Analysis] Residual Predictor Importance")
    importance = model_resid.feature_importances_
    for i, feat in enumerate(features):
        print(f"{feat}: {importance[i]:.4f}")

    # Save
    out_data = {
        'base_performance': {'R2': r2_base, 'RMSE': rmse_base},
        'hybrid_performance': {'R2': r2_final, 'RMSE': rmse_final},
        'improvement_pct': improv_r2,
        'dm_test': {'stat': stat, 'p_value': p_val, 'winner': winner}
    }
    
    out_path = 'src/experiments/sci/v21_hybrid_residual.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
