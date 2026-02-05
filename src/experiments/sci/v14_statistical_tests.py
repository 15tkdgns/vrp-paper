"""
V14 Exp: Statistical Significance Verification (SCI Phase 4)
Purpose: To provide statistical rigor to the model comparison results.

Key Tests:
1. Bootstrap Confidence Intervals (95%) for R2, RMSE, MAE.
2. Diebold-Mariano Test for predictive accuracy significance.

Models Compared:
- Baseline: HAR-Ridge
- Proposed: Random Forest (Universal Model)
- Competitor: XGBoost
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import xgboost as xgb
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity',
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
        
        # RV calculation
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Extended Features (Full version matching v8)
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        df['RS_Asymmetry'] = df['LogRV_Down'] - df['LogRV_Up']
        
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1']
        
        # Technicals
        window = 14
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1) / 100.0
        
        window_bb = 20
        ma = price.rolling(window_bb).mean()
        std = price.rolling(window_bb).std()
        df['BB_Width'] = ((2 * std) / (ma + 1e-6)).shift(1)
        
        # Moments
        df['Skewness'] = ret.rolling(22).skew().shift(1).fillna(0)
        df['Kurtosis'] = ret.rolling(22).kurt().shift(1).fillna(0)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).reset_index(drop=True)

def diebold_mariano_test(real, pred1, pred2, h=1, power=2):
    """
    Diebold-Mariano test.
    stat > 1.96: Model 1 better (Loss1 < Loss2 significantly? No, H0: equal. Sign depends on order)
    d = e1^2 - e2^2. 
    If d > 0, e1 > e2 => Model 2 is better. 
    If d < 0, e1 < e2 => Model 1 is better.
    """
    e1 = real - pred1
    e2 = real - pred2
    
    d = (np.abs(e1)**power) - (np.abs(e2)**power) 
    
    d_mean = np.mean(d)
    T = len(d)
    
    # Simple variance for h=1 (approx for now, ideally Newey-West)
    gamma0 = np.var(d, ddof=1)
    std_error = np.sqrt(gamma0 / T)
    
    dm_stat = d_mean / std_error
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_boot=1000):
    scores = []
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(y_true))
    
    for _ in range(n_boot):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        score = metric_func(y_true[sample_idx], y_pred[sample_idx])
        scores.append(score)
        
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean_val = np.mean(scores)
    
    return mean_val, lower, upper

def run_experiment():
    print("="*80)
    print("V14: Statistical Significance Analysis (SCI Phase 4)")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS + ['^VIX']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    # Split
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    features = [
        'LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 
        'LogRV_Down', 'LogRV_Up', 'RS_Asymmetry',
        'VRP_Proxy', 'RSI', 'BB_Width', 'Skewness', 'Kurtosis'
    ]
    
    X_train = train_data[features]
    y_train = train_data['Target']
    X_test = test_data[features]
    y_test = test_data['Target']
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 2. Model Training
    print("\n[Step 2] Training Models...")
    
    # M1: Baseline (HAR-Ridge)
    m1 = Ridge(alpha=1.0)
    m1.fit(X_train_s, y_train)
    p1 = m1.predict(X_test_s)
    
    # M2: Random Forest (Proposed)
    m2 = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=SEED) # Tuned up
    m2.fit(X_train_s, y_train)
    p2 = m2.predict(X_test_s)
    
    # M3: XGBoost (Competitor)
    m3 = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=SEED) # Tuned up
    m3.fit(X_train_s, y_train)
    p3 = m3.predict(X_test_s)
    
    predictions = {
        'HAR-Ridge': p1,
        'Random Forest': p2,
        'XGBoost': p3
    }
    
    # 3. Bootstrap CI
    print("\n[Step 3] Calculating Bootstrap Confidence Intervals (95%)...")
    metrics = {
        'R2': r2_score,
        'RMSE': lambda y, p: np.sqrt(mean_squared_error(y, p)),
        'MAE': mean_absolute_error
    }
    
    ci_results = {}
    
    print(f"{'Model':<15} {'Metric':<6} {'Mean':<8} {'95% CI':<20}")
    print("-" * 55)
    
    for name, pred in predictions.items():
        ci_results[name] = {}
        for metric_name, func in metrics.items():
            mean_val, lower, upper = bootstrap_confidence_interval(y_test.values, pred, func)
            ci_results[name][metric_name] = {'mean': mean_val, 'lower': lower, 'upper': upper}
            print(f"{name:<15} {metric_name:<6} {mean_val:.4f}   [{lower:.4f}, {upper:.4f}]")
            
    # 4. Diebold-Mariano Tests
    print("\n[Step 4] Diebold-Mariano Tests (Target: Random Forest)")
    print(f"{'Comparison':<30} {'DM Stat':<10} {'p-value':<10} {'Result':<15}")
    print("-" * 70)
    
    comparisons = [
        ('Random Forest', 'HAR-Ridge'),
        ('Random Forest', 'XGBoost')
    ]
    
    dm_results = {}
    
    for m_a, m_b in comparisons:
        pred_a = predictions[m_a]
        pred_b = predictions[m_b]
        
        stat, p_val = diebold_mariano_test(y_test.values, pred_a, pred_b, h=1)
        
        significance = "Significant" if p_val < 0.05 else "Not Sig"
        print(f"{m_a} vs {m_b:<15} {stat:.4f}     {p_val:.4f}     {significance}")
        
        dm_results[f"{m_a}_vs_{m_b}"] = {'stat': stat, 'p_value': p_val, 'significance': significance}

    # Save Results
    final_output = {
        'confidence_intervals': ci_results,
        'dm_tests': dm_results
    }
    
    out_path = 'src/experiments/sci/v14_statistical_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
