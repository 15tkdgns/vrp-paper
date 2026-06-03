import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import norm
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=22):
    """
    Diebold-Mariano test for forecast comparison.
    H0: The two models have the same forecast accuracy.
    H1: Model 1 is more accurate than Model 2 (one-sided).
    """
    e1 = (y_true - y_pred1)**2
    e2 = (y_true - y_pred2)**2
    d = e1 - e2
    
    mean_d = np.mean(d)
    T = len(d)
    
    # Auto-covariance for h-step ahead forecasts
    # For h=22, we need Newey-West like adjustment
    # Simplified version:
    gamma0 = np.var(d)
    def autocov(k):
        if k == 0: return gamma0
        return np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))

    # Variance estimate for multi-step (H-1 lags)
    var_d = gamma0
    for k in range(1, h):
        var_d += 2 * autocov(k)
        
    if var_d <= 0: var_d = 1e-8 # Stability
    
    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = norm.cdf(dm_stat) # Probability that d < 0 (Model 1 better)
    
    return dm_stat, p_value

def run_experiment():
    print("="*80)
    print("V31: Diebold-Mariano Statistical Significance Test")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Simple processing
    pooled_data = []
    from arch import arch_model # Need this for GARCH
    
    for asset in ASSETS:
        ret = np.log(raw[asset] / raw[asset].shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH
        am = arch_model(ret * 100, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        garch_vol = res.conditional_volatility / 100
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_lag1': garch_vol.shift(1),
            'Naive': log_rv.shift(1), # Persistence
            'Target': log_rv.shift(-22)
        }).dropna()
        pooled_data.append(d)
        
    data = pd.concat(pooled_data).sort_index()
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    y_test = test['Target'].values
    
    # 1. Hybrid (V29)
    feats_v29 = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Garch_lag1']
    sc_v29 = StandardScaler()
    X_train_v29 = sc_v29.fit_transform(train[feats_v29])
    X_test_v29 = sc_v29.transform(test[feats_v29])
    model_v29 = Ridge(alpha=1.0).fit(X_train_v29, train['Target'])
    pred_v29 = model_v29.predict(X_test_v29)
    
    # 2. HAR-Only
    feats_har = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    sc_har = StandardScaler()
    X_train_har = sc_har.fit_transform(train[feats_har])
    X_test_har = sc_har.transform(test[feats_har])
    model_har = Ridge(alpha=1.0).fit(X_train_har, train['Target'])
    pred_har = model_har.predict(X_test_har)
    
    # 3. Naive (Persistence)
    pred_naive = test['Naive'].values
    
    print("\n[DM Test Results]")
    print(f"{'Comparison':<30} | {'DM Stat':<10} | {'p-value':<10} | {'Sig (0.05)':<10}")
    print("-" * 75)
    
    comparisons = [
        ("V29 vs HAR-Only", pred_v29, pred_har),
        ("V29 vs Naive", pred_v29, pred_naive),
        ("HAR-Only vs Naive", pred_har, pred_naive)
    ]
    
    outputs = []
    for label, p1, p2 in comparisons:
        stat, pval = diebold_mariano_test(y_test, p1, p2)
        sig = "Yes" if pval < 0.05 else "No"
        print(f"{label:<30} | {stat:>10.4f} | {pval:>10.4f} | {sig:<10}")
        outputs.append({'Comparison': label, 'DM_Stat': stat, 'P_Value': pval})
        
    out_path = 'src/experiments/verification/v31_dm_test.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    run_experiment()
