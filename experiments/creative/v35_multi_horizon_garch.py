import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from arch import arch_model
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def fit_garch_horizon(returns, horizon_name):
    # Scale for numerical stability
    ret_scaled = returns * 100
    try:
        am = arch_model(ret_scaled, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility / 100
        return pd.Series(cond_vol, index=returns.index, name=f'Garch_{horizon_name}')
    except Exception as e:
        print(f"GARCH fit failed for {horizon_name}: {e}")
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def run_experiment():
    print("="*80)
    print("V35: Multi-Horizon GARCH Experiment")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = int_raw = raw.ffill()
    
    pooled_data = []
    print("Extracting features with Multi-Horizon GARCH...")
    
    for asset in ASSETS:
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        
        # 1. Base Features (HAR)
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # 2. Multi-Horizon GARCH
        # Horizon 1: Daily returns (Standard)
        garch_d = fit_garch_horizon(ret_daily, 'Daily')
        
        # Horizon 2: Weekly returns (5-day agg)
        ret_weekly = ret_daily.resample('W').sum()
        garch_w_raw = fit_garch_horizon(ret_weekly, 'Weekly')
        garch_w = garch_w_raw.reindex(ret_daily.index, method='ffill')
        
        # Horizon 3: Monthly returns (Month End)
        ret_monthly = ret_daily.resample('ME').sum()
        
        if len(ret_monthly) > 24: 
             garch_m_raw = fit_garch_horizon(ret_monthly, 'Monthly')
             garch_m = garch_m_raw.reindex(ret_daily.index, method='ffill')
        else:
             garch_m = pd.Series(0, index=ret_daily.index)

        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            
            # Original GARCH
            'Garch_Daily': garch_d.shift(1),
            
            # New Multi-Horizon GARCH Features
            'Garch_Weekly': garch_w.shift(1),
            'Garch_Monthly': garch_m.shift(1),
            
            'Target': log_rv.shift(-22),
            'Asset': asset
        })
        
        # Debug nan
        if d.isna().sum().sum() > 0:
             # Just dropping is fine, GARCH shift creates NaNs at start
             pass

        d = d.dropna()
        if len(d) > 0:
            pooled_data.append(d)
        else:
            print(f"Warning: {asset} dropped all data.")
        
    if not pooled_data:
        print("Error: No data available after processing!")
        return

    data = pd.concat(pooled_data).sort_index()
    print(f"Total processed samples: {len(data)}")
    
    # Train/Test Split
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    y_train = train_df['Target']
    y_test = test_df['Target']
    
    # Experiment 1: Baseline (V29)
    feats_v29 = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Garch_Daily']
    sc_v29 = StandardScaler()
    X_train_v29 = sc_v29.fit_transform(train_df[feats_v29])
    X_test_v29 = sc_v29.transform(test_df[feats_v29])
    
    model_v29 = Ridge(alpha=1.0).fit(X_train_v29, y_train)
    pred_v29 = model_v29.predict(X_test_v29)
    r2_v29 = r2_score(y_test, pred_v29)
    
    # Experiment 2: V35 (Multi-Horizon)
    feats_v35 = feats_v29 + ['Garch_Weekly', 'Garch_Monthly']
    sc_v35 = StandardScaler()
    X_train_v35 = sc_v35.fit_transform(train_df[feats_v35])
    X_test_v35 = sc_v35.transform(test_df[feats_v35])
    
    model_v35 = Ridge(alpha=1.0).fit(X_train_v35, y_train)
    pred_v35 = model_v35.predict(X_test_v35)
    r2_v35 = r2_score(y_test, pred_v35)
    
    print("\n[Results]")
    print(f"V29 Baseline R2: {r2_v29:.5f}")
    print(f"V35 Multi-Garch R2: {r2_v35:.5f}")
    print(f"Improvement: {r2_v35 - r2_v29:.5f}")
    
    res = {
        'V29_R2': r2_v29,
        'V35_R2': r2_v35,
        'Improvement': r2_v35 - r2_v29
    }
    
    with open('src/experiments/creative/v35_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_experiment()
