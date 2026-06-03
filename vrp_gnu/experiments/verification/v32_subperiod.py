import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import os
import json
import warnings
from arch import arch_model

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def run_experiment():
    print("="*80)
    print("V32: Sub-period Stability Analysis")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    pooled_data = []
    print("Extracting features...")
    for asset in ASSETS:
        ret = np.log(raw[asset] / raw[asset].shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        am = arch_model(ret * 100, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        garch_vol = res.conditional_volatility / 100
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_lag1': garch_vol.shift(1),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        pooled_data.append(d)
        
    data = pd.concat(pooled_data).sort_index()
    
    # Define Sub-periods
    periods = {
        'Full Period': data.index,
        'Pre-COVID (2010-2019)': data[(data.index < '2020-01-01')].index,
        'COVID Crisis (2020-2021)': data[(data.index >= '2020-01-01') & (data.index < '2022-01-01')].index,
        'Inflation/Stable (2022-2025)': data[(data.index >= '2022-01-01')].index
    }
    
    # Training of V29 on Full dataset (or we can re-train per period, but typically we check OOS stability of a trained model)
    # Most rigorous: 80% train, then check performance on the three sub-periods in the 20% test set?
    # Or check full sample stability. Let's do 80/20 split and then check where the test set falls.
    # Actually, the user asked for sub-period analysis of the model. 
    # Let's use a 70% Train (Pre-Transition) and 30% Test (Transition + COVID + Stable).
    
    split_idx = int(len(data) * 0.7)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Garch_lag1']
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[feats])
    y_train = train_df['Target']
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    
    print(f"\nTraining Samples: {len(train_df)} (Up to {train_df.index.max()})")
    
    results_periods = []
    
    for name, idx in periods.items():
        subset = data.loc[idx]
        if subset.empty: continue
        
        X_sub = sc.transform(subset[feats])
        y_sub = subset['Target']
        
        preds = model.predict(X_sub)
        r2 = r2_score(y_sub, preds)
        rmse = np.sqrt(np.mean((y_sub - preds)**2))
        
        results_periods.append({
            'Period': name,
            'R2': r2,
            'RMSE': rmse,
            'Samples': len(subset)
        })
        
    df_res = pd.DataFrame(results_periods)
    print("\n[Regime Stability Results]")
    print(df_res.to_string(index=False))
    
    out_path = 'src/experiments/verification/v32_subperiod.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results_periods, f, indent=2)

if __name__ == "__main__":
    run_experiment()
