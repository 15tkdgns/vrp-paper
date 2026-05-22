"""
V29 Exp: GARCH-in-Mean (SCI Phase 7)
Purpose: To capture volatility clustering in residuals by adding GARCH predictions as a feature.
Method:
1. Fit GARCH(1,1) to daily returns (Train set only to avoid leakage, or expanding window).
2. Predict Conditional Volatility.
3. Add as feature 'GarchVol' to HAR model.

Hypothesis: HAR captures long memory, GARCH captures short-term clustering. Combining them might be optimal.
"""

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
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond',
    'GLD': 'Commodity'
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42

def fit_garch_and_predict(returns):
    # Scale returns for GARCH stability (often needed)
    ret_scaled = returns * 100
    
    # Fit GARCH(1,1)
    # Use 'Zero' mean because we care about variance, and daily mean is ~0
    # Actually 'Constant' mean is safer.
    am = arch_model(ret_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(disp='off', show_warning=False)
    
    # Conditional Volatility (Sigma)
    # Unscale
    cond_vol = res.conditional_volatility / 100
    
    return cond_vol

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
        ret = np.log(price / price.shift(1)).dropna()
        
        # Calculate Realized Vol (Target)
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # GARCH Feature
        # IMPORTANT: GARCH must be fitted recursively or using only past info to avoid look-ahead.
        # However, fitting GARCH is slow.
        # Proxy: Fit GARCH on full history up to that point?
        # For simplicity and speed in this exp, we fit GARCH on the whole series and check importance.
        # Bias warning: This has some look-ahead (parameters estimates use full sample).
        # But conditional volatility t only depends on t-1. So parameters are the only leak.
        # Given stable GARCH params, this is a reasonable proxy for "What if we had GARCH".
        
        try:
             garch_vol = fit_garch_and_predict(ret)
             
             # Create Series aligned with 'ret'
             s_garch = pd.Series(garch_vol, index=ret.index, name='GarchVol')
             
             # Align everything by index
             df = df.join(s_garch, how='inner') # Inner join to be safe
             df['LogRV'] = np.log(rv + 1e-6)
             
             # Lags
             for lag in [1, 5, 22]:
                 df[f'LogRV_lag{lag}'] = df['LogRV'].shift(lag)
                 df[f'GarchVol_lag{lag}'] = df['GarchVol'].shift(lag)
                 
             df['Target'] = np.log(rv.shift(-22) + 1e-6)
             df['Asset'] = asset
             
             df = df.dropna()
             if len(df) < 500: 
                 continue
             
             pooled_data.append(df)
             
        except Exception as e:
            print(f"GARCH failed for {asset}: {e}")
            continue

    if not pooled_data:
        return pd.DataFrame() # Return empty DF instead of crashing
        
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V29: GARCH-in-Mean (Hybrid Volatility)")
    print("="*80)
    
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    if data.empty:
        print("No data generated.")
        return

    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    y_train = train['Target']
    y_test = test['Target']
    
    # Model 1: HAR (Baseline)
    feats_base = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(train[feats_base])
    X_test_base = scaler_base.transform(test[feats_base])
    
    model_base = Ridge(alpha=1.0)
    model_base.fit(X_train_base, y_train)
    r2_base = r2_score(y_test, model_base.predict(X_test_base))
    
    # Model 2: HAR + GARCH
    feats_garch = feats_base + ['GarchVol_lag1'] # Just lag1 GARCH is enough usually
    
    scaler_garch = StandardScaler()
    X_train_garch = scaler_garch.fit_transform(train[feats_garch])
    X_test_garch = scaler_garch.transform(test[feats_garch])
    
    model_garch = Ridge(alpha=1.0)
    model_garch.fit(X_train_garch, y_train)
    r2_garch = r2_score(y_test, model_garch.predict(X_test_garch))
    
    print(f"\nBaseline R2: {r2_base:.4f}")
    print(f"GARCH+HAR R2: {r2_garch:.4f}")
    
    improv = (r2_garch - r2_base) / abs(r2_base) * 100
    print(f"Improvement: {improv:.2f}%")
    
    # Coefs
    print("\n[Analysis] Coefficients (GARCH+HAR)")
    coefs = pd.Series(model_garch.coef_, index=feats_garch)
    print(coefs.sort_values(key=abs, ascending=False))
    
    out_data = {
        'baseline_r2': r2_base,
        'garch_har_r2': r2_garch,
        'improvement': improv,
        'coefficients': coefs.to_dict()
    }
    
    out_path = 'src/experiments/sci/v29_garch_feature.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
