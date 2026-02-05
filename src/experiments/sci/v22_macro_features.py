"""
V22 Exp: Macro-Financial Feature Expansion (SCI Phase 5)
Purpose: To break the R2 ceiling by injecting Exogenous Macro Information.
New Features:
1. VIX (^VIX): Market Fear directly.
2. Yield Spread (^TNX - ^IRX): Proxy for Yield Curve (10Y - 13W).
3. Dollar Index (DX-Y.NYB): Global Liquidity.
4. Oil (CL=F): Inflation Expectation.

Hypothesis: Macro variables provide "State Information" that modulates volatility regimes.
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
MACRO_TICKERS = {
    'VIX': '^VIX',
    'Yield10Y': '^TNX',
    'Yield13W': '^IRX', # Proxy for short rate
    'Dollar': 'DX-Y.NYB',
    'Oil': 'CL=F'
}
SEED = 42

def load_macro_data(start, end):
    print("Downloading Macro Data...")
    tickers = list(MACRO_TICKERS.values())
    raw = yf.download(tickers, start=start, end=end, progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
        
    df = pd.DataFrame(index=close.index)
    
    # Map back to names
    inv_map = {v: k for k, v in MACRO_TICKERS.items()}
    for col in close.columns:
        # yfinance columns might be simple ticker or ticker
        clean_col = col.replace('^', '') # Tickers in cols might not have ^ if yfinance stripped it? usually does not.
        # Actually yfinance returns columns as the Tickers requested.
        
        # Let's match carefully
        matched_name = None
        for k, v in MACRO_TICKERS.items():
            if col == v:
                matched_name = k
                break
        
        if matched_name:
             df[matched_name] = close[col]
             
    # Fill missing
    df = df.ffill().bfill()
    
    # Derived Features
    # Yield Curve: 10Y - 13W (approx 10Y - 3M spread)
    if 'Yield10Y' in df.columns and 'Yield13W' in df.columns:
        df['YieldSpread'] = df['Yield10Y'] - df['Yield13W']
    else:
        print("Warning: Yield data missing for spread calc.")
        
    # Log Returns of others
    for col in ['VIX', 'Dollar', 'Oil']:
        if col in df.columns:
            df[f'{col}_Ret'] = np.log(df[col] / df[col].shift(1))
            
    return df.dropna()

def feature_engineering(data, macro_df):
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
        
        # Endogenous Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        
        # Merge Macro Features (join by index)
        df = df.join(macro_df, how='inner')
        
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V22: Macro-Financial Feature Expansion")
    print("="*80)
    
    start_date = '2010-01-01'
    end_date = '2025-01-01'
    
    # 1. Macro Data
    macro_df = load_macro_data(start_date, end_date)
    # Select Macro Features to use
    # Levels (VIX, YieldSpread) and Changes (Dollar_Ret, Oil_Ret)?
    # Usually VIX level is informative. Spread level is informative.
    macro_features = ['VIX', 'YieldSpread', 'Dollar_Ret', 'Oil_Ret']
    # Filter available
    macro_features = [f for f in macro_features if f in macro_df.columns]
    
    print(f"Using Macro Features: {macro_features}")
    
    # 2. Asset Data
    print("\n[Step 1] Preparing Asset Data...")
    tickers = ASSETS
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    data = feature_engineering(raw, macro_df)
    
    # Time Series Split
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    base_features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    all_features = base_features + macro_features
    
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(train_df[base_features])
    X_test_base = scaler_base.transform(test_df[base_features])
    
    scaler_macro = StandardScaler()
    X_train_macro = scaler_macro.fit_transform(train_df[all_features])
    X_test_macro = scaler_macro.transform(test_df[all_features])
    
    y_train = train_df['Target']
    y_test = test_df['Target']
    
    # 3. Model Training
    print("\n[Step 2] Training Models")
    
    # Base Model
    model_base = Ridge(alpha=1.0, random_state=SEED)
    model_base.fit(X_train_base, y_train)
    pred_base = model_base.predict(X_test_base)
    r2_base = r2_score(y_test, pred_base)
    
    # Macro Model
    model_macro = Ridge(alpha=1.0, random_state=SEED)
    model_macro.fit(X_train_macro, y_train)
    pred_macro = model_macro.predict(X_test_macro)
    r2_macro = r2_score(y_test, pred_macro)
    
    # 4. Results
    print(f"\nBase Model R2: {r2_base:.4f}")
    print(f"Macro Model R2: {r2_macro:.4f}")
    
    improv = (r2_macro - r2_base) / abs(r2_base) * 100
    print(f"Improvement: {improv:.2f}%")
    
    # Feature Importance of Macro
    print("\n[Analysis] Feature Coefficients (Macro Model)")
    coefs = pd.Series(model_macro.coef_, index=all_features)
    print(coefs.sort_values(key=abs, ascending=False))

    out_data = {
        'base_r2': r2_base,
        'macro_r2': r2_macro,
        'improvement': improv,
        'coefficients': coefs.to_dict()
    }
    
    out_path = 'src/experiments/sci/v22_macro_features.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
