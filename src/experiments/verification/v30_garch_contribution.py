import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
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
    ret_scaled = returns * 100
    am = arch_model(ret_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(disp='off', show_warning=False)
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
        
        price = close[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        try:
             garch_vol = fit_garch_and_predict(ret)
             s_garch = pd.Series(garch_vol, index=ret.index, name='GarchVol')
             
             df = pd.DataFrame(index=ret.index)
             df['LogRV'] = np.log(rv + 1e-6)
             df['GarchVol'] = s_garch
             
             # Features
             df['HAR_D'] = df['LogRV'].shift(1)
             df['HAR_W'] = df['LogRV'].shift(5)
             df['HAR_M'] = df['LogRV'].shift(22)
             df['Garch_L1'] = df['GarchVol'].shift(1)
             
             df['Target'] = np.log(rv.shift(-22) + 1e-6)
             df['Asset'] = asset
             
             df = df.dropna()
             if len(df) < 500: continue
             pooled_data.append(df)
             
        except Exception as e:
            print(f"GARCH failed for {asset}: {e}")
            continue

    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V30: GARCH Contribution Analysis (Ablation Study)")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    y_train = train['Target']
    y_test = test['Target']
    
    results = []
    
    configurations = {
        'HAR-Only': ['HAR_D', 'HAR_W', 'HAR_M'],
        'GARCH-Only': ['Garch_L1'],
        'Hybrid-V29': ['HAR_D', 'HAR_W', 'HAR_M', 'Garch_L1']
    }
    
    models = {}
    
    for name, feats in configurations.items():
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feats])
        X_test = scaler.transform(test[feats])
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'Params': len(feats)
        })
        models[name] = {'preds': preds, 'r2': r2}
        
    df_res = pd.DataFrame(results)
    print("\n[Ablation Results]")
    print(df_res.to_string(index=False))
    
    # Calculate relative contributions
    r2_full = models['Hybrid-V29']['r2']
    r2_har = models['HAR-Only']['r2']
    r2_garch = models['GARCH-Only']['r2']
    
    gain_from_garch = (r2_full - r2_har) / abs(r2_har) * 100
    overlap_info = (r2_har + r2_garch - r2_full) # Purely indicative
    
    print(f"\n[Analysis]")
    print(f"R2 Gain from adding GARCH to HAR: {gain_from_garch:.2f}%")
    
    out_path = 'src/experiments/verification/v30_garch_contribution.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
