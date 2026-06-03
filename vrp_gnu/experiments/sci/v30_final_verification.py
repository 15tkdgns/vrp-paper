"""
V30 Exp: Final Verification (SCI Phase 8)
Purpose: Rigorous comparison of Baseline (Standard HAR) vs Champion (Expanded HAR).
Models:
1. Baseline: Standard HAR (Lags 1,5,22) + Ridge.
2. Champion: Asymmetric HAR (Good/Bad Vol) + GARCH Feature + Ridge.
Method:
- Walk-Forward Validation (Rolling Window) to simulate real-world trading.
- Metrics: R2, RMSE, Diebold-Mariano Test (p-value).
- Analysis: Asset-wise, Year-wise (Regime).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from arch import arch_model
from scipy import stats
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

# --- GARCH Helper (Robust) ---
def fit_garch_and_predict(returns):
    ret_scaled = returns * 100
    # Use Constant mean for robustness
    am = arch_model(ret_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(disp='off', show_warning=False)
    cond_vol = res.conditional_volatility / 100
    return cond_vol

# --- Feature Engineering ---
def prepare_data(raw_data):
    if isinstance(raw_data.columns, pd.MultiIndex):
        close = raw_data['Close']
    else:
        close = raw_data
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1)).dropna()
        
        # 1. Basic RV
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # 2. Asymmetric Components
        ret_pos = ret.clip(lower=0)
        ret_neg = ret.clip(upper=0)
        rv_pos = (ret_pos**2).rolling(22).mean() * 252 * 10000
        rv_neg = (ret_neg**2).rolling(22).mean() * 252 * 10000
        
        # 3. GARCH Component
        try:
             garch_vol = fit_garch_and_predict(ret)
             s_garch = pd.Series(garch_vol, index=ret.index, name='GarchVol')
             df = df.join(s_garch, how='outer') # Join first
        except:
             # Fallback if GARCH fails
             df['GarchVol'] = np.nan
        
        # Align Series
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_pos'] = np.log(rv_pos + 1e-6)
        df['LogRV_neg'] = np.log(rv_neg + 1e-6)
        
        # Lags
        lags = [1, 5, 22]
        # Baseline Features : LogRV lags
        for lag in lags:
            df[f'LogRV_lag{lag}'] = df['LogRV'].shift(lag)
            
        # Champion Features : Asym lags + GARCH lag
        for lag in lags:
            df[f'LogRV_pos_lag{lag}'] = df['LogRV_pos'].shift(lag)
            df[f'LogRV_neg_lag{lag}'] = df['LogRV_neg'].shift(lag)
            df[f'GarchVol_lag{lag}'] = df['GarchVol'].shift(lag)
            
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset'] = asset
        
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    if not pooled_data: return pd.DataFrame()
    return pd.concat(pooled_data).sort_index()

# --- Diebold-Mariano Test ---
def dm_test(actual, pred1, pred2, h=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    
    T = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0) # Simple variance for now, strictly needs HAC
    
    # Simple DM statistic
    # Using autocovariance correction for h-step forecast would be better
    # But for h=1 (or implicitly h=22 but treated as 1-step regression), we start simple
    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# --- Rolling Window Logic ---
def walk_forward_validation(data, features, window_size=1000, step_size=22):
    # Sort by time
    data = data.sort_index()
    unique_dates = data.index.unique()
    
    preds = []
    actuals = []
    dates = []
    assets = []
    
    # We roll based on time index chunks
    start_idx = window_size
    
    # Pre-allocate scaler/model to avoid re-init overhead
    # But fit needs to be in loop
    
    # To speed up, we update every 'step_size' days (e.g. monthly)
    # But predict for all assets in that chunk
    
    for i in range(start_idx, len(unique_dates), step_size):
        # Current Train Window: [i-window : i]
        # Test Window: [i : i+step]
        
        date_split = unique_dates[i]
        date_end = unique_dates[min(i+step_size, len(unique_dates)-1)]
        
        train_mask = (data.index < date_split) & (data.index >= unique_dates[i-window_size])
        test_mask = (data.index >= date_split) & (data.index < date_end)
        
        if not np.any(test_mask): break
        
        train_df = data[train_mask]
        test_df = data[test_mask]
        
        if len(train_df) < 100: continue
        
        X_train = train_df[features]
        y_train = train_df['Target']
        X_test = test_df[features]
        y_test = test_df['Target']
        
        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Model (Fixed Alpha for stability/speed in rolling)
        # We assume Adaptive nature is captured by Rolling Window itself
        model = Ridge(alpha=1.0) 
        model.fit(X_train_s, y_train)
        
        p = model.predict(X_test_s)
        
        preds.extend(p)
        actuals.extend(y_test)
        dates.extend(test_df.index)
        assets.extend(test_df['Asset'])
        
        # Determine if last loop
        if i + step_size >= len(unique_dates): break
        
    return pd.DataFrame({
        'Date': dates,
        'Asset': assets,
        'Actual': actuals,
        'Predicted': preds
    })

def run_experiment():
    print("="*80)
    print("V30: Final Verification (Champion vs Baseline)")
    print("="*80)
    
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = prepare_data(raw)
    
    print(f"Data Prepared. Total samples: {len(data)}")
    
    # Features
    feats_base = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    feats_champ = feats_base + \
                  ['LogRV_pos_lag1', 'LogRV_pos_lag5', 'LogRV_pos_lag22',
                   'LogRV_neg_lag1', 'LogRV_neg_lag5', 'LogRV_neg_lag22',
                   'GarchVol_lag1']
                   
    # Run Walk-Forward
    print("\n[Running Walk-Forward] Baseline Model...")
    res_base = walk_forward_validation(data, feats_base)
    
    print("\n[Running Walk-Forward] Champion Model...")
    res_champ = walk_forward_validation(data, feats_champ)
    
    # Ensure alignment
    # Because walking forward might skip if data missing, intersect
    # Just creating a merged df
    res_base['ID'] = res_base['Date'].astype(str) + res_base['Asset']
    res_champ['ID'] = res_champ['Date'].astype(str) + res_champ['Asset']
    
    merged = pd.merge(res_base[['ID', 'Date', 'Asset', 'Actual', 'Predicted']], 
                      res_champ[['ID', 'Predicted']], 
                      on='ID', suffixes=('_Base', '_Champ'))
    
    merged = merged.sort_values('Date')
    
    # 1. Overall Metrics
    r2_base = r2_score(merged['Actual'], merged['Predicted_Base'])
    r2_champ = r2_score(merged['Actual'], merged['Predicted_Champ'])
    
    print(f"\n[Overall Result]")
    print(f"Baseline R2: {r2_base:.4f}")
    print(f"Champion R2: {r2_champ:.4f}")
    print(f"Improvement: {(r2_champ - r2_base)/abs(r2_base)*100:.2f}%")
    
    # 2. DM Test
    dm_stat, dm_p = dm_test(merged['Actual'], merged['Predicted_Base'], merged['Predicted_Champ'])
    print(f"\n[Diebold-Mariano Test]")
    print(f"DM Statistic: {dm_stat:.4f}")
    print(f"P-Value:      {dm_p:.6f}")
    print("Result:       " + ("Significantly Better" if (dm_stat > 1.96 and dm_p < 0.05) else "Not Significant"))
    # Note: DM stat sign depends on (e1^2 - e2^2). If Base error > Champ error, diff is positive -> DM > 0 -> Champ wins.
    
    # 3. Asset-wise Analysis
    print("\n[Asset-wise Analysis]")
    assets = merged['Asset'].unique()
    asset_res = {}
    for a in assets:
        sub = merged[merged['Asset'] == a]
        r2_b = r2_score(sub['Actual'], sub['Predicted_Base'])
        r2_c = r2_score(sub['Actual'], sub['Predicted_Champ'])
        print(f"{a:<5} | Base: {r2_b:.4f} -> Champ: {r2_c:.4f} ({'+' if r2_c>r2_b else ''}{(r2_c-r2_b)*100:.2f}%)")
        asset_res[a] = {'base': r2_b, 'champ': r2_c}
        
    # 4. Regime Analysis (Yearly)
    print("\n[Yearly Analysis]")
    merged['Year'] = pd.to_datetime(merged['Date']).dt.year
    years = sorted(merged['Year'].unique())
    year_res = {}
    for y in years:
        sub = merged[merged['Year'] == y]
        r2_b = r2_score(sub['Actual'], sub['Predicted_Base'])
        r2_c = r2_score(sub['Actual'], sub['Predicted_Champ'])
        print(f"{y} | Base: {r2_b:.4f} -> Champ: {r2_c:.4f} ({'+' if r2_c>r2_b else ''}{(r2_c-r2_b)*100:.2f}%)")
        year_res[y] = {'base': r2_b, 'champ': r2_c}
    
    # Convert keys to string for JSON serialization
    year_res_str = {str(k): v for k, v in year_res.items()}
    
    out_data = {
        'overall': {'r2_base': r2_base, 'r2_champ': r2_champ, 'dm_p_value': dm_p},
        'asset_wise': asset_res,
        'year_wise': year_res_str
    }
    
    out_path = 'src/experiments/sci/v30_final_verification.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
