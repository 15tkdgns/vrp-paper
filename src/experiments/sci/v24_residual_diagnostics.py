"""
V24 Exp: Residual Diagnostics (SCI Phase 6)
Purpose: To diagnose WHY HAR-Ridge error exists.
Tests:
1. Ljung-Box: Check for serial correlation in residuals (Guidance for Lag optimization).
2. ARCH Test (Engle's): Check for volatility clustering in residuals (Guidance for GARCH).
3. Jarque-Bera: Check for Normality (Guidance for Robust/Transform).
4. Asymmetry check: RMSE in Bull vs Bear markets.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
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

def run_experiment():
    print("="*80)
    print("V24: Residual Diagnostics")
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
    
    # 2. Train Model
    print("\n[Step 2] Training Baseline HAR-Ridge...")
    model = Ridge(alpha=1.0, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 3. Calculate Residuals
    residuals = y_test - y_pred
    
    # 4. Diagnostics
    print("\n[Step 3] Running Tests...")
    results = {}
    
    # A. Ljung-Box (Autocorrelation)
    # H0: No autocorrelation. p < 0.05 => Autocorrelation exists.
    lb_df = acorr_ljungbox(residuals, lags=[5, 10, 22], return_df=True)
    print("\n1. Ljung-Box Test (Autocorrelation):")
    print(lb_df)
    results['ljung_box_pvalues'] = lb_df['lb_pvalue'].to_dict()
    
    # B. ARCH Test (Heteroskedasticity)
    # H0: No ARCH effect. p < 0.05 => Volatility clustering exists in residuals.
    # LM test (Engle's)
    # Need consistent length
    arch_stat, arch_pvalue, _, _ = het_arch(residuals)
    print(f"\n2. ARCH Test p-value: {arch_pvalue:.4f}")
    results['arch_pvalue'] = arch_pvalue
    
    # C. Jarque-Bera (Normality)
    # H0: Normal distribution. p < 0.05 => Non-normal.
    jb_stat, jb_pvalue = jarque_bera(residuals)
    print(f"\n3. Jarque-Bera Test p-value: {jb_pvalue:.4f}")
    results['jb_pvalue'] = jb_pvalue
    
    # D. Asymmetry Analysis (Bull vs Bear Prediction Error)
    # Define Bull/Bear by Target > Mean Target (locally or globally?)
    # Let's use simple global mean for now
    mean_target = np.mean(y_test)
    bull_mask = y_test < mean_target # Low Vol = Bull typically
    bear_mask = y_test >= mean_target # High Vol = Bear
    
    rmse_bull = np.sqrt(mean_squared_error(y_test[bull_mask], y_pred[bull_mask]))
    rmse_bear = np.sqrt(mean_squared_error(y_test[bear_mask], y_pred[bear_mask]))
    
    print(f"\n4. Asymmetry Analysis (RMSE):")
    print(f"   Low Vol (Bull): {rmse_bull:.4f}")
    print(f"   High Vol (Bear): {rmse_bear:.4f}")
    results['rmse_bull'] = rmse_bull
    results['rmse_bear'] = rmse_bear
    
    # Recommendations based on results
    recs = []
    if any(p < 0.05 for p in results['ljung_box_pvalues'].values()):
        recs.append("Modify Lag Structure (Autocorrelation found)")
    if arch_pvalue < 0.05:
        recs.append("Consider GARCH-in-Mean (Volatility Clustering found)")
    if jb_pvalue < 0.05:
         recs.append("Consider Box-Cox or Robust Scaling (Non-normality found)")
    if abs(rmse_bear - rmse_bull) > 0.1:
         recs.append("Consider Regime-Switching or Asymmetric Features")
         
    print("\n[Conclusion] Recommendations:")
    for r in recs:
        print(f"- {r}")
        
    results['recommendations'] = recs

    # Save
    out_path = 'src/experiments/sci/v24_residual_diagnostics.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data := results, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
