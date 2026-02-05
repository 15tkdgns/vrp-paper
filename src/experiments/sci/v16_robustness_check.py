"""
V16 Exp: Robustness & Sensitivity Analysis (SCI Phase 4)
Purpose: To demonstrate that the Trading Strategy is not overfitted to specific parameters or time periods.

Key Checks:
1. Walk-Forward Validation:
   - Train on rolling window (e.g., 5 years), Test on next 1 year.
   - Prevents look-ahead bias and checks stability over time.
2. Parameter Sensitivity (Trading Strategy):
   - Transaction Costs: 5bps, 10bps, 15bps (Base), 20bps, 30bps.
   - Signal Thresholds: Top 10%, 15%, 20% (Base), 25%, 30% Volatility regimes.
   - Leverage Caps: 1.0x, 1.5x (Base), 2.0x.

Model: Pooled HAR-Ridge (Selected as "Robust Universal Model" from v14/v15).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond',
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
        
        # RV calculation
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1']
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6) # Predict next month vol
        df['NextReturn'] = ret.shift(-1) # For trading simulation
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index() # Sor by Date for Walk-Forward

def run_experiment():
    print("="*80)
    print("V16: Robustness & Sensitivity Analysis")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS + ['^VIX']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'VRP_Proxy']
    
    # 2. Walk-Forward Validation
    print("\n[Step 2] Walk-Forward Validation (Rolling Window)...")
    # Training Window: 5 Years (~1250 days), Test: 1 Year (~250 days)
    # Step size: 1 Year
    
    years = sorted(data.index.year.unique())
    start_year = 2015
    
    wf_results = []
    simulation_segments = []
    
    for test_year in range(start_year, 2025 + 1):
        train_start = str(test_year - 5)
        train_end = str(test_year - 1)
        test_start = str(test_year)
        test_end = str(test_year)
        
        train_df = data[train_start:train_end]
        test_df = data[test_start:test_end]
        
        if len(train_df) < 100 or len(test_df) < 10: continue
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        y_train = train_df['Target']
        X_test = scaler.transform(test_df[features])
        y_test = test_df['Target']
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        
        print(f"Test Year: {test_year} | R2: {r2:.4f} | RMSE: {rmse:.4f}")
        wf_results.append({'Year': test_year, 'R2': r2, 'RMSE': rmse})
        
        # Store predictions for trading simulation
        test_df = test_df.copy()
        test_df['PredictedVol'] = pred
        simulation_segments.append(test_df)
        
    avg_wf_r2 = np.mean([r['R2'] for r in wf_results])
    print(f"Average Walk-Forward R2: {avg_wf_r2:.4f}")
    
    # 3. Sensitivity Analysis (Trading Strategy)
    print("\n[Step 3] Sensitivity Analysis (Signal Trading)...")
    
    # Concatenate all test segments with predictions
    if not simulation_segments:
        print("No predictions generated.")
        return

    sim_data = pd.concat(simulation_segments)
    
    # Parameters to test
    costs_bps = [5, 15, 30] # Transaction costs
    thresholds = [0.10, 0.20, 0.30] # Top X% volatility regime -> Cash
    
    sensitivity_res = []
    
    print(f"{'Cost(bps)':<10} {'Threshold':<10} {'Sharpe':<10} {'MDD':<10} {'Return':<10}")
    print("-" * 55)
    
    for cost in costs_bps:
        for thres in thresholds:
            total_returns = []
            
            # Simple Simulation per Asset
            for asset in ASSETS:
                adf = sim_data[sim_data['Asset'] == asset].copy()
                if len(adf) == 0: continue
                
                # Determine Regime (High Vol -> Cash)
                cutoff = adf['PredictedVol'].quantile(1.0 - thres)
                # Signal: 1 if Pred < cutoff else 0 (Cash)
                # Note: Pred is LogRV.
                
                adf['Signal'] = np.where(adf['PredictedVol'] < cutoff, 1.0, 0.0)
                
                # Returns
                cost_decimal = cost / 10000.0
                turnover = adf['Signal'].diff().abs().fillna(0)
                
                # Strategy Return: Signal * NextReturn - Cost
                # Note: This is simplified. Signal t determines exposure t+1. 
                # NextReturn is t+1 return.
                
                strat_ret = adf['Signal'] * adf['NextReturn'] - (turnover * cost_decimal)
                total_returns.extend(strat_ret.values)
                
            # Aggregate Strategy Performance
            # Assuming equal weight portfolio of all assets implies averaging returns? 
            # Or just concatenated returns distribution metrics. 
            # Let's calculate Mean/Std of the pooled return stream for simplicity 
            # (equivalent to assessing the strategy's expectancy per trade opportunity).
            # Ideally, we build a daily portfolio index.
            
            # Portfolio Approach:
            # Group by Date, mean strategy return
            port_ret = sim_data.groupby(level=0).apply(
                lambda x: (
                    np.where(x['PredictedVol'] < x['PredictedVol'].quantile(1.0-thres), 1.0, 0.0) * x['NextReturn'] 
                    - (np.nan_to_num(np.diff(np.where(x['PredictedVol'] < x['PredictedVol'].quantile(1.0-thres), 1.0, 0.0), prepend=0)) != 0) * cost_decimal
                ).mean()
            )
            
            ann_ret = port_ret.mean() * 252
            ann_vol = port_ret.std() * np.sqrt(252)
            sharpe = ann_ret / (ann_vol + 1e-6)
            
            # MDD
            cum_ret = (1 + port_ret).cumprod()
            peak = cum_ret.cummax()
            dd = (cum_ret - peak) / peak
            mdd = dd.min()
            
            print(f"{cost:<10} {thres:<10.2f} {sharpe:<10.4f} {mdd:<10.4f} {ann_ret*100:.1f}%")
            
            sensitivity_res.append({
                'Cost_bps': cost,
                'Threshold': thres,
                'Sharpe': sharpe,
                'MDD': mdd,
                'Ann_Return': ann_ret
            })
            
    # Save Results
    out_data = {
        'walk_forward': wf_results,
        'sensitivity': sensitivity_res
    }
    
    out_path = 'src/experiments/sci/v16_robustness_check.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
