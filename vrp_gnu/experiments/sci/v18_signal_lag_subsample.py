"""
V18 Exp: Signal Efficiency & Sub-sample Analysis (SCI Phase 4.1)
Purpose: To diagnose the "Profitability Failure" identified in v16.
Focus:
1. Signal Lag: Does the predictive power decay instantly? (t+1 vs t+2 vs t+3)
2. Sub-sample Performance: Crisis (High Vol) vs Normal vs Sideways.
3. Break-even Cost: What transaction cost makes this strategy viable?
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
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
        
        # Returns for Trading (t+1 to t+5)
        df['Ret_t1'] = ret.shift(-1)
        df['Ret_t2'] = ret.shift(-2)
        df['Ret_t3'] = ret.shift(-3)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def get_market_regime(date):
    year = date.year
    if year == 2020: return "Crisis (COVID)"
    if year == 2022: return "Crisis (Inflation)"
    if 2017 <= year <= 2019: return "Normal (Bull)"
    if year >= 2023: return "Sideways/Mixed"
    return "Other"

def run_experiment():
    print("="*80)
    print("V18: Signal Efficiency & Sub-sample Analysis")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    # Train/Test Split (Fixed for analysis)
    # Train: 2010-2016, Test: 2017-2024
    train_df = data[data.index.year <= 2016]
    test_df = data[data.index.year >= 2017].copy()
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df['Target']
    X_test = scaler.transform(test_df[features])
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    test_df['PredictedVol'] = model.predict(X_test)
    
    # 2. Signal Lag Analysis
    print("\n[Analysis 1] Signal Decay (Lag Analysis)")
    # Strategy: Cash if High Vol (Top 20%)
    threshold = test_df['PredictedVol'].quantile(0.80)
    test_df['Signal'] = np.where(test_df['PredictedVol'] < threshold, 1.0, 0.0) # 1=Invest, 0=Cash
    
    lags = [1, 2, 3]
    lag_results = {}
    
    print(f"{'Lag (Days)':<10} {'Ann. Return':<12} {'Sharpe':<10}")
    print("-" * 40)
    
    for lag in lags:
        ret_col = f'Ret_t{lag}'
        # Signal generated at t, executed at t+lag against return of t+lag?
        # No, signal at t decides exposure for t+lag return.
        # If execution is immediate (t+1), we capture Ret_t1.
        # If delayed by 1 day (execute at t+2), we capture Ret_t2.
        
        # Simple strategy return without cost for decay check
        strat_ret = test_df['Signal'] * test_df[ret_col]
        
        # Portfolio mean return
        port_daily = strat_ret.groupby(level=0).mean()
        ann_ret = port_daily.mean() * 252
        ann_vol = port_daily.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-6)
        
        print(f"t+{lag:<9} {ann_ret*100:6.2f}%      {sharpe:.4f}")
        lag_results[f't+{lag}'] = {'Return': ann_ret, 'Sharpe': sharpe}

    # 3. Sub-sample Analysis (Regime Performance)
    print("\n[Analysis 2] Sub-sample Performance (Regimes)")
    
    test_df['Regime'] = test_df.index.map(get_market_regime)
    regimes = ["Normal (Bull)", "Crisis (COVID)", "Crisis (Inflation)", "Sideways/Mixed"]
    
    subsample_results = {}
    
    print(f"{'Regime':<20} {'Ann. Return':<12} {'Sharpe':<10} {'Count'}")
    print("-" * 60)
    
    # Base strategy (t+1 execution, 10bps cost assumption for reality check)
    cost_bps = 10
    cost_dec = cost_bps / 10000.0
    
    # Calculate daily turnover per asset
    # Note: Turnover needs Diff of signal. Since predictions are per asset-day...
    # We need to preserve asset structure.
    
    regime_stats = []
    
    for regime in regimes:
        regime_mask = test_df['Regime'] == regime
        if not regime_mask.any(): continue
        
        regime_data = test_df[regime_mask]
        
        # Calculate returns
        # Strategy Return = Signal * Ret_t1 - Cost * Turnover
        # Turnover approx: abs(Signal_t - Signal_t-1).
        # We assume independent daily rebalancing logic for simplicity in this aggregate view
        
        # Simply: Daily Portfolio Return (Signal * Ret) - Daily Cost Estimate
        # Cost estimate: Signal change probability ~ 10% (from v13) -> 10bps * 0.1 * 2 (buy/sell) ~ 2bps/day?
        # Let's use the raw return for "Potential" and comment on cost.
        
        daily_rets = (regime_data['Signal'] * regime_data['Ret_t1']).groupby(level=0).mean()
        
        ann_ret = daily_rets.mean() * 252
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-6)
        count = len(daily_rets)
        
        print(f"{regime:<20} {ann_ret*100:6.2f}%      {sharpe:6.4f}     {count}")
        subsample_results[regime] = {'Return': ann_ret, 'Sharpe': sharpe}
        
    # 4. Break-even Cost Analysis
    print("\n[Analysis 3] Break-even Cost Analysis")
    # For the whole test period
    
    # Total accumulated return (sum of log returns approx)
    daily_raw_ret = (test_df['Signal'] * test_df['Ret_t1']).groupby(level=0).mean()
    total_raw_ret = daily_raw_ret.sum() # approx
    
    # Total Turnover
    # We need accurate turnover.
    # Group by Asset, calculate diff
    turnover_series = test_df.groupby('Asset')['Signal'].diff().abs().fillna(0)
    total_turnover = turnover_series.sum() / len(ASSETS) # Avg turnover per asset across period
    
    # Break-even Cost (bps)
    # Total Return - (Total Turnover * Cost) = 0
    # Cost = Total Return / Total Turnover
    
    if total_turnover > 0:
        be_cost = (total_raw_ret / total_turnover) * 10000
        print(f"Total Raw Return (Log): {total_raw_ret:.4f}")
        print(f"Total Turnover (factor): {total_turnover:.2f}")
        print(f"Break-even Cost: {be_cost:.2f} bps")
    else:
        be_cost = 0
        print("No turnover.")

    # Save
    out_data = {
        'signal_lag': lag_results,
        'subsample': subsample_results,
        'break_even_bps': be_cost
    }
    
    out_path = 'src/experiments/sci/v18_signal_lag_subsample.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
