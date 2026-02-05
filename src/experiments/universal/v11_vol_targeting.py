"""
V11 Exp: Volatility Targeting Strategy Backtest
Purpose: Implement and backtest Volatility Targeting strategy using Random Forest predictions.

Strategy Logic:
- Target Volatility: 10% (Annualized)
- Rebalancing: Daily
- Weight Calculation:
    w_t = Target_Vol / Predicted_Vol_{t+1}
    (Leverage Capped at 2.0x)

Benchmark: Buy & Hold (Full Equity)
Assets: Tested on SPY (Equity) and TLT (Bond) separately.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'TLT', 'GLD', 'QQQ', 'IEF'] # Representative assets
TARGET_VOL = 0.10 # 10% Annualized Vol
MAX_LEVERAGE = 2.0

def get_financial_data(tickers, start='2010-01-01', end='2025-01-01'):
    print(f"Downloading data for {tickers}...")
    try:
        raw = yf.download(tickers, start=start, end=end, progress=False, timeout=60)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close']
        else:
            close = raw
        # If single asset, yfinance might return Series or DataFrame with different structure
        # Ensure consistent DataFrame
        if len(tickers) == 1:
             close = pd.DataFrame(close)
             close.columns = tickers
        else:
             close.columns = [c.replace('^', '') for c in close.columns]
        
        return close.ffill()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def prepare_features(price_series):
    """
    Generate features for a single asset.
    Returns X (features), y (target next day vol), and returns_series
    """
    df = pd.DataFrame({'Price': price_series})
    df['Ret'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Realized Volatility (22-day rolling)
    rolling_window = 22
    # Annualized RV for features
    df['RV_Daily'] = df['Ret']**2
    df['RV'] = np.sqrt(df['RV_Daily'].rolling(rolling_window).mean() * 252)
    
    # Features
    df['LogRV'] = np.log(df['RV'] + 1e-6)
    df['LogRV_lag1'] = df['LogRV'].shift(1)
    df['LogRV_lag5'] = df['LogRV'].shift(5)
    df['LogRV_lag22'] = df['LogRV'].shift(22)
    
    # Momentum
    df['Ret_1M'] = df['Price'].pct_change(22)
    
    # Target: Next period volatility (proxy by 22-day forward RV or just next day squared return? 
    # Usually Vol Targeting uses next day forecast to size position for tomorrow. 
    # Let's predict next day realized volatility (or short term proxy)
    # Ideally we validly predict next month vol.
    # Standard Vol Targeting: Estimate Vol_t, Size Position for t+1.
    # We will use predicted RV_t+1 (based on info up to t) as estimate.
    
    # Target: Next 22-day realized vol (average)
    df['Target_Vol'] = df['RV'].shift(-22) # Trying to predict average vol of next month
    
    df = df.dropna()
    return df

def run_backtest():
    print("="*70)
    print("V11 Experiment: Volatility Targeting Backtest")
    print("="*70)
    
    close_data = get_financial_data(ASSETS + ['^VIX'])
    if close_data.empty:
        print("No data available.")
        return

    results = {}
    time_series_data = []
    
    for asset in ASSETS:
        if asset not in close_data.columns: continue
        print(f"\nProcessing {asset}...")
        
        df = prepare_features(close_data[asset])
        if len(df) < 500: continue
        
        # Train/Test Split (Time Series)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Ret_1M']
        X_train = train_df[features]
        y_train = train_df['Target_Vol'] # Predicting Volatility Level directly or Log? Let's predict Level for sizing
        X_test = test_df[features]
        y_test_real = test_df['Target_Vol']
        
        # Model Training (Random Forest)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Prediction
        pred_vol = model.predict(X_test)
        
        # --- Strategy Logic ---
        # Weight = Target / Predicted
        # Ensure predicted vol is not zero
        pred_vol = np.maximum(pred_vol, 0.001) 
        
        weights = TARGET_VOL / pred_vol
        weights = np.minimum(weights, MAX_LEVERAGE) # Cap leverage
        
        # Strategy Returns
        # We take position at t based on prediction t+1? No.
        # At close of t, we predict Vol_{t+1...}. We set weight for t+1 returns.
        # Test df 'Ret' is return at t. shift(-1) is return at t+1. 
        # Actually prepare_features aligns rows.
        # current row index i: has features from t-1... to predict t... 
        # Wait, prepare_features: LogRV_lag1 is shift(1) of LogRV. LogRV is at t. So Lag1 is t-1.
        # So row i has info up to t-1. 
        # 'Target_Vol' is shift(-22). 
        # Let's align carefully.
        # We want to trade at close of day t, using info up to t.
        # Features should be current available info. 
        # df['LogRV'] is info at t.
        
        # For simulation:
        # X_test row i corresponds to time t. We have info up to t.
        # We predict Volatility for t+1 (or t+1..t+22).
        # We set weight for Return_{t+1}.
        
        # Re-alignment for backtest
        # Predictions are made at index i.
        # Weights apply to Return at index i+1.
        
        # Returns for strategy
        market_rets = test_df['Ret'].shift(-1).fillna(0) # Return of next day (t+1)
        
        # Aligned weights
        strategy_rets = weights * market_rets
        
        # Benchmark (Buy & Hold)
        benchmark_rets = market_rets
        
        # Evaluation
        strat_cum = (1 + strategy_rets).cumprod()
        bench_cum = (1 + benchmark_rets).cumprod()
        
        strat_sharpe = np.sqrt(252) * strategy_rets.mean() / (strategy_rets.std() + 1e-9)
        bench_sharpe = np.sqrt(252) * benchmark_rets.mean() / (benchmark_rets.std() + 1e-9)
        
        strat_mdd = (strat_cum / strat_cum.cummax() - 1).min()
        bench_mdd = (bench_cum / bench_cum.cummax() - 1).min()
        
        actual_vol = strategy_rets.std() * np.sqrt(252)
        
        print(f"   Shape: {strat_sharpe:.4f} vs {bench_sharpe:.4f} (Bench)")
        print(f"   MDD:   {strat_mdd:.4f} vs {bench_mdd:.4f} (Bench)")
        print(f"   Vol:   {actual_vol:.4f} (Target: {TARGET_VOL})")
        
        results[asset] = {
            'Sharpe': strat_sharpe,
            'Bench_Sharpe': bench_sharpe,
            'MDD': strat_mdd,
            'Bench_MDD': bench_mdd,
            'Realized_Vol': actual_vol,
            'Return': strat_cum.iloc[-1] - 1
        }
        
        # Prepare time series data
        ts_df = pd.DataFrame({
            'Date': strat_cum.index,
            'Asset': asset,
            'Strategy': strat_cum.values,
            'Benchmark': bench_cum.values
        })
        time_series_data.append(ts_df)

    # Save summary
    out_path = 'src/experiments/universal/v11_vol_targeting.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Save time series
    if time_series_data:
        ts_path = 'src/experiments/universal/v11_timeseries.csv'
        final_ts = pd.concat(time_series_data, ignore_index=True)
        final_ts.to_csv(ts_path, index=False)
        print(f"Time series saved to {ts_path}")

if __name__ == "__main__":
    run_backtest()
