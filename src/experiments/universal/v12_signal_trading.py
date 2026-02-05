"""
V12 Exp: Signal-based Trading Strategy
Purpose: Implement a trading strategy using VRP signal and Volatility Regimes.

Strategy Logic:
1. Signal Generation:
   - Calculate VRP = VIX - Realized_Vol(22d)
   - Predict Next Month Volatility (Pred_Vol) using Random Forest
   
2. Trading Rules (Regime Switching):
   - Stable Regime (Pred_Vol < Percentile_80 and VRP > 0):
     -> Long Equity (100% or Leveraged)
   - Unstable Regime (Pred_Vol > Percentile_80):
     -> Cash / Defensive (0% Equity) or Hedge
   - Fear Regime (VRP < 0):
     -> Reduce Exposure (50% Equity) or Wait

Assets: SPY, QQQ (Equity Focus)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import json
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'QQQ']

def get_data(tickers):
    print(f"Downloading data for {tickers}...")
    try:
        raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False, timeout=60)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close']
        else:
            close = raw
        if len(tickers) == 1:
             close = pd.DataFrame(close)
             close.columns = tickers
        else:
             close.columns = [c.replace('^', '') for c in close.columns]
        return close.ffill()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def run_signal_strategy():
    print("="*70)
    print("V12 Experiment: Signal-based Trading Strategy")
    print("="*70)
    
    # Needs VIX for VRP
    tickers = ASSETS + ['^VIX']
    data = get_data(tickers)
    
    if data.empty:
        print("No data.")
        return
        
    results = {}
    time_series_data = []
    
    for asset in ASSETS:
        if asset not in data.columns: continue
        print(f"\nAnalyzing {asset}...")
        
        df = pd.DataFrame({'Price': data[asset]})
        vix = data['VIX']
        
        # Features
        df['Ret'] = np.log(df['Price'] / df['Price'].shift(1))
        df['RV'] = df['Ret'].rolling(22).std() * np.sqrt(252) * 100 # Annualized %
        
        # VRP = VIX - RV (Both in %)
        # Align VIX (implied for next 30 days) with RV (recent realized)
        # Or better: VRP_t = VIX_t - RV_{t-22:t}
        df['VIX'] = vix
        df['VRP'] = df['VIX'] - df['RV']
        
        # ML Feature: Lagged RV
        df['LogRV'] = np.log(df['RV'] + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        
        # Target: Next Month Vol
        df['Target_Vol'] = df['RV'].shift(-22)
        
        df = df.dropna()
        if len(df) < 500: continue
        
        # Train ML Model (Rolling or Split? Using Split for simplicity)
        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(train[['LogRV_lag1', 'VRP']], train['Target_Vol'])
        
        pred_vol = model.predict(test[['LogRV_lag1', 'VRP']])
        
        # Thresholds defined on Train 
        vol_threshold = train['Target_Vol'].quantile(0.80)
        
        # Strategy Logic on Test
        # 1. High Vol Regime -> Cash
        # 2. VRP < 0 -> Reduce Risk
        # 3. Else -> Long
        
        signals = []
        for i in range(len(test)):
            p_vol = pred_vol[i]
            vrp_val = test['VRP'].iloc[i] # Current VRP
            
            if p_vol > vol_threshold:
                signals.append(0.0) # Cash in high vol regime
            elif vrp_val < 0:
                signals.append(0.5) # Reduce size if VRP negative (Market pricing < Realized?) 
                # Actually VIX < RV usually means Complacency or backwardation, sometimes bullish but correction prone.
                # Let's stick to simple logic: VRP < 0 is "Premium is negative", risk of spike.
            else:
                signals.append(1.0) # Normal Long
                
        signals = np.array(signals)
        
        # Returns
        market_ret = test['Ret'].shift(-1).fillna(0) # Next day return
        strat_ret = signals * market_ret
        
        # Metrics
        cum_strat = (1 + strat_ret).cumprod()
        cum_market = (1 + market_ret).cumprod()
        
        sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-9)
        bench_sharpe = np.sqrt(252) * market_ret.mean() / (market_ret.std() + 1e-9)
        
        mdd = (cum_strat / cum_strat.cummax() - 1).min()
        bench_mdd = (cum_market / cum_market.cummax() - 1).min()
        
        print(f"   MDD:    {mdd:.4f} vs {bench_mdd:.4f}")
        print(f"   Return: {cum_strat.iloc[-1]-1:.2%}")
        
        results[asset] = {
            'Sharpe': sharpe,
            'Bench_Sharpe': bench_sharpe,
            'MDD': mdd,
            'Return': cum_strat.iloc[-1] - 1
        }
        
        # Prepare time series data
        ts_df = pd.DataFrame({
            'Date': cum_strat.index,
            'Asset': asset,
            'Strategy': cum_strat.values,
            'Benchmark': cum_market.values
        })
        time_series_data.append(ts_df)
        
    out_path = 'src/experiments/universal/v12_signal_trading.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save time series
    if time_series_data:
        ts_path = 'src/experiments/universal/v12_timeseries.csv'
        final_ts = pd.concat(time_series_data, ignore_index=True)
        final_ts.to_csv(ts_path, index=False)
        print(f"Time series saved to {ts_path}")

if __name__ == "__main__":
    run_signal_strategy()
