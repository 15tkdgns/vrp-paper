"""
V13 Exp: Unified Backtest with Transaction Costs
Purpose: Comprehensive backtest of the best strategy (Volatility Targeting) including transaction costs.

Settings:
- Strategy: Volatility Targeting (10% Target Vol)
- Model: Random Forest (trained on window)
- Transaction Cost: 10 bps (0.10%) per trade (Buy/Sell)
- Slippage: 5 bps (Estimated)
- Total Cost: 15 bps (0.0015) per turnover

Metrics:
- Sharpe Ratio (Post-Cost)
- Sortino Ratio
- Maximum Drawdown (MDD)
- Turnover Rate
- Win Rate
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import json
import os


# Config
ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD', 'IEF']
TARGET_VOL = 0.10
MAX_LEVERAGE = 2.0
COST_PER_TURNOVER = 0.0015 # 15 bps

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

def run_unified_backtest():
    print("="*70)
    print("V13 Experiment: Unified Backtest (Vol Target + Transaction Costs)")
    print("="*70)
    
    data = get_data(ASSETS + ['^VIX'])
    if data.empty: return

    results = {}
    time_series_data = []
    
    for asset in ASSETS:
        if asset not in data.columns: continue
        print(f"\nBacktesting {asset} with costs...")
        
        # 1. Prepare Data
        df = pd.DataFrame({'Price': data[asset]})
        df['Ret'] = np.log(df['Price'] / df['Price'].shift(1))
        df['RV'] = df['Ret'].rolling(22).std() * np.sqrt(252)
        df['LogRV'] = np.log(df['RV'] + 1e-6)
        
        # Features
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['Target'] = df['RV'].shift(-22) # Predict next month avg vol
        
        df = df.dropna()
        if len(df) < 500: continue
        
        # 2. Split
        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        # 3. Model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(train[['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']], train['Target'])
        pred_vol = model.predict(test[['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']])
        
        # 4. Strategy with Cost
        pred_vol = np.maximum(pred_vol, 0.001)
        target_weights = TARGET_VOL / pred_vol
        target_weights = np.minimum(target_weights, MAX_LEVERAGE)
        
        # Shift weights to align with returns (Position taken at t close for t+1)
        # Weights calculated at i affect returns at i+1
        # Test Ret is at i. We align:
        # returns array: test['Ret'].values[1:]
        # weights array: target_weights[:-1]
        
        returns = test['Ret'].values[1:] # t+1 returns
        weights = target_weights[:-1]   # t weights
        
        # Turnover Calculation
        # Turnover_t = |w_t - w_{t-1} * (1+r_{t-1})| 
        # Approx: |w_t - w_{t-1}| if daily rebalancing
        
        weight_diff = np.abs(np.diff(weights, prepend=weights[0]))
        costs = weight_diff * COST_PER_TURNOVER
        
        gross_strategy_ret = weights * returns
        net_strategy_ret = gross_strategy_ret - costs
        
        # Benchmark
        bench_ret = returns
        
        # Metrics
        cum_net = (1 + net_strategy_ret).cumprod()
        cum_bench = (1 + bench_ret).cumprod()
        
        sharpe = np.sqrt(252) * net_strategy_ret.mean() / (net_strategy_ret.std() + 1e-9)
        bench_sharpe = np.sqrt(252) * bench_ret.mean() / (bench_ret.std() + 1e-9)
        
        mdd = (cum_net / np.maximum.accumulate(cum_net) - 1).min()
        
        annual_turnover = np.mean(weight_diff) * 252
        
        print(f"   Net Sharpe: {sharpe:.4f} (Bench: {bench_sharpe:.4f})")
        print(f"   Net MDD:    {mdd:.4f}")
        print(f"   Turnover:   {annual_turnover:.2f}x/year")
        print(f"   Cost Drag:  {costs.mean()*252:.2%}/year")
        
        results[asset] = {
            'Net_Sharpe': sharpe,
            'Net_MDD': mdd,
            'Turnover': annual_turnover,
            'Cost_Drag': costs.mean()*252,
            'Process': 'Vol Targeting + 15bps Cost'
        }
        
        # Prepare time series data
        ts_df = pd.DataFrame({
            'Date': cum_net.index,
            'Asset': asset,
            'Strategy': cum_net.values,
            'Benchmark': cum_bench.values
        })
        time_series_data.append(ts_df)
        
    out_path = 'src/experiments/universal/v13_backtest.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Save time series
    if time_series_data:
        ts_path = 'src/experiments/universal/v13_timeseries.csv'
        final_ts = pd.concat(time_series_data, ignore_index=True)
        final_ts.to_csv(ts_path, index=False)
        print(f"Time series saved to {ts_path}")

if __name__ == "__main__":
    run_unified_backtest()
