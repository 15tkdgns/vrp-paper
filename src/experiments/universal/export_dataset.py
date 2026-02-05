
import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings

warnings.filterwarnings('ignore')

# Extended asset universe (Same as v5_extended_experiments.py)
ASSETS = [
    # US Equity Sectors
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLB', 'XLP', 'XLU', 'XLY', 'XLRE',
    # Global
    'EFA', 'EEM', 'VGK', 'EWJ', 'FXI',
    # Bonds
    'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG', 'AGG',
    # Commodities
    'GLD', 'SLV', 'USO', 'DBC', 'UNG',
    # Volatility
    'VXX',
]

def self_compute_rsi(prices, period=14):
    """Compute RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def export_dataset():
    print("="*70)
    print("Exporting V5 Universal Model Dataset to CSV")
    print("="*70)
    
    # 1. Download Extended Data
    tickers = ASSETS + ['^VIX', '^VIX3M', '^SKEW', '^TNX', '^DJI', 'DX-Y.NYB']
    print(f"Downloading {len(tickers)} tickers...")
    
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '').replace('-Y.NYB', '') for c in close.columns]
    close = close.ffill()
    
    valid_assets = [a for a in ASSETS if a in close.columns]
    print(f"\nValid assets: {len(valid_assets)}")
    
    pooled_data = []
    
    # Feature list matching the model
    # basic_features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom']
    # global_features = ['LogVIX', 'VIX_Term', 'SKEW', 'TNX', 'DXY']
    # technical_features = ['RSI', 'MACD', 'BB_Width']
    # cross_asset_features = ['SPY_RV', 'TLT_RV', 'GLD_RV']
    
    # Compute cross-asset RVs
    cross_rvs = {}
    for ca in ['SPY', 'TLT', 'GLD']:
        if ca in close.columns:
            ret = np.log(close[ca] / close[ca].shift(1))
            cross_rvs[ca] = np.log(ret.rolling(22).std() * np.sqrt(252) * 100 + 1e-6)
    
    for asset in valid_assets:
        df = pd.DataFrame(index=close.index)
        
        # Target Calculation (Log-RV)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        # --- Feature Engineering ---
        
        # 1. Basic features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        
        # 2. Global features
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1) if 'VIX' in close.columns else 0
        df['VIX_Term'] = (close.get('VIX3M', close['VIX']) - close['VIX']).shift(1) if 'VIX' in close.columns else 0
        df['SKEW'] = (close['SKEW'] / 100 - 1).shift(1) if 'SKEW' in close.columns else 0
        df['TNX'] = close['TNX'].shift(1) / 10 if 'TNX' in close.columns else 0
        df['DXY'] = (close.get('DX', 100) / 100 - 1).shift(1)
        
        # 3. Technical indicators
        price = close[asset]
        df['RSI'] = self_compute_rsi(price, 14).shift(1) / 100
        df['MACD'] = (price.ewm(span=12).mean() - price.ewm(span=26).mean()).shift(1) / price.shift(1)
        bb_mid = price.rolling(20).mean()
        bb_std = price.rolling(20).std()
        df['BB_Width'] = ((bb_std * 2) / bb_mid).shift(1)
        
        # 4. Cross-asset features
        df['SPY_RV'] = cross_rvs.get('SPY', pd.Series(0, index=df.index)).shift(1)
        df['TLT_RV'] = cross_rvs.get('TLT', pd.Series(0, index=df.index)).shift(1)
        df['GLD_RV'] = cross_rvs.get('GLD', pd.Series(0, index=df.index)).shift(1)
        
        # Target (Log RV t+22)
        df['Target_LogRV_22d'] = np.log(rv.shift(-22) + 1e-6)
        
        # Metadata
        df['Asset'] = asset
        df['Date'] = df.index
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Reorder columns
        cols = ['Date', 'Asset', 'Target_LogRV_22d', 
                'LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom',
                'LogVIX', 'VIX_Term', 'SKEW', 'TNX', 'DXY',
                'RSI', 'MACD', 'BB_Width',
                'SPY_RV', 'TLT_RV', 'GLD_RV']
        
        if len(df) > 100:
            pooled_data.append(df[cols])
            
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    print(f"Total samples: {len(full_df)}")
    
    # Save to CSV
    output_dir = 'experiments/07_v2_methodology/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'universal_model_dataset.csv')
    
    full_df.to_csv(output_path, index=False)
    print(f"Dataset exported to: {output_path}")

if __name__ == "__main__":
    export_dataset()
