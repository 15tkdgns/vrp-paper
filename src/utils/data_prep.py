import yfinance as yf
import pandas as pd
import os

ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'AGG', 'GLD', 'SLV', 'USO']
CACHE_FILE = 'src/data/ohlcv_cache.csv'

def download_data():
    print(f"Starting data download for {ASSETS}...")
    try:
        data = yf.download(ASSETS, start='2008-01-01', end='2025-01-01', progress=True)
        # Ensure directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        
        # Save as multi-index CSV or flattened? Flattened is often easier for simple scripts.
        # But yfinance returns MultiIndex if multiple assets.
        # Let's flatten to 'Date, Asset, Open, High, Low, Close, Volume' style for robustness.
        
        # data['Close'] is a DataFrame with assets as columns
        close_prices = data['Close']
        close_prices.to_csv(CACHE_FILE)
        print(f"Successfully cached data to {CACHE_FILE}")
        print(f"Shape: {close_prices.shape}")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
