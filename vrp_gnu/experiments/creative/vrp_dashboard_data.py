"""
Cross-Asset VRP Dashboard Data Generator
Generates VRP data for 19+ assets and saves to CSV.
Run this script first, then render vrp_dashboard.qmd.
"""
import sys, os
sys.path.append('/root/vrp/src')

import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Asset Universe (19 assets)
# ============================================
ASSET_CONFIG = {
    # Equity
    'SPY': {'name': 'S&P 500', 'group': 'Equity', 'iv_index': '^VIX'},
    'QQQ': {'name': 'NASDAQ 100', 'group': 'Equity', 'iv_index': '^VXN'},
    'IWM': {'name': 'Russell 2000', 'group': 'Equity', 'iv_index': '^RVX'},
    'EFA': {'name': 'EAFE Developed', 'group': 'Equity', 'iv_index': None},
    'EEM': {'name': 'Emerging Mkt', 'group': 'Equity', 'iv_index': None},
    # Bond
    'TLT': {'name': '20Y+ Treasury', 'group': 'Bond', 'iv_index': None},
    'IEF': {'name': '7-10Y Treasury', 'group': 'Bond', 'iv_index': None},
    'AGG': {'name': 'US Agg Bond', 'group': 'Bond', 'iv_index': None},
    'HYG': {'name': 'High Yield', 'group': 'Bond', 'iv_index': None},
    # Grain
    'WEAT': {'name': 'Wheat', 'group': 'Grain', 'iv_index': None},
    'CORN': {'name': 'Corn', 'group': 'Grain', 'iv_index': None},
    'SOYB': {'name': 'Soybean', 'group': 'Grain', 'iv_index': None},
    # Livestock
    'COW': {'name': 'Livestock', 'group': 'Livestock', 'iv_index': None},
    # Energy
    'USO': {'name': 'Crude Oil', 'group': 'Energy', 'iv_index': '^OVX'},
    'UNG': {'name': 'Natural Gas', 'group': 'Energy', 'iv_index': None},
    # Metal
    'GLD': {'name': 'Gold', 'group': 'Metal', 'iv_index': '^GVZ'},
    'SLV': {'name': 'Silver', 'group': 'Metal', 'iv_index': None},
    # Other
    'VNQ': {'name': 'US Real Estate', 'group': 'Other', 'iv_index': None},
    'UUP': {'name': 'US Dollar', 'group': 'Other', 'iv_index': None},
}

def fit_garch_iv(returns):
    """GARCH(1,1) conditional variance as IV proxy"""
    try:
        ret = returns * 100
        am = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility / 100
        ann_var = (cond_vol ** 2) * 252
        return ann_var
    except:
        return None

def main():
    # Collect all tickers
    etf_tickers = list(ASSET_CONFIG.keys())
    iv_tickers = [v['iv_index'] for v in ASSET_CONFIG.values() if v['iv_index']]
    all_tickers = list(set(etf_tickers + iv_tickers))

    print(f"Downloading {len(all_tickers)} tickers...", flush=True)
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)

    # Clean MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        new_cols = []
        for price_type, ticker in raw.columns:
            new_cols.append((price_type, ticker.replace('^', '')))
        raw.columns = pd.MultiIndex.from_tuples(new_cols)
    raw = raw.ffill()

    def get_close(ticker):
        t = ticker.replace('^', '')
        if isinstance(raw.columns, pd.MultiIndex):
            if t in raw['Close'].columns:
                return raw[('Close', t)].dropna()
        return None

    # Calculate VRP for each asset
    vrp_ts_records = []  # time series
    summary_records = []  # summary table

    for ticker, info in ASSET_CONFIG.items():
        close = get_close(ticker)
        if close is None or len(close) < 100:
            print(f"  {ticker}: Insufficient data, skipping", flush=True)
            continue

        # Realized Variance (22-day, annualized)
        ret = np.log(close / close.shift(1))
        rv = (ret ** 2).rolling(22).mean() * 252

        # Implied Variance
        iv_type = 'GARCH'
        iv_idx = info.get('iv_index')
        if iv_idx:
            iv_close = get_close(iv_idx)
            if iv_close is not None and len(iv_close) > 100:
                iv_var = (iv_close / 100) ** 2
                iv_type = 'IV Index'
                vrp = (iv_var - rv).dropna()
                vrp_smooth = vrp.rolling(22).mean()
                
                for date, val in vrp_smooth.dropna().items():
                    vrp_ts_records.append({
                        'Date': date, 'Ticker': ticker,
                        'Group': info['group'], 'Name': info['name'],
                        'VRP': val * 100, 'IV_Type': iv_type
                    })
                
                recent = vrp.dropna().tail(22)
                summary_records.append({
                    'Group': info['group'], 'Ticker': ticker,
                    'Name': info['name'], 'IV_Source': iv_type,
                    'Current_RV_pct': rv.dropna().iloc[-1] * 100,
                    'Recent_VRP_pct': recent.mean() * 100,
                    'Avg_VRP_pct': vrp.dropna().mean() * 100,
                })
                print(f"  {ticker} ({info['name']}): IV Index", flush=True)
                continue

        # Fallback: GARCH
        ret_clean = ret.dropna()
        garch_iv = fit_garch_iv(ret_clean)
        if garch_iv is not None:
            vrp = (garch_iv - rv).dropna()
            vrp_smooth = vrp.rolling(22).mean()
            
            for date, val in vrp_smooth.dropna().items():
                vrp_ts_records.append({
                    'Date': date, 'Ticker': ticker,
                    'Group': info['group'], 'Name': info['name'],
                    'VRP': val * 100, 'IV_Type': iv_type
                })
            
            recent = vrp.dropna().tail(22)
            summary_records.append({
                'Group': info['group'], 'Ticker': ticker,
                'Name': info['name'], 'IV_Source': iv_type,
                'Current_RV_pct': rv.dropna().iloc[-1] * 100,
                'Recent_VRP_pct': recent.mean() * 100,
                'Avg_VRP_pct': vrp.dropna().mean() * 100,
            })
            print(f"  {ticker} ({info['name']}): GARCH", flush=True)
        else:
            print(f"  {ticker} ({info['name']}): Failed", flush=True)

    # Save
    os.makedirs('/root/vrp/paper/csv', exist_ok=True)

    ts_df = pd.DataFrame(vrp_ts_records)
    ts_df.to_csv('/root/vrp/paper/csv/vrp_timeseries.csv', index=False)
    print(f"\nSaved time series: {len(ts_df)} rows", flush=True)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv('/root/vrp/paper/csv/vrp_summary.csv', index=False)
    print(f"Saved summary: {len(summary_df)} rows", flush=True)

if __name__ == "__main__":
    main()
