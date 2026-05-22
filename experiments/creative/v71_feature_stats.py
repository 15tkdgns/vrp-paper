
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import yfinance as yf
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('/root/vrp/src')
from experiments.creative.v71_advanced_data import (
    ASSET_GROUPS, ALL_ASSETS, IV_TICKERS, 
    compute_parkinson_vol, compute_garman_klass_vol, 
    compute_rogers_satchell_vol, compute_volume_features,
    fit_garch
)

# Define feature metadata directly to avoid CSV parsing errors
FEATURE_METADATA = [
    ['RogersSatchell_22', 'HF Proxy', 0.731, 'Drift-independent OHLC volatility estimator. Best proxy for realized volatility.', 'ln(H/C)ln(H/O) + ln(L/C)ln(L/O)', 'Rogers & Satchell (1991)'],
    ['GarmanKlass_22', 'HF Proxy', 0.290, 'OHLC volatility estimator using all 4 prices. 7.4x more efficient than Close-to-Close.', '0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2', 'Garman & Klass (1980)'],
    ['Parkinson_22', 'HF Proxy', 0.116, 'Range-based volatility estimator using High-Low.', '1/(4*ln(2)) * ln(H/L)^2', 'Parkinson (1980)'],
    ['IV_VIX', 'IV Surface', 0.024, 'CBOE VIX Index (30-day implied volatility). Market fear gauge.', 'VIX index value', 'CBOE'],
    ['IV_VIX3M', 'IV Surface', 0.022, 'CBOE VIX3M Index (3-month implied volatility).', '-', 'CBOE'],
    ['IV_VIX9D', 'IV Surface', 0.018, 'CBOE VIX9D Index (9-day implied volatility).', '-', 'CBOE'],
    ['IV_VIX_TermSlope', 'IV Surface', 0.015, 'Term structure slope between VIX and VIX3M.', 'log(VIX) - log(VIX3M)', '-'],
    ['IV_VRP', 'IV Surface', 0.021, 'Variance Risk Premium proxy.', 'VIX^2 - RV_SPY', 'Bollerslev et al. (2009)'],
    ['LogRV_lag1', 'Base (HAR)', 0.023, 'Daily realized volatility lag (yesterday).', 'log(RV_t-1)', 'Corsi (2009)'],
    ['LogRV_lag5', 'Base (HAR)', 0.019, 'Weekly realized volatility lag (average of past 5 days).', 'log(mean(RV_t-1...t-5))', 'Corsi (2009)'],
    ['LogRV_lag22', 'Base (HAR)', 0.015, 'Monthly realized volatility lag (average of past 22 days).', 'log(mean(RV_t-1...t-22))', 'Corsi (2009)'],
    ['Garch_Daily', 'Base (GARCH)', 0.025, 'Conditional volatility from GARCH(1,1) on daily returns.', 'sigma_t^2 = omega + alpha*r^2 + beta*sigma^2', 'Bollerslev (1986)'],
    ['Garch_Weekly', 'Base (GARCH)', 0.030, 'Conditional volatility from GARCH(1,1) on weekly returns.', '-', 'Bollerslev (1986)'],
    ['AltVol_Amihud', 'Alt Data', 0.008, 'Illiquidity measure (price impact per volume).', 'log(|ret| / DollarVol)', 'Amihud (2002)'],
    ['AltVol_Kyle_Lambda', 'Alt Data', 0.005, "Kyle's Lambda proxy (market impact).", 'log(|ret| / |delta Vol|)', 'Kyle (1985)'],
    ['AltVol_Order_Imbalance', 'Alt Data', 0.004, 'VPIN proxy (Buy/Sell volume imbalance).', '(BuyVol - SellVol) / TotalVol', 'Easley et al. (2012)'],
    ['SPY_LogRV', 'Cross-Asset', 0.045, 'S&P 500 Realized Volatility. Systematic risk factor.', 'log(RV_SPY)', '-'],
    ['Range_Close_Ratio', 'HF Proxy', 0.033, 'Ratio of range-based vol to close-based vol. Information efficiency metric.', 'log(Parkinson) - log(RV)', '-'],
    ['Parkinson_5', 'HF Proxy', 0.062, 'Short-term range-based volatility (5-day window).', '-', 'Parkinson (1980)'],
    ['Overnight_Vol', 'HF Proxy', 0.012, 'Overnight gap volatility.', 'std(log(Open/Close_prev))', '-'],
    ['Overnight_Ret', 'HF Proxy', 0.009, 'Overnight gap return.', 'log(Open/Close_prev)', '-'],
    ['LogRV_Std5', 'Base (HAR)', 0.010, 'Volatility of Volatility (short-term).', 'std(logRV, 5)', '-'],
    ['LogRV_Std22', 'Base (HAR)', 0.011, 'Volatility of Volatility (long-term).', 'std(logRV, 22)', '-'],
    ['RV_Mom5', 'Base (HAR)', 0.014, 'Short-term volatility momentum.', 'logRV_t - logRV_t-5', '-'],
    ['RV_Mom22', 'Base (HAR)', 0.012, 'Long-term volatility momentum.', 'logRV_t - logRV_t-22', '-'],
    ['Ret_lag1', 'Base (HAR)', 0.008, 'Previous day return (Leverage effect).', 'r_t-1', '-'],
    ['Ret_abs_lag1', 'Base (HAR)', 0.007, 'Previous day absolute return (Magnitude).', '|r_t-1|', '-'],
    ['Corr_SPY', 'Cross-Asset', 0.015, 'Correlation with S&P 500 (Beta proxy).', 'corr(r_asset, r_SPY, 22)', '-'],
    ['IV_VIX_chg', 'IV Surface', 0.005, 'Daily change in VIX.', 'VIX_t - VIX_t-1', '-'],
    ['IV_VIX_ma5', 'IV Surface', 0.011, '5-day moving average of VIX.', 'mean(VIX, 5)', '-'],
    ['IV_VIX_std5', 'IV Surface', 0.009, '5-day standard deviation of VIX (VVIX proxy).', 'std(VIX, 5)', '-'],
    ['IV_VIX_ShortSlope', 'IV Surface', 0.006, 'Short-term term structure (VIX9D vs VIX).', 'log(VIX9D) - log(VIX)', '-'],
    ['IV_VRP_ma22', 'IV Surface', 0.018, 'Long-term VRP trend.', 'mean(VRP, 22)', '-'],
    ['AltVol_Vol_Ratio', 'Alt Data', 0.003, 'Volume momentum ratio.', 'log(Vol_ma5 / Vol_ma22)', '-'],
    ['AltVol_PV_Corr', 'Alt Data', 0.002, 'Price-Volume correlation.', 'corr(ret, delta Vol, 22)', '-'],
    ['AltVol_Vol_Surprise', 'Alt Data', 0.003, 'Volume surprise measure.', '(Vol - Vol_ma22) / Vol_std22', '-'],
    ['LogRV_lag10', 'Base (HAR)', 0.016, 'Bi-weekly realized volatility lag.', 'log(mean(RV_t-1...t-10))', 'Corsi (2009)']
]

def download_ohlcv():
    # Try to load raw cache first
    raw_cache = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
    if os.path.exists(raw_cache):
        print(f"Loading raw OHLCV from {raw_cache}...", flush=True)
        return pd.read_pickle(raw_cache)
    
    print("Downloading raw data...", flush=True)
    all_tickers = ALL_ASSETS + IV_TICKERS
    raw = yf.download(all_tickers, start='2010-01-01', end='2025-01-01', progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        new_cols = []
        for price_type, ticker in raw.columns:
            ticker_clean = ticker.replace('^', '')
            new_cols.append((price_type, ticker_clean))
        raw.columns = pd.MultiIndex.from_tuples(new_cols)
    
    raw = raw.ffill()
    return raw

def generate_features():
    print("Generating features...", flush=True)
    raw = download_ohlcv()
    
    if isinstance(raw.columns, pd.MultiIndex):
        price_types = raw.columns.get_level_values(0).unique()
        available_tickers = raw.columns.get_level_values(1).unique()
    else:
        price_types = ['Close']
        available_tickers = raw.columns

    has_ohlc = all(pt in price_types for pt in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in price_types
    has_vix = 'VIX' in available_tickers
    has_vix3m = 'VIX3M' in available_tickers
    has_vix9d = 'VIX9D' in available_tickers
    
    # IV Surface Features
    iv_features = {}
    if has_vix:
        vix = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        iv_features['VIX'] = np.log(vix + 1e-6)
        iv_features['VIX_chg'] = iv_features['VIX'].diff()
        iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
        iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
    
    if has_vix3m:
        vix3m = raw[('Close', 'VIX3M')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX3M']
        iv_features['VIX3M'] = np.log(vix3m + 1e-6)
        if has_vix:
            iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
            
    if has_vix9d:
        vix9d = raw[('Close', 'VIX9D')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX9D']
        iv_features['VIX9D'] = np.log(vix9d + 1e-6)
        if has_vix:
            iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
            
    # SPY reference
    if isinstance(raw.columns, pd.MultiIndex):
        spy_close = raw[('Close', 'SPY')]
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    else:
        spy_close = raw['SPY']
        spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)
    
    if has_vix:
        vix_raw = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        vrp = (vix_raw**2 / 100) - spy_rv / 10000
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    pooled_data = []
    available_assets = [a for a in ALL_ASSETS if a in available_tickers]
    
    for i, asset in enumerate(available_assets):
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[('Close', asset)]
            open_p = raw[('Open', asset)] if has_ohlc else None
            high = raw[('High', asset)] if has_ohlc else None
            low = raw[('Low', asset)] if has_ohlc else None
            volume = raw[('Volume', asset)] if has_volume else None
        else:
            close = raw[asset]
            open_p = high = low = volume = None
        
        ret_daily = np.log(close / close.shift(1)).dropna()
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        
        # Weekly GARCH
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')
        
        feat_dict = {
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv.rolling(5).std().shift(1),
            'LogRV_Std22': log_rv.rolling(22).std().shift(1),
            'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
            'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
        }
        
        if has_ohlc and open_p is not None:
            park_5 = compute_parkinson_vol(high, low, window=5)
            park_22 = compute_parkinson_vol(high, low, window=22)
            feat_dict['Parkinson_5'] = np.log(park_5 + 1e-6).shift(1)
            feat_dict['Parkinson_22'] = np.log(park_22 + 1e-6).shift(1)
            
            gk_22 = compute_garman_klass_vol(open_p, high, low, close, window=22)
            feat_dict['GarmanKlass_22'] = np.log(gk_22 + 1e-6).shift(1)
            
            rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, window=22)
            feat_dict['RogersSatchell_22'] = np.log(rs_22 + 1e-6).shift(1)
            
            feat_dict['Range_Close_Ratio'] = (np.log(park_22 + 1e-6) - log_rv).shift(1)
            
            overnight_ret = np.log(open_p / close.shift(1))
            feat_dict['Overnight_Vol'] = overnight_ret.rolling(22).std().shift(1)
            feat_dict['Overnight_Ret'] = overnight_ret.shift(1)
        
        for iv_name, iv_val in iv_features.items():
            feat_dict[f'IV_{iv_name}'] = iv_val.shift(1)
            
        if has_volume and volume is not None:
            vol_feats = compute_volume_features(volume, close, ret_daily, window=22)
            for vf_name, vf_val in vol_feats.items():
                feat_dict[f'AltVol_{vf_name}'] = vf_val.shift(1)
                
        if asset != 'SPY':
            feat_dict['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
        else:
            feat_dict['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)
            
        feat_dict['Asset'] = asset
        d = pd.DataFrame(feat_dict).dropna()
        pooled_data.append(d)
        
    return pd.concat(pooled_data).sort_index().reset_index(drop=True)

def calculate_stats():
    df = generate_features()
    
    exclude_cols = ['Asset', 'Class', 'Target']
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Calculating stats for {len(features)} features...", flush=True)
    
    stats_list = []
    for feat in features:
        data = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) == 0: continue
        
        stats = {
            'Feature': feat,
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data),
            'Obs_Count': len(data)
        }
        stats_list.append(stats)
        
    stats_df = pd.DataFrame(stats_list)
    
    desc_df = pd.DataFrame(FEATURE_METADATA, columns=['Feature', 'Group', 'Importance', 'Description', 'Formula', 'Reference'])
    final_df = pd.merge(desc_df, stats_df, on='Feature', how='left')
        
    output_path = '/root/vrp/paper/csv/v71_features_stats.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved extended stats to {output_path}", flush=True)

if __name__ == "__main__":
    calculate_stats()
