"""
V4 Exp 01: Log-RV Distribution Analysis
Purpose: Validate that ln(RV) is more suitable for MSE-based learning than raw RV

Analysis:
1. Compare Skewness/Kurtosis of RV vs ln(RV) for all 23 assets
2. Visual check: Histograms
3. Correlation analysis in Log-space
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from scipy.stats import skew, kurtosis
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'EFA', 'EEM', 'IOO',
    'TLT', 'IEF', 'SHY', 'TIP', 'ZROZ',
    'GLD', 'USO', 'SLV', 'DBC',
]

def run_log_rv_analysis():
    print("="*70)
    print("V4 Exp 01: Log-RV Distribution Analysis")
    print("="*70)
    
    # 1. Download Data
    tickers = ASSETS
    print(f"Downloading {len(tickers)} tickers...")
    
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
        
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    stats_data = []
    
    # Prepare figure for histograms (Sample 4 assets)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    sample_assets = ['SPY', 'TLT', 'GLD', 'USO']
    
    # 2. Compute RV and Stats
    print("\nComputing Statistics...")
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        # Compute RV (22-day)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        rv = rv.dropna()
        
        # Log Transformation
        log_rv = np.log(rv + 1e-6) # Add epsilon to avoid log(0)
        
        # Stats
        s_rv, k_rv = skew(rv), kurtosis(rv)
        s_log, k_log = skew(log_rv), kurtosis(log_rv)
        
        stats_data.append({
            'Asset': asset,
            'Skew_RV': s_rv, 'Kurt_RV': k_rv,
            'Skew_LogRV': s_log, 'Kurt_LogRV': k_log,
            'Skew_Improvement': abs(s_rv) - abs(s_log) 
        })
        
        # Plot for sample assets
        if asset in sample_assets:
            idx = sample_assets.index(asset)
            
            # Raw RV
            sns.histplot(rv, kde=True, ax=axes[0, idx], color='blue')
            axes[0, idx].set_title(f'{asset} Raw RV (Skew={s_rv:.2f})')
            
            # Log RV
            sns.histplot(log_rv, kde=True, ax=axes[1, idx], color='green')
            axes[1, idx].set_title(f'{asset} Log RV (Skew={s_log:.2f})')

    # 3. Create DataFrame
    df_stats = pd.DataFrame(stats_data).set_index('Asset')
    
    print("\n" + "="*70)
    print("DISTRIBUTION STATISTICS (RV vs Log-RV)")
    print("="*70)
    print(df_stats[['Skew_RV', 'Skew_LogRV', 'Kurt_RV', 'Kurt_LogRV']].round(2))
    
    print("\n[Average Statistics]")
    print(df_stats.mean().round(4))
    
    # 4. Save Results
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v4_exp01_dist_plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nSaved distribution plot to {plot_path}")
    
    res_path = 'experiments/07_v2_methodology/results/v4_exp01_stats.json'
    df_stats.to_json(res_path, orient='index', indent=2)
    print(f"Saved stats to {res_path}")
    
    # 5. Conclusion
    avg_skew_improve = df_stats['Skew_Improvement'].mean()
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if avg_skew_improve > 0.5:
        print(f"✅ Log-transformation significantly improves normality (Avg Skew reduction: {avg_skew_improve:.2f})")
        print("   → Highly recommended for Deep Learning models (MSE loss)")
    else:
        print(f"⚠️ Improvement is marginal ({avg_skew_improve:.2f})")

if __name__ == "__main__":
    run_log_rv_analysis()
