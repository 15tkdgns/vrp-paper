"""
V4 Exp 07: Scaling Law Analysis
Purpose: Test if volatility follows universal scaling laws across time horizons

Analysis:
1. Compute RV at multiple horizons (1, 5, 22, 66 days)
2. Test: Var(logRV_delta) ~ delta^alpha (scaling)
3. Estimate Hurst exponent H = (alpha+1)/2
"""
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE',
    'EFA', 'EEM',
    'TLT', 'IEF', 'TIP',
    'GLD', 'USO', 'SLV',
]

ASSET_CLASSES = {
    'US_Equity': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE'],
    'Global_Equity': ['EFA', 'EEM'],
    'Bonds': ['TLT', 'IEF', 'TIP'],
    'Commodities': ['GLD', 'USO', 'SLV'],
}

def run_scaling_analysis():
    print("="*70)
    print("V4 Exp 07: Scaling Law Analysis (Hurst Exponent)")
    print("="*70)
    
    # 1. Download Data
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close = close.ffill()
    
    # Time horizons for scaling analysis
    horizons = [1, 5, 10, 22, 44, 66]
    
    results = []
    
    for asset in ASSETS:
        if asset not in close.columns:
            continue
            
        # Compute returns
        ret = np.log(close[asset] / close[asset].shift(1))
        
        # Compute RV at different horizons
        horizon_vars = []
        for h in horizons:
            rv_h = ret.rolling(h).std() * np.sqrt(252) * 100
            log_rv = np.log(rv_h + 1e-6)
            var_log_rv = log_rv.var()
            horizon_vars.append(var_log_rv)
        
        # Log-log regression: log(Var) ~ alpha * log(delta)
        log_horizons = np.log(horizons)
        log_vars = np.log(horizon_vars)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_horizons, log_vars)
        
        # Hurst exponent
        alpha = slope
        hurst = (alpha + 1) / 2
        
        # Get asset class
        asset_class = 'Unknown'
        for cls, assets in ASSET_CLASSES.items():
            if asset in assets:
                asset_class = cls
                break
        
        results.append({
            'Asset': asset,
            'Class': asset_class,
            'Alpha': alpha,
            'Hurst': hurst,
            'R2': r_value**2,
        })
        
    df_results = pd.DataFrame(results)
    
    # 2. Results
    print("\n" + "="*70)
    print("SCALING LAW RESULTS")
    print("="*70)
    print(df_results.round(4).to_string())
    
    # 3. Summary by Class
    print("\n" + "="*70)
    print("HURST EXPONENT BY ASSET CLASS")
    print("="*70)
    
    for cls in ASSET_CLASSES.keys():
        sub = df_results[df_results['Class'] == cls]
        if len(sub) == 0:
            continue
        avg_hurst = sub['Hurst'].mean()
        std_hurst = sub['Hurst'].std()
        print(f"  {cls:15}: H = {avg_hurst:.4f} ± {std_hurst:.4f}")
    
    # Overall
    overall_hurst = df_results['Hurst'].mean()
    overall_std = df_results['Hurst'].std()
    print(f"\n  Overall:        H = {overall_hurst:.4f} ± {overall_std:.4f}")
    
    # 4. Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if overall_std < 0.1:
        print("✅ STRONG UNIVERSALITY: Hurst exponents are tightly clustered")
        print(f"   All assets share similar long-memory structure (H ≈ {overall_hurst:.2f})")
    elif overall_std < 0.2:
        print("⚠️ MODERATE UNIVERSALITY: Hurst exponents show some variation")
    else:
        print("❌ NO UNIVERSALITY: Scaling behavior differs significantly across assets")
    
    if overall_hurst > 0.5:
        print(f"\n   H > 0.5 indicates PERSISTENCE (volatility clustering)")
    elif overall_hurst < 0.5:
        print(f"\n   H < 0.5 indicates MEAN-REVERSION")
    
    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hurst by asset
    colors = [CLASS_COLORS.get(r['Class'], 'gray') for _, r in df_results.iterrows()]
    axes[0].bar(df_results['Asset'], df_results['Hurst'], color=colors)
    axes[0].axhline(0.5, color='red', linestyle='--', label='H=0.5 (Random Walk)')
    axes[0].set_title('Hurst Exponent by Asset')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    
    # Distribution
    axes[1].hist(df_results['Hurst'], bins=10, edgecolor='black')
    axes[1].axvline(overall_hurst, color='red', linestyle='--', label=f'Mean H={overall_hurst:.3f}')
    axes[1].set_title('Distribution of Hurst Exponents')
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v4_exp07_scaling.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp07_scaling.json'
    df_results.to_json(out_path, orient='records', indent=2)
    print(f"Saved to {out_path}")

CLASS_COLORS = {
    'US_Equity': 'blue',
    'Global_Equity': 'green',
    'Bonds': 'red',
    'Commodities': 'orange'
}

if __name__ == "__main__":
    run_scaling_analysis()
