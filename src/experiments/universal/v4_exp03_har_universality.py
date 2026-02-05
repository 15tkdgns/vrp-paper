"""
V4 Exp 03: HAR Coefficient Universality
Purpose: Test if HAR coefficients (a1, a2, a3) are universal across assets

Hypothesis: The ratio a1:a2:a3 and signs are consistent across all assets,
demonstrating a "universal dynamic law" of volatility.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV',
    'EFA', 'EEM',
    'TLT', 'IEF', 'TIP',
    'GLD', 'USO', 'SLV', 'DBC',
]

ASSET_CLASSES = {
    'US_Equity': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV'],
    'Global_Equity': ['EFA', 'EEM'],
    'Bonds': ['TLT', 'IEF', 'TIP'],
    'Commodities': ['GLD', 'USO', 'SLV', 'DBC'],
}

def run_har_universality():
    print("="*70)
    print("V4 Exp 03: HAR Coefficient Universality Test")
    print("="*70)
    
    # 1. Download Data
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close = close.ffill()
    
    # 2. Estimate HAR coefficients for each asset
    results = []
    
    for asset in ASSETS:
        if asset not in close.columns:
            continue
            
        # Compute Log-RV
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        log_rv = np.log(rv + 1e-6)
        
        # HAR features
        df = pd.DataFrame(index=close.index)
        df['HAR_d'] = log_rv.shift(1)  # Daily (yesterday)
        df['HAR_w'] = log_rv.rolling(5).mean().shift(1)  # Weekly
        df['HAR_m'] = log_rv.rolling(22).mean().shift(1)  # Monthly
        df['Target'] = log_rv.shift(-22)  # 22-day ahead
        
        df = df.dropna()
        if len(df) < 500:
            continue
        
        # OLS Regression
        X = df[['HAR_d', 'HAR_w', 'HAR_m']].values
        y = df['Target'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        a0 = model.intercept_
        a1, a2, a3 = model.coef_
        r2 = model.score(X, y)
        
        # Get asset class
        asset_class = 'Unknown'
        for cls, assets in ASSET_CLASSES.items():
            if asset in assets:
                asset_class = cls
                break
        
        results.append({
            'Asset': asset,
            'Class': asset_class,
            'a0_intercept': a0,
            'a1_daily': a1,
            'a2_weekly': a2,
            'a3_monthly': a3,
            'R2': r2,
            'a1_ratio': a1 / (a1 + a2 + a3),  # Relative weight
            'a2_ratio': a2 / (a1 + a2 + a3),
            'a3_ratio': a3 / (a1 + a2 + a3),
        })
    
    df_results = pd.DataFrame(results)
    
    # 3. Analysis
    print("\n" + "="*70)
    print("HAR COEFFICIENTS BY ASSET")
    print("="*70)
    print(df_results[['Asset', 'Class', 'a1_daily', 'a2_weekly', 'a3_monthly', 'R2']].round(4).to_string())
    
    # 4. Coefficient Ratio Analysis
    print("\n" + "="*70)
    print("COEFFICIENT RATIO ANALYSIS (a1 : a2 : a3)")
    print("="*70)
    
    for cls in ASSET_CLASSES.keys():
        sub = df_results[df_results['Class'] == cls]
        if len(sub) == 0:
            continue
        
        avg_a1 = sub['a1_daily'].mean()
        avg_a2 = sub['a2_weekly'].mean()
        avg_a3 = sub['a3_monthly'].mean()
        std_a1 = sub['a1_daily'].std()
        std_a2 = sub['a2_weekly'].std()
        std_a3 = sub['a3_monthly'].std()
        
        print(f"\n{cls}:")
        print(f"  a1 (daily):   {avg_a1:.4f} ± {std_a1:.4f}")
        print(f"  a2 (weekly):  {avg_a2:.4f} ± {std_a2:.4f}")
        print(f"  a3 (monthly): {avg_a3:.4f} ± {std_a3:.4f}")
        
        # Ratio
        total = avg_a1 + avg_a2 + avg_a3
        print(f"  Ratio: {avg_a1/total:.2f} : {avg_a2/total:.2f} : {avg_a3/total:.2f}")
    
    # 5. Cross-Class Comparison
    print("\n" + "="*70)
    print("UNIVERSALITY TEST")
    print("="*70)
    
    # Check if signs are consistent
    sign_a1 = (df_results['a1_daily'] > 0).mean()
    sign_a2 = (df_results['a2_weekly'] > 0).mean()
    sign_a3 = (df_results['a3_monthly'] > 0).mean()
    
    print(f"Sign Consistency (proportion positive):")
    print(f"  a1 (daily):   {sign_a1*100:.1f}%")
    print(f"  a2 (weekly):  {sign_a2*100:.1f}%")
    print(f"  a3 (monthly): {sign_a3*100:.1f}%")
    
    # CV (Coefficient of Variation) - lower = more universal
    cv_a1 = df_results['a1_daily'].std() / df_results['a1_daily'].mean()
    cv_a2 = df_results['a2_weekly'].std() / df_results['a2_weekly'].mean()
    cv_a3 = df_results['a3_monthly'].std() / df_results['a3_monthly'].mean()
    
    print(f"\nCoefficient of Variation (lower = more universal):")
    print(f"  CV(a1): {cv_a1:.2f}")
    print(f"  CV(a2): {cv_a2:.2f}")
    print(f"  CV(a3): {cv_a3:.2f}")
    
    # 6. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, col in enumerate(['a1_daily', 'a2_weekly', 'a3_monthly']):
        sns.boxplot(data=df_results, x='Class', y=col, ax=axes[i])
        axes[i].set_title(f'{col} by Asset Class')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v4_exp03_har_coefs.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    
    # 7. Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if sign_a1 > 0.9 and sign_a2 > 0.7 and sign_a3 > 0.7:
        print("✅ STRONG UNIVERSALITY: Sign structure is consistent across 90%+ of assets")
    elif sign_a1 > 0.7:
        print("⚠️ PARTIAL UNIVERSALITY: Daily coefficient is universal, but weekly/monthly vary")
    else:
        print("❌ NO UNIVERSALITY: Coefficient signs differ significantly across assets")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp03_har_universality.json'
    df_results.to_json(out_path, orient='records', indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_har_universality()
