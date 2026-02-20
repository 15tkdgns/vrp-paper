"""
Step 4: Regime & Subsample Analysis
=====================================
리뷰 포인트 4: 기간별 서브샘플 + VIX quantile 레짐별 성능
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.experiments.creative.v71_sci.data_builder import (
    build_dataset, ASSET_GROUPS
)

def run():
    print("="*70, flush=True)
    print("STEP 4: Regime & Subsample Analysis", flush=True)
    print("="*70, flush=True)
    
    ds = build_dataset()
    data = ds['data']
    feats = ds['feats']
    har3 = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    results = {}
    
    # ===== 4a. Period Subsample =====
    print("\n--- 4a. Period Subsample Analysis ---", flush=True)
    
    periods = [
        ('2010-2014', '2010', '2015'),
        ('2015-2019', '2015', '2020'),
        ('2020-2024', '2020', '2025'),
    ]
    
    period_results = {}
    for label, start, end in periods:
        mask = (data['Date'] >= start) & (data['Date'] < end)
        subset = data[mask].reset_index(drop=True)
        if len(subset) < 500:
            print(f"  {label}: insufficient data ({len(subset)}), skipping", flush=True)
            continue
        
        split = int(len(subset) * 0.7)
        tr, te = subset.iloc[:split], subset.iloc[split:]
        
        # V36
        sc36 = StandardScaler()
        sc36.fit(tr[har3])
        preds_36 = np.full(len(te), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = tr[tr['Class'] == cls]
            te_idx = te['Class'] == cls
            if len(tr_c) < 50 or te_idx.sum() == 0: continue
            m = Ridge(alpha=100.0).fit(sc36.transform(tr_c[har3]), tr_c['Target'])
            preds_36[te_idx.values] = m.predict(sc36.transform(te.loc[te_idx, har3]))
        r2_36 = r2_score(te['Target'].values, preds_36) if not np.isnan(preds_36).all() else np.nan
        
        # V71
        sc71 = StandardScaler()
        sc71.fit(tr[feats])
        preds_71 = np.full(len(te), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = tr[tr['Class'] == cls]
            te_idx = te['Class'] == cls
            if len(tr_c) < 50 or te_idx.sum() == 0: continue
            m = Ridge(alpha=100.0).fit(sc71.transform(tr_c[feats]), tr_c['Target'])
            preds_71[te_idx.values] = m.predict(sc71.transform(te.loc[te_idx, feats]))
        r2_71 = r2_score(te['Target'].values, preds_71) if not np.isnan(preds_71).all() else np.nan
        
        period_results[label] = {
            'n_total': len(subset), 'n_train': len(tr), 'n_test': len(te),
            'r2_v36': float(r2_36) if not np.isnan(r2_36) else None,
            'r2_v71': float(r2_71) if not np.isnan(r2_71) else None,
        }
        print(f"  {label}: V36={r2_36:.5f}, V71={r2_71:.5f}, N={len(subset)}", flush=True)
    
    results['period_subsample'] = period_results
    
    # ===== 4b. VIX Quantile Regime =====
    print("\n--- 4b. VIX Quantile Regime Analysis ---", flush=True)
    
    # Use full 80/20 split
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    # Train models on full training set
    sc36 = StandardScaler()
    sc36.fit(train_df[har3])
    sc71 = StandardScaler()
    sc71.fit(train_df[feats])
    
    models_36 = {}
    models_71 = {}
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        if len(tr_c) < 100: continue
        models_36[cls] = Ridge(alpha=100.0).fit(sc36.transform(tr_c[har3]), tr_c['Target'])
        models_71[cls] = Ridge(alpha=100.0).fit(sc71.transform(tr_c[feats]), tr_c['Target'])
    
    # Generate predictions for full test set
    preds_36_full = np.full(len(test_df), np.nan)
    preds_71_full = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        te_idx = test_df['Class'] == cls
        if te_idx.sum() == 0 or cls not in models_36: continue
        preds_36_full[te_idx.values] = models_36[cls].predict(sc36.transform(test_df.loc[te_idx, har3]))
        preds_71_full[te_idx.values] = models_71[cls].predict(sc71.transform(test_df.loc[te_idx, feats]))
    
    # VIX quantile based on IV_VIX feature
    if 'IV_VIX' in test_df.columns:
        vix_values = test_df['IV_VIX'].values
        q50 = np.percentile(vix_values, 50)
        q90 = np.percentile(vix_values, 90)
        
        regimes = {
            'Low VIX (0-50%)': vix_values <= q50,
            'Mid VIX (50-90%)': (vix_values > q50) & (vix_values <= q90),
            'High VIX (90-100%)': vix_values > q90,
        }
        
        regime_results = {}
        print(f"\n  {'Regime':<25} {'N':>6} {'V36 R²':>8} {'V71 R²':>8} {'Delta':>8}", flush=True)
        print("  " + "-"*58, flush=True)
        
        for regime_name, mask in regimes.items():
            if mask.sum() < 30: continue
            actual = test_df['Target'].values[mask]
            p36 = preds_36_full[mask]
            p71 = preds_71_full[mask]
            
            r2_36 = r2_score(actual, p36)
            r2_71 = r2_score(actual, p71)
            
            regime_results[regime_name] = {
                'n': int(mask.sum()),
                'r2_v36': float(r2_36),
                'r2_v71': float(r2_71),
                'delta': float(r2_71 - r2_36),
            }
            print(f"  {regime_name:<25} {mask.sum():>6} {r2_36:>8.5f} {r2_71:>8.5f} {r2_71 - r2_36:>+8.5f}", flush=True)
        
        results['vix_regime'] = regime_results
    
    out_path = 'src/experiments/creative/v71_sci/results_04_regime.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    
    return results

if __name__ == '__main__':
    run()
