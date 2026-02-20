"""
Step 3: Per-Asset & Per-Class Performance
==========================================
리뷰 포인트 6: 자산별/클래스별 R² (V36 vs V71)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.experiments.creative.v71_sci.data_builder import (
    build_dataset, get_train_test, ASSET_GROUPS
)

def run():
    print("="*70, flush=True)
    print("STEP 3: Per-Asset & Per-Class Performance", flush=True)
    print("="*70, flush=True)
    
    ds = build_dataset()
    data = ds['data']
    feats = ds['feats']
    train_df, test_df = get_train_test(data)
    
    har3 = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    asset_results = {}
    class_results = {}
    
    for asset in sorted(data['Asset'].unique()):
        cls = data[data['Asset'] == asset]['Class'].iloc[0]
        tr_a = train_df[train_df['Asset'] == asset]
        te_a = test_df[test_df['Asset'] == asset]
        
        if len(tr_a) < 100 or len(te_a) < 30:
            continue
        
        # V36 (HAR-3)
        sc36 = StandardScaler()
        X_tr36 = sc36.fit_transform(tr_a[har3])
        m36 = Ridge(alpha=100.0).fit(X_tr36, tr_a['Target'])
        p36 = m36.predict(sc36.transform(te_a[har3]))
        r2_36 = r2_score(te_a['Target'], p36)
        
        # V71 (All-37)
        sc71 = StandardScaler()
        X_tr71 = sc71.fit_transform(tr_a[feats])
        m71 = Ridge(alpha=100.0).fit(X_tr71, tr_a['Target'])
        p71 = m71.predict(sc71.transform(te_a[feats]))
        r2_71 = r2_score(te_a['Target'], p71)
        
        asset_results[asset] = {
            'class': cls,
            'n_test': len(te_a),
            'r2_v36': float(r2_36),
            'r2_v71': float(r2_71),
            'improvement': float(r2_71 - r2_36),
        }
    
    # Print asset-level results
    print(f"\n{'Asset':<6} {'Class':<10} {'N_test':>7} {'V36 R²':>8} {'V71 R²':>8} {'Delta':>8}", flush=True)
    print("-"*52, flush=True)
    for asset in sorted(asset_results.keys()):
        r = asset_results[asset]
        print(f"{asset:<6} {r['class']:<10} {r['n_test']:>7} {r['r2_v36']:>8.5f} {r['r2_v71']:>8.5f} {r['improvement']:>+8.5f}", flush=True)
    
    # Class-level aggregation
    print(f"\n{'Class':<12} {'V36 R² (mean)':>14} {'V71 R² (mean)':>14} {'Delta':>8}", flush=True)
    print("-"*52, flush=True)
    for cls in ASSET_GROUPS:
        cls_assets = [a for a, r in asset_results.items() if r['class'] == cls]
        if not cls_assets: continue
        v36_mean = np.mean([asset_results[a]['r2_v36'] for a in cls_assets])
        v71_mean = np.mean([asset_results[a]['r2_v71'] for a in cls_assets])
        v36_std = np.std([asset_results[a]['r2_v36'] for a in cls_assets])
        v71_std = np.std([asset_results[a]['r2_v71'] for a in cls_assets])
        class_results[cls] = {
            'v36_mean': float(v36_mean), 'v36_std': float(v36_std),
            'v71_mean': float(v71_mean), 'v71_std': float(v71_std),
            'delta': float(v71_mean - v36_mean),
        }
        print(f"{cls:<12} {v36_mean:>8.5f}+/-{v36_std:.3f} {v71_mean:>8.5f}+/-{v71_std:.3f} {v71_mean - v36_mean:>+8.5f}", flush=True)
    
    out = {'per_asset': asset_results, 'per_class': class_results}
    out_path = 'src/experiments/creative/v71_sci/results_03_asset_class.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    
    return out

if __name__ == '__main__':
    run()
