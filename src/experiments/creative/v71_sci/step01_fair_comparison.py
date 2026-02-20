"""
Step 1: Fair Model Comparison
==============================
리뷰 포인트 2: 동일 피처 세트에서 Ridge vs XGBoost vs MLP 공정 비교
- Base only (14개 피처)
- All features (37개 피처)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from src.experiments.creative.v71_sci.data_builder import (
    build_dataset, get_train_test, ASSET_GROUPS
)

def run():
    print("="*70, flush=True)
    print("STEP 1: Fair Model Comparison (동일 피처 세트 공정 비교)", flush=True)
    print("="*70, flush=True)
    
    ds = build_dataset()
    data = ds['data']
    train_df, test_df = get_train_test(data)
    
    feature_sets = {
        'Base (14)': ds['base_feats'],
        'All (37)': ds['feats'],
    }
    
    results = {}
    
    for fs_name, feat_list in feature_sets.items():
        print(f"\n--- Feature Set: {fs_name} ({len(feat_list)} features) ---", flush=True)
        
        sc = StandardScaler()
        sc.fit(train_df[feat_list])
        
        val_split = int(len(train_df) * 0.8)
        tr_inner, va_inner = train_df.iloc[:val_split], train_df.iloc[val_split:]
        
        model_results = {}
        
        for model_name, model_fn in [
            ('Ridge', lambda: Ridge(alpha=100.0)),
            ('XGBoost', None),  # handled separately
            ('MLP', lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                learning_rate='adaptive', random_state=42, verbose=False
            )),
        ]:
            print(f"  Training {model_name}...", flush=True)
            preds = np.full(len(test_df), np.nan)
            
            if model_name == 'XGBoost':
                try:
                    from xgboost import XGBRegressor
                    for cls in ASSET_GROUPS:
                        tr_c = train_df[train_df['Class'] == cls]
                        te_idx = test_df['Class'] == cls
                        if len(tr_c) < 100 or te_idx.sum() == 0: continue
                        
                        # Tune on inner split
                        tr_i = tr_inner[tr_inner['Class'] == cls]
                        va_i = va_inner[va_inner['Class'] == cls]
                        best_r2, best_cfg = -999, {'n_estimators': 200, 'max_depth': 4}
                        if len(va_i) > 30:
                            for n_est in [100, 200, 300]:
                                for md in [3, 4, 5]:
                                    m = XGBRegressor(
                                        n_estimators=n_est, max_depth=md, learning_rate=0.03,
                                        subsample=0.8, colsample_bytree=0.8,
                                        reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                                        random_state=42, verbosity=0
                                    ).fit(sc.transform(tr_i[feat_list]), tr_i['Target'])
                                    r2 = r2_score(va_i['Target'], m.predict(sc.transform(va_i[feat_list])))
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_cfg = {'n_estimators': n_est, 'max_depth': md}
                        
                        m = XGBRegressor(
                            n_estimators=best_cfg['n_estimators'], max_depth=best_cfg['max_depth'],
                            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                            random_state=42, verbosity=0
                        ).fit(sc.transform(tr_c[feat_list]), tr_c['Target'])
                        preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feat_list]))
                except ImportError:
                    print("    XGBoost not available, skipping", flush=True)
                    continue
            
            elif model_name == 'Ridge':
                # Per-class alpha tuning
                alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
                for cls in ASSET_GROUPS:
                    tr_c = train_df[train_df['Class'] == cls]
                    te_idx = test_df['Class'] == cls
                    if len(tr_c) < 100 or te_idx.sum() == 0: continue
                    
                    tr_i = tr_inner[tr_inner['Class'] == cls]
                    va_i = va_inner[va_inner['Class'] == cls]
                    best_r2, best_a = -999, 100.0
                    if len(va_i) > 30:
                        for a in alphas:
                            m = Ridge(alpha=a).fit(sc.transform(tr_i[feat_list]), tr_i['Target'])
                            r2 = r2_score(va_i['Target'], m.predict(sc.transform(va_i[feat_list])))
                            if r2 > best_r2: best_r2, best_a = r2, a
                    
                    m = Ridge(alpha=best_a).fit(sc.transform(tr_c[feat_list]), tr_c['Target'])
                    preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feat_list]))
            
            else:  # MLP
                for cls in ASSET_GROUPS:
                    tr_c = train_df[train_df['Class'] == cls]
                    te_idx = test_df['Class'] == cls
                    if len(tr_c) < 100 or te_idx.sum() == 0: continue
                    m = model_fn()
                    m.fit(sc.transform(tr_c[feat_list]), tr_c['Target'])
                    preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feat_list]))
            
            valid = ~np.isnan(preds)
            if valid.sum() > 0:
                r2 = r2_score(test_df['Target'].values[valid], preds[valid])
                rmse = np.sqrt(mean_squared_error(test_df['Target'].values[valid], preds[valid]))
                mae = mean_absolute_error(test_df['Target'].values[valid], preds[valid])
                model_results[model_name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
                print(f"    {model_name}: R²={r2:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}", flush=True)
        
        results[fs_name] = model_results
    
    # Summary table
    print("\n" + "="*70, flush=True)
    print(f"{'Feature Set':<15} {'Model':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8}", flush=True)
    print("-"*55, flush=True)
    for fs_name in results:
        for model_name, metrics in results[fs_name].items():
            print(f"{fs_name:<15} {model_name:<10} {metrics['r2']:>8.5f} {metrics['rmse']:>8.5f} {metrics['mae']:>8.5f}", flush=True)
    
    out_path = 'src/experiments/creative/v71_sci/results_01_fair_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_path}", flush=True)
    
    return results

if __name__ == '__main__':
    run()
