"""
Step 2: Additional Metrics & Diebold-Mariano Test
===================================================
리뷰 포인트 3: R² 외 추가 메트릭(RMSE, MAE, QLIKE) + DM test 통계적 유의성
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.experiments.creative.v71_sci.data_builder import (
    build_dataset, get_train_test, train_ridge_perclass, ASSET_GROUPS
)

def qlike_loss(actual, predicted):
    """QLIKE loss: 변동성 예측 문헌의 표준 손실함수"""
    # For log-scale: exp transform back
    act = np.exp(actual)
    pred = np.exp(predicted)
    pred = np.clip(pred, 1e-10, None)
    return np.mean(act / pred - np.log(act / pred) - 1)

def diebold_mariano_test(e1, e2, horizon=22):
    """
    Diebold-Mariano test (HAC with Newey-West)
    H0: 두 모델의 예측력에 차이가 없다
    
    Parameters
    ----------
    e1, e2 : array-like
        Forecast errors from models 1 and 2
    horizon : int
        Forecast horizon in trading days. Determines HAC bandwidth (h-1).
        Must match the prediction horizon to correctly account for
        serial correlation in overlapping forecast errors.
        예: 22d 예측 → horizon=22, 60d 예측 → horizon=60
    """
    d = e1**2 - e2**2  # MSE loss differential
    n = len(d)
    
    # Newey-West HAC (bandwidth = horizon - 1)
    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    
    hac_var = gamma_0
    for k in range(1, horizon):
        weight = 1 - k / horizon  # Bartlett kernel
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / (n - 1)
        hac_var += 2 * weight * gamma_k
    
    hac_var = max(hac_var, 1e-15)
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def run():
    print("="*70, flush=True)
    print("STEP 2: Additional Metrics & DM Test", flush=True)
    print("="*70, flush=True)
    
    ds = build_dataset()
    data = ds['data']
    feats = ds['feats']
    base_feats = ds['base_feats']
    
    train_df, test_df = get_train_test(data)
    
    # --- V36-like model (base 3 features: HAR only) ---
    har3_feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    preds_v36, _, _, _ = train_ridge_perclass(train_df, test_df, har3_feats)

    # --- V68-like model (base 14 features) ---
    preds_base, _, _, _ = train_ridge_perclass(train_df, test_df, base_feats)
    
    # --- V71 model (all 37 features) ---
    preds_v71, _, _, _ = train_ridge_perclass(train_df, test_df, feats)
    
    # --- V71 + XGBoost Ensemble ---
    try:
        from xgboost import XGBRegressor
        sc = StandardScaler()
        sc.fit(train_df[feats])
        preds_xgb = np.full(len(test_df), np.nan)
        
        val_split = int(len(train_df) * 0.8)
        tr_inner, va_inner = train_df.iloc[:val_split], train_df.iloc[val_split:]
        
        for cls in ASSET_GROUPS:
            tr_c = train_df[train_df['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_c) < 100 or te_idx.sum() == 0: continue
            m = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc.transform(tr_c[feats]), tr_c['Target'])
            preds_xgb[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))
        
        # Weighted ensemble (0.7 Ridge + 0.3 XGB)
        best_w, best_r2 = 0.7, -999
        for w in np.arange(0.0, 1.01, 0.05):
            blend = w * preds_v71 + (1-w) * preds_xgb
            r2 = r2_score(test_df['Target'].values, blend)
            if r2 > best_r2: best_w, best_r2 = w, r2
        preds_ens = best_w * preds_v71 + (1-best_w) * preds_xgb
        has_ens = True
    except ImportError:
        preds_ens = preds_v71
        has_ens = False
    
    actual = test_df['Target'].values
    
    models = {
        'V36 (HAR-3)': preds_v36,
        'V68 (Base-14)': preds_base,
        'V71 Ridge (All-37)': preds_v71,
    }
    if has_ens:
        models['V71 Ensemble'] = preds_ens
    
    # --- Compute all metrics ---
    print("\n--- Multi-Metric Comparison ---", flush=True)
    print(f"{'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8}", flush=True)
    print("-"*58, flush=True)
    
    results = {}
    for name, preds in models.items():
        r2 = r2_score(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        ql = qlike_loss(actual, preds)
        results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'qlike': ql}
        print(f"{name:<22} {r2:>8.5f} {rmse:>8.5f} {mae:>8.5f} {ql:>8.5f}", flush=True)
    
    # --- Diebold-Mariano Tests ---
    # NOTE: horizon=22 (trading days) matches the 22d forward RV prediction target.
    # For multi-horizon experiments, this should be set to the respective horizon h.
    TARGET_HORIZON = 22  # trading days (22d = ~1 calendar month)
    print(f"\n--- Diebold-Mariano Test (Newey-West HAC, bandwidth={TARGET_HORIZON-1}) ---", flush=True)
    print(f"{'Comparison':<40} {'DM-stat':>10} {'p-value':>10} {'Bonf.':>10} {'Result':>10}", flush=True)
    print("-"*85, flush=True)
    
    dm_results = {}
    best_model_name = 'V71 Ensemble' if has_ens else 'V71 Ridge (All-37)'
    best_preds = models[best_model_name]
    
    comparisons = [
        (best_model_name, 'V36 (HAR-3)'),
        (best_model_name, 'V68 (Base-14)'),
        ('V71 Ridge (All-37)', 'V36 (HAR-3)'),
        ('V71 Ridge (All-37)', 'V68 (Base-14)'),
    ]
    n_comparisons = len(comparisons)  # For Bonferroni correction
    
    for model_a, model_b in comparisons:
        e_a = actual - models[model_a]
        e_b = actual - models[model_b]
        dm_stat, p_val = diebold_mariano_test(e_a, e_b, horizon=TARGET_HORIZON)
        p_bonf = min(p_val * n_comparisons, 1.0)  # Bonferroni correction
        sig = "***" if p_bonf < 0.01 else "**" if p_bonf < 0.05 else "*" if p_bonf < 0.1 else "n.s."
        label = f"{model_a} vs {model_b}"
        dm_results[label] = {
            'dm_stat': dm_stat, 'p_value': p_val,
            'p_bonferroni': p_bonf, 'significance': sig
        }
        print(f"{label:<40} {dm_stat:>10.4f} {p_val:>10.6f} {p_bonf:>10.6f} {sig:>10}", flush=True)
    
    print(f"\n  Note: Bonferroni correction applied for {n_comparisons} comparisons", flush=True)
    
    all_results = {
        'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'dm_tests': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                        for kk, vv in v.items()} for k, v in dm_results.items()},
    }
    
    out_path = 'src/experiments/creative/v71_sci/results_02_metrics_dm.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    
    return all_results

if __name__ == '__main__':
    run()
