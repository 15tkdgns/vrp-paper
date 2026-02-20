"""
Step 5: Feature Correlation & Coefficient Analysis
====================================================
리뷰 포인트 5: 피처 상관 히트맵 + 상위 피처 계수 부호 해석
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.experiments.creative.v71_sci.data_builder import (
    build_dataset, get_train_test, ASSET_GROUPS
)

def run():
    print("="*70, flush=True)
    print("STEP 5: Feature Correlation & Coefficient Analysis", flush=True)
    print("="*70, flush=True)
    
    ds = build_dataset()
    data = ds['data']
    feats = ds['feats']
    train_df, test_df = get_train_test(data)
    
    # ===== 5a. Feature Correlation Matrix (Top features) =====
    print("\n--- 5a. Feature Correlation Matrix ---", flush=True)
    
    # Top-15 important features from V71 robustness results
    top_feats = [
        'RogersSatchell_22', 'GarmanKlass_22', 'Parkinson_22',
        'SPY_LogRV', 'Range_Close_Ratio', 'Garch_Weekly', 'Garch_Daily',
        'IV_VIX', 'LogRV_lag1', 'IV_VIX3M',
        'IV_VRP_ma22', 'IV_VIX9D', 'LogRV_lag10', 'Parkinson_5',
        'IV_VIX_TermSlope'
    ]
    top_feats = [f for f in top_feats if f in feats]
    
    corr_matrix = train_df[top_feats].corr()
    
    print(f"\nCorrelation matrix ({len(top_feats)} x {len(top_feats)}):", flush=True)
    # Print condensed
    print(f"\n{'':>20}", end='', flush=True)
    for f in top_feats[:10]:
        print(f"{f[:8]:>9}", end='', flush=True)
    print(flush=True)
    
    for i, f1 in enumerate(top_feats[:10]):
        print(f"{f1[:20]:<20}", end='', flush=True)
        for j, f2 in enumerate(top_feats[:10]):
            v = corr_matrix.loc[f1, f2]
            print(f"{v:>9.3f}", end='', flush=True)
        print(flush=True)
    
    # HF Proxy internal correlations
    hf_feats_in_top = [f for f in top_feats if any(x in f for x in ['Rogers', 'Garman', 'Parkinson', 'Range'])]
    print(f"\n  HF Proxy 내부 상관관계:", flush=True)
    for i, f1 in enumerate(hf_feats_in_top):
        for f2 in hf_feats_in_top[i+1:]:
            print(f"    {f1} <-> {f2}: {corr_matrix.loc[f1, f2]:.4f}", flush=True)
    
    # IV Surface internal correlations
    iv_feats_in_top = [f for f in top_feats if f.startswith('IV_')]
    print(f"\n  IV Surface 내부 상관관계:", flush=True)
    for i, f1 in enumerate(iv_feats_in_top):
        for f2 in iv_feats_in_top[i+1:]:
            print(f"    {f1} <-> {f2}: {corr_matrix.loc[f1, f2]:.4f}", flush=True)
    
    # ===== 5b. Coefficient Sign Analysis =====
    print("\n--- 5b. Ridge Coefficient Analysis (per class) ---", flush=True)
    
    sc = StandardScaler()
    sc.fit(train_df[feats])
    
    coef_results = {}
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        if len(tr_c) < 100: continue
        
        m = Ridge(alpha=100.0).fit(sc.transform(tr_c[feats]), tr_c['Target'])
        coefs = dict(zip(feats, m.coef_))
        
        # Sort by absolute value
        sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        coef_results[cls] = {k: float(v) for k, v in sorted_coefs}
        
        print(f"\n  [{cls}] Top-10 coefficients (standardized):", flush=True)
        print(f"  {'Feature':<25} {'Coef':>10} {'Sign':>6} {'Interpretation'}", flush=True)
        print(f"  {'-'*70}", flush=True)
        for feat_name, coef_val in sorted_coefs[:10]:
            sign = "+" if coef_val > 0 else "-"
            interp = interpret_coefficient(feat_name, coef_val)
            print(f"  {feat_name:<25} {coef_val:>10.5f} {sign:>6} {interp}", flush=True)
    
    # Save
    out = {
        'correlation_matrix': {f1: {f2: float(corr_matrix.loc[f1, f2]) 
                                    for f2 in top_feats} for f1 in top_feats},
        'coefficients_per_class': coef_results,
    }
    
    out_path = 'src/experiments/creative/v71_sci/results_05_correlation_coef.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    
    return out

def interpret_coefficient(feat, coef):
    """피처별 계수 부호에 대한 경제적 해석"""
    sign = "+" if coef > 0 else "-"
    interpretations = {
        'RogersSatchell_22': f"{'과거 변동성 높으면 미래도 높음 (변동성 지속성)' if coef > 0 else '역전 효과'}",
        'GarmanKlass_22': f"{'OHLC 기반 변동성 지속' if coef > 0 else '반전'}",
        'Parkinson_22': f"{'범위 변동성 지속' if coef > 0 else '반전'}",
        'Parkinson_5': f"{'단기 범위 변동성 -> 미래 변동성' if coef > 0 else '반전'}",
        'SPY_LogRV': f"{'시장 변동성 전이 효과' if coef > 0 else '역상관'}",
        'Range_Close_Ratio': f"{'Range 정보 > Close 정보' if coef > 0 else 'Close 정보 우위'}",
        'Garch_Daily': f"{'조건부 변동성 지속' if coef > 0 else '반전'}",
        'Garch_Weekly': f"{'주간 변동성 하위 트렌드 지속' if coef > 0 else '반전'}",
        'IV_VIX': f"{'공포 지수 -> 실현 변동성 예측' if coef > 0 else '역방향'}",
        'IV_VIX3M': f"{'중기 IV -> 미래 RV' if coef > 0 else '기간구조 효과'}",
        'IV_VIX_TermSlope': f"{'콘탱고 -> 변동성 상승' if coef > 0 else '백워데이션 = 위기 신호'}",
        'IV_VRP': f"{'VRP 높으면 미래 변동성 상승' if coef > 0 else 'VRP = 하락 재료'}",
        'IV_VRP_ma22': f"{'평활 VRP -> 추세' if coef > 0 else '반전'}",
        'LogRV_lag1': f"{'전일 변동성 지속 (HAR effect)' if coef > 0 else '반전'}",
        'LogRV_lag10': f"{'중기 변동성 지속' if coef > 0 else '반전'}",
        'Corr_SPY': f"{'시장 연동 클수록 변동성 높음' if coef > 0 else '분산 효과'}",
        'Ret_abs_lag1': f"{'레버리지 효과 (큰 수익률 -> 변동성)' if coef > 0 else '역효과'}",
    }
    return interpretations.get(feat, f"({sign})")

if __name__ == '__main__':
    run()
