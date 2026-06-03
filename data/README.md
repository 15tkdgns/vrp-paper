# 실험 결과 아카이브

## 디렉토리 구조

```
paper/csv/
├── benchmark/           # Multi-horizon 벤치마크 결과
├── robustness/          # 강건성 검증 (split/alpha/CV/seed/DM test)
├── feature_analysis/    # 피처 중요도/ablation 분석
├── model_evolution/     # V31~V74 모델별 실험 결과
├── sci_review/          # SCI 리뷰 대응 5단계 분석
├── methodology/         # III장 방법론 검증 (VIX/배당/IV 지평)
├── CATALOG.json         # 전체 파일 카탈로그 (메타데이터)
├── *.csv                # 피처 통계/예측값 등 CSV 데이터
└── README.md            # 이 파일
```

## 카테고리별 요약

### benchmark/
Multi-horizon(1d~252d) 벤치마크. 7개 모델(V71, V50, HAR-3, LASSO, HAR-CJ, RF, V50 Orig)
비교. Non-overlapping forward RV 타겟 기준.

### robustness/
- V71: 다중 split, alpha 민감도, 5-fold CV, 데이터 누출 검증, DM test
- V50: 하이퍼파라미터 민감도, 5-seed, extreme market, 11자산 확장
- 학술: GVZ/OVX 대체, block bootstrap DM, ffill 비교, 9개 모델 비교

### feature_analysis/
7개 방법(Ridge Coef, Permutation, XGBoost, SHAP, MI, Spearman, RFE)의
합의 피처 중요도 순위 및 서브셋 ablation.

### model_evolution/
V31(Mamba)~V74(Deep Learning)까지 총 70+회 실험의 핵심 결과.
Phase 1(HAR/GARCH) → Phase 6(ElasticNet) 진화 과정 추적.

### sci_review/
SCI 심사 대응 5단계: 공정 비교, 추가 메트릭, 자산별 분석, 레짐 안정성, 계수 해석.

### methodology/
III장 보강을 위한 정량 검증: VIX 가용기간, 배당 영향, IV 지평별 비교.

## 기존 루트 파일 (유지)
`paper/csv/` 루트의 CSV/JSON 파일은 Quarto 대시보드(.qmd)에서 직접 참조하므로 유지.
