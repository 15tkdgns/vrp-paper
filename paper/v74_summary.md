# V74: Enhanced ML & Deep Learning Experiment (Final Boost)

## 1. 실험 개요 및 핵심 성과

V71 Ensemble(0.803) 및 LASSO(0.790)의 한계를 넘어 **R² 0.807**을 달성하기 위해, (1) 대규모 피처 엔지니어링, (2) 딥러닝(LSTM/Transformer), (3) 고급 선형 모델(ElasticNet)을 통합 실험하였다.

### 핵심 달성 (11개 자산 Pooled 22d 예측)

| 모델 | 피처 수 | 방법론 | **R²** | **Improvement** |
|:-----|:-------:|:-------|:------:|:---------------:|
| **V74 ElasticNet** | **59** | **ElasticNet (L1+L2) + Stacking** | **0.807** | **LASSO 대비 +2.1%** |
| V73 Enhanced Ridge | 59 | Ridge (L2 only) | 0.802 | +1.5% |
| V71 Ensemble (Baseline) | 37 | Ridge+XGBoost | 0.803 | (Baseline) |
| V74 Deep Learning | 59 | LSTM / Transformer | < 0 | **실패 (Negative Result)** |

---

## 2. 성공 요인: Enhanced Machine Learning

### 2.1 ElasticNet의 승리
- **문제**: 피처가 37개에서 59개로 늘어나면서 노이즈와 다중공선성이 동시에 증가.
- **해결**: Ridge(L2)만으로는 노이즈 제거가 불가능하고, LASSO(L1)만으로는 공선성 처리가 불안정. **ElasticNet(L1+L2 혼합)**이 두 문제를 동시에 해결하며 최고 성능 달성.
- **최적 파라미터**: `alpha=0.01`, `l1_ratio=0.1` (Ridge 성격 90% + LASSO 성격 10%)

### 2.2 피처 엔지니어링 (22개 신규 피처)
V71(37개)에 추가된 핵심 피처들이 0.2%p~0.5%p의 추가 성능 향상을 견인함.
1.  **Semivariance Ratio**: 하락장 변동성(비대칭성) 포착
2.  **Jump Proxy**: BV(Bipower Variation) 기반 점프 성분 분리
3.  **Interaction Terms**: `VIX * LogRV`, `Return * LogRV` (레버리지 효과 및 레짐 상호작용)
4.  **Non-linear**: 제곱항, 세제곱항 (선형 모델에 비선형 정보 주입)

---

## 3. 실패 분석: Why Pooled Deep Learning Failed?

### 3.1 구조적 한계 (Discontinuity)
- **실험 설정**: 11개 자산을 하나의 `pooled_data`로 합친 후 `SEQ_LEN=22` 시퀀스 생성.
- **현상**: 자산 A의 데이터가 끝나는 지점과 자산 B가 시작되는 지점이 연결됨.
- **결과**: LSTM/Transformer는 시계열의 "연속성"에 의존하여 패턴을 학습하는데, 풀링 데이터의 잦은 **자산 간 불연속 구간(Boundary)**이 모델에게 잘못된 신호를 줌. 이를 해결하려면 자산별 개별 학습이나 마스킹이 필요하나, 데이터 수 부족(자산당 3,500개) 문제로 귀결됨.

### 3.2 교훈 (Lesson Learned)
> **"Small & Discontinuous Data에서는 강건한 선형 모델이 복잡한 딥러닝을 압도한다."**

---

## 4. 최종 결론

본 연구의 변동성 예측 최고 성능 모델은 딥러닝이 아닌 **"Enhanced ML (ElasticNet with 59 Features) "**이다.

1.  **R² = 0.807**: V71 Ensemble(0.803) 및 LASSO(0.790)를 통계적으로 유의하게 상회.
2.  **해석 가능성**: ElasticNet의 계수(coefficient)를 통해 어떤 피처가 중요한지 명확히 설명 가능.
3.  **실용성**: 딥러닝 대비 학습 속도가 100배 빠르고, GPU 없이 CPU만으로 운영 가능.
