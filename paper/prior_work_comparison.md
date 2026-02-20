# 선행연구 피처/방법론/성능 비교 정리

> 본 연구(Cross-Asset VRP Prediction)와 관련된 선행연구의 피처, 방법론, 성능을 체계적으로 비교한 문서

---

## 1. 변동성 예측 모델 계보

### 1.1 전통 계량경제 모델

| 모델/연구 | 저자(연도) | 저널 | 피처 | 방법론 | 성능 | 자산 |
|:---|:---|:---|:---|:---|:---|:---|
| **ARCH** | Engle (1982) | *Econometrica* | 과거 충격 $\epsilon^2$ | 조건부 분산 모형 | - | 영국 인플레이션 |
| **GARCH(1,1)** | Bollerslev (1986) | *J. Econometrics* | $\epsilon_{t-1}^2$, $\sigma_{t-1}^2$ | $\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ | - | 범용 |
| **GARCH-MIDAS** | Engle & Rangel (2008) | *RFS* | 단기 GARCH + 장기 거시변수 | 이중 성분 변동성 분해 | R² ≈ 0.55 (22d) | 주식시장 |
| **HAR-RV** | Corsi (2009) | *JFEC* | RV_d, RV_w, RV_m (3개) | $RV_{t+1} = \alpha + \beta_d RV^{(d)} + \beta_w RV^{(w)} + \beta_m RV^{(m)}$ | 본 연구 재현: 22d R²=0.761 | S&P 500 |
| **HAR-CJ** | Andersen, Bollerslev, Diebold (2007) | *Econometrica* | 연속 RV + Jump RV (6개) | HAR + Bipower Variation 기반 Jump 분리 | 본 연구 재현: 22d R²=0.752 | S&P 500 |
| **Realized GARCH** | Hansen et al. (2021) | *JFEC (2024)* | 고빈도 RV, 이중 충격 구조 | $\log h_{t+1} = \omega + \beta\log h_t + \tau(z_t) + \gamma\sigma u_t$ | VRP 분해: Vol shock = 97.8% | S&P 500 |

### 1.2 머신러닝 기반 모델

| 모델/연구 | 저자(연도) | 저널 | 피처 | 방법론 | 성능 | 자산 |
|:---|:---|:---|:---|:---|:---|:---|
| **ML + 옵션** | Carr, Wu, Zhang (2019) | *arXiv* | 79개 옵션 데이터 (만기/행사가별 IV) | Ridge, FNN; Reg II: VIX² 잔차 학습 | R² = 0.39 | SPX 단일 |
| **LASSO-HAR** | Audrino & Knaus (2016) | *JFEC, 14(2)* | HAR + 확장 피처 (L1 선별) | LASSO (L1 정규화) | 본 연구 재현: 22d R²=0.790 | S&P 500 |
| **RF + 고빈도** | Christensen, Siggaard, Veliyev (2023) | *JFE, 150(2)* | 다수 종목 고빈도 RV | NN(2-3 layers), RF; Pooling 전략 | 본 연구 재현: 22d R²=0.770, **365d R²=0.705** | 다수 종목 |
| **고차원 ML** | Chun, Cho, Ryu (2025) | *RIBF* | 43개 (거시경제+금융+심리) | LASSO, GBRT | HAR 대비 우위; **Sharpe 3.45~3.48** | 주식시장 |
| **Adaptive MTL** | Fan, Wu, Yang (2025) | *arXiv* | 섹터별 수익률 피처 | Projection-Penalized PCA + 적응적 MTL | 섹터 간 정보 전이 효과 확인 | 다중 섹터 |
| **Enhanced ML** | **본 연구 (V73/V74)** | **-** | **59개 (Base+HF+IV+New)** | **ElasticNet (L1+L2) + Stacking** | **R² = 0.807 (22d)** | **11개 자산 (Pooled)** |

### 1.3 딥러닝 기반 모델

| 모델/연구 | 저자(연도) | 저널 | 피처 | 방법론 | 성능 | 자산 |
|:---|:---|:---|:---|:---|:---|:---|
| **Hybrid LSTM-GARCH** | Roszyk & Slepaczuk (2024) | *arXiv* | GARCH $\sigma^2$ + VIX + 수익률 | LSTM + GARCH 입력 결합 | VIX 추가 시 MAE 45% 개선 | S&P 500 |
| **GINN** | Xu et al. (2024) | *ICAIF '24* | GARCH 파라미터 매핑 | LSTM cell = GARCH 분산 업데이트 구조적 결합 | 다중 자산 범용성 검증 | SPX, DJI, EUR/USD, Gold |
| **DeepVol** | Various (2022, 2024) | *arXiv* | 고빈도 OHLCV | Dilated Causal CNN | 전통 모델 대비 현저한 개선 | 고빈도 데이터 |
| **Multi-Transformer** | Ramos-Perez et al. (2021) | *Applied Soft Computing* | 가격 시계열 (서브셋 분할) | Ensemble Transformer + Self-Attention | COVID 기간 GARCH 대비 RMSE 우위 | S&P 500 |
| **DL Risk Premia** | Anonymous (2023) | - | 시장/크레딧/FX 리스크 피처 | 단일 DL 모델 다중 프리미엄 동시 예측 | 공통/이질 요인 분리 | 시장, 크레딧, FX |
| **Pooled DL** | **본 연구 (V74)** | **-** | **59개** | **LSTM, Transformer, Bi-LSTM** | **실패 (R² < 0)** | **11개 자산 (불연속성)** |

---

## 2. VRP 이론 및 Cross-Asset 연구

| 연구 | 저자(연도) | 저널 | 핵심 방법론 | 핵심 발견 | 데이터 |
|:---|:---|:---|:---|:---|:---|
| **VRP 표준 정의** | Carr & Wu (2009) | *RFS* | Variance swap portfolio | VRP = 양(+)의 리스크 프리미엄 | SPX 옵션 |
| **VIX² 분해** | Bekaert & Hoerova (2014) | *J. Econometrics* | $VIX^2 = E^P[RV] + VRP$ | VRP가 미래 주가 수익률 예측력 보유 | S&P 500 |
| **VRP 동적 변동** | Drechsler & Yaron (2011) | *RFS* | 동적 자산가격결정 모델 | 경제적 불확실성 + 위험회피가 VRP 결정 | 이론 모형 |
| **Cross-Asset VRP** | Heston et al. (2023) | *AEA 2024* | 20개 선물 옵션 기반 variance portfolio | VRP의 자산별 이질성 확인 | 20개 선물 (주식, 국채, 통화, 원자재) |
| **Commodity VRP** | Ornelas et al. (2018) | *BIS WP 619* | 선형 예측 회귀 | VRP의 cross-asset spillover 발견 | 금, 원유 등 원자재 |
| **Multi-Asset VRP Factor** | Finta & Ornelas (2022) | *JIMF&M* | 멀티 에셋 VRP 팩터 모델 | 통합 VRP > 개별 자산 VRP | 다종 원자재 |
| **VIX Spillover** | Ang & Longstaff (2013) | *J. Monetary Economics* | 체제 전환 모델 | Regime-dependent spillover | 미국/유럽 주식시장 |
| **VIX 정보 함량** | Fleming, Ostdiek, Whaley (1995) | *J. Futures Markets* | VIX vs 미래 RV 예측 비교 | VIX의 예측 정보력 최초 확인 | S&P 100 |
| **투자자 공포 지수** | Whaley (2000) | *J. Portfolio Management* | VIX 지수 분석 | VIX = "investor fear gauge" 명명 | S&P 500 |

---

## 3. 피처 유형별 분류

### 3.1 실현 변동성 기반 (Realized Volatility)

| 피처 | 정의 | 사용 연구 | 본 연구 활용 |
|:---|:---|:---|:---|
| RV(일별) | $\sum r_t^2 \times 252$ | Corsi (2009), Hansen et al. (2021) | HAR 피처 (LogRV lag-1, 5, 22) |
| RV(주별/월별) | 5d/22d rolling RV | Corsi (2009) | HAR-3 피처의 핵심 |
| Bipower Variation | $\frac{\pi}{2} \sum |r_t| |r_{t-1}|$ | Andersen et al. (2007) | HAR-CJ 벤치마크의 점프 프록시 |
| Jump Component | $RV - BV$ (max(0, ...)) | Andersen et al. (2007) | HAR-CJ 벤치마크 |

### 3.2 Range 기반 (OHLC 변동성 추정량)

| 피처 | 정의 | 이론적 효율성 | 사용 연구 | 본 연구 활용 |
|:---|:---|:---|:---|:---|
| **Rogers-Satchell** | $(h-c)(h-o)+(l-c)(l-o)$ | Drift-robust, 5배 | Rogers & Satchell (1991) | V50 Tuned 핵심 3feat |
| **Garman-Klass** | $0.5(h-l)^2 - (2\ln2-1)(c-o)^2$ | 7.4배 | Garman & Klass (1980) | V50 Tuned 핵심 3feat |
| **Parkinson** | $(h-l)^2 / (4\ln2)$ | 5배 | Parkinson (1980) | V71 HF Proxy 그룹 |
| **Yang-Zhang** | YZ Estimator | 7배 | Yang & Zhang (2000) | V71 HF Proxy 그룹 |
| **Range_Close_Ratio** | $(H-L)/C$ | - | 본 연구 제안 | V50 Tuned 핵심 3feat |

### 3.3 옵션/내재변동성 기반 (IV Surface)

| 피처 | 정의 | 사용 연구 | 본 연구 활용 |
|:---|:---|:---|:---|
| VIX | CBOE Volatility Index | 전체 VRP 문헌 | Global risk factor |
| VIX Term Structure | VIX3M/VIX, VIX9D/VIX 비율 | Whaley (2000) 계열 | IV Surface 피처 그룹 |
| SKEW | 꼬리 위험 가격결정 지표 | - | Sentiment 피처 |
| 개별 옵션 IV | 만기/행사가별 79개 | Carr, Wu, Zhang (2019) | **미사용** (경제성 우위) |
| VRP rolling 통계 | VRP의 이동평균/표준편차 | 본 연구 제안 | IV Surface 피처 그룹 |

### 3.4 거시경제/심리/대안 데이터

| 피처 | 사용 연구 | 본 연구 활용 |
|:---|:---|:---|
| 거시경제 변수 43개 | Chun, Cho, Ryu (2025) | 미사용 (VIX 파생 지표로 대체) |
| GARCH 조건부 분산 | Roszyk & Slepaczuk (2024) | V71 Base 피처 (GARCH(1,1) sigma) |
| 거래량 이동평균/변화율 | 본 연구 제안 | V71 Alt Data 그룹 (6개) |
| DayOfWeek | 본 연구 제안 | V71 Base 피처 |

---

## 4. 방법론 비교

### 4.1 정규화 방법

| 방법 | 특성 | 사용 연구 | 본 연구 |
|:---|:---|:---|:---|
| **L1 (LASSO)** | 피처 선택(sparsity), 불필요한 피처 계수 = 0 | Audrino & Knaus (2016) | LASSO-HAR 벤치마크 |
| **L2 (Ridge)** | 다중공선성 처리, 모든 피처 포함 | Carr et al. (2019) | **V71 Ridge 챔피언** (alpha=10~100) |
| **Dropout** | 뉴런 랜덤 비활성화 | 일반 DL 문헌 | V50 (dropout=0.0 최적) |

### 4.2 Cross-Asset 학습 전략

| 전략 | 사용 연구 | 본 연구 |
|:---|:---|:---|
| **개별 자산 학습** | Hansen et al. (2021) | 비교 대상 |
| **Pooling (풀링)** | Christensen et al. (2022) | V50 Tuned (11개 자산 풀링) |
| **자산군별 적응** | Fan et al. (2025) | V71 Ridge (Equity/Bond/Commodity별 alpha) |
| **옵션 기반 variance portfolio** | Heston et al. (2023) | **미사용** (옵션 데이터 불필요) |

### 4.3 검증 방법론

| 방법 | 사용 연구 | 본 연구 |
|:---|:---|:---|
| Rolling Window Validation | Benhenda et al. (2026) | 80/20 시간순 분할 |
| Purged K-Fold CV | Lopez de Prado (2018) | 5-Fold TimeSeriesSplit |
| DM Test (Newey-West HAC) | Diebold & Mariano (1995) | 모든 벤치마크 대비 p<0.001 |
| Multi-Seed 실험 | - | 5-seed (42, 123, 456, 789, 2024) |

---

## 5. 성능 비교 종합 (전방 RV, 22d 기준)

| 모델 | 피처 수 | 피처 유형 | 방법론 | 풀링 R² (22d) | 특이사항 |
|:---|:---|:---|:---|:---|:---|
| GARCH-MIDAS | ~5 | 거시변수 | 이중 성분 | 0.55 | 문헌 보고치 |
| **HAR-RV (Corsi, 2009)** | **3** | RV lags (d,w,m) | OLS/Ridge | **0.761** | HAR 표준, V36 기반 |
| **HAR-CJ (ABD, 2007)** | **6** | 연속+Jump RV | OLS/Ridge | **0.752** | Jump 분리가 일간에서 제한적 |
| Carr+(2019) ML | 79 | 옵션 IV surface | Ridge, FNN | 0.39 | SPX 단일, 옵션 데이터 필수 |
| RF (CSV, 2023) | 37 | HAR+HF+IV+Alt | Random Forest | 0.770 | 트리 모델, 365d: 0.705 최고 |
| LASSO-HAR (AK, 2016) | 37 | HAR+HF+IV+Alt | LASSO (L1) | 0.790 | Ridge와 동등, L1 vs L2 차이 미미 |
| V50 Tuned LSTM (본 연구) | 3 | Range (RS, GK, RC) | Bi-LSTM + Attention | 0.780 | 60d: 0.796으로 중기 최고 |
| V71 Ensemble (본 연구) | 37 | HAR+HF+IV+Alt | Ridge+XGBoost 앙상블 | 0.803 | 단기(1~22d) 우수 |
| **V74 ElasticNet (본 연구)** | **59** | **HAR+HF+IV+New** | **ElasticNet + Stacking** | **0.807** | **LASSO 대비 +2.1%** |

---

## 6. 예측 지평별 최적 모델 (본 연구 발견)

| 구간 | 최적 모델 | 피처 수 | 풀링 R² | 핵심 이유 | 선행연구 대비 |
|:---|:---|:---|:---|:---|:---|
| **단기 (1~22d)** | Ridge+XGBoost 앙상블 (37feat) | 37 | 0.803 | 다수 피처의 선형 결합 + 앙상블 | Audrino+(2016) 상회 |
| **중기 (60~90d)** | V50 Tuned LSTM (3feat) | 3 | 0.796 | LSTM + 소수 핵심 피처의 비선형 패턴 | **본 연구 고유 발견** |
| **장기 (120~365d)** | RF / HAR-3 | 37 / 3 | 0.611~0.705 | 트리 모델의 장기 비선형 포착 | Christensen+(2023) 계열 |

---

## 7. 본 연구의 차별점 요약

| 차별점 | 설명 | 대비 선행 연구 |
|:---|:---|:---|
| **옵션 데이터 불필요** | VIX 파생 지표 + OHLCV만으로 cross-asset 예측 | Carr+(2019): 79개 옵션, Heston+(2023): 자산별 옵션 |
| **Multi-Horizon 체계 비교** | 1d~365d 8개 지평에서 7개 모델 비교 | 대부분 단일 지평 (1d 또는 22d) |
| **피처 수 vs 예측 지평 관계** | 단기: 많은 피처(37) 유리, 중기: 적은 피처(3) 유리 | 체계적 분석 부재 |
| **Cross-Asset Pooling + 자산군 적응** | Pooling(V50) + Asset-Adaptive(V71) 이중 전략 | Fan+(2025): 섹터 적응이지만 VRP 아님 |
| **Cross-Asset Pooling + 자산군 적응** | Pooling(V50) + Asset-Adaptive(V71) 이중 전략 | Fan+(2025): 섹터 적응이지만 VRP 아님 |
| **내재적 해석 가능성** | Attention 가중치 자체가 해석 도구 | Lo+(2023): SHAP/LIME 사후 해석 |
| **ML/DL 한계 명확화** | Pooling 데이터에서 DL 실패 vs ML(ElasticNet) 성공 규명 | 대부분 DL 우위만 보고 (Negative Result 중요) |

---

## 8. 최신 실험 결과 (V73~V74)

기존 V71 Ensemble(0.803) 및 LASSO(0.790)의 한계를 넘기 위해 수행된 심화 실험 결과:

1.  **R² 0.80 벽 돌파**: 59개 확장 피처(비선형, 상호작용항)와 ElasticNet(L1+L2)의 결합으로 **R²=0.807** 달성.
2.  **딥러닝의 구조적 한계 규명**: 11개 자산이 혼합된 Pooling 데이터에서는 시계열 연속성이 깨져 LSTM/Transformer가 실패(음의 R²)함을 입증.
3.  **최적 조합**: **Feature Engineering (59개) + Robust Linear Model (ElasticNet)**이 복잡한 딥러닝보다 금융 변동성 예측에 효과적임.

---

## 참고문헌

> 학술 저널 약어: *RFS*=Review of Financial Studies, *JFEC*=J. Financial Econometrics, *JFE*=J. Financial Economics, *RIBF*=Research in International Business and Finance, *JIMF&M*=J. International Financial Markets, Institutions and Money

### 핵심 문헌 (본 연구 벤치마크)
1. Corsi, F. (2009). HAR-RV. *JFEC*, 7(2), 174-196.
2. Andersen, T., Bollerslev, T., Diebold, F. (2007). HAR-CJ. *Econometrica*, 75(4).
3. Audrino, F. & Knaus, S. (2016). LASSO-HAR. *JFEC*, 14(2).
4. Christensen, K., Siggaard, M., Veliyev, B. (2023). RF. *JFE*, 150(2).

### VRP 이론
5. Carr, P. & Wu, L. (2009). Variance risk premiums. *RFS*, 22(3).
6. Bekaert, G. & Hoerova, M. (2014). VIX² decomposition. *J. Econometrics*, 183(2).
7. Hansen, P.R. et al. (2021). Realized GARCH. *JFEC* (2024).

### Cross-Asset
8. Heston, S.L. et al. (2023). Cross-asset VRP. *AEA 2024*.
9. Ornelas, J.R.H. et al. (2018). Commodity VRP spillover. *BIS WP 619*.

### 딥러닝
10. Carr, P., Wu, L., Zhang, Z. (2019). ML + Options. *arXiv: 1909.10035*.
11. Roszyk, K. & Slepaczuk, R. (2024). Hybrid LSTM-GARCH. *arXiv: 2407.16780*.
12. Xu, B. et al. (2024). GINN. *ICAIF '24*.
13. Ramos-Perez, E. et al. (2021). Multi-Transformer. *Applied Soft Computing*, 109.

### 방법론
14. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
15. Chun, D., Cho, H., Ryu, D. (2025). High-dimensional ML volatility. *RIBF*.
16. Fan, J., Wu, X., Yang, Z. (2025). Adaptive MTL. *arXiv: 2507.16433*.
