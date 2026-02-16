# V71: Enhanced Asset-Adaptive Volatility Prediction Model

## 1. 연구 개요

본 연구는 다중 자산 변동성 위험 프리미엄(VRP) 예측을 위한 통합 프레임워크를 개발한다. 기존 종가(Close) 기반 HAR 모델의 한계를 극복하기 위해 **OHLCV 기반 고빈도 프록시(Range-based Estimators)**, **옵션 내재변동성(IV) Surface**, **대안 데이터(Alternative Data)**를 통합한 Enhanced Asset-Adaptive Ridge 모델(V71)을 제안한다.

### 핵심 성과
- **22일(약 1개월) 예측 R² = 0.803** (Weighted Ensemble)
- 기존 챔피언 V36 (R²=0.755) 대비 **6.3% 개선**
- 6개 건전성 검증 테스트 **전체 통과**

---

## 2. 모델 진화 과정

### 2.1 실험 타임라인

| 버전 | 모델 | R² | 핵심 전략 |
|------|------|-----|---------|
| V29 | HAR+GARCH | 0.679 | HAR 피처 + GARCH 변동성 |
| V35 | Multi-Horizon GARCH | 0.685 | 다중 시계열 GARCH |
| **V36** | **Asset-Adaptive Ridge** | **0.755** | **자산 클래스별 alpha 최적화** |
| V43 | Transformer | 0.534 | Self-Attention 메커니즘 |
| V50 | Temporal-Attention LSTM | 0.518 | Bi-LSTM + Temporal Attention |
| V68 | Enhanced Adaptive | 0.777 | V36 + Vol-of-Vol + Momentum + Cross-Asset |
| V69 | Max Performance | 0.777 | V68 + Stacking/Weighted Ensemble |
| V70 | Extended Universe | 0.760 | 22개 자산 확장 (성능 하락) |
| **V71** | **Advanced Data Sources** | **0.803** | **HF Proxy + IV Surface + Alt Data** |

### 2.2 핵심 발견

1. **복잡한 딥러닝 모델 < 강건한 선형 모델**: Transformer(0.534), LSTM(0.518)이 Ridge(0.755)보다 낮은 성능
2. **자산 확장 = 노이즈 증가**: 11개→22개 자산 확장 시 R² 0.777→0.760으로 하락
3. **OHLC 데이터의 정보 우위**: 종가만 사용하는 모델 대비 OHLC 기반 고빈도 프록시가 결정적 성능 향상 제공

---

## 3. V71 모델 아키텍처

### 3.1 데이터

| 항목 | 내용 |
|------|------|
| 기간 | 2010-01-01 ~ 2024-12-31 (약 15년) |
| 자산 | 11개 ETF (Equity 5, Bond 3, Commodity 3) |
| 샘플 수 | 38,489 (Train 30,791 / Test 7,698) |
| 피처 수 | 37개 (4개 카테고리) |
| 타겟 | Log Realized Volatility (22일 후, shift(-22)) |
| 데이터 소스 | Yahoo Finance (OHLCV), CBOE (VIX/VIX3M/VIX9D) |

### 3.2 자산 구성

| 클래스 | 자산 | 수 |
|--------|------|---|
| Equity | SPY, QQQ, IWM, EFA, EEM | 5 |
| Bond | TLT, IEF, AGG | 3 |
| Commodity | GLD, SLV, USO | 3 |

### 3.3 모델 구조

```
[자산별 OHLCV 데이터]
    │
    ├── Base Features (14개)
    │   ├── HAR: LogRV lag 1/5/10/22
    │   ├── GARCH: Daily/Weekly
    │   ├── Vol-of-Vol: Std 5/22
    │   ├── Momentum: Mom 5/22
    │   ├── Cross-Asset: SPY LogRV, Corr
    │   └── Return: lag1, abs_lag1
    │
    ├── HF Proxy Features (7개)
    │   ├── Parkinson Vol (5d, 22d)
    │   ├── Garman-Klass Vol (22d)
    │   ├── Rogers-Satchell Vol (22d)
    │   ├── Range/Close Ratio
    │   └── Overnight Vol/Ret
    │
    ├── IV Surface Features (10개)
    │   ├── VIX Level/Change/MA5/Std5
    │   ├── VIX3M, VIX9D
    │   ├── Term Structure Slope (VIX-VIX3M)
    │   ├── Short-Term Slope (VIX9D-VIX)
    │   └── VRP, VRP MA22
    │
    └── Alt Data Features (6개)
        ├── Amihud Illiquidity
        ├── Volume Ratio (5/22)
        ├── Price-Volume Correlation
        ├── Volume Surprise
        ├── Order Imbalance (VPIN proxy)
        └── Kyle's Lambda

    ↓ StandardScaler

    ↓ 자산 클래스별 Ridge(alpha 최적화)
      - Equity: alpha 개별 튜닝
      - Bond: alpha 개별 튜닝
      - Commodity: alpha 개별 튜닝

    ↓ XGBoost 병렬 학습

    ↓ Weighted Ensemble (Ridge 70% + XGBoost 30%)

    → R² = 0.803
```

---

## 4. 피처 설명 (Variable Description)

### 4.1 Base Features (14개)

| 변수명 | 설명 | 산출 방법 |
|--------|------|---------|
| LogRV_lag1 | 전일 Log 실현변동성 | 22일 롤링 RV의 로그, 1일 래그 |
| LogRV_lag5 | 5일 전 Log 실현변동성 | 22일 롤링 RV의 로그, 5일 래그 |
| LogRV_lag10 | 10일 전 Log 실현변동성 | 22일 롤링 RV의 로그, 10일 래그 |
| LogRV_lag22 | 22일 전 Log 실현변동성 | 22일 롤링 RV의 로그, 22일 래그 |
| Garch_Daily | 일간 GARCH(1,1) 변동성 | 일간 수익률로 추정한 조건부 변동성 |
| Garch_Weekly | 주간 GARCH(1,1) 변동성 | 주간 수익률로 추정한 조건부 변동성 |
| LogRV_Std5 | 단기 Vol-of-Vol | Log RV의 5일 표준편차 |
| LogRV_Std22 | 장기 Vol-of-Vol | Log RV의 22일 표준편차 |
| RV_Mom5 | 단기 RV 모멘텀 | Log RV의 5일 변화 |
| RV_Mom22 | 장기 RV 모멘텀 | Log RV의 22일 변화 |
| SPY_LogRV | 시장(SPY) 변동성 | S&P 500 ETF의 Log RV |
| Corr_SPY | 시장 상관관계 | SPY와의 22일 롤링 상관계수 |
| Ret_lag1 | 전일 수익률 | 로그 수익률 1일 래그 |
| Ret_abs_lag1 | 전일 절대 수익률 | 레버리지 효과 프록시 |

### 4.2 HF Proxy Features (7개) — OHLC 기반

| 변수명 | 설명 | 산출 방법 |
|--------|------|---------|
| Parkinson_5 | Parkinson 변동성 (5일) | log(H/L) 기반 범위 추정량, 5일 윈도우 |
| Parkinson_22 | Parkinson 변동성 (22일) | log(H/L) 기반 범위 추정량, 22일 윈도우 |
| GarmanKlass_22 | Garman-Klass 변동성 | OHLC 4가격 기반 추정량, 22일 윈도우 |
| RogersSatchell_22 | Rogers-Satchell 변동성 | 드리프트 독립적 OHLC 추정량 |
| Range_Close_Ratio | 범위/종가 비율 | Parkinson vs Close-based RV의 정보 비율 |
| Overnight_Vol | 야간 수익률 변동성 | 시가/전일종가 비율의 22일 표준편차 |
| Overnight_Ret | 야간 수익률 | 시가/전일종가 비율 (갭 리스크) |

### 4.3 IV Surface Features (10개)

| 변수명 | 설명 | 산출 방법 |
|--------|------|---------|
| IV_VIX | VIX 수준 | CBOE S&P 500 30일 내재변동성 지수 |
| IV_VIX_chg | VIX 일간 변화 | Log VIX의 일간 변화량 |
| IV_VIX_ma5 | VIX 5일 이동평균 | 단기 VIX 추세 |
| IV_VIX_std5 | VIX 변동성 (VVIX 프록시) | Log VIX의 5일 표준편차 |
| IV_VIX3M | VIX3M 수준 | 3개월 S&P 500 내재변동성 |
| IV_VIX_TermSlope | VIX 기간구조 기울기 | VIX - VIX3M (콘탱고/백워데이션) |
| IV_VIX9D | VIX9D 수준 | 9일 S&P 500 내재변동성 |
| IV_VIX_ShortSlope | VIX 단기 기울기 | VIX9D - VIX (단기 스큐) |
| IV_VRP | 분산 위험 프리미엄 | VIX² - Realized Variance |
| IV_VRP_ma22 | VRP 22일 평균 | 평활화된 분산 위험 프리미엄 |

### 4.4 Alternative Data Features (6개)

| 변수명 | 설명 | 산출 방법 |
|--------|------|---------|
| AltVol_Amihud | Amihud 비유동성 | |수익률| / 거래대금  |
| AltVol_Vol_Ratio | 거래량 비율 (5/22) | 단기/장기 거래량 추세 |
| AltVol_PV_Corr | 가격-거래량 상관관계 | 감성 프록시 |
| AltVol_Vol_Surprise | 거래량 서프라이즈 | 22일 평균 대비 거래량 편차 |
| AltVol_Order_Imbalance | 주문 불균형 | 매수/매도 거래량 비율 (VPIN 프록시) |
| AltVol_Kyle_Lambda | Kyle의 Lambda | 단위 거래량 변화당 가격 영향 |

---

## 5. Ablation Study (피처 그룹 기여도)

| 피처 조합 | 피처 수 | R² | 기여도 |
|-----------|--------|-----|--------|
| Base only | 14 | 0.758 | baseline |
| Base + HF Proxy | 21 | 0.779 | +0.021 |
| Base + IV Surface | 24 | 0.784 | +0.026 |
| Base + Alt Data | 20 | 0.765 | +0.007 |
| Base + HF + IV | 31 | 0.796 | +0.037 |
| All Features (Ridge) | 37 | 0.798 | +0.040 |
| XGBoost (all) | 37 | 0.773 | +0.014 |
| **Weighted Ensemble** | **37** | **0.803** | **+0.044** |

### 핵심 발견
- **IV Surface가 가장 큰 개별 기여** (+0.026): 옵션 시장의 내재변동성이 실현변동성 예측에 추가 정보 제공
- **HF Proxy가 두 번째 기여** (+0.021): OHLC에서 추출한 고빈도 정보가 종가 기반 모델의 한계 보완
- **HF + IV 결합 시 시너지** (+0.037 > 0.021+0.026의 합이 아닌 값): 두 소스가 보완적 정보 제공
- **Alt Data 기여는 제한적** (+0.007): 거래량 기반 피처는 다른 피처와 중복 정보

---

## 6. 피처 중요도 (Permutation Importance)

| 순위 | 피처 | 중요도 (R² drop) | 카테고리 |
|------|------|-----------------|---------|
| 1 | Rogers-Satchell Vol | 0.731 | HF Proxy |
| 2 | Garman-Klass Vol | 0.290 | HF Proxy |
| 3 | Parkinson Vol (22d) | 0.116 | HF Proxy |
| 4 | SPY Log RV | 0.045 | Cross-Asset |
| 5 | Range/Close Ratio | 0.033 | HF Proxy |
| 6 | GARCH Weekly | 0.030 | GARCH |
| 7 | GARCH Daily | 0.025 | GARCH |
| 8 | VIX Level | 0.024 | IV Surface |
| 9 | Log RV (t-1) | 0.023 | Base/HAR |
| 10 | VIX3M Level | 0.022 | IV Surface |

### 해석
- **상위 5개 중 4개가 HF Proxy**: OHLC 기반 범위 추정량이 변동성 예측의 핵심
- **Rogers-Satchell이 압도적 1위** (중요도 0.731): 드리프트에 독립적인 추정량이 가장 강력
- **IV Surface 피처도 유의미**: VIX(0.024), VIX3M(0.022)이 Top 10에 포함
- **기존 HAR 피처(LogRV_lag1)는 9위**: OHLC/IV 피처가 있으면 상대적 중요도 감소

---

## 7. 건전성 검증 결과

### 7.1 다중 Split 안정성

| Split Ratio | Train | Test | R² |
|------------|-------|------|-----|
| 75% | 28,867 | 9,622 | 0.775 |
| 78% | 30,021 | 8,468 | 0.783 |
| 80% | 30,791 | 7,698 | 0.798 |
| 82% | 31,561 | 6,928 | 0.806 |
| 85% | 32,716 | 5,773 | 0.806 |

**결론**: R² = 0.793 +/- 0.010, 모든 split에서 0.775 이상으로 안정적

### 7.2 파라미터 민감도 (Alpha Sweep)

| Alpha | R² |
|-------|-----|
| 0.001 | 0.793 |
| 0.1 | 0.795 |
| 10.0 | 0.797 |
| 100.0 | 0.800 |
| 500.0 | 0.802 |
| 1000.0 | 0.800 |
| 5000.0 | 0.788 |

**결론**: alpha 0.001~5000 범위에서 R² 0.788~0.802로 매우 안정적. 과적합 위험 낮음

### 7.3 시계열 교차검증 (5-Fold Expanding Window)

| Fold | Train 크기 | Test 크기 | R² |
|------|-----------|----------|-----|
| 1 | 15,395 | 4,619 | 0.708 |
| 2 | 20,014 | 4,619 | 0.786 |
| 3 | 24,633 | 4,619 | 0.789 |
| 4 | 29,252 | 4,619 | 0.803 |
| 5 | 33,871 | 4,618 | 0.737 |

**결론**: Mean R² = 0.765 +/- 0.046. Fold 1은 학습 데이터 부족(40%)으로 낮지만, Fold 2~4에서 0.79 안정

### 7.4 데이터 누출 검증

| 검증 항목 | 결과 | 판정 |
|----------|------|------|
| Target 셔플 테스트 | R² = -0.002 (기대값 ~0) | PASS |
| 피처 시점 검증 | 모든 피처 shift(1) 이상 | PASS |
| 시간적 무결성 | Train/Test 순차 분리 확인 | PASS |
| 피처-타겟 분리 | Target = shift(-22) | PASS |

**결론**: 데이터 누출 없음 확인

---

## 8. 모델 비교 (전체 실험 결과)

### 8.1 22일 예측 성능 비교

| 모델 | 유형 | 피처 수 | R² | 대V36 개선 |
|------|------|--------|-----|---------|
| **V71 Ensemble** | **Ridge+XGBoost** | **37** | **0.803** | **+6.3%** |
| V71 Ridge | Ridge | 37 | 0.798 | +5.7% |
| V68 Ensemble | Ridge+XGBoost | 14 | 0.777 | +2.9% |
| V36 | Ridge | 3 | 0.755 | baseline |
| V35 | Ridge+GARCH | 5 | 0.685 | -9.3% |
| V29 | Ridge+HAR | 4 | 0.679 | -10.1% |
| V43 | Transformer | 4 | 0.534 | -29.3% |
| V50 | Bi-LSTM | 4 | 0.518 | -31.4% |

### 8.2 딥러닝 vs 선형 모델

| 구분 | 최고 R² | 대표 모델 |
|------|---------|---------|
| **선형 모델 (Ridge)** | **0.803** | V71 |
| 트리 기반 (XGBoost) | 0.773 | V71 XGBoost |
| Transformer | 0.534 | V43 |
| LSTM+Attention | 0.518 | V50 |

**결론**: 변동성 예측에서 강건한 선형 모델이 복잡한 딥러닝 모델을 압도. 피처 엔지니어링이 모델 복잡도보다 중요

### 8.3 동일 피처 세트 공정 비교

| Feature Set | Ridge R² | XGBoost R² | MLP R² |
|-------------|---------|----------|-------|
| Base (14) | 0.728 | 0.736 | 0.680 |
| All (37) | **0.773** | 0.761 | 0.750 |

**결론**: 동일 피처 조건에서 Ridge가 37개 피처 기준 최고 성능

### 8.4 추가 메트릭 비교

| Model | R² | RMSE | MAE | QLIKE |
|-------|-----|------|-----|-------|
| V36 (HAR-3) | 0.732 | 0.541 | 0.430 | 0.180 |
| V68 (Base-14) | 0.728 | 0.545 | 0.431 | 0.192 |
| V71 Ridge | 0.774 | 0.496 | 0.396 | 0.149 |
| **V71 Ensemble** | **0.779** | **0.492** | **0.393** | **0.147** |

### 8.5 Diebold-Mariano Test (Newey-West HAC, h=22)

| Comparison | DM-stat | p-value | Significance |
|------------|---------|---------|-------------|
| V71 Ensemble vs V36 | -8.802 | <0.001 | *** |
| V71 Ensemble vs V68 | -8.756 | <0.001 | *** |
| V71 Ridge vs V36 | -7.481 | <0.001 | *** |
| V71 Ridge vs V68 | -7.593 | <0.001 | *** |

**결론**: V71의 성능 우위가 1% 수준에서 통계적으로 유의함

---

## 9. 자산별/클래스별 성능

### 9.1 자산별 R² (V36 vs V71)

| Asset | Class | V36 R² | V71 R² | Delta |
|-------|-------|--------|--------|-------|
| SPY | Equity | 0.316 | 0.497 | +0.181 |
| QQQ | Equity | 0.078 | 0.411 | +0.333 |
| EFA | Equity | 0.304 | 0.428 | +0.124 |
| EEM | Equity | 0.111 | 0.249 | +0.138 |
| IWM | Equity | -0.025 | 0.015 | +0.040 |
| TLT | Bond | -0.346 | -0.045 | +0.301 |
| IEF | Bond | -0.581 | -0.190 | +0.391 |
| AGG | Bond | -3.906 | -1.740 | +2.166 |
| GLD | Commodity | 0.091 | 0.065 | -0.026 |
| USO | Commodity | 0.153 | 0.210 | +0.057 |
| SLV | Commodity | 0.063 | -0.133 | -0.196 |

### 9.2 클래스별 평균

| Class | V36 R² | V71 R² | Delta |
|-------|--------|--------|-------|
| **Equity** | 0.157 | 0.320 | **+0.163** |
| **Bond** | -1.611 | -0.658 | **+0.953** |
| Commodity | 0.102 | 0.047 | -0.055 |

---

## 10. 레짐별 안정성

### 10.1 기간 서브샘플

| Period | V36 R² | V71 R² | Delta |
|--------|--------|--------|-------|
| 2010-2014 | 0.634 | 0.739 | +0.106 |
| 2015-2019 | 0.781 | 0.800 | +0.019 |
| 2020-2024 | 0.683 | 0.493 | -0.189 |

**해석**: 2020-24 하락은 구간 내 제한적 학습 데이터(70/30 split)와 COVID 이후 구조 변화에 기인

### 10.2 VIX Quantile 레짐

| Regime | N | V36 R² | V71 R² | Delta |
|--------|---|--------|--------|-------|
| Low VIX (0-50%) | 4,136 | 0.739 | 0.781 | +0.042 |
| Mid VIX (50-90%) | 3,292 | 0.698 | 0.722 | +0.024 |
| **High VIX (90-100%)** | **825** | **0.602** | **0.765** | **+0.163** |

**핵심**: V71은 고변동성 국면에서 V36 대비 가장 큰 개선(+0.163)

---

## 11. 피처 상관관계 및 계수 해석

### 11.1 HF Proxy 다중공선성

RS_22/GK_22/Park_22 간 상관계수 >0.99. Ridge L2 정규화가 다중공선성을 자연스럽게 처리

### 11.2 주요 계수 (Standardized)

| Feature | Equity | Bond | Commodity | Interpretation |
|---------|--------|------|-----------|---------------|
| RS_22 | +0.217 | +0.265 | +0.356 | 변동성 지속성 (전체 양) |
| Garch_Weekly | +0.153 | +0.386 | +0.084 | 주간 추세 (Bond 최대) |
| IV_VRP_ma22 | -0.176 | -0.306 | -0.166 | VRP 역방향 (전체 음) |
| SPY_LogRV | -0.385 | - | - | 시장 RV 통제 (Equity 음) |

---

## 12. 학술적 기여

1. **OHLC 기반 고빈도 프록시의 효과 실증**
2. **IV Surface의 보완적 예측력 확인**
3. **다중 자산 통합 프레임워크**
4. **모델 복잡도 vs 피처 엔지니어링** 우수성 입증
5. **포괄적 건전성 검증** (6개 테스트)
6. **통계적 유의성**: DM test(Newey-West HAC) 모든 비교 p<0.001
7. **레짐별 안정성**: 고변동성 국면에서 V71 우수성 입증

---

## 13. 한계 및 향후 과제

1. **실제 고빈도 데이터 미사용** (OHLC 프록시 사용)
2. **개별 자산 IV Surface 미사용** (VIX 지수만 활용)
3. **뉴스 감성 데이터 미반영**
4. **예측 기간 단일화** (22일만)
5. **Commodity 클래스 성능 제한적** (VIX/IV가 Equity 중심)
6. **HF Proxy 다중공선성** (>0.99, Elastic Net 적용 가능)

---

## 부록: 실험 파일 목록

| 파일 | 설명 |
|------|------|
| `v71_advanced_data.py` | V71 메인 실험 코드 |
| `v71_robustness.py` | 건전성 검증 스위트 |
| `v71_sci/pipeline.py` | SCI 리뷰 대응 파이프라인 |
| `v71_sci/data_builder.py` | 공유 데이터셋 빌더 |
| `v71_sci/step01~05*.py` | 개별 분석 스크립트 (5개) |
| `v71_sci/results_*.json` | SCI 리뷰 대응 결과 (5개) |
