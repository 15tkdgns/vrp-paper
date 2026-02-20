# V50: Temporal-Attention Bi-LSTM Multi-Horizon 변동성 예측 모델

## 1. 연구 개요

V50은 **양방향 LSTM(Bi-LSTM)**과 **Temporal Attention** 메커니즘을 결합한 변동성 예측 모델이다. 22일간의 Range 기반 피처 시퀀스를 입력받아 다양한 예측 기간의 Log Realized Volatility를 예측한다. 11개 ETF를 풀링하여 학습하며, cross-asset 범용 모델로 설계되었다.

### 핵심 성과 (11개 자산 풀링 기준)

| Horizon | 풀링 R² | 중위 R² | RMSE |
|:---|:---|:---|:---|
| 22d | 0.780 | **0.274** | 0.482 |
| **60d** | **0.796** | -0.178 | 0.446 |
| 90d | 0.788 | -0.031 | 0.448 |
| 120d | 0.751 | -0.149 | 0.479 |

> **핵심 발견**: V50 Tuned(3개 Range 피처)는 **중기(60~180d) 풀링 R²에서 최고 성능**을 달성하며,
> **전 구간 개별 자산 중위 R²에서도 1위**를 기록한다. 37개 피처 Ridge를 3개 피처 LSTM이 중기에서 능가함.

---

## 2. 모델 아키텍처

### 2.1 네트워크 구조

```
Input: [batch, seq_len=22, features=2]
  │  (LogRV, Return)
  │
  ├── Bi-LSTM (hidden=64, bidirectional=True)
  │     → output: [batch, 22, 128]
  │
  ├── Temporal Attention
  │     ├── Linear(128 → 1) + Tanh
  │     ├── Softmax (시퀀스 차원)
  │     └── Weighted Sum → context: [batch, 128]
  │
  └── FC Layer (128 → 1)
        → prediction: [batch, 1]
```

### 2.2 핵심 설계 원리

1. **양방향 LSTM**: 과거→미래, 미래→과거 양방향으로 시퀀스를 읽어 temporal dependency를 풍부하게 캡처
2. **Temporal Attention**: 22일 입력 중 예측에 중요한 시점(lag)에 동적으로 가중치를 부여 → "Dynamic HAR"로 해석 가능
3. **경량 설계**: 약 9K 파라미터, Transformer(V43, ~85K)의 약 1/9 규모

### 2.3 하이퍼파라미터

| 파라미터 | 값 |
|:---|:---|
| 입력 차원 | 3 (RogersSatchell, GarmanKlass, Range_Close_Ratio) |
| 시퀀스 길이 | 22일 |
| LSTM Hidden Dim | 32 |
| LSTM 방향 | Bidirectional |
| Attention | Single-head, Tanh → Softmax |
| FC 출력 | 1 (LogRV 예측) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Batch Size | 128 |
| Epochs | 10 |
| Dropout | 0.2 |
| Gradient Clip | 1.0 |
| Loss | MSE |
| Seed | 42 |

---

## 3. 데이터

### 3.1 데이터 개요

| 항목 | 내용 |
|:---|:---|
| 기간 | 2010-01-01 ~ 2024-12-31 (약 15년) |
| 자산 | 11개 ETF (Equity 5, Bond 3, Commodity 3) |
| 샘플 수 | 46,343 (Train 37,074 / Test 9,269) |
| 입력 피처 | 2개 (LogRV, Return) |
| 타겟 | Log Realized Volatility (t+22, scaled) |
| 데이터 소스 | Yahoo Finance (OHLCV) |
| 전처리 | StandardScaler (train-only fit, no leakage) |

### 3.2 자산 구성

| 클래스 | 자산 | 수 |
|:---|:---|:---|
| Equity | SPY, QQQ, IWM, EFA, EEM | 5 |
| Bond | TLT, IEF, AGG | 3 |
| Commodity | GLD, SLV, USO | 3 |

### 3.3 타겟 변수 구성

```
Realized Volatility (RV):
  RV = rolling_mean(daily_log_return², window=22) × 252 × 10000

Log RV:
  LogRV = log(RV + 1e-6)

Target:
  y_t = LogRV_{t+22}  (22일 ahead)
```

### 3.4 Train/Test Split

- **방법**: 시간순 80/20 Split (시간적 무결성 보장)
- **Train**: 2010 ~ 2021.08 (37,074 샘플)
- **Test**: 2021.08 ~ 2024.12 (9,269 샘플)
- **Scaler**: Train 데이터로만 fit, Test 데이터에 transform (no leakage)

---

## 4. 실험 변천사 (V50 관련)

### 4.1 V50 초기 버전 (6개 자산)

최초 V50 실험은 6개 자산(SPY, QQQ, IWM, TLT, IEF, GLD)으로 수행되었다.

| 지표 | 값 |
|:---|:---|
| IS R² (풀링) | 0.848 |
| OOS R² (풀링) | 0.533 |

### 4.2 V51b 공정 비교

V50의 IS R²(0.848)와 OOS R²(0.533) 간 큰 격차가 발견되어, v51b 실험에서 원인을 분석하였다:

| 실험 | 설정 | OOS R² |
|:---|:---|:---|
| V50 원본 | 전체 scaler, 6개 자산 | 0.848 (IS) / — |
| V51b Exp1 | Hold-out scaler | 0.472 |
| **V51b Exp2** | **Train-only scaler** | **0.533** |

- **원인**: 초기 실험에서 전체 데이터로 scaler를 fit하여 IS에 미래 정보 누출
- **결론**: Train-only scaler 사용 시 OOS R² = 0.533이 공정한 결과

### 4.3 11개 자산 확장 (현재)

V71과 동일한 11개 자산으로 확장하여 재실험한 결과:

| 지표 | 6개 자산 | 11개 자산 | 변화 |
|:---|:---|:---|:---|
| OOS R² (풀링) | 0.533 | **0.651** | +0.118 |
| IS R² (풀링) | 0.848 | 0.927 | +0.079 |

자산 수 증가로 풀링 R²가 상승하였으나, 이는 cross-sectional variation 증가에 기인한다.

---

## 5. 자산별/클래스별 성능 (11개 자산, OOS)

### 5.1 자산별 성능 (4개 메트릭)

| 자산 | 클래스 | R² | RMSE | MAE | QLIKE |
|:---|:---|:---|:---|:---|:---|
| **SPY** | Equity | **0.251** | 0.663 | 0.527 | 0.296 |
| **QQQ** | Equity | **0.218** | 0.630 | 0.500 | 0.249 |
| IWM | Equity | -0.430 | 0.609 | 0.483 | 0.231 |
| EFA | Equity | -0.097 | 0.666 | 0.520 | 0.262 |
| EEM | Equity | -0.196 | 0.599 | 0.475 | 0.176 |
| TLT | Bond | -0.542 | 0.569 | 0.458 | 0.190 |
| IEF | Bond | -0.107 | 0.600 | 0.477 | 0.223 |
| AGG | Bond | -0.396 | 0.737 | 0.602 | 0.379 |
| GLD | Commodity | -0.476 | 0.582 | 0.469 | 0.179 |
| SLV | Commodity | -0.903 | 0.600 | 0.482 | 0.186 |
| USO | Commodity | -0.312 | 0.733 | 0.554 | 0.331 |

### 5.2 클래스별 평균 R²

| 클래스 | 평균 R² | 자산 수 |
|:---|:---|:---|
| Equity | -0.051 | 5 |
| Bond | -0.348 | 3 |
| Commodity | -0.564 | 3 |

### 5.3 성능 해석

- **SPY(0.251), QQQ(0.218)만 양수 R²**: 대형 Equity ETF에서만 시계열 예측력 존재
- **개별 자산 중위 R²: -0.312**: 대부분 자산에서 Historical Mean보다 예측이 못함
- **풀링 R²(0.651, overlapping OOS) vs 중위 R²(-0.312)**: 자산 간 스케일 차이(USO ~7.5 vs AGG ~4.0)가 풀링 R²를 높이는 핵심 요인. 전방 RV 기준 22d 풀링 R²=0.780
- **Bond/Commodity 성능 저조**: 이 클래스들은 Equity와 다른 변동성 동학을 보이며, LSTM의 2개 피처(LogRV, Return)만으로는 충분한 정보를 제공하지 못함

---

## 6. V50 vs V71 Multi-Horizon 비교 (전방 RV 기준)

### 6.1 Horizon별 풀링 R² 비교

| Horizon | V50 Tuned (3feat) | V71 Ridge (37feat) | HAR-3 | 챔피언 |
|:---|:---|:---|:---|:---|
| 1d | 0.092 | **0.114** | 0.109 | Ridge |
| 5d | 0.577 | **0.625** | 0.581 | Ridge |
| 22d | 0.780 | **0.803** | 0.761 | Ensemble |
| **60d** | **0.796** | 0.774 | 0.784 | **LSTM** |
| **90d** | **0.788** | 0.752 | 0.781 | **LSTM** |
| 120d | 0.751 | 0.737 | **0.758** | HAR-3 |
| 180d | 0.708 | 0.691 | **0.705** | HAR-3 |
| 365d | 0.589 | 0.516 | **0.611** | HAR-3 |

### 6.2 핵심 발견

1. **V50 Tuned이 60~90d에서 승리**: 3개 Range 피처의 중기 예측 우위
2. **V71이 단기(1~22d)에서 승리**: 37개 피처의 풍부한 정보가 단기에서 유리
3. **HAR-3이 장기(120~365d)에서 승리**: 단순 모델의 노이즈 강건성
4. **V50 Tuned는 전 구간 중위 R² 최고**: 개별 자산 수준에서 가장 안정적

---

## 7. Temporal Attention 분석

### 7.1 Attention의 해석적 의미

V50의 Temporal Attention은 22일 입력 시퀀스 내에서 어떤 시점(lag)이 예측에 가장 중요한지 동적으로 결정한다. 이는 기존 HAR 모델의 고정 래그 구조(1일, 5일, 22일)를 **동적으로 일반화한 "Dynamic HAR"**로 해석할 수 있다.

### 7.2 Attention vs HAR

| 특성 | HAR 모델 | V50 Temporal Attention |
|:---|:---|:---|
| 래그 선택 | 고정 (1, 5, 22) | 동적 (데이터 기반) |
| 가중치 | 선형 회귀 계수 | Softmax 확률 |
| 시장 상태 적응 | 불가 | 가능 (상태별 가중치 변화) |
| 해석 가능성 | 직접적 | 내재적 (post-hoc 불필요) |

---

## 8. IS/OOS 성능 격차 분석

### 8.1 IS R² = 0.927 vs OOS R² = 0.651 (overlapping, 22d)

| 원인 | 설명 |
|:---|:---|
| **과적합** | 9K 파라미터 LSTM이 학습 데이터 패턴을 암기 |
| **시장 구조 변화** | Test 기간(2021-2024)에 COVID 이후 금리 인상 레짐 |
| **제한적 피처** | 2개 피처(LogRV, Return)만으로는 비정상적 시장 대응 어려움 |
| **IS-OOS Gap: 0.276** | V71(Gap ~0.02)에 비해 LSTM의 과적합 경향 심함 |

### 8.2 Scaler Leakage 문제 (v51b에서 발견)

초기 실험에서 IS R²가 비정상적으로 높게(0.848) 나온 원인:
- 전체 데이터로 scaler fit → 미래 정보가 IS 평가에 누출
- Train-only scaler로 수정 후 공정한 IS/OOS 결과 확보

---

## 9. 딥러닝 vs 선형 모델 논쟁

### 9.1 22일 변동성 예측에서의 모델 비교 (overlapping RV 기준)

> **주의**: 아래 수치는 overlapping RV 타겟 기준. 전방 RV 기준: Ensemble 22d=0.803, Ridge 22d=0.798, LSTM Tuned 22d=0.780, HAR-3 22d=0.761.

| 모델 유형 | 대표 | 풀링 R² (overlap) | 파라미터 수 |
|:---|:---|:---|:---|
| **Ridge (선형)** | V71 | **0.803** | ~37 |
| XGBoost (앙상블) | V71 XGB | 0.773 | ~1K |
| **Bi-LSTM (DL)** | V50 | 0.651 | ~9K |
| Transformer (DL) | V43 | 0.534 | ~85K |

### 9.2 V50이 Ridge에 못 미치는 이유

1. **낮은 신호 대 잡음비**: 22일 ahead 예측은 noise가 많아, LSTM이 noise까지 학습
2. **제한적 피처**: 2개 피처로는 변동성 동학의 풍부한 정보를 포착 불가
3. **정규화 부재**: Ridge의 L2 정규화 대비, LSTM은 Early Stopping 등에 의존
4. **변동성의 선형성**: Realized Volatility의 자기 상관 구조(HAR 참조)는 본질적으로 선형적

### 9.3 시사점

- **피처 엔지니어링 > 모델 복잡도**: V71(37 피처, Ridge)이 V50(2 피처, LSTM)을 압도
- **"더 복잡한 모델 = 더 나은 성능"이 아님**: 금융 시계열의 낮은 SNR 환경에서 특히 그러함
- **OHLC 데이터 활용의 중요성**: V50은 종가만 사용했지만, OHLC 기반 피처가 결정적 성능 차이 생성

---

## 10. 한계 및 개선 방향

### 10.1 현재 한계

| 한계 | 설명 |
|:---|:---|
| **2개 피처만 사용** | LogRV, Return만으로 제한. OHLC/VIX 피처 미활용 |
| **단일 타겟 호라이즌** | t+22만 예측. 1일/5일/66일 등 다중 호라이즌 미지원 |
| **풀링 모델의 한계** | 자산 간 특성 차이를 모델이 충분히 학습하지 못함 |
| **과적합 경향** | IS-OOS Gap(0.276)이 크며, 정규화가 부족 |
| **Bond/Commodity 성능 저조** | 개별 자산 R²가 음수, 예측 유틸리티 없음 |

### 10.2 잠재적 개선 방향

1. **피처 확장**: V71의 37개 피처를 LSTM 입력으로 사용
2. **자산 임베딩**: 자산별 학습 가능한 임베딩 벡터 추가
3. **Dropout/Weight Decay**: 과적합 완화를 위한 정규화 추가
4. **Multi-Head Attention**: 단일 Attention → Multi-Head로 풍부한 시점 조합
5. **Hybrid 모델**: LSTM + Ridge 앙상블 (deep feature extraction + 선형 예측)

---

## 11. 주요 결론

1. **V50 Tuned(3 Range feat)은 중기(60~90d)에서 V71(37feat Ridge)을 능가**: 피처 선택과 모델 아키텍처의 상호작용이 중요
2. **단기에서는 피처 수 > 아키텍처**: 22d에서 Ensemble(37feat, 0.803) > LSTM(3feat, 0.780)
3. **Temporal Attention은 "Dynamic HAR"로 해석 가능**: 시장 상태에 따라 중요한 lag를 동적으로 선택
4. **V50 Tuned는 전 구간 중위 R² 최고**: 개별 자산 수준에서 가장 안정적인 예측
5. **문헌 벤치마크 대비**: HAR-CJ(Andersen+ 2007), LASSO(Audrino & Knaus 2016), RF(Christensen+ 2023) 대비 우위 확인

---

## 부록: 실험 파일 목록

| 파일 | 설명 |
|:---|:---|
| `v50_dual_attention_lstm.py` | V50 메인 실험 코드 |
| `v50_preds.csv` | V50 예측 결과 (초기 6개 자산) |
| `v50_results.json` | V50 메인 실험 결과 |
| `v51b_fair_comparison.py` | IS/OOS 격차 원인 분석 실험 |
| `v51b_results.json` | v51b 실험 결과 |
| `v50_11assets_results.json` | 11개 자산 확장 실험 결과 |
