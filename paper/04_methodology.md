# Methodology (방법론)

> **Cross-Asset Volatility Risk Premium Prediction Using Machine Learning**

---

## 1. 연구 개요 (Research Overview)

본 연구는 Cross-Asset Volatility Risk Premium(VRP)을 예측하기 위한 머신러닝 프레임워크를 제안한다. 핵심 연구 질문은 다음과 같다:

1. Global VIX와 Cross-Asset Realized Volatility만으로 다수 자산군의 VRP를 효과적으로 예측할 수 있는가?
2. 예측 기간(1일~365일)에 따라 최적 모델 아키텍처와 피처 전략이 체계적으로 달라지는가?
3. LSTM+Attention이 소수 핵심 피처와 결합될 때, 다수 피처 선형 모델을 능가할 수 있는가?

---

## 2. 데이터 (Data)

### 2.1 자산 유니버스

본 연구는 3개 자산 클래스에 걸쳐 총 11개 자산의 일별 가격 데이터를 사용한다.

| 자산 클래스 | Ticker | 설명 | 비고 |
|:-----------|:-------|:-----|:-----|
| **Equity** | SPY | S&P 500 ETF | 미국 대형주 |
| | QQQ | Nasdaq-100 ETF | 기술주 중심 |
| | IWM | Russell 2000 ETF | 미국 소형주 |
| | EFA | MSCI EAFE ETF | 선진 해외시장 |
| | EEM | MSCI Emerging Markets ETF | 신흥시장 |
| **Bond** | TLT | 20+ Year Treasury Bond ETF | 장기 국채 |
| | IEF | 7-10 Year Treasury Bond ETF | 중기 국채 |
| | AGG | Aggregate Bond ETF | 종합 채권 |
| **Commodity** | GLD | Gold ETF | 금 |
| | SLV | Silver ETF | 은 |
| | USO | US Oil Fund ETF | 원유 |

### 2.2 데이터 기간 및 출처

- **기간**: 2010년 1월 ~ 2025년 1월 (약 15년, 약 3,780 거래일)
- **출처**: Yahoo Finance (yfinance API)를 통한 일별 OHLCV 데이터
- **전처리**: 결측치는 forward-fill 방식으로 보간

### 2.3 데이터 분할

시계열 데이터의 시간적 순서를 보존하기 위해 **시간 기반 분할(Temporal Split)**을 사용한다.

- **학습 세트 (Training Set)**: 전체 데이터의 80% (시간순 기준 앞부분)
- **테스트 세트 (Test Set)**: 전체 데이터의 20% (시간순 기준 뒷부분)
- **교차 검증**: 5-Fold TimeSeriesSplit (확장 윈도우 방식)

```
전체 데이터 타임라인:
|[=================== Training (80%) ====================|= Test (20%) =|
2010                                                      ~2022          2025

5-Fold TimeSeriesSplit:
Fold 1: |==Train==|=Val=|
Fold 2: |=====Train=====|=Val=|
Fold 3: |=========Train=========|=Val=|
Fold 4: |=============Train=============|=Val=|
Fold 5: |=================Train=================|=Val=|
```

---

## 3. 타겟 변수 정의 (Target Variable)

### 3.1 전방 실현 변동성 (Forward Realized Volatility)

본 연구의 예측 타겟은 **전방 실현 변동성**이다. 이는 피처(t-1 이전 데이터)와 타겟 간 데이터 중첩을 완전히 방지하기 위한 설계이다.

$$\text{target}(t, h) = \log\left(\text{mean}\left(r^2[t+1 : t+h]\right) \times 252\right)$$

여기서:
- $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$: 일별 로그 수익률
- $h$: 예측 기간 (horizon)
- $252$: 연환산 계수 (연간 거래일 수)

### 3.2 Multi-Horizon 예측 기간

본 연구는 8개 예측 기간을 대상으로 체계적으로 비교한다:

| 예측 기간 | h | 타겟 정의 | 경제적 의미 |
|:---------|:--|:---------|:-----------|
| **1일** | 1 | $\text{target}(t, 1)$ | 단기 트레이딩 및 일별 리스크 관리 |
| **5일** | 5 | $\text{target}(t, 5)$ | 주간 포트폴리오 모니터링 |
| **22일** | 22 | $\text{target}(t, 22)$ | 월간 포트폴리오 재조정 |
| **60일** | 60 | $\text{target}(t, 60)$ | 분기 스윙 전략 |
| **90일** | 90 | $\text{target}(t, 90)$ | 분기 리밸런싱 |
| **120일** | 120 | $\text{target}(t, 120)$ | 중기 전략적 배분 |
| **180일** | 180 | $\text{target}(t, 180)$ | 반기 리밸런싱 |
| **365일** | 365 | $\text{target}(t, 365)$ | 연간 전략적 자산 배분 |

전방 윈도우(forward window)를 사용하므로, 학습 시점 t의 모든 피처는 t-1 이전 데이터에서 산출되어 look-ahead bias를 원천 방지한다.

---

## 4. 피처 엔지니어링 (Feature Engineering)

### 4.1 HAR (Heterogeneous Autoregressive) 피처 (3개)

Corsi (2009)의 HAR 구조에 기반하여, 세 가지 시간 스케일의 실현 변동성 래그를 핵심 피처로 사용한다:

| 피처 | 정의 | 경제적 의미 |
|------|------|------------|
| $LogRV_{t-1}^{(d)}$ | 전일 로그 실현 변동성 | 단기 트레이더 반응 (일간 수준) |
| $LogRV_{t-5}^{(w)}$ | 5일 전 로그 실현 변동성 | 중기 투자자 경향 (주간 수준) |
| $LogRV_{t-22}^{(m)}$ | 22일 전 로그 실현 변동성 | 장기 투자자 전략 (월간 수준) |

이 세 래그는 이질적 시장 가설(Heterogeneous Market Hypothesis; Muller et al., 1997)에 기반한다.

### 4.2 Range 기반 피처 (V50 Tuned 핵심, 3개)

OHLC 가격 데이터에서 추출한 범위 기반 변동성 추정량:

| 피처 | 정의 | 특성 |
|------|------|------|
| **RogersSatchell** | $\text{RS}_t = \sqrt{\frac{1}{22}\sum_{i=0}^{21}[(h_i - c_i)(h_i - o_i) + (l_i - c_i)(l_i - o_i)]}$ | drift-robust, 이론적 효율성 5배 |
| **GarmanKlass** | $\text{GK}_t = \sqrt{\frac{1}{22}\sum_{i=0}^{21}[0.5(h_i-l_i)^2 - (2\ln 2 - 1)(c_i-o_i)^2]}$ | efficiency 7.4배 |
| **Range_Close_Ratio** | $(H_t - L_t) / C_t$ | 일중 가격 범위의 상대적 크기 |

V50 Tuned 모델은 37개 후보 피처에서 피처 선택(feature selection)을 통해 이 3개만 사용한다.

### 4.3 확장 피처 (V71 Ridge 전용, 37개)

37개 피처는 4개 그룹으로 구성된다:

| 그룹 | 피처 수 | 주요 내용 |
|------|---------|----------|
| **Base** | 14 | HAR(3) + ReturnLag(1) + GARCH(2) + VIX(5) + VRP(2) + DayOfWeek(1) |
| **HF Proxy** | 7 | RS/GK/Park/Yang-Zhang 22d + 비율 피처 |
| **IV Surface** | 10 | VIX 기간구조(VIX3M/VIX9D) + VRP rolling 통계  |
| **Alt Data** | 6 | 거래량 이동평균, 거래량 변화율, 거래량 비율 |

### 4.4 피처 정규화 (Feature Normalization)

모든 피처는 **StandardScaler**를 통해 자산별로 정규화된다:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

여기서 $\mu$, $\sigma$는 학습 세트에서 산출된 평균 및 표준편차이다.

---

## 5. 모델 아키텍처 (Model Architectures)

### 5.1 V71 Ridge (단기 1~22d 챔피언, 37 피처)

#### 5.1.1 설계 원리

단기(1~22일) 예측에서 37개 피처의 다양한 정보원(HAR + OHLC + IV + Alt)을 효과적으로 결합하기 위해, L2 정규화된 Ridge 회귀를 채택한다. 고차원 피처 간 다중공선성(특히 HF Proxy 간 상관>0.99)을 L2 정규화가 자연스럽게 처리한다.

$$\hat{Y}_{t+h} = \beta_0 + \sum_{j=1}^{37} \beta_j X_{t,j}$$

학습 목적 함수:

$$\min_{\beta} \|X\beta - y\|_2^2 + \alpha \|\beta\|_2^2$$

#### 5.1.2 자산 클래스별 적응적 정규화

자산군 간 변동성 역학의 이질성을 반영하여, 클래스별 독립 Ridge 모델을 학습한다:

| 클래스 | 포함 자산 | 최적 Alpha | 근거 |
|--------|----------|-----------|------|
| Equity | SPY, QQQ, IWM, EFA, EEM | **100.0** | 높은 노이즈, 시장 효율성 |
| Bond | TLT, IEF, AGG | **10.0** | 금리 추세 명확, 중간 정규화 |
| Commodity | GLD, SLV, USO | **10.0** | 매크로 요인 의존, 중간 노이즈 |

### 5.2 V50 Tuned (중기 60~90d 챔피언, 3 Range 피처)

#### 5.2.1 아키텍처 개요

V50 Tuned 모델은 **양방향 LSTM(Bi-LSTM)**에 **Temporal Attention Mechanism**을 결합한 구조이다. 37개 후보에서 피처 선택으로 3개 Range 피처(RogersSatchell, GarmanKlass, Range_Close_Ratio)만 사용한다.

```
입력 시퀀스 X ∈ R^{22×3}
     │
     ▼
┌─────────────────────────┐
│    Bi-directional LSTM   │
│  Forward:  h→ = LSTM→(x) │
│  Backward: h← = LSTM←(x) │
│  H = [h→; h←] ∈ R^{22×2d}│
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Attention Mechanism    │
│  e_t = tanh(W_a · H_t)  │
│  α_t = softmax(e_t)     │
│  c = Σ α_t · H_t        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Fully Connected Layer  │
│   ŷ = W_fc · c + b      │
└──────────┬──────────────┘
           │
           ▼
      Predicted LogRV
```

#### 5.2.2 Bi-directional LSTM

양방향 LSTM은 입력 시퀀스를 순방향과 역방향 두 방향으로 처리한다:

**순방향 (Forward LSTM)**:
$$\vec{h}_t = \text{LSTM}_{\rightarrow}(x_t, \vec{h}_{t-1})$$

**역방향 (Backward LSTM)**:
$$\overleftarrow{h}_t = \text{LSTM}_{\leftarrow}(x_t, \overleftarrow{h}_{t+1})$$

**결합 은닉 상태**:
$$H_t = [\vec{h}_t ; \overleftarrow{h}_t] \in \mathbb{R}^{2d_h}$$

#### LSTM 셀 내부 동작

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate Cell State)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State Update)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State)}$$

#### 5.2.3 Temporal Attention Mechanism

모든 시점의 은닉 상태에 동적 가중치를 부여하여 Context Vector를 생성한다.

**에너지 점수**: $e_t = \tanh(W_a H_t + b_a)$

**Attention 가중치**: $\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^{T} \exp(e_j)}$

**Context Vector**: $c = \sum_{t=1}^{T} \alpha_t \cdot H_t$

**경제적 해석**: Attention 가중치 $\alpha_t$는 HAR 모델의 고정 래그(1, 5, 22일)를 **Dynamic HAR**로 일반화한 것으로 해석할 수 있다.

#### 5.2.4 하이퍼파라미터

| 파라미터 | 최적값 | 탐색 범위 |
|---------|--------|----------|
| hidden_dim ($d_h$) | **32** | {32, 64, 128} |
| learning_rate ($\eta$) | **0.001** | {0.0001, 0.001, 0.01} |
| dropout ($p$) | **0.0** | {0.0, 0.3, 0.5} |
| seq_len ($L$) | **22** | {10, 22, 44} |
| batch_size ($B$) | 64 | 고정 |
| epochs | 20 | 고정 |
| optimizer | Adam | 고정 |
| loss function | MSE | 고정 |
| **입력 피처** | **3 (Range)** | RS, GK, RC |

최적 hidden_dim이 32(최소값)로 선택된 것은 금융 시계열의 낮은 신호 대 잡음비 환경에서 간소한 구조(parsimonious architecture)가 일반화에 유리함을 시사한다.

#### 5.2.5 모델 파라미터 수

$d_{in} = 3$ (입력 차원), $d_h = 32$ (은닉 차원)일 때:

| 구성 요소 | 파라미터 수 | 산출 근거 |
|----------|-----------|----------|
| Bi-LSTM | $4 \times (d_h^2 + d_{in} \times d_h + d_h) \times 2$ | 양방향, 4개 게이트 |
| Attention | $2d_h + 1$ | 선형 변환 + 바이어스 |
| FC Layer | $2d_h + 1$ | Context → 스칼라 |
| **합계** | **~9,000** | --- |

Transformer(V43, ~85,000)의 약 **1/9** 수준으로, 더 적은 파라미터로 더 높은 성능을 달성한다.

### 5.3 V50 Original (Bi-LSTM + Attention, 2 피처)

V50 Tuned의 원본 버전. LogRV_lag1, Ret_lag1만 사용하는 LSTM 모델로, 피처 선택 효과를 검증하기 위한 대조군이다.

### 5.4 HAR-3 (Ridge, 3 HAR 피처)

Corsi (2009)의 HAR 모형 기반. LogRV lag-1, lag-5, lag-22를 사용하는 가장 단순한 벤치마크 모델이다.

$$\hat{Y}_{t+h} = \beta_0 + \beta_d LogRV_{t-1}^{(d)} + \beta_w LogRV_{t-5}^{(w)} + \beta_m LogRV_{t-22}^{(m)}$$

### 5.5 LASSO-HAR (Audrino & Knaus, 2016)

L1 정규화를 적용한 HAR 확장 모델. Ridge(V71)와의 L1 vs L2 정규화 효과 비교를 위한 벤치마크.

$$\min_{\beta} \|X\beta - y\|_2^2 + \lambda \|\beta\|_1$$

### 5.6 HAR-CJ (Andersen, Bollerslev, Diebold, 2007)

연속(Continuous) 변동성과 점프(Jump) 변동성을 분리한 HAR 확장. Bipower Variation 기반 점프 프록시를 사용한다.

### 5.7 Random Forest (Christensen, Siggaard, Veliyev, 2023)

트리 기반 ML 모델. 37개 피처를 사용하며, 초장기(365d) 예측에서 비선형 패턴 포착 능력으로 최고 성능을 달성한다.

---

## 6. 학습 절차 (Training Procedure)

### 6.1 V50 Tuned 학습

#### 6.1.1 데이터 Pooling

11개 자산의 시퀀스를 단일 데이터셋에 합산(pooling)한 후, 날짜 순으로 정렬하여 시간적 순서를 보존한다. 이는 자산 간 공통 변동성 역학을 학습하기 위한 설계이다.

```python
# 의사 코드
all_sequences = []
for asset in ALL_11_ASSETS:
    for t in range(len(data) - SEQ_LEN - h):
        X = data[t : t+SEQ_LEN]        # 입력 시퀀스 (22, 3)
        y = target_fwd_rv(t, h)         # 전방 RV 타겟
        all_sequences.append((Date, Asset, X, y))

all_sequences.sort(by='Date')  # 시간순 정렬
train, test = temporal_split(all_sequences, ratio=0.8)
```

#### 6.1.2 Mini-Batch 학습

| 항목 | 설정 |
|------|------|
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Batch Size | 64 |
| Epochs | 20 |
| Shuffle | True (배치 내에서만; 전체 시간 순서는 분할에서 보존) |

### 6.2 V71 Ridge 학습

#### 6.2.1 데이터 Pooling (자산 클래스별)

11개 자산의 37개 피처를 합산한 후, 자산 클래스별로 분리하여 각각 독립적인 Ridge 모델을 학습한다.

```
data_all = concatenate([features_37(asset) for asset in ALL_11_ASSETS])
data_all.sort(by='Date')
train, test = temporal_split(data_all, ratio=0.8)

for class in {Equity, Bond, Commodity}:
    train_class = train[train.Class == class]
    model[class] = Ridge(alpha=α_class).fit(train_class)
```

---

## 7. 평가 지표 (Evaluation Metrics)

### 7.1 풀링 R² (Pooling R²)

$$R^2 = 1 - \frac{\sum_{t}(y_t - \hat{y}_t)^2}{\sum_{t}(y_t - \bar{y})^2}$$

11개 자산의 예측값을 풀링하여 단일 R²를 산출. **cross-sectional variation을 포함**하므로 자산 간 변동성 수준 차이가 R²에 기여한다.

### 7.2 중위 R² (Median R²)

11개 자산 각각의 개별 R²를 산출한 후 중위값을 보고. **순수 시계열 예측력**을 반영하며, 풀링 R²보다 보수적인 지표이다.

### 7.3 보조 지표

| 지표 | 정의 | 특성 |
|------|------|------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$ | 큰 오차에 민감 |
| **MAE** | $\frac{1}{n}\sum|y_t - \hat{y}_t|$ | 이상치에 강건 |
| **QLIKE** | $\frac{1}{n}\sum\left[\frac{y_t}{\hat{y}_t} - \ln\frac{y_t}{\hat{y}_t} - 1\right]$ | 변동성 예측 특화 |
| **Diebold-Mariano Test** | 두 모델 예측 오차의 유의한 차이 검정 | Newey-West HAC 적용 |

### 7.4 Diebold-Mariano 검정

$$d_t = L(e_t^A) - L(e_t^B)$$
$$DM = \frac{\bar{d}}{\hat{\sigma}_d / \sqrt{n}} \sim N(0, 1)$$

22일 horizon에서의 직렬 상관을 고려하여 Newey-West HAC(bandwidth=22)를 적용한다.

---

## 8. 강건성 검증 (Robustness Validation)

### 8.1 검증 프레임워크

```
Phase 1: Hyperparameter Sensitivity
   ├── Stage 1: hidden_dim × lr 그리드 (3×3 = 9 조합)
   └── Stage 2: dropout × seq_len 그리드 (3×3 = 9 조합)

Phase 2: 5-Fold TimeSeriesSplit Cross-Validation
   └── 시간순 확장 윈도우, 5개 fold

Phase 3: Multi-Seed / Bootstrap Robustness
   ├── V50: 5개 독립 랜덤 시드 실험 (42, 123, 456, 789, 2024)
   └── V71 Ridge: 5회 부트스트랩 리샘플링 (Ridge는 결정론적이므로)
```

### 8.2 V50 하이퍼파라미터 민감도 분석

#### Stage 1: hidden_dim x lr

| hidden_dim | lr=0.0001 | lr=0.001 | lr=0.01 |
|:-----------|:----------|:---------|:--------|
| 32 | R²=0.9816 | **R²=0.9913** | R²=0.9911 |
| 64 | R²=0.9827 | R²=0.9902 | R²=0.9902 |
| 128 | R²=0.9810 | R²=0.9891 | R²=0.9896 |

#### Stage 2: dropout x seq_len (hidden_dim=32, lr=0.001 고정)

| dropout | seq_len=10 | seq_len=22 | seq_len=44 |
|:--------|:-----------|:-----------|:-----------|
| 0.0 | R²=0.9859 | **R²=0.9913** | R²=0.9873 |
| 0.3 | R²=0.9855 | R²=0.9912 | R²=0.9873 |
| 0.5 | R²=0.9854 | R²=0.9908 | R²=0.9866 |

### 8.3 5-Fold 교차 검증 결과

#### V50

| Fold | R² | RMSE | MAE |
|------|-----|------|-----|
| 1 | 0.9797 | 0.1445 | 0.0998 |
| 2 | 0.9838 | 0.1156 | 0.0774 |
| 3 | 0.9802 | 0.1452 | 0.1010 |
| 4 | 0.9926 | 0.1095 | 0.0774 |
| 5 | 0.9913 | 0.0797 | 0.0534 |
| **Mean +/- Std** | **0.9855 +/- 0.0061** | **0.1189 +/- 0.0273** | |

#### V71 Ridge

| Fold | R² | RMSE | MAE |
|------|-----|------|-----|
| 1 | 0.7849 | 0.6747 | 0.5140 |
| 2 | 0.7975 | 0.6073 | 0.4736 |
| 3 | 0.7862 | 0.6346 | 0.4924 |
| 4 | 0.6769 | 0.8849 | 0.6350 |
| 5 | 0.7679 | 0.5008 | 0.3991 |
| **Mean +/- Std** | **0.7627 +/- 0.0491** | **0.6605 +/- 0.1411** | |

### 8.4 시드 / 부트스트랩 안정성

#### V50: 5-Seed 실험

| Seed | R² | RMSE | MAE |
|------|-----|------|-----|
| 42 | 0.9913 | 0.0823 | 0.0503 |
| 123 | 0.9914 | 0.0819 | 0.0545 |
| 456 | 0.9891 | 0.0923 | 0.0725 |
| 789 | 0.9907 | 0.0853 | 0.0531 |
| 2024 | 0.9906 | 0.0861 | 0.0560 |
| **Mean +/- Std** | **0.9906 +/- 0.0009** | **0.0856 +/- 0.0042** | |

#### V71 Ridge: 5-Bootstrap 실험

| Bootstrap | R² | RMSE |
|-----------|-----|------|
| 1 | 0.7734 | 0.5134 |
| 2 | 0.7726 | 0.5143 |
| 3 | 0.7729 | 0.5139 |
| 4 | 0.7725 | 0.5144 |
| 5 | 0.7726 | 0.5143 |
| **Mean +/- Std** | **0.7728 +/- 0.0004** | **0.5141 +/- 0.0004** |

---

## 9. 구현 환경 (Implementation Details)

### 9.1 소프트웨어 환경

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.x |
| scikit-learn | 1.x |
| pandas | 2.x |
| numpy | 1.x |
| matplotlib + seaborn | 시각화 |
| OS | Ubuntu 24.04 (WSL) |

### 9.2 핵심 소스코드 경로

| 파일 | 역할 |
|------|------|
| `src/experiments/creative/v50_dual_attention_lstm.py` | V50 모델 정의 및 실험 |
| `src/experiments/creative/v71_advanced_data.py` | V71 모델 정의 및 실험 |
| `src/experiments/creative/multi_horizon_benchmark.py` | Multi-Horizon 벤치마크 실험 |
| `src/experiments/creative/multi_horizon_benchmark_results.json` | 벤치마크 결과 (Ground Truth) |
| `src/experiments/verification/v66_v50_robustness_hparam_cv.py` | V50 강건성 검증 |
| `src/data/ohlcv_cache.csv` | OHLCV 데이터 캐시 |

### 9.3 재현성 (Reproducibility)

- 모든 실험에서 5개 고정 시드(42, 123, 456, 789, 2024)를 사용
- PyTorch 및 NumPy의 난수 생성기를 실험 시작 시 초기화
- 데이터 분할은 시간 기반으로 결정론적
- 모든 결과는 JSON 형식으로 저장
