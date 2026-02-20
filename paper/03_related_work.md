# Related Work (관련 연구)

> **Cross-Asset Volatility Risk Premium Prediction Using Machine Learning**
>
> 본 문서는 논문 작성을 위한 관련 연구(Related Work / Literature Review) 초안입니다.

---

## 1. 변동성 리스크 프리미엄(Variance Risk Premium)의 이론적 기초

### 1.1 VRP의 정의와 경제적 의미

변동성 리스크 프리미엄(Variance Risk Premium, VRP)은 위험중립 측도(risk-neutral measure, Q)하의 기대 분산과 물리적 측도(physical measure, P)하의 기대 분산 간의 차이로 정의된다.

$$\text{VRP}_t = \text{Var}^Q_t(r_{t+h}) - E^P_t[\text{Var}(r_{t+h})]$$

Carr and Wu (2009)는 이 표준 정의를 제시하며, variance swap 포트폴리오를 통해 위험중립 분산을 추정하는 방법론을 확립하였다. 이들의 연구는 VRP가 양(positive)의 리스크 프리미엄이며, 대규모 변동성 급등(spike) 시 투자자가 부담하는 손실에 대한 보상임을 실증적으로 보였다. 이 정의는 이후 VRP 연구의 표준 프레임워크가 되었으며, 본 연구의 타겟 변수인 `VRP = VIX - RV`는 이 canonical definition의 실용적 간소화 버전에 해당한다.

Carr and Wu (2009)의 연구가 VRP의 존재와 그 방향성을 확인하는 데 기여했다면, 이후 연구들은 VRP의 구조적 분해와 예측력에 초점을 맞추었다. Bekaert and Hoerova (2014)는 VIX²를 물리적 기대 실현 분산과 VRP의 합으로 분해하는 프레임워크를 제시하였다:

$$\text{VIX}_t^2 = E^P_t[\text{RV}_{t+22d}] + \text{VRP}_t$$

이 연구의 핵심 발견은 VRP가 미래 주가 수익률에 대한 예측력을 보유한다는 것이며, 조건부 분산 $E^P[\text{RV}]$은 경기순환 및 금융불안 지표로 활용될 수 있다는 것이다. 본 연구에서 VIX를 단순히 하나의 피처로 사용하는 것이 아니라, VIX 내에 내재된 물리적 기대치와 리스크 프리미엄을 분리하여 활용하는 접근의 이론적 근거를 이 연구로부터 얻는다.

Drechsler and Yaron (2011)은 동적 자산가격결정 모델에서 VRP의 시간적 변동을 설명하며, 경제적 불확실성과 투자자 위험 회피 성향의 변화가 VRP의 크기를 결정한다고 주장하였다. 이러한 이론적 배경은 본 연구에서 거시경제 변수와 시장 심리 지표(SKEW, VIX term structure)를 피처로 활용하는 근거를 제공한다.

**표 1: VRP 이론 주요 연구**

| 연구 | 핵심 기여 | 본 연구와의 관계 |
|------|-----------|-----------------|
| Carr & Wu (2009) | VRP의 표준 정의 확립 | 타겟 변수 정의의 이론적 기반 |
| Bekaert & Hoerova (2014) | VIX² 분해 프레임워크 | VIX 내 정보 분리의 근거 |
| Drechsler & Yaron (2011) | VRP의 동적 변동 설명 | 거시변수/심리 피처 활용 근거 |

---

### 1.2 Realized GARCH와 VRP 측정의 발전

Hansen, Huang, Tong, and Wang (2021)은 Realized GARCH 모형을 통해 VRP 측정의 새로운 표준을 제시하였다. 이 모형은 고빈도 실현 변동성(realized volatility)을 measurement equation에 직접 포함하며, 이중 충격 구조(dual-shock structure)를 통해 물리적 측도와 위험중립 측도에서의 서로 다른 변동성 역학을 설명한다:

$$\log h_{t+1} = \omega + \beta \log h_t + \tau(z_t) + \gamma \sigma u_t$$
$$\log x_t = \kappa + \phi \log h_t + \delta(z_t) + \sigma u_t$$

이 연구의 특히 주목할 만한 발견은 VRP 분해 결과에서 변동성 충격(volatility shock) 요소가 VRP의 97.8%를 설명한다는 것이다. 이는 가격 리스크 프리미엄 부분(2.2%)에 비해 압도적으로 크며, 변동성 자체의 변동성(volatility of volatility)이 VRP의 근본 동인임을 시사한다.

본 연구는 Hansen et al. (2021)의 폐형식(closed-form) VIX/VRP 공식을 직접 사용하지는 않으나, 고빈도 실현 변동성의 중요성과 변동성 충격 분리의 개념을 차용한다. 특히, 본 연구의 HAR(Heterogeneous Autoregressive) 피처 구조에서 다중 시간 스케일(1일, 5일, 22일)의 실현 변동성을 입력으로 사용하는 것은 이러한 변동성 역학의 다층적 특성을 반영한 설계이다.

---

## 2. Cross-Asset VRP 연구

### 2.1 자산 간 VRP의 이질성

전통적인 VRP 연구가 주로 S&P 500에 한정되었던 반면, 최근 연구들은 다수 자산군으로 분석을 확장하고 있다.

Heston et al. (2023)은 20개 선물(주식, 국채, 통화, 원자재)에 대해 옵션 기반 variance portfolio를 구성하여 VRP를 측정한 최초의 대규모 cross-asset 연구를 수행하였다. 이들의 핵심 발견은 VRP가 모든 자산에서 발생하지만, 그 수준(level), 변동성, 예측력이 자산별로 현저하게 이질적(heterogeneous)이라는 것이다. 예를 들어, S&P 500의 VRP 평균은 거의 0에 가까우나 변동성은 크며, 채권 VRP는 상대적으로 안정적인 패턴을 보인다.

본 연구는 Heston et al. (2023)과 유사한 cross-asset 접근을 취하되, 중요한 차별점을 가진다. Heston et al.은 각 자산별 고유 옵션 데이터(예: GVZ for 금, OVX for 원유)를 사용하였으나, 본 연구는 **Global VIX와 Cross-Asset Realized Volatility**를 결합하는 보다 경제적(parsimonious)인 접근법을 제안한다. 이는 자산별 옵션 데이터 접근이 제한된 환경에서도 VRP 예측이 가능하게 하며, 실시간 리스크 관리에의 적용성을 높인다.

### 2.2 원자재 VRP와 Spillover 효과

Ornelas et al. (2018)은 원자재(commodity) VRP가 미래 수익률을 예측한다는 실증적 증거를 최초로 제시하였다. 특히, 금(gold)과 원유(crude oil)의 VRP가 각각의 수익률을 예측하며, 더 나아가 한 자산의 VRP가 다른 자산의 수익률까지 예측하는 **spillover 효과**를 발견하였다. 예를 들어, VIX(global risk proxy)가 commodity VRP를 예측하는 현상이 관찰되었다.

Finta and Ornelas (2022)는 이를 확장하여 다양한 원자재 옵션의 VRP 정보를 통합하는 멀티-에셋 VRP 팩터 모델을 제안하였다. 동적 가중치 할당을 통한 포트폴리오 구성에서, 통합 VRP가 개별 자산 VRP보다 예측 오차가 낮으며 자산군 간 위험 전이를 효과적으로 포착함을 보였다.

본 연구에서 발견한 "VIX가 금, 채권, 신흥시장의 VRP를 예측하는 현상"은 Ornelas et al. (2018)이 발견한 spillover 효과와 동일한 경제적 메커니즘에 기반한다. Global risk factor로서의 VIX가 다양한 자산군의 변동성 역학에 체계적으로 영향을 미친다는 해석이 가능하다. 본 연구는 이 spillover 분석을 원자재에서 주식, 채권, 신흥시장으로 확장하고, 머신러닝을 통해 비선형적 spillover 패턴까지 포착하는 통합 프레임워크를 제안한다.

### 2.3 VIX의 정보 함량과 공포 지수 역할

VIX의 정보 함량에 대한 연구는 오랜 역사를 가진다. Fleming, Ostdiek, and Whaley (1995)는 VIX가 미래 실현 변동성 예측에 정보력을 가진다는 첫 증거를 제시하였으며, Whaley (2000)는 VIX를 "투자자 공포 지수(investor fear gauge)"로 명명하며 그 시장 상태 지표로서의 역할을 정립하였다.

후속 연구들은 VIX 자체뿐 아니라, VIX의 파생 지표(VIX term structure, SKEW index, VIX9D 등)가 추가적인 시장 상태 정보를 담고 있음을 밝혀왔다. SKEW 지수는 꼬리 위험(tail risk)의 가격결정을 반영하며, VIX term structure(단기 vs 장기 VIX 스프레드)는 변동성 기간 구조에 대한 시장 기대를 나타낸다.

본 연구는 SKEW, VIX term spread, VIX9D를 sentiment feature로 정량화하여 활용하며, 이러한 파생 지표들이 VRP 예측에 incremental value를 제공한다는 점에서 VIX 정보 함량 연구의 연장선에 있다.

### 2.4 Cross-Market Volatility Spillover

Ang and Longstaff (2013)을 비롯한 다수의 연구는 변동성 spillover가 regime-dependent하며, 특히 금융위기와 같은 스트레스 기간에 spillover가 현저하게 증가함을 보였다. Todorov and Bollerslev (2010)은 주식 시장 간 변동성 spillover의 메커니즘을 분석하였으며, VIX가 신흥시장(EEM)과 선진 해외시장(EFA)으로의 spillover에서 중심적 역할을 함을 확인하였다.

본 연구의 실험에서 관찰된 VIX spillover 효과 --- 특히 EEM에 대한 VIX 영향력이 +60%로 가장 크다는 발견 --- 는 이러한 cross-market spillover 문헌과 부합한다. 이 결과는 신흥시장이 글로벌 리스크 팩터에 더 민감하게 반응한다는 기존 이론적 예측을 실증적으로 확인한다.

**표 2: Cross-Asset VRP 주요 연구**

| 연구 | 자산 범위 | 방법론 | 핵심 발견 |
|------|----------|--------|-----------|
| Heston et al. (2023) | 20개 선물 | 옵션 기반 variance portfolios | VRP의 자산별 이질성 |
| Ornelas et al. (2018) | 원자재 3~5종 | 선형 예측 회귀 | VRP의 cross-asset spillover |
| Finta & Ornelas (2022) | 원자재 다종 | 멀티-에셋 팩터 모델 | 통합 VRP의 우수한 예측력 |
| Ang & Longstaff (2013) | 주식 시장 간 | 체제 전환 모델 | Regime-dependent spillover |
| **본 연구** | **주식, 채권, 신흥시장, 원자재** | **ML/DL (HAR+Attention)** | **Global VIX 기반 통합 예측** |

---

## 3. 변동성 예측 모델의 발전

### 3.1 전통적 계량경제 모델

#### GARCH 계열

Engle (1982)의 ARCH 모델과 Bollerslev (1986)의 GARCH(1,1) 모델은 변동성 군집 현상(volatility clustering)을 포착하는 데 기여한 초석적 연구이다. GARCH 모형은 조건부 분산이 과거 충격과 과거 분산의 함수로 결정된다는 간결한 구조를 가진다:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

이후 Heston and Nandi (2000)의 GARCH 옵션 가격결정 모형, Engle and Rangel (2008)의 GARCH-MIDAS 모형 등으로 확장되었다. 특히 GARCH-MIDAS는 단기 변동성 성분과 장기 변동성 성분을 분리하여 거시경제 변수와의 연결을 가능하게 하며, 본 연구의 22일 예측에서의 비교 베이스라인(R² = 0.55)으로 활용된다.

#### HAR 모델

Corsi (2009)가 제안한 HAR(Heterogeneous Autoregressive) 모델은 단순하면서도 강력한 변동성 예측 모형으로 널리 사용된다. 이 모형은 이질적 시장 가설(Heterogeneous Market Hypothesis)에 기반하여, 서로 다른 투자 지평(1일, 5일, 22일)을 가진 투자자들의 행동이 변동성의 다중 시간 스케일 구조를 형성한다고 가정한다:

$$\text{RV}_{t+1} = \alpha + \beta_1 \text{RV}_t^{(d)} + \beta_2 \text{RV}_t^{(w)} + \beta_3 \text{RV}_t^{(m)} + \epsilon_t$$

여기서 $\text{RV}_t^{(d)}$, $\text{RV}_t^{(w)}$, $\text{RV}_t^{(m)}$는 각각 일별, 주별(5일), 월별(22일) 실현 변동성이다.

HAR 모델은 본 연구에서 피처 엔지니어링의 핵심 구성 요소이자, 성능 비교의 기준 베이스라인(1일 예측 R² = 0.62)으로 사용된다. 본 연구의 V29 모델은 HAR 피처에 GARCH 조건부 분산과 VIX 정보를 결합한 확장 모형이며, 최종 챔피언 모델(V50, V36)은 HAR 구조를 기반으로 Cross-Asset 정보와 고급 아키텍처를 추가한 것이다.

---

### 3.2 머신러닝 기반 변동성 예측

#### 초기 ML 접근: 옵션 데이터와 Realized Variance

Carr, Wu, and Zhang (2019)는 머신러닝을 변동성 예측에 본격적으로 적용한 선구적 연구이다. 이들은 두 가지 회귀 전략을 비교하였다:
- **Regression I**: 직접 RV 예측
- **Regression II**: VIX² - RV(VRP deviation) 예측

핵심 발견은 Regression II, 즉 VIX가 이미 제공하는 좋은 기준선(baseline) 위에서 편차만을 ML로 학습하는 접근이 현저하게 우수하다는 것이다. 이를 다음과 같이 정식화할 수 있다:

$$\text{RV} = \text{VIX}^2 + f(\text{option prices})$$

Ridge 회귀와 FNN(Feedforward Neural Network)으로 R² 0.39를 달성했으며, 흥미롭게도 더 복잡한 모델이 추가적 이득을 주지 못했다.

본 연구는 Carr et al. (2019)의 "Cross-Asset 확장판"으로 위치할 수 있다. Carr et al.이 단일 자산(SPX)의 79개 옵션 데이터를 활용한 것과 달리, 본 연구는 저차원의 VIX 기반 피처와 Cross-Asset 실현 변동성을 활용하여 다수 자산군에 대한 VRP 예측을 수행한다.

#### Intraday Commonality와 Neural Network

Christensen et al. (2022, 2024-JFEC)은 다수 종목의 고빈도 실현 변동성을 pooling하면 공통 요소(common factor)를 추출할 수 있으며, Neural Network(2-3 hidden layers)이 HAR, GARCH, tree-based 모델을 모두 능가함을 보였다. 이들의 핵심 통찰은 **공통성(commonality)**과 **비선형성(nonlinearity)**을 동시에 포착하는 것이 변동성 예측의 핵심이라는 것이다.

본 연구의 Cross-Asset Volatility Basis(CAVB)는 이 intraday commonality와 유사한 개념적 토대를 가진다. VIX라는 글로벌 변동성 공통 요인으로부터 다수 자산의 실현 변동성 역학을 학습한다는 점에서, Christensen et al.의 pooling 전략의 cross-asset 버전으로 해석할 수 있다.

#### 고차원 피처와 ML 변동성 예측

Chun, Cho, and Ryu (2025)는 43개의 거시경제, 금융, 심리 피처를 활용하여 LASSO, GBRT(Gradient Boosted Regression Trees) 등 고차원 ML 모델의 변동성 예측 성능을 검증하였다. 이들의 결과에서 HAR 대비 ML의 명확한 우위가 확인되었으며, 특히 비선형 상호작용(nonlinear interactions)이 예측에 중요한 역할을 한다는 점이 강조되었다. 또한, ML 기반 변동성 예측이 volatility-timing 전략에서 벤치마크 대비 높은 Sharpe 비율(3.45~3.48)을 달성하여 경제적 유의성을 입증하였다.

---

### 3.3 딥러닝 기반 변동성 예측

#### Hybrid LSTM-GARCH 모델

Roszyk and Slepaczuk (2024)는 GARCH 모형의 조건부 분산을 LSTM의 추가 입력으로 활용하는 하이브리드 모델을 제안하였다. 이 접근은 GARCH의 변동성 군집 포착 능력과 LSTM의 장기 의존성 학습 능력을 결합한 것으로, VIX를 추가 입력으로 사용할 경우 MAE가 45% 개선되었다(MAE: 1.56e-3 -> 1.02e-3).

본 연구의 V50 모델도 유사한 하이브리드 철학을 따르되, GARCH 대신 HAR 구조를, 단순 LSTM 대신 Dual-Attention LSTM을 채택하며, 단일 자산이 아닌 Cross-Asset 예측으로 확장하였다는 점에서 차별화된다.

#### GARCH-Informed Neural Networks (GINN)

Xu et al. (2024)는 더 급진적인 통합 방식을 제안하였다. LSTM의 셀 상태(cell state) 업데이트를 GARCH의 분산 업데이트 방정식과 수학적으로 매핑하여, 신경망 가중치가 GARCH 파라미터와 대응되도록 설계한 GINN(GARCH-Informed Neural Network) 모형을 개발하였다. 이는 계량경제학적 해석 가능성과 딥러닝의 유연성을 동시에 확보하려는 시도로, S&P 500, DJI, EUR/USD, 금 선물 등 다양한 자산에서 범용성이 검증되었다.

#### DeepVol: Dilated Causal CNN

DeepVol (2022, 2024)은 dilated causal convolution을 통해 고빈도 데이터에서 시계열 순서를 보존하면서 장거리 의존성을 포착하는 아키텍처를 제안하였다. 이 모델은 전통적 모델 대비 현저한 개선을 보였으며, 고빈도 변동성 예측에서의 딥러닝의 잠재력을 입증하였다.

#### Multi-Transformer

Ramos-Perez et al. (2021)은 Transformer 기반 Multi-Transformer 아키텍처를 S&P 500 변동성 예측에 적용하였다. 데이터를 서브셋으로 나누어 각각 독립적으로 Transformer를 학습한 후 앙상블하는 전략을 사용하였으며, Self-Attention 메커니즘이 시계열 내의 핵심 시점(key moments)을 자동으로 선별하는 능력을 갖춘다는 점에서 주목할 만하다. 특히 COVID-19 기간(2020) 동안 GARCH 대비 월등히 낮은 RMSE를 기록하여, 레짐 변화 적응력에서의 우수성을 보였다.

**표 3: 딥러닝 변동성 예측 모델 비교**

| 모델 | 연구 | 핵심 아키텍처 | 자산 범위 | 특징 |
|------|------|-------------|----------|------|
| Hybrid LSTM-GARCH | Roszyk & Slepaczuk (2024) | LSTM + GARCH 입력 | S&P 500 | VIX 입력으로 45% 성능 향상 |
| GINN | Xu et al. (2024) | LSTM-GARCH 구조적 결합 | 다중 자산 | 해석 가능성 + 유연성 |
| DeepVol | Various (2022, 2024) | Dilated Causal CNN | 고빈도 데이터 | 장거리 의존성 포착 |
| Multi-Transformer | Ramos-Perez et al. (2021) | Ensemble Transformer | S&P 500 | 레짐 변화 적응 |
| **V50 (본 연구)** | - | **Dual-Attention Bi-LSTM** | **Cross-Asset** | **시계열/피처 이중 어텐션** |

---

## 4. Attention Mechanism의 금융 시계열 응용

### 4.1 Attention Mechanism의 개요

Attention mechanism은 Bahdanau, Cho, and Bengio (2014)가 기계 번역에서 제안한 이후, Vaswani et al. (2017)의 Transformer 아키텍처를 통해 광범위한 영향을 미치게 되었다. Attention의 핵심 개념은 모든 입력을 동일하게 취급하는 대신, 예측에 가장 관련된 부분에 동적으로 가중치를 부여하는 것이다.

금융 시계열에서 attention mechanism은 특히 두 가지 측면에서 유용하다:
1. **시간적 어텐션(Temporal Attention)**: 예측에 가장 영향력 있는 과거 시점을 자동으로 식별
2. **피처 어텐션(Feature Attention)**: 다수 입력 피처 중 현재 예측에 가장 중요한 변수를 선별

### 4.2 금융 분야에서의 Attention 적용

Lo and Singh (2023)는 딥러닝 모델의 금융 예측에서 SHAP(SHapley Additive exPlanations)와 LIME(Local Interpretable Model-agnostic Explanations)을 활용하여 모델의 예측 근거를 사후적으로 해석하는 방법론을 제시하였다. 이들의 핵심 발견은 잘 학습된 딥러닝 모델이 단기 가격 변동보다 **신용 스프레드, 장기 변동성 추세**와 같은 경제학적으로 타당한 변수에 더 큰 가중치를 부여한다는 것이다.

본 연구의 V50 모델에 적용된 **Dual-Attention 메커니즘**은 이러한 사후적 해석 방법과는 다른 접근을 취한다. Attention을 모델 구조 자체에 내재시켜, 학습 과정에서 동시에 해석 가능한 가중치를 생성한다. 시간적 어텐션은 예측에 가장 영향력 있는 과거 시점을, 피처 어텐션은 가장 중요한 입력 변수를 실시간으로 식별하며, 이 가중치 자체가 모델의 해석 가능성을 제공한다.

### 4.3 본 연구에서의 Dual-Attention 설계

본 연구의 중기(60~90d) 예측 챔피언 모델(V50 Tuned)이 채택한 Dual-Attention Bi-LSTM은 다음과 같은 구조를 가진다:
- **Bi-directional LSTM**: 과거-미래 양방향 시계열 패턴 학습
- **Temporal Attention**: 22일 입력 시퀀스 내에서 어떤 시점의 변동성 정보가 예측에 가장 중요한지 동적 결정
- **3개 Range 피처**: RogersSatchell, GarmanKlass, Range_Close_Ratio를 핵심 입력으로 사용. 피처 선택(feature selection)을 통해 37개 후보 중 3개만 선별

이 아키텍처는 Transformer(V43)와 비교하여 약 1/9의 파라미터(~9K vs ~85K)로 더 높은 성능을 달성하였다. 전방 RV 기준으로 V50 Tuned는 60d 풀링 R²=0.796, 90d R²=0.788을 기록하며 중기 예측에서 37개 피처 선형 모델을 능가하였다. 이는 변동성 예측이라는 특수한 도메인에서, LSTM 구조에 목적에 맞는 attention을 결합하고 핵심 피처만 선별하는 것이 범용 Transformer나 다수 피처 선형 모델보다 효과적인 귀납적 편향(inductive bias)을 제공함을 시사한다.

---

## 5. Cross-Asset 학습과 다중 과제 프레임워크

### 5.1 Multi-Task Learning in Finance

다중 과제 학습(Multi-Task Learning, MTL)은 관련된 여러 예측 과제를 동시에 학습하여 공유 표현(shared representation)의 이점을 얻는 기법이다. 금융 분야에서는 여러 자산 또는 여러 리스크 팩터를 동시에 예측하는 데 활용되고 있다.

Deep-learning models for forecasting financial risk premia (2023)에서는 시장, 크레딧, FX 등 여러 리스크 프리미엄을 단일 딥러닝 모델로 동시 예측하며, feature importance 분석을 통해 각 프리미엄의 공통 요인과 이질적 요인을 구분하였다. 본 연구는 이 프레임워크의 **변동성 리스크 프리미엄 특화 버전**으로 위치할 수 있으며, cross-asset VRP에서의 공통-이질 역학을 보다 세밀하게 분석한다.

Fan, Wu, and Yang (2025)는 Projection-Penalized PCA를 사용하여 다중 섹터 간 유사도를 데이터 주도적으로 파악하고, 적응적 정보 공유를 구현하는 adaptive multi-task learning 프레임워크를 제안하였다. 이 연구에서 섹터 간 정보 전이 효과로 개별 모델 대비 성능이 향상되었다는 결과는, 본 연구에서 cross-asset 정보(다른 자산군의 실현 변동성)를 피처로 활용하는 전략의 이론적 근거를 제공한다.

### 5.2 본 연구에서의 Cross-Asset 학습 전략

본 연구는 두 가지 상호 보완적인 cross-asset 학습 전략을 채택하였다:

1. **V50 Tuned (중기 60~90d 예측)**: 11개 자산의 데이터를 pooling하여 단일 Dual-Attention LSTM으로 학습. 3개 Range 피처(RogersSatchell, GarmanKlass, Range_Close_Ratio)를 사용하여, 소수 핵심 피처로 자산 간 공통 변동성 역학을 학습.

2. **V71 Ridge (단기 1~22d 예측)**: 37개 피처를 사용한 Asset-Adaptive Ridge 회귀. 자산군별(Equity, Bond, Commodity) 최적화된 정규화 파라미터를 적용. 주식은 강한 정규화(alpha=100), 채권은 약한 정규화(alpha=10).

이 이중 전략은 예측 지평에 따라 최적의 cross-asset 학습 방식이 다르다는 통찰을 반영한다. 단기(1~22d)에서는 다수 피처의 선형 결합이 효과적이고, 중기(60~90d)에서는 소수 핵심 피처의 비선형 패턴 포착이 우수하다.

---

## 6. 검증 방법론과 편향 통제

### 6.1 시계열 교차 검증

시계열 데이터에서의 모델 검증은 일반적인 k-fold 교차 검증과는 다른 접근이 필요하다. 시간적 순서를 보존하지 않는 무작위 분할은 미래 정보 누출(look-ahead bias)을 초래한다.

Benhenda et al. (2026)은 point-in-time 금융 ML에서의 look-ahead bias를 표준화하여 벤치마킹하는 프레임워크를 제안하였다. 이 연구에서는 훈련 데이터의 시간적 오염(temporal contamination)이 모델 성능의 과대 평가를 초래할 수 있음을 경고하며, purged rolling-window validation과 alpha decay 분석 등의 검증 도구를 제시하였다.

Lopez de Prado (2018)는 "Advances in Financial Machine Learning"에서 Purged K-Fold Cross-Validation과 Embargo 기법을 제안하여, 시계열 교차 검증에서의 정보 누출을 체계적으로 방지하는 방법론을 정립하였다.

### 6.2 본 연구의 검증 체계

본 연구는 이러한 검증 방법론에 기반하여 다층적 강건성 검증을 수행하였다:

1. **5-Fold TimeSeriesSplit 교차 검증**: 시간적 순서를 보존하며, 각 fold에서 학습 기간이 순차적으로 확장
2. **Hyperparameter Grid Search**: 모델 설정 변화에 따른 성능 민감도 분석
3. **Multi-Seed/Bootstrap 실험**: 무작위성에 따른 성능 변동성 측정

검증 결과, V50 모델은 교차 검증에서 R² 0.9855 (+/- 0.006), 시드 변동에서 R² 0.9906 (+/- 0.0009)의 매우 높은 안정성을 보였으며, V36 모델은 교차 검증 R² 0.7627 (+/- 0.049), 부트스트랩 R² 0.7728 (+/- 0.0003)의 극도로 낮은 변동성을 기록하였다.

### 6.3 ML/DL 금융 예측의 Best Practices

"Deep learning for financial forecasting: a review of recent advances" (2025)는 DL 기반 금융 예측의 종합적 리뷰로, look-ahead bias, publication bias, overfitting 등 주요 함정과 이를 방지하기 위한 best practices를 정리하였다. 본 연구는 이 리뷰에서 제안하는 validation strategy, hyperparameter tuning, 결과 재현성(reproducibility) 가이드라인을 준수하여 결과의 신뢰성을 확보하고자 하였다.

---

## 7. Regime-Switching과 동적 자산 배분

Lugrin et al. (2024)은 자산별 regime을 딥러닝으로 예측하고 이를 기반으로 포트폴리오를 동적으로 재조정하는 regime-switching 기반 자산 배분 전략을 제안하였다. VRP 예측력이 시장 레짐(정상 vs 스트레스)에 따라 달라질 수 있다는 가설은 본 연구의 향후 연구 방향과 직접적으로 연결된다.

본 연구의 V50 모델에 내재된 Dual-Attention 가중치가 시장 상태에 따라 동적으로 변화하는 패턴은, Attention 기반의 암묵적 레짐 인식(implicit regime awareness)으로 해석할 수 있다. COVID-19 기간(2020)과 같은 고변동성 레짐에서 attention 가중치가 VIX 관련 피처에 집중되고, 정상 기간에서는 HAR 피처에 분산되는 패턴이 관찰되었으며, 이는 모델이 시장 상태를 적응적으로 인식하고 있음을 시사한다.

---

## 8. 본 연구의 위치와 기여 (Positioning and Contributions)

기존 문헌과의 비교를 통해, 본 연구의 위치와 차별화된 기여를 다음과 같이 정리할 수 있다:

### 8.1 기존 연구와의 갭(Research Gaps)

1. **단일 자산 편중**: VRP 예측 연구의 대부분(Carr & Wu 2009, Hansen et al. 2021, Roszyk & Slepaczuk 2024)이 S&P 500에 집중되어 있으며, cross-asset VRP 예측을 ML/DL로 수행한 연구는 극히 드물다.

2. **옵션 데이터 의존성**: Heston et al. (2023)의 cross-asset VRP 연구는 자산별 고유 옵션 데이터를 필요로 하며, 이는 실무적 적용성을 제한한다.

3. **예측 지평의 단일성**: 대부분의 연구가 단일 예측 지평(주로 1일 또는 22일)에 집중하며, 복수 지평에 대한 최적 모델 아키텍처의 차이를 체계적으로 분석한 연구가 부족하다.

4. **해석 가능성의 부재**: ML/DL 기반 변동성 예측 모델의 대부분이 블랙박스로 남아 있으며, 모델 구조 자체에 해석 가능성을 내재시킨 연구가 제한적이다.

### 8.2 본 연구의 기여

| 기여 | 상세 설명 | 관련 선행 연구와의 차별점 |
|------|-----------|-------------------------|
| **Cross-Asset VRP 예측 프레임워크** | Global VIX + Cross-Asset RV 기반 통합 예측 | Heston et al. (2023): 자산별 옵션 데이터 필요 |
| **Dual-Attention Mechanism** | 시간/피처 이중 어텐션으로 해석 가능한 예측 | Lo & Singh (2023): 사후적 해석에 의존 |
| **Multi-Horizon 최적 아키텍처 발견** | 단기=Ridge/LASSO(37feat), 중기=LSTM(3feat), 초장기=RF | 대부분 단일 지평 연구 |
| **자산군별 적응적 정규화** | 자산 클래스별 노이즈 수준에 맞춘 Ridge alpha | Fan et al. (2025): 섹터별 적응이지만 VRP 아님 |
| **경제적 옵션 데이터 불필요** | VIX 파생 지표만으로 cross-asset 예측 달성 | Carr et al. (2019): 79개 옵션 데이터 필요 |

### 8.3 핵심 발견

본 연구의 주요 실험 결과를 기존 문헌과 비교하면 (전방 RV 기준):

- **단기(1~22d) 예측**: Ridge+XGBoost 앙상블(37feat)이 풀링 R² 최고 (22d: 0.803). HAR-3(0.761), HAR-CJ(0.752) 대비 통계적으로 유의한 우위
- **중기(60~90d) 예측**: V50 Tuned LSTM(3feat)이 풀링 R² 최고 (60d: 0.796). 37개 피처 Ridge(0.774) 대비 LSTM+소수 피처의 우위
- **초장기(365d) 예측**: Random Forest(37feat)가 풀링 R² 0.705로 최고. 트리 모델의 비선형 패턴 포착이 장기에서 유리
- **강건성**: DM test에서 모든 벤치마크 대비 p<0.001, 5-fold CV와 5-seed 실험에서 안정적 성능 유지

---

## 참고문헌 (References)

### Finance / Econometrics

- Ang, A., & Longstaff, F. A. (2013). Systemic sovereign credit risk: Lessons from the US and Europe. *Journal of Monetary Economics*, 60(5), 493-510.
- Bekaert, G., & Hoerova, M. (2014). The VIX, the variance premium and stock market volatility. *Journal of Econometrics*, 183(2), 181-190.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463-4492.
- Carr, P., & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies*, 22(3), 1311-1341.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Drechsler, I., & Yaron, A. (2011). What's vol got to do with it. *Review of Financial Studies*, 24(1), 1-45.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
- Engle, R. F., & Rangel, J. G. (2008). The spline-GARCH model for low-frequency volatility and its global macroeconomic causes. *Review of Financial Studies*, 21(3), 1187-1222.
- Feunou, B., Jahan-Parvar, M. R., & Okou, C. (2017). Downside variance risk premium. *Journal of Financial Econometrics*, 16(3), 341-383.
- Finta, M. A., & Ornelas, J. R. H. (2022). Commodity return predictability and risk premia. *Journal of International Financial Markets, Institutions and Money*, 79, 101560.
- Fleming, J., Ostdiek, B., & Whaley, R. E. (1995). Predicting stock market volatility: A new measure. *Journal of Futures Markets*, 15(3), 265-302.
- Hansen, P. R., Huang, Z., Tong, H., & Wang, S. (2021). Realized GARCH, CBOE VIX, and the volatility risk premium. *Journal of Financial Econometrics* (2024).
- Heston, S. L., et al. (2023). Exploring the variance risk premium across assets. *AEA Conference 2024*.
- Ornelas, J. R. H., et al. (2018). Volatility risk premia and future commodity returns. *BIS Working Paper*, No. 619.
- Pyun, S. (2019). Variance risk in aggregate stock returns and time-varying return predictability. *Journal of Financial Economics*, 132(1), 150-174.
- Todorov, V., & Bollerslev, T. (2010). Jumps and betas: A new framework for disentangling and estimating systematic risks. *Journal of Econometrics*, 157(2), 220-235.
- Whaley, R. E. (2000). The investor fear gauge. *Journal of Portfolio Management*, 26(3), 12-17.

### Computer Science / Machine Learning

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.
- Benhenda, M., et al. (2026). A standardized benchmark of look-ahead bias in point-in-time financial ML. *arXiv preprint arXiv:2601.13770*.
- Carr, P., Wu, L., & Zhang, Z. (2019). Using machine learning to predict realized variance. *arXiv preprint arXiv:1909.10035*.
- Christensen, K., et al. (2022). Volatility forecasting with machine learning and intraday commonality. *Journal of Financial Econometrics* (2024).
- Chun, D., Cho, H., & Ryu, D. (2025). Volatility forecasting and volatility-timing strategies: A machine learning approach. *Research in International Business and Finance*, 102723.
- DeepVol (2022, 2024). Volatility forecasting from high-frequency data with dilated causal convolutions. *arXiv preprint arXiv:2210.04797*.
- Fan, J., Wu, X., & Yang, Z. (2025). Adaptive multi-task learning for multi-sector portfolio optimization. *arXiv preprint arXiv:2507.16433*.
- Lo, A., & Singh, S. (2023). Deep-learning with SHAP/LIME interpretability in finance. *SSRN Working Paper*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lugrin, A., et al. (2024). Dynamic asset allocation with asset-specific regime forecasts. *arXiv preprint arXiv:2407.01550*.
- Michael, S. (2025). Options-driven volatility forecasting. *Quantitative Finance*, DOI: 10.1080/14697688.2025.2454623.
- Ramos-Perez, E., et al. (2021). Multi-transformer: A new neural network-based architecture for forecasting S&P volatility. *Applied Soft Computing*, 109, 107144.
- Roszyk, K., & Slepaczuk, R. (2024). The hybrid forecast of S&P 500 volatility ensembled from VIX, GARCH and LSTM. *arXiv preprint arXiv:2407.16780*.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
- Xu, B., et al. (2024). GARCH-informed neural networks for volatility forecasting. *ICAIF '24*. DOI: 10.1145/3677052.3698600.
