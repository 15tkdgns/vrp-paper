# V29: Hybrid Volatility Champion Model (Ridge-GARCH-HAR)

**V29**는 현재까지의 실험에서 가장 높은 성능($R^2 \approx 0.73$)을 기록한 변동성 예측 모델입니다. 이 모델은 금융 데이터의 특성인 **장기 기억성(Long Memory)**과 **변동성 군집현상(Volatility Clustering)**을 동시에 포착하도록 설계되었습니다.

## 1. 모델 핵심 구조

V29는 선형 회귀에 L2 제약을 더한 **Ridge Regression**을 기반으로 하며, 세 가지 주요 요소가 결합된 하이브리드 접근법을 사용합니다.

### A. HAR (Heterogeneous Auto-Regressive) 구조
Müller et al. (1997)과 Corsi (2009)가 제안한 방식으로, 서로 다른 시간 지평을 가진 투자자 집단의 행동을 모방합니다.
*   **Day Lag** ($logRV_{t-1}$): 단기 트레이더의 반응 관찰.
*   **Week Lag** ($logRV_{t-5}$): 중기 투자자의 경향 관찰.
*   **Month Lag** ($logRV_{t-22}$): 장기 투자자의 전략 및 거시 지표 반영.

### B. GARCH(1,1) Feature
잔차의 변동성 군집현상을 명시적으로 모델링하기 위해 GARCH 예측값을 피처로 도입했습니다.
$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$
*   **$\sigma_t$ (Conditional Volatility)**: 이전 시점의 수익률 충격($\epsilon_{t-1}$)과 이전 변동성($\sigma_{t-1}$)의 가중 평균입니다.
*   HAR이 "자산 가격의 레벨"에 민감하다면, GARCH는 "충격의 전파"에 민감하여 상호 보완적인 정보력을 제공합니다.

### C. Ridge Regularization
$$ \min_{w} ||Xw - y||_2^2 + \alpha ||w||_2^2 $$
*   피처 간의 다중공선성(Multicollinearity) 문제를 해결하고, 노이즈가 많은 금융 데이터에서 과적합(Overfitting)을 방지합니다.

---

## 2. 수학적 정식화 (Formulation)

### 예측 목표 (Target)
향후 한 달(22거래일) 동안의 **실현 로그 변동성(Log Realized Volatility)**을 예측합니다.
$$ Y_{t+22} = \log\left(\sqrt{\frac{252}{22} \sum_{i=1}^{22} r_{t+i}^2} \times 10000\right) $$

### 선형 결합 구조
$$ \hat{Y}_{t+22} = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-5} + \beta_m RV_{t-22} + \beta_g GARCH_{t-1} $$
*   데이터는 **StandardScaler**를 통해 정규화되어 각 피처의 영향력을 동등하게 비교합니다.

---

## 3. 왜 딥러닝(Mamba, KAN)보다 뛰어난가?

Phase 9 실험 결과, 최신 딥러닝 아키텍처보다 V29가 압도적으로 높았던 이유는 다음과 같습니다.

1.  **Low Signal-to-Noise**: 금융 데이터는 신호보다 노이즈가 압도적으로 많습니다. 딥러닝 모델은 이 노이즈를 "패턴"으로 오해하여 과적합되지만, Ridge는 단순한 추세만 포착하여 일반화 성능이 높습니다.
2.  **Explicit Domain Logic**: HAR과 GARCH는 금융 시계열의 검증된 특성을 수학적으로 직접 주입합니다. 딥러닝이 이를 스스로 데이터에서 찾아내기에는 금융 데이터의 샘플 수가(수만 개 수준) 너무 적습니다.
3.  **Stability**: 파라미터 수가 극히 적어(5개 내외), 시장 국면(Regime)이 바뀌어도 안정적인 성능을 유지합니다.

---

## 4. 구현 참고 (`src/experiments/sci/v29_garch_feature.py`)
*   **Python Libraries**: `arch` (GARCH fitting), `sklearn.linear_model.Ridge`, `yfinance`.
*   **데이터 스케일링**: 수익률에 100을 곱하여 GARCH 최적화 수렴도를 높인 후 다시 결과값을 복원하여 사용합니다.

---
*이 모델은 현재 "Universal Volatility Model" 대시보드의 엔진으로 사용되고 있으며, SCI 논문의 주력 모델입니다.*
