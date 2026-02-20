<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 성능 최소 10% 이상 올릴 수 있는 시도 제시

솔직히 말하면, **풀링 R² 기준으로 LASSO보다 “10%p 이상” 올리는 건 거의 불가능한 타겟**에 가깝습니다.
현실적으로는 (1) **loss 기준 10%+ 개선(QLIKE, MSE)** 이나, (2) **특정 레짐/자산에서 10%+ 개선**을 목표로 잡는 게 맞습니다.[^1][^2]

그 전제를 깔고, “진짜 점프를 노려볼 만한” 시도들을 정리해보면 아래 세 축입니다.

***

## 1. 정보량 자체를 늘리는 시도 (데이터 쪽)

### 1-1. 진짜 고빈도 RV·점프 분리 도입

지금은 OHLC 기반 Range proxy라 **틱/5분 데이터의 정보를 proxy로만 쓰는 상태**입니다.[^3]
실제 intraday RV·bipower variation·jump variation을 쓰면, 22일 RV의 예측 신호 자체가 강해져서 LASSO–Ridge 모두 같이 올라가지만, **구조가 더 잘 설계된 쪽(V71)이 더 이득**을 볼 가능성이 큽니다.[^4][^5]

시도 아이디어:

- 5분 수익률로
    - Realized Variance,
    - Bipower Variation (continuous vol),
    - Jump variation,
    - Upside/Downside semivariance
를 계산해서 현재 HF Proxy 블록에 추가.
- 특히 **downside semivariance + jump**는 VRP 문헌에서 추가 예측력 있는 걸로 알려져 있음.[^5][^1]

이건 데이터 수집/연산 비용은 크지만, 정보량 측면에서 가장 “10% 이상 점프” 가능성이 있는 축입니다.

### 1-2. 옵션 체인 기반 IV surface (개별 자산)

현재는 SPX VIX/VIX3M/VIX9D만 쓰고 있고, 나머지 10개 ETF에는 “자산 고유 IV 정보”가 없습니다.[^3]
개별 ETF 옵션에서:

- At-the-money IV term structure (1M, 3M, 6M)
- Moneyness-by-maturity grid에서 level/slope/curvature (skew)
- 자산별 VRP (IV² − RV)

를 빼와서, **자산별 IV Surface 블록**을 새로 만들면, 특히 Bond/Commodity 쪽에서 LASSO 대비 큰 갭을 만들 여지가 큽니다.[^1]

***

## 2. 구조적으로 LASSO보다 앞설 수 있는 모델링 시도

### 2-1. Multi-horizon / path-dependent 학습

LASSO는 보통 “단일 horizon 회귀”입니다.
문헌에서는 **여러 horizon(1, 5, 22, 60일 등)을 동시에 학습하는 multi-horizon 모델이 단일 horizon 대비 loss 기준 10%+ 개선**을 자주 보여줍니다.[^6][^7][^8]

시도 아이디어:

- 타깃을 $y^{(1)}, y^{(5)}, y^{(22)}, y^{(60)}$ 네 개로 두고,
    - Shared feature block + horizon-specific coefficient deviation 구조(“Ridge 기반 multi-task”)로 학습.
- 22일 예측 시에는:
    - 직접 예측 $\hat{y}^{(22)}$ + $\hat{y}^{(1)}$·누적, $\hat{y}^{(5)}$·누적을 함께 쓰는 **path-dependent aggregator** (예: 22일 = 4×5일 + 2×1일 조합).

이렇게 하면 LASSO처럼 “각 horizon 독립 회귀”보다 **정보를 재사용**해서, 특히 중·장기 horizon에서 loss 기준 10% 이상 개선이 현실적인 영역입니다.[^7][^6]

### 2-2. Regime-specific / mixture-of-experts Ridge

지금은 하나의 Ridge가 Low/High VIX 레짐을 모두 커버합니다.[^3]
VIX quantile에 따라 성능이 달라지는 게 이미 보였으니, **레짐별 전문가 모델**을 쓰면 LASSO보다 훨씬 유연해집니다.[^9][^3]

시도 아이디어:

- VIX 또는 IV_VRP를 기준으로 2~3개 레짐(Low/Medium/High VIX) 정의.
- 각 레짐마다 별도의 Ridge/Elastic Net을 학습하고, 테스트 시 현재 VIX로 라우팅.
- 부드럽게 하려면 **mixture-of-experts**: softmax(VIX, VRP)을 weight로 해서 세 Ridge 출력 가중합.

LASSO는 기본적으로 global linear model이라, 이런 **레짐별 곡선**을 표현하는 데 약합니다.
특히 High VIX 구간에서 이미 V71이 V36 대비 +0.163 R² 개선이 있으니, 레짐 전용 모델을 쓰면 여기서 LASSO 대비 10% 이상 상대 개선(QLIKE 기준)은 충분히 노려볼 만합니다.[^3]

### 2-3. Factor + idiosyncratic 구조 (low-rank + Ridge)

Cross-asset RV/VRP 문헌에서는 **공통 인자 + 자산별 잔차** 구조가 표준입니다.[^10][^9]

시도 아이디어:

- 11개 자산 LogRV를 패널로 보고,
    - PCA/PPCA 등으로 공통 factor 1–3개 추출.
    - Factor를 타깃으로 먼저 예측(Ridge + HF/IV/Alt).
    - 잔차(idiosyncratic)를 각 자산별 Ridge로 별도 예측.
- 최종 예측 = factor part + idiosyncratic part.

LASSO를 단일 회귀로 둘 때보다, 이렇게 구조를 반영하면 **“자산 간 공통 패턴 vs 개별 특이 패턴”을 분리해서 더 효율적으로 학습**할 수 있고, 패널 데이터 연구에서 오차 10–20% 감소가 자주 보고됩니다.[^11][^10]

***

## 3. 학습 목표를 “R²”가 아닌 “리스크 지표”로 바꾸는 시도

마지막으로, “성능 10% 증가”를 **R²가 아니라 QLIKE/MSE 기준**으로 정의하면 훨씬 현실적입니다. 실제 변동성 예측 논문도 주로 QLIKE, MSE 감소율을 봅니다.[^2][^1]

시도 아이디어:

- 손실 함수를 MSE → QLIKE 또는 “High VIX 구간에 가중치 준 MSE”로 바꿔서 학습.
- 동일 데이터에서 LASSO는 여전히 MSE 기반일 때,
    - V71은 **리스크 관리에 더 중요한 구간(고변동성)**에서 loss를 더 줄이도록 학습.

이 경우:

- “QLIKE 기준 LASSO 대비 10–20% loss 감소, DM test p<0.01”
- “High VIX 구간에서 tail risk 예측력 우위”

같은 스토리를 논문에서 내세울 수 있고, 숫자적으로도 현실적인 목표입니다.[^12][^1][^3]

***

### 한 줄로 정리하면

- **데이터 측면**: intraday RV/점프 + 개별 자산 옵션 IV surface 추가.
- **모델 측면**: multi-horizon multi-task Ridge, VIX 레짐 mixture-of-experts, factor+idiosyncratic 구조.
- **목표 측면**: R² 말고 QLIKE·MSE 감소율 10%+를 타겟.

이 세 축 중 **한두 개만 제대로 구현해도, LASSO 대비 “실질적인 10% 이상 개선”을 논문에서 설득력 있게 주장할 수 있는 구간**까지는 충분히 갈 수 있습니다.

<div align="center">⁂</div>

[^1]: https://www.sciencedirect.com/science/article/abs/pii/S0927539820300608

[^2]: https://www.garp.org/hubfs/Whitepapers/a2r1W000000iDb0QAE_RiskIntell.6.20.19.Whitepaper.Volatility.pdf

[^3]: v71_summary.md

[^4]: https://arxiv.org/html/2210.04797v3

[^5]: https://www.sciencedirect.com/science/article/abs/pii/S0927539899000134

[^6]: https://thesis.eur.nl/pub/38972/Vijn-S.-454110-.pdf

[^7]: https://www.semanticscholar.org/paper/Evaluating-Multi-Horizon-Volatility-Forecasting-of-Hu-Quaedvlieg/6fbb8889b6a16026ed54d8f2f9daa23282fdb943

[^8]: https://arxiv.org/html/2602.01157v2

[^9]: https://ux-tauri.unisg.ch/RePEc/usg/econwp/EWP-1501.pdf

[^10]: https://arxiv.org/pdf/2206.14114.pdf

[^11]: https://www.bis.org/publ/work667.pdf

[^12]: 05_experimental_results.md

