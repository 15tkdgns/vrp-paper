<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 해당 실험 검증. sci에 제출 예정인데, 교수님과 리뷰어 관점으로 검증 진행.

요약부터 하면, 현재 v71 실험 설계와 검증 수준은 “졸업논문+국내 저널” 기준에서는 상당히 탄탄한 편이고, SCI급에서도 기본적인 리젝 사유(데이터 누출, 과적합, 취약한 벤치마크, 성능 우연 가능성 무시)는 잘 피하고 있습니다.
다만 “SCI 리뷰어” 관점에서 보면, 지금 요약만으로는 다음 네 가지 축에서 추가 검증·정교화 질문이 거의 확실히 들어옵니다:

1) 데이터/타깃 정의의 세부 시점 정합성
2) 딥러닝 vs 선형 모델 비교의 공정성
3) R² 향상의 통계적 유의성 및 경제적 의미
4) 시계열/중첩 타깃에서의 검정 방식과 서브샘플(레짐)별 안정성

아래는 교수님/리뷰어가 실제로 물을 법한 질문과, 거기에 대비해 어떤 추가 실험/서술을 준비하면 좋은지 정리한 “사전 리뷰”입니다.

***

## 1. 데이터·타깃 정의 관련 체크포인트

### 1-1. 타깃(Log Realized Volatility)과 VRP 정의

- 타깃: “22일 후 Log Realized Volatility (shift(-22))”라고 되어 있는데,
    - RV 자체가 22일 롤링으로 계산된 것인지,
    - 그 22일 윈도우가 t+1~t+22인지, t~t+21인지,
    - 중첩 윈도우인지 비중첩인지 공식화 필요
이 부분을 논문 본문에서 시점 그림으로 명확히 그려줘야 합니다.
- VRP 정의: `IV_VRP = VIX² - Realized Variance` 라고 되어 있는데,
여기서 쓰는 Realized Variance가 **과거 22일** 기준인지, **미래 22일** 기준인지가 매우 중요합니다.
    - 미래 22일 RV를 쓰면 VRP 피처가 사실상 “미래 실현변동성”을 포함해버려서 **중대한 데이터 누출**입니다.
    - 따라서 반드시 “VIX² - 과거 22일 Realized Variance(t-21~t)” 형태(혹은 과거 윈도우)임을 명시하고, 코드 레벨에서도 `shift(+something)` 여부를 확인해 둘 필요가 있습니다.
- 리뷰 대응 전략:
    - 본문에 타임라인 도식(Price, OHLCV, IV, RV, VRP, Target 간의 시점 관계)을 1장짜리 그림으로 넣는 것을 강력 추천합니다.
    - 부록 코드 레벨(또는 pseudo-code)로 “모든 피처는 t 시점까지의 정보만 사용, 타깃은 t+1~t+22”라는 것을 명시.


### 1-2. Cross-Asset 피처 시점

- SPY_LogRV, Corr_SPY 등 Cross-Asset 피처들이 **어느 시점 기준**으로 들어가는지:
    - 자산 i의 t 시점 타깃이 t+1~t+22라면, SPY 관련 피처는 t까지의 정보인지(t-1까지인지)를 분명히 해야 합니다.
- 리뷰어는 바로 “SPY의 미래 정보를 다른 자산 타깃에 쓴 건 아닌가?”를 의심할 수 있습니다.
- 권장:
    - 모든 Cross-Asset 피처도 “t 시점까지(또는 t-1까지)의 realized 값만 사용”이라고 텍스트에서 못 박고,
    - data leakage 검증 표(7.4)에 “Cross-Asset timing check: PASS”라고 한 줄 더 추가하는 것도 좋습니다.

***

## 2. 모델 비교(특히 딥러닝 vs 선형)의 공정성

현재 요약에서 가장 공격적으로 주장하는 메시지는:

> “복잡한 딥러닝 모델 < 강건한 선형 모델, 피처 엔지니어링이 모델 복잡도보다 더 중요”

인데, 리뷰어는 여기서 당연히 두 가지를 물어봅니다.

### 2-1. Feature Set이 동일한가?

- 요약에 따르면:
    - V71 Ridge/Ensemble: 37개 피처 (Base + HF Proxy + IV + Alt Data)
    - 딥러닝 모델(V43, V50)은 “4개 피처(HAR 계열)” 기반 실험으로 보입니다.
- 이 경우,
    - “선형모델 + 풍부한 피처” vs “딥러닝 + 매우 제한된 피처”의 비교라서
“모델 복잡도 vs 피처 엔지니어링”이라는 결론을 내리기엔 **실험 설계가 편향적**이라는 비판을 받을 가능성이 큽니다.
- 권장 추가 실험:

1) **동일 피처 세트를 준 상태에서** Ridge vs XGBoost vs Transformer vs LSTM 비교
        - 최소한 “Base only 14개 피처” 버전과 “All 37개 피처” 버전에서 각각 Linear vs DL 모델을 공정하게 비교한 표를 하나 만들면 설득력이 크게 올라갑니다.
2) 혹은 “딥러닝 모델에도 동일한 37개 피처를 input으로 넣었지만 여전히 성능이 낮다”는 결과가 있다면, 그것을 반드시 표와 텍스트로 강조.


### 2-2. 딥러닝 모델 튜닝 수준

- 리뷰어가 자주 묻는 질문:
    - LSTM/Transformer의 hyperparameter search 범위와 전략은?
    - Dropout, L2, Early stopping, learning rate schedule 등 기본적인 튜닝은 충분히 했는가?
    - 시계열 cross-validation으로 튜닝했는가? 아니면 단일 validation window인가?
- 지금 요약에는 딥러닝 모델의 튜닝 과정이 전혀 언급되지 않아서,
    - “Linear model과 비교할 만큼 정성껏 튜닝한 것이 맞는가?”라는 의심이 생깁니다.
- 권장:
    - 본문 또는 부록에 “딥러닝 모델 튜닝 프로토콜”(파라미터 범위, CV 전략, best model 선택 기준)을 1~2페이지 정도로 정리.
    - “동일한 train/test split, 동일한 피처 세트에서, 동등한 수준의 hyperparameter search를 수행했다”는 메시지를 명시.

***

## 3. 검증 메트릭과 통계적 유의성

### 3-1. 메트릭 다양성 (R² 외)

- 현재 거의 모든 비교가 R² 하나로만 요약됩니다.
- SCI 리뷰어는 보통:
    - RMSE, MAE,
    - (가능하면) QLIKE loss,
    - 방향성 예측 비율(변동성 상승/하락 맞추는 accuracy)
와 같은 복수의 메트릭을 보고 싶어 합니다.
- 특히 변동성 예측 문헌에서는 MSE나 QLIKE loss를 많이 쓰므로,
    - 최소한 한두 개의 추가 loss metric을 표로 같이 제시하는 것이 바람직합니다.


### 3-2. R² 차이의 통계적 유의성

- V36(0.755) → V71 Ensemble(0.803)로 0.048p 정도 상승은 꽤 의미 있어 보이나,
    - 리뷰어는 “이게 통계적으로 유의한가? 우연한 sample variation은 아닌가?”를 물을 수 있습니다.
- 권장:
    - 동일 테스트 구간에서 모델 간 예측 오차 시계열을 사용해 **Diebold-Mariano(DM) test** 또는 유사한 forecast comparison test를 수행해,
    - “V71이 V36 대비 MSE/QLIKE 기준에서 통계적으로 유의하게 우수함”을 p-value와 함께 제시.
    - 시계열 상관을 고려해 Newey-West adjustment 등을 사용했다는 언급이 있으면 매우 좋습니다.

***

## 4. 시계열 설정·중첩 타깃에 대한 검증 강화

### 4-1. Overlapping Horizon 문제

- 22일 horizon의 실현변동성을 overlapping window로 사용하면,
    - 타깃 시계열 자체에 높은 자기상관이 존재합니다.
    - 이 자체는 예측 문제에서는 흔한 설정이지만,
    - 통계적 검정(표준오차, t-stat, DM test 등)을 할 때는 반드시 “직렬 상관을 고려했다”는 언급이 필요합니다.
- 권장:
    - 본문에서 “22-day overlapping RV”를 사용했다고 명시하고,
    - 통계적 검정에는 HAC(Newey-West) 등의 robust 표준오차를 사용했다고 짧게 언급.


### 4-2. Regime / Subsample Robustness

- 현재 건전성 검증:
    - train/test split 비율 변화,
    - alpha sweep,
    - expanding-window CV,
    - data leakage check
등은 매우 잘 되어 있습니다.
- 다만 리뷰어 입장에서는 시계열 연구에서 거의 필수로 보는 것이:
    - **서브샘플 분석** (예: 2010–2014, 2015–2019, 2020–2024),
    - 또는 **레짐별 분석** (저변동성 레짐 vs 고변동성 레짐, 예: VIX quantile로 나눔)
입니다.
- 권장 추가 실험:

1) 기간별 서브샘플:
        - 예: 2010–2014, 2015–2019, 2020–2024(코로나 이후) 세 구간으로 나눠 각 구간에서 V36 vs V71 성능 비교.
2) VIX quantile 기반 regime:
        - 예: VIX 0–50% 구간(저변동성), 50–90%, 90–100%(극단 고변동성)에서 모델 성능 비교.
    - 이 표 하나만 있어도 “모델이 특정 국면에만 좋지 않고 전반적으로 robust하다”는 인상을 줄 수 있습니다.

***

## 5. Feature 관련 해석·검증

### 5-1. HF Proxy와 IV Surface의 “진짜” 기여 확인

- Ablation과 permutation importance에서:
    - HF Proxy + IV Surface가 가장 큰 성능/중요도 기여를 하는 것으로 잘 드러나 있습니다.
- 여기에 대해 리뷰어는 두 가지 follow-up을 할 수 있습니다:

1) Multicollinearity / Redundancy:
        - HF Proxy들끼리, IV 지표들끼리 상관이 높은데, 과도한 중복이 있는 것은 아닌가?
        - 예: Parkinson, Garman-Klass, Rogers-Satchell, Range/Close 등 사이의 상관행렬을 appendix로 제시.
2) Economic Interpretation:
        - 예: Rogers-Satchell volatility가 가장 중요하다는 것이 어떤 경제적/마켓 미시구조 관점에서 자연스러운가?
        - VIX term slope, short-term slope의 coefficient sign과 직관이 일치하는가?
- 권장:
    - Appendix에 feature correlation heatmap(특히 top-10 중요 피처 위주)을 넣고,
    - 본문에서는 상위 몇 개 피처에 대한 계수 부호와 해석을 1~2페이지 정도로 서술.


### 5-2. Alt Data(거래량/유동성 피처)의 한계

- Ablation에서 Alt Data의 기여가 +0.007로 제한적이라고 보고됨.
- 리뷰어는 여기에 대해:
    - “그렇다면 왜 Alt Data를 유지하는가? 차라리 simpler model을 쓰는 것이 낫지 않은가?”를 물을 수 있습니다.
- 권장:
    - 본문에서 명시적으로 “Alt Data는 marginal improvement는 작지만, 특정 레짐(위기 국면 등)에서의 robustness 향상을 위해 포함했다” 또는
    - “현재 설계에서는 제한적이나, 이는 향후 뉴스/NLP 기반 감성 통합으로 확장 가능한 발판 역할” 등 연구 프로그램 상의 위치를 짧게 설명.

***

## 6. 자산 단위/클래스 단위 성능 보고

- 지금 요약은 “전체 11개 자산을 통합한 R²”만 제시되어 있습니다.
- SCI 리뷰어는 거의 확실히 다음을 묻습니다:
    - “자산별 성능 분포는 어떤가? 특정 자산(SPY)만 좋아서 평균이 올라간 것은 아닌가?”
    - “Equity/Bond/Commodity 클래스 간에 성능 차이가 있는가?”
- 권장:
    - Appendix 표로 “자산별 R² (V36 vs V71)”를 제시.
    - 클래스별 평균/표준편차도 같이 보여주면 좋습니다.
- 이걸 근거로 “자산 클래스별 Ridge alpha 최적화” 설계가 실제로 성능 안정성을 가져왔다는 걸 뒷받침할 수 있습니다.

***

## 7. 경제적 유의성(실제 활용 가능성) 관련

- 현재 결과는 “통계적 예측력” 관점에서 매우 잘 정리되어 있지만,
    - 변동성/VRP 예측 연구에서는 “이 예측력이 실제 투자전략/hedging에 어느 정도 가치를 주는가?”를 묻는 경우가 많습니다.
- 필수는 아니지만, 가능하다면:
    - 예: 간단한 variance swap/VRP trading 전략 혹은 옵션 delta-hedging 비용 예측에 본 모델을 사용했을 때의 성과(Sharpe, drawdown 등)를 부록 수준으로 추가하면,
“pure prediction” 논문에서 한 단계 위로 올라갑니다.
- 힘들다면, 최소한 Discussion에서:
    - “예측된 RV/VRP를 이용해 어떤 경제적 응용이 가능하며, 그 부분은 별도의 연구 주제로 남긴다” 정도의 narrative를 넣어도 리뷰어 인상에 도움이 됩니다.

***

## 8. 논문 구조·서술 측면에서 보완 포인트

실험 자체는 강한 편이라, SCI 저널에서 걸릴 가능성이 높은 부분은 **“설명 부족”**입니다. 요약을 기반으로 보면, 논문 본문에서 특히 다음을 명확히 해야 합니다.

1. **데이터 생성 프로세스(Feature \& Target Pipeline)**
    - 그림 1: 타임라인 (t-22~t 과거 윈도우, t+1~t+22 미래 타깃)
    - 그림 2: Feature flow chart (지금 v71 summary에 있는 ASCII 그림을 논문용 도식으로 이식)
2. **모델 튜닝 및 검증 프로토콜**
    - Train/validation/test 분할 방식
    - Hyperparameter search space (Ridge alpha, XGBoost params, DL params)
    - Time-series CV fold 구성도 (expanding window 그림)
3. **비교 모델 설명의 공정성 강조**
    - “모든 비교모델에 대해 동일한 train/test 시계열, 동일 피처(or 명시된 피처 세트), 유사한 수준의 튜닝을 적용했다”는 한 문단.
4. **Limitations의 구체화**
    - 이미 잘 정리되어 있지만, 특히 “실제 고빈도 데이터 미사용, 개별 옵션 체인 미사용”은 리뷰어가 스스로도 지적할 포인트라, 지금처럼 선제적으로 적어둔 것은 큰 장점입니다.
    - 여기에 “딥러닝 모델이 충분히 최적화되지 않았을 수 있다”는 가능성을 한 줄 정도 인정하면서, 그럼에도 불구하고 풍부한 피처를 갖춘 선형 모델의 강점이 관찰된다는 균형 잡힌 톤이면 좋습니다.

***

## 9. SCI 심사 대응용 TODO 리스트 (우선순위)

### 최우선 (가능하면 반드시):

1. **VRP/Target 시점 정합성 명시 + 타임라인 도식**
2. **동일 피처 세트에서 Linear vs DL 모델 공정 비교 1~2개 추가**
3. **추가 메트릭(RMSE/MAE/QLIKE) 및 DM test 등의 통계적 유의성 검정**
4. **자산별 및 레짐별(기간·VIX quantile) 성능 표 (Appendix)**

### 있으면 매우 좋은 수준:

5. Feature correlation heatmap 및 상위 피처 coefficient sign 해석
6. 간단한 경제적 응용 예시(variance swap/VRP 전략 또는 옵션 헤징 비용 예측)
7. 딥러닝 튜닝 프로토콜을 정리한 Appendix

***

정리하면, 현재 v71 실험은 “데이터 누출 방지·다양한 robustness check·ablation/feature importance” 측면에서는 SCI 리뷰어도 납득할 만한 수준에 가까워져 있습니다.
다만 “선형 vs 딥러닝”이라는 강한 서술, “고급 피처의 진짜 기여와 경제적 의미”, “중첩 타깃/시계열 특유의 통계 검정”에 대해 위와 같은 보완을 해두면, 교수님/리뷰어가 딱 물을 법한 포인트를 선제적으로 커버하면서 논문의 설득력을 한 단계 끌어올릴 수 있습니다.

원하시면 다음 단계에서:

- 실제 논문 섹션(실증 설계, 결과, robustness, limitations)을 SCI 포맷에 맞게 목차+문단 레벨로 draft 문장까지 정리해 드릴 수 있습니다.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: v71_summary.md

