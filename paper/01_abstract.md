# Abstract

본 연구는 Cross-Asset Volatility Risk Premium(VRP) 예측을 위한 통합 머신러닝 프레임워크를 제안한다. 기존 VRP 연구가 주로 S&P 500 단일 자산에 한정되고 자산별 고유 옵션 데이터를 필요로 하는 반면, 본 연구는 OHLCV 데이터와 Global VIX를 활용하여 주식, 채권, 원자재 등 다수 자산군의 VRP를 통합적으로 예측하는 경제적(parsimonious) 접근법을 제시한다.

11개 ETF(SPY, QQQ, IWM, EFA, EEM, TLT, IEF, AGG, GLD, SLV, USO)의 2010-2024년 일별 데이터를 대상으로 70회 이상의 체계적 실험을 수행하였다. 전방 Realized Volatility를 타겟으로 사용하여 데이터 중첩 문제를 완전히 방지하고, 1일부터 365일까지 8개 예측 기간(horizon)에 걸쳐 7개 모델을 체계적으로 비교하였다.

핵심 발견은 다음과 같다: (1) **예측 기간에 따라 최적 모델이 달라진다**. 단기(1~22일)에서는 37개 피처를 활용한 Ridge+XGBoost 가중 앙상블(V71, 22d 풀링 R²=0.803)이, 중기(60~180일)에서는 3개 Range 기반 피처를 사용한 Bi-LSTM+Attention 모델(60d 풀링 R²=0.796)이, 초장기(365일)에서는 Random Forest(풀링 R²=0.705)가 최고 성능을 달성하였다. 59개 확장 피처의 ElasticNet(V74, 22d R²=0.807)이 최종 최고 성능을 기록하였다. (2) **LSTM은 소수 핵심 피처에 최적화될 때 다수 피처 선형 모델을 능가**하며, 특히 개별 자산 수준(중위 R²)에서 전 구간 최고 성능을 보였다. (3) 문헌 벤치마크(HAR-RV, HAR-CJ, LASSO-HAR, Random Forest) 대비 통계적으로 유의한 성능 우위를 확인하였다(DM test, p<0.001).

본 연구의 주요 기여는 다음과 같다: (1) 전방 RV 타겟을 통한 엄밀한 multi-horizon 평가 프레임워크 제안, (2) OHLC 기반 Range 추정량(RogersSatchell, GarmanKlass)의 중기 변동성 예측에서의 핵심적 역할 실증, (3) 예측 기간별 최적 모델 전략의 체계적 제시, (4) 기존 문헌이 다루지 않는 60일~365일 장기 예측 분석.

**Keywords**: Volatility Risk Premium, Cross-Asset, Realized Volatility, Range-based Estimators, Multi-Horizon Forecasting, LSTM, Ridge Regression, Machine Learning
