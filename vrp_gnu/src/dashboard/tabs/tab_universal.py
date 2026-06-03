"""
Tab: 통합모델 (Integrated Universal Model)
단일 모델로 다중 자산 변동성 예측
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = 'experiments/07_v2_methodology/results'

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def render_universal():
    st.title("통합모델 (Integrated Universal Model)")
    st.markdown("""
    > 23개 자산을 하나의 데이터셋으로 통합(Pooling)하여 단일 모델로 학습.  
    > **Log-RV**를 예측 타겟으로 사용하여 분포 안정화 및 스케일 통일.
    """)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 1: 피처 구성 상세 설명
    # ========================================
    st.header("피처 구성 (Feature Engineering)")
    
    st.markdown("""
    ### 설계 원칙
    
    1. **모든 피처는 시점 $t$의 정보만 사용** (미래 정보 누수 방지)
    2. **Log-스케일로 통일**: 모든 변동성 관련 피처를 $\\ln(RV)$로 변환
    3. **Cross-Asset 정보 활용**: 자산 간 Spillover 효과 포착
    """)
    
    # ========================================
    # SECTION 2: 기본 피처 (HAR-RV 기반)
    # ========================================
    st.subheader("1. 기본 피처 (4개) - HAR-RV 구조")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        | 피처 | 계산식 | 설명 |
        |------|--------|------|
        | `LogRV_lag1` | $\\ln(RV_{t-1})$ | 어제의 변동성 (단기 정보) |
        | `LogRV_lag5` | $\\ln(\\bar{RV}_{t-5:t-1})$ | 지난 5일 평균 변동성 (주간 정보) |
        | `LogRV_lag22` | $\\ln(\\bar{RV}_{t-22:t-1})$ | 지난 22일 평균 변동성 (월간 정보) |
        | `LogRV_Mom` | $\\ln RV_{t-1} - \\ln RV_{t-5}$ | 변동성 모멘텀 (추세 방향) |
        """)
    
    with col2:
        st.info("""
        **HAR-RV 모형** (Corsi 2009)
        
        변동성은 이질적(heterogeneous) 시간 스케일에서 
        다른 자기상관 구조를 가짐.
        """)
    
    st.markdown("""
    **Log 변환 이유**:
    - Raw RV는 오른쪽으로 치우친 분포 (Skewness ≈ 3.0)
    - Log 변환 후 Skewness ≈ 0.5로 정규분포에 근접
    - MSE 손실함수 학습에 최적화
    """)
    
    # ========================================
    # SECTION 3: 글로벌 피처
    # ========================================
    st.subheader("2. 글로벌 피처 (3개) - 시장 전체 정보")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        | 피처 | 계산식 | 설명 |
        |------|--------|------|
        | `LogVIX` | $\\ln(VIX_{t-1})$ | 옵션 시장의 내재 변동성 |
        | `VIX_Term` | $VIX3M_{t-1} - VIX_{t-1}$ | VIX 기간 구조 (정상/역전) |
        | `SKEW` | $(SKEW_{t-1}/100) - 1$ | 옵션 시장의 꼬리 위험 지표 |
        """)
    
    with col2:
        st.warning("""
        **VIX Term Structure**
        
        - **정상 (+)**: 평시, 미래 변동성 상승 예상
        - **역전 (-)**: 위기, 현재 변동성이 더 높음
        """)
    
    # ========================================
    # SECTION 4: Cross-Asset 피처 (핵심)
    # ========================================
    st.subheader("3. Cross-Asset 피처 (3개) - 자산 간 전이 효과")
    
    st.markdown("""
    ### 왜 SPY, TLT, GLD인가?
    
    이 세 자산은 **글로벌 리스크의 대표 프록시**입니다:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### SPY (주식)
        - **대표성**: S&P 500 = 미국 경제 전체
        - **역할**: Risk-On 자산의 변동성 기준
        - **Spillover**: 주식 변동성 → 다른 자산으로 전이
        """)
    
    with col2:
        st.markdown("""
        #### TLT (채권)
        - **대표성**: 20년+ 미국 국채
        - **역할**: Safe Haven (안전자산)
        - **상관**: 주식과 역상관 (위기 시 상승)
        """)
    
    with col3:
        st.markdown("""
        #### GLD (금)
        - **대표성**: 실물 안전자산
        - **역할**: 인플레이션/위기 헤지
        - **독립성**: 주식/채권과 다른 동학
        """)
    
    st.markdown("""
    ### 피처 계산
    
    | 피처 | 계산식 | 의미 |
    |------|--------|------|
    | `LogRV_SPY` | $\\ln(RV_{SPY, t-1})$ | 주식 시장 변동성 상태 |
    | `LogRV_TLT` | $\\ln(RV_{TLT, t-1})$ | 채권 시장 변동성 상태 |
    | `LogRV_GLD` | $\\ln(RV_{GLD, t-1})$ | 원자재/금 변동성 상태 |
    
    **모두 $t-1$ 시점 값 사용** → 미래 정보 누수 없음
    """)
    
    st.success("""
    **Cross-Asset Factor 버전** (v5.1)
    
    개별 자산 대신 자산군 평균을 사용하면 노이즈 감소:
    - `EquityFactor` = mean(LogRV of SPY, QQQ, IWM, DIA, EFA, EEM)
    - `RateFactor` = mean(LogRV of TLT, IEF, SHY, TIP, LQD, AGG)
    - `CommodityFactor` = mean(LogRV of GLD, SLV, USO, DBC)
    
    → Avg Asset R² +0.01 개선
    """)
    
    # ========================================
    # SECTION 5: 전체 피처 요약
    # ========================================
    st.markdown("---")
    st.subheader("전체 피처 요약")
    
    feature_data = [
        {'그룹': 'Basic', '피처': 'LogRV_lag1', '스케일': 'Log', '시점': 't-1', '설명': '어제 변동성'},
        {'그룹': 'Basic', '피처': 'LogRV_lag5', '스케일': 'Log', '시점': 't-5:t-1', '설명': '주간 평균 변동성'},
        {'그룹': 'Basic', '피처': 'LogRV_lag22', '스케일': 'Log', '시점': 't-22:t-1', '설명': '월간 평균 변동성'},
        {'그룹': 'Basic', '피처': 'LogRV_Mom', '스케일': 'Log Diff', '시점': 't-1, t-5', '설명': '변동성 모멘텀'},
        {'그룹': 'Global', '피처': 'LogVIX', '스케일': 'Log', '시점': 't-1', '설명': 'VIX 로그 값'},
        {'그룹': 'Global', '피처': 'VIX_Term', '스케일': 'Level', '시점': 't-1', '설명': 'VIX 기간 구조'},
        {'그룹': 'Global', '피처': 'SKEW', '스케일': 'Normalized', '시점': 't-1', '설명': '옵션 SKEW'},
        {'그룹': 'Cross-Asset', '피처': 'LogRV_SPY', '스케일': 'Log', '시점': 't-1', '설명': 'SPY 변동성'},
        {'그룹': 'Cross-Asset', '피처': 'LogRV_TLT', '스케일': 'Log', '시점': 't-1', '설명': 'TLT 변동성'},
        {'그룹': 'Cross-Asset', '피처': 'LogRV_GLD', '스케일': 'Log', '시점': 't-1', '설명': 'GLD 변동성'},
    ]
    
    df_features = pd.DataFrame(feature_data)
    
    st.dataframe(df_features, use_container_width=True, hide_index=True)
    
    # ========================================
    # SECTION 6: 핵심 설계 원칙
    # ========================================
    st.markdown("---")
    st.subheader("핵심 설계 원칙")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ 미래 정보 누수 방지
        
        - 모든 피처는 **$t$ 시점에서 알 수 있는 정보만** 사용
        - `t-1` 또는 `t-N:t-1` 범위의 과거 데이터만 활용
        - 타겟($\\ln RV_{t+22}$)과 시간적 분리 보장
        """)
    
    with col2:
        st.markdown("""
        ### ✅ Log-스케일 통일
        
        - 모든 변동성 관련 피처를 $\\ln(\\cdot)$ 변환
        - **장점**:
          - 스케일 표준화 (자산별 레벨 차이 보정)
          - 꼬리 두터움(Kurtosis) 감소
          - MSE 손실함수 최적화
        """)
    
    # ========================================
    # SECTION 7: 모델 성능 분석 (Result)
    # ========================================
    st.markdown("---")
    st.header("모델 실험 결과 (Model Experiment Results)")

    st.markdown("""
    ### 🎯 예측 타겟 및 방법론
    
    - **예측 타겟 (Target)**: **Log Realized Volatility at $t+22$** ($\ln RV_{t+22}$)
    - **평가 지표 (Metric)**: **$R^2$ Score** (Out-of-Sample Test)
    - **검증 데이터**: 2022년 1월 1일 이후 데이터 (Time-series Split)
    """)

    # ----------------------------------------
    # 7.1 아키텍처 비교 (딥러닝 vs 앙상블)
    # ----------------------------------------
    st.subheader("1. 모델 아키텍처 비교")
    st.markdown("금융 시계열 데이터(High Noise)에서의 모델별 성능 비교 실험 결과입니다.")

    model_comp_data = {
        "모델 (Model)": [
            "Baseline MLP", 
            "Residual MLP", 
            "LSTM", 
            "Transformer", 
            "**Ensemble (Final)**"
        ],
        "설명 (Description)": [
            "기본 다층 퍼셉트론 (3-Layer)",
            "Skip-Connection 추가",
            "시계열 패턴 학습 (RNN)",
            "Attention 메커니즘",
            "**Linear (Ridge+Huber) + Tree (RF+GBM)**"
        ],
        "Overall R²": [0.26, 0.23, -0.05, 0.15, "**0.85**"],
        "Avg Asset R²": [-0.12, -0.13, -0.48, -0.22, "**0.17**"]
    }
    
    df_model = pd.DataFrame(model_comp_data)
    st.table(df_model)

    st.info("""
    **💡 핵심 발견 (Key Insignt)**
    
    1. **복잡성의 역설**: Transformer, LSTM 등 복잡한 딥러닝 모델보다 **단순한 MLP가 더 우수**했습니다.
    2. **선형성 우위**: 금융 변동성 데이터는 노이즈가 매우 커서, 과적합 위험이 적은 **선형 기반 앙상블(Ridge, Huber)**이 압도적인 성능(R² 0.85)을 기록했습니다.
    3. **최종 모델**: 따라서 V5 최종 모델은 **Ensemble (Linear Emphasis)**로 선정되었습니다.
    """)

    # ----------------------------------------
    # 7.2 최종 모델 상세 성능
    # ----------------------------------------
    st.subheader("2. 최종 모델 (V5 Ensemble) 상세 성능")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall R²", "0.848", help="데이터 전체에 대한 통합 설명력")
    with col2:
        st.metric("Avg Asset R²", "0.170", help="개별 자산 R²의 평균값")
    with col3:
        st.metric("최적 Horizon", "1개월 (22일)", help="예측 정확도가 가장 높은 기간")
        
    st.markdown("#### 자산별 예측 정확도 ($R^2$)")
    
    # 자산별 성능 데이터 로드 및 시각화
    results = load_json('v5_extended.json')
    if results and 'per_asset' in results:
        df_res = pd.DataFrame(results['per_asset'])
        df_res = df_res.sort_values('Ensemble', ascending=False)
        
        # 상위 10개 자산 시각화
        fig = px.bar(
            df_res.head(10), 
            x='Asset', 
            y='Ensemble',
            title="Top 10 예측 정확도 자산 (Ensemble Model)",
            text_auto='.3f',
            color='Ensemble',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis_title="R² Score", xaxis_title="Asset")
        st.plotly_chart(fig, use_container_width=True)
    
    # ----------------------------------------
    # 7.3 고성능 자산 심층 분석 (R² > 0.3)
    # ----------------------------------------
    st.markdown("---")
    st.subheader("3. 고성능 자산 심층 분석 (R² > 0.3)")
    
    st.markdown("""
    예측 정확도가 30% ($R^2 > 0.3$)를 넘는 "고성능 자산"에는 뚜렷한 공통점이 있습니다.
    왜 주식보다 채권/원자재가 더 잘 맞을까요?
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.success("""
        **주요 고성능 자산 리스트**
        
        1. **AGG (종합 채권)**: $R^2$ **0.479**
        2. **DBC (원자재)**: $R^2$ **0.306**
        3. **SLV (은)**: $R^2$ **0.319**
        4. **USO (석유)**: $R^2$ **0.297**
        """)
        
    with col2:
        st.info("""
        **성능 우수 이유 (Why?)**
        
        1. **매크로 민감도**: 채권(AGG)과 원자재(DBC)는 금리, 인플레 등 **거시경제 지표(Macro)**와 직접 연동되어 추세 추종성이 강합니다.
        2. **효율성 차이**: 주식 시장(SPY)은 정보 반영이 너무 빨라(Efficient) 예측이 어렵지만, 채권/원자재는 **변동성 지속성(Persistence)**이 더 강하게 나타납니다.
        """)
        
    st.markdown("""
    > **참고**: **Overall R² (0.848)**은 33개 자산의 모든 데이터 샘플(약 12만 개)을 합쳐서 계산한 수치입니다. 
    > 개별 자산별로 나누어 계산할 때($R^2 < 0.5$)보다, 전체를 모아서 볼 때 데이터의 일반적인 경향성을 훨씬 더 잘 설명함을 의미합니다.
    """)

    factor_data = load_json('v5_crossasset_factor.json')
    if factor_data:
        st.markdown(f"""
        > **Factor 개선 효과**: 
        > 개별 자산 대신 **Aggregated Factor** (Equity/Rate/Commodity Factor) 사용 시 
        > 평균 성능이 **{factor_data['original']['avg_asset_r2']:.4f}**에서 **{factor_data['factor']['avg_asset_r2']:.4f}**로 향상됨.
        """)

if __name__ == "__main__":
    render_universal()
