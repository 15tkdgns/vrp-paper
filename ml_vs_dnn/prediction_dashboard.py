import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')

# 딥러닝 라이브러리 임포트 (텐서플로우)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# 시계열 통계 라이브러리 임포트 (statsmodels)
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

st.set_page_config(page_title="주식 가격 예측 성능 비교", layout="wide")

st.title("머신러닝 및 딥러닝 주식 예측 모델 성능 비교 대시보드")
st.markdown("LSTM, GRU, Linear Regression, ARIMA, SVM 모델을 활용하여 주식 예측 성능을 산출하고 비교합니다.")

if not DL_AVAILABLE:
    st.error("TensorFlow 라이브러리가 설치되어 있지 않아 LSTM, GRU 모델을 사용할 수 없습니다.")
if not ARIMA_AVAILABLE:
    st.error("statsmodels 라이브러리가 설치되어 있지 않아 ARIMA 모델을 사용할 수 없습니다.")

st.sidebar.header("설정 메뉴")
uploaded_file = st.sidebar.file_uploader("주식 데이터셋 업로드 (CSV 포맷)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("업로드된 데이터 미리보기")
    st.dataframe(df.head())
    
    columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("예측 대상 단일 컬럼(Target) 선택", columns)
    
    seq_length = st.sidebar.slider("과거 참조 기간 (Sequence Length)", min_value=5, max_value=60, value=20)
    train_ratio = st.sidebar.slider("학습 데이터 비율", min_value=0.5, max_value=0.9, value=0.8)
    epochs = st.sidebar.slider("딥러닝 학습 에포크(Epochs)", min_value=1, max_value=50, value=5)
    
    if st.sidebar.button("모델 학습 및 평가 시작"):
        # 원본 타겟 데이터 준비
        data_raw = df[target_col].values.astype(float)
        
        # 스케일링 준비
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_raw.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:(i + seq_length), 0])
            y.append(scaled_data[i + seq_length, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        train_size = int(len(X) * train_ratio)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # 복원을 위한 실제값 세팅
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        plot_df = pd.DataFrame({"Actual": y_test_inv})
        
        # 1. Linear Regression (머신러닝)
        status_text.text("Linear Regression 모델 학습 중...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_pred_inv = scaler.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
        
        mse_lr = mean_squared_error(y_test_inv, lr_pred_inv)
        mae_lr = mean_absolute_error(y_test_inv, lr_pred_inv)
        results.append({"Model": "Linear Regression", "Type": "Machine Learning", "MSE": mse_lr, "MAE": mae_lr})
        plot_df["Linear Regression"] = lr_pred_inv
        progress_bar.progress(20)
        
        # 2. SVM (머신러닝)
        status_text.text("SVM (SVR) 모델 학습 중...")
        svm_model = SVR(kernel='rbf')
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_pred_inv = scaler.inverse_transform(svm_pred.reshape(-1, 1)).flatten()
        
        mse_svm = mean_squared_error(y_test_inv, svm_pred_inv)
        mae_svm = mean_absolute_error(y_test_inv, svm_pred_inv)
        results.append({"Model": "SVM", "Type": "Machine Learning", "MSE": mse_svm, "MAE": mae_svm})
        plot_df["SVM"] = svm_pred_inv
        progress_bar.progress(40)
        
        # 3. ARIMA (전통 시계열)
        status_text.text("ARIMA 모델 학습 중...")
        if ARIMA_AVAILABLE:
            try:
                # ARIMA는 단변량 시계열을 그대로 투입하는 것이 적합함 (스케일링 제외 원본데이터 적용)
                # 시퀀스 생성으로 인한 오프셋을 고려하여 train 구간과 test 구간 분리
                arima_train_raw = data_raw[:train_size + seq_length]
                
                # 예측 시간을 고려하여 간단히 전체 테스트 기간 예측 진행
                model = ARIMA(arima_train_raw, order=(5, 1, 0))
                model_fit = model.fit()
                arima_pred_raw = model_fit.forecast(steps=len(y_test_inv))
                
                mse_arima = mean_squared_error(y_test_inv, arima_pred_raw)
                mae_arima = mean_absolute_error(y_test_inv, arima_pred_raw)
                results.append({"Model": "ARIMA", "Type": "Traditional Time Series", "MSE": mse_arima, "MAE": mae_arima})
                plot_df["ARIMA"] = arima_pred_raw
            except Exception as e:
                st.warning(f"ARIMA 모델 학습 중 오류 발생: {e}")
        progress_bar.progress(60)
        
        # 4. LSTM (딥러닝)
        status_text.text("LSTM 모델 학습 중...")
        if DL_AVAILABLE:
            X_train_dl = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test_dl = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            lstm_model = Sequential()
            lstm_model.add(LSTM(32, input_shape=(seq_length, 1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train_dl, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            lstm_pred = lstm_model.predict(X_test_dl)
            lstm_pred_inv = scaler.inverse_transform(lstm_pred).flatten()
            
            mse_lstm = mean_squared_error(y_test_inv, lstm_pred_inv)
            mae_lstm = mean_absolute_error(y_test_inv, lstm_pred_inv)
            results.append({"Model": "LSTM", "Type": "Deep Learning", "MSE": mse_lstm, "MAE": mae_lstm})
            plot_df["LSTM"] = lstm_pred_inv
        progress_bar.progress(80)
        
        # 5. GRU (딥러닝)
        status_text.text("GRU 모델 학습 중...")
        if DL_AVAILABLE:
            gru_model = Sequential()
            gru_model.add(GRU(32, input_shape=(seq_length, 1)))
            gru_model.add(Dense(1))
            gru_model.compile(optimizer='adam', loss='mse')
            gru_model.fit(X_train_dl, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            gru_pred = gru_model.predict(X_test_dl)
            gru_pred_inv = scaler.inverse_transform(gru_pred).flatten()
            
            mse_gru = mean_squared_error(y_test_inv, gru_pred_inv)
            mae_gru = mean_absolute_error(y_test_inv, gru_pred_inv)
            results.append({"Model": "GRU", "Type": "Deep Learning", "MSE": mse_gru, "MAE": mae_gru})
            plot_df["GRU"] = gru_pred_inv
        progress_bar.progress(100)
        status_text.text("모든 모델 학습이 완료되었습니다.")
        
        # 결과값 테이블 표출 (인터랙티브 정렬 지원)
        st.subheader("예측 성능 비교 결과 (MSE / MAE 정렬 지원)")
        st.markdown("테이블 상단의 **MSE** 또는 **MAE** 항목을 클릭하시면 성능 지표 순서별 정렬이 가능합니다.")
        
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by="MSE").reset_index(drop=True)
        st.dataframe(res_df, use_container_width=True)
        
        # 선 그래프 표출
        st.subheader("실제 가격 대비 예측 모델 결과 시각화")
        st.line_chart(plot_df)

else:
    st.info("좌측 사이드바에서 비교할 주식 가격 데이터셋 파일(CSV)을 업로드해 주십시오.")

st.markdown("---")
st.header("선행연구 대조: 변동성 예측 모델 계보와 본 연구의 우위성")
st.markdown("과거 전통적 시계열 모델부터 머신러닝, 그리고 딥러닝(2026)에 이르기까지 22거래일 지평(22d)의 R² 성능 발전 계보입니다.")

# CSV 데이터 로드
try:
    history_df = pd.read_csv("ml_vs_dnn_prior_studies_comparison.csv")
    
    # 모델명 전처리를 통해 차트 표시 최적화
    history_df['Model_Year'] = history_df['Model'] + " (" + history_df['Year'].astype(str) + ")"
    
    # Plotly 시각화 (Category 별로 색상 다르게)
    fig = px.bar(history_df, 
                 x='Model_Year', 
                 y='Metric(22d_R2)', 
                 color='Category',
                 text='Metric(22d_R2)',
                 hover_data=['Reference', 'Key_Notes'],
                 title='선행연구 대비 모델 지표(R²) 성과 계보')
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="모델 (발표 연도)", yaxis_title="22d 예측 성과 (R²)", 
                      xaxis={'categoryorder':'array', 'categoryarray': history_df['Model_Year'].tolist()},
                      height=500, paper_bgcolor='white', plot_bgcolor='white')
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(history_df, use_container_width=True)
    
except Exception as e:
    st.error(f"선행연구 파일을 로드하는 데 실패했습니다. 파일이 같은 폴더 내에 있는지 확인해 주세요. 오류: {e}")

st.markdown("---")
st.header("실제 SCI급 저널 등재 논문 예측 성과 비교")
st.markdown("""
**[논문 추출에 적용된 엄격한 조건 (Search Query Framework)]**
*   **주제어 (Topic):** "stock price prediction" OR "stock volatility forecasting" OR "variance risk premium" OR "VRP"
*   **기간 (Publication Year):** 2016년 ~ 2026년 최신 문헌
*   **학술지 등급 (Quality):** SCI/SSCI 급 저널만 선별 (JCR Impact Factor 1.0 이상 우수 국제 학술지)
*   **학문 분야 (Categories):** Business, Economics, Operations Research & Management Science 한정
*   **분석 대상:** R², RMSE, Accuracy 등 예측 성능 검증 지표가 확실하게 기재된 Peer-reviewed Article (Preprint 및 Conference 제외)

위 조건을 통과한 실제 검증 논문 데이터(`real_sci_papers_verified.csv`)를 기반으로, 평가 지표(R², Accuracy, RMSE)별 성능 수치를 추출해 **ML(머신러닝)과 DNN(딥러닝)** 그룹을 색상으로 명확히 구분하여 비교 시각화했습니다. 
""")

# 엄격 검증된 실제 논문 데이터 로드 및 시각화
try:
    file_to_load = "real_sci_papers_verified.csv"
    real_df = pd.read_csv(file_to_load)
    
    # 고유 식별을 위한 Y축 키 생성 (바 차트 Aggregate 방지용)
    real_df['Model_Journal'] = real_df['Model'] + " (" + real_df['Journal'] + ", " + real_df['Year'].astype(str) + ")"
    
    # 왼쪽 두 줄 (모델명), 오른쪽 두 줄 (논문/저널명) 포맷팅
    def wrap_model(text):
        if ' / ' in text:
            return text.replace(' / ', '<br>/ ')
        elif '-' in text and len(text) > 10:
            return text.replace('-', '<br>-')
        else:
            return text

    real_df['Model_Split'] = real_df['Model'].apply(wrap_model)
    real_df['Journal_Split'] = "<b>" + real_df['Journal'] + "</b><br>(" + real_df['Year'].astype(str) + "년)"
    
    # Performance 열에서 숫자만 정규식으로 추출 (예: 'R2 0.40' -> 0.40)
    real_df['Metric_Value'] = real_df['Performance'].str.extract(r'([0-9]+\.[0-9]+|[0-9]+)').astype(float)
    # 54% 같은 퍼센트 수치는 100으로 나누어 소수점 통일
    real_df.loc[real_df['Performance'].str.contains('%', na=False), 'Metric_Value'] = real_df['Metric_Value'] / 100.0
    
    # 어떤 지표인지 추출 (R2, Accuracy, RMSE, MAE 등)
    real_df['Metric_Type'] = real_df['Performance'].str.extract(r'([A-Za-z2²]+)')[0].str.replace('R2', 'R²').str.upper()
    
    # ML(초록), DNN(파랑), ML/DNN(청록/녹색계열) 카테고리에 따른 색상 맵핑
    color_map = {'ML': '#2ca02c', 'DNN': '#1f77b4', 'ML/DNN': '#17becf'}
    
    # 왼쪽 Y축: 예측 대상(Target) 변수, 데이터프레임 원래 컬럼 사용
    # 막대 위 글자: 모델명 + IF 점수
    real_df['Bar_Text'] = "<b>" + real_df['Model'] + "</b>   (IF: " + real_df['ImpactFactor'].astype(str) + ")"
    
    # v2: JCR 임팩트 팩터 (IF) 단독 표출
    fig2 = px.bar(real_df, 
                 x='ImpactFactor', 
                 y='Model_Journal', # Y축 틱 레이블용 고유키 유지
                 color='Category',
                 color_discrete_map=color_map,
                 orientation='h',
                 text='Bar_Text',   # 막대그래프 위에 모델명+IF점수 표기
                 hover_data=['Title', 'Model', 'Metric_Value', 'Citations'],
                 title='SCI 논문 객관적 가치 평가: [v2] JCR 임팩트 팩터 (IF / 연구영향력)')
    
    fig2.update_traces(
        textposition='inside',      # 막대그래프 내부/위쪽으로 오버레이 
        insidetextanchor='end',     # 막대 우측 정렬
        textfont=dict(size=18, color='white') # 가독성을 위해 흰색 또는 밝은색 (일부 검정 혼합)
    )
    
    # 오른쪽에 논문/저널명 2줄 표기를 위한 Annotations 설정
    annotations = []
    for _, row in real_df.iterrows():
        annotations.append(dict(
            xref='paper', yref='y',
            x=1.01, y=row['Model_Journal'], # 실제 Y축 좌표 매칭
            text=row['Journal_Split'],       # 들어갈 텍스트 (오른쪽 두 줄)
            xanchor='left', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='black'),
            align='left'
        ))

    fig2.update_layout(
        yaxis_categoryorder='total ascending', 
        height=750, 
        yaxis_title="",
        xaxis_title="JCR Impact Factor",
        yaxis=dict(
            tickmode='array',
            tickvals=real_df['Model_Journal'],
            ticktext="<br>" + real_df['Target'] + "<br>", # 실제로 보여줄 왼쪽 표기 (예측 대상 변수명 Target)
            tickfont=dict(size=18, color='black') # 검정/큰 글씨
        ),
        xaxis=dict(tickfont=dict(size=14, color='black')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=220, r=450, t=120, b=50), # 우측 논문명 표시 여백 확보 (r=450), 좌측 Target(220)
        legend=dict(
            title_font_size=20, 
            font=dict(size=18, color='black'),
            orientation="h",       # 가로 배열 범례
            yanchor="bottom", y=1.02, # 상단 배치
            xanchor="right", x=1
        ),
        annotations=annotations
    )
    
    st.markdown("**그래프 범례 안내**")
    st.markdown("- **ML (초록색):** 전통적 머신러닝 최적화 (LASSO, XGBoost, SVM 등)\n- **DNN (파란색):** 딥러닝 아키텍처 (LSTM, CNN, Attention 등)\n- **ML/DNN (청록/녹색):** 머신러닝과 딥러닝, 트리 구조 등을 혼합한 앙상블 모형")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("실제 논문 데이터셋 원본")
    st.dataframe(real_df, use_container_width=True)
    
except Exception as e:
    st.error(f"실제 SCI 논문 데이터 파일을 읽는 데 실패했습니다. 오류: {e}")
