import pandas as pd
import numpy as np
import os

path = "C:/Users/15tkd/OneDrive/바탕 화면/vrp_paper"
os.makedirs(os.path.join(path, "tables_csv"), exist_ok=True)

# 1. Checklist CSV
checklist = pd.DataFrame({
    'Figure': ['Figure 1. 흐름도', 'Figure 2. 검증 절차', 'Figure 3. 기간별 성능', 'Figure 4. 22d 성능', 'Figure 5. 누적 기여도', 'Figure 6. 중요도 기간이동', 'Figure 7. VIX 분위수 비교', 'Figure 8. 모델 합의도', 'Figure 9. 누적성과', 'Figure 10. 자산군 분포'],
    '데이터상태': ['데이터 불필요', '데이터 불필요', '확인된 수치만 추출(나머지 NaN)', '확인된 수치만 추출(나머지 NaN)', '확인된 수치 일부추출(나머지 NaN)', '상세 수치 미확인(전체 NaN)', '확인된 수치 전체추출', '일부 기간만 추출(나머지 NaN)', '제외됨', '개별 자산 R2 미확인(전체 NaN)']
})
checklist.to_csv(os.path.join(path, "tables_csv", "00_figure_checklist.csv"), index=False, encoding='utf-8-sig')

# 3. Fig 3 Data
df3 = pd.DataFrame({
    'Predict_Horizon': [1, 5, 22, 60, 90, 120, 180, 252],
    'WEns': [np.nan, np.nan, 0.803, np.nan, np.nan, np.nan, np.nan, np.nan],
    'BiLSTM-A': [0.092, 0.577, 0.780, 0.796, 0.788, np.nan, 0.708, 0.589],
    'HAR-3': [np.nan, np.nan, 0.761, np.nan, np.nan, np.nan, np.nan, np.nan],
    'RF': [np.nan, np.nan, 0.759, np.nan, np.nan, np.nan, 0.719, 0.705],
    'Ridge': [np.nan, 0.621, 0.762, np.nan, np.nan, np.nan, np.nan, np.nan] 
})
df3.to_csv(os.path.join(path, "tables_csv", "Figure3_기간별_성능.csv"), index=False, encoding='utf-8-sig', float_format='%.3f', na_rep='NaN')

# 4. Fig 4 Data
df4 = pd.DataFrame({
    'Model': ['ElasticNet-59', 'Ridge(pooled)', 'WEns', 'BiLSTM-A', 'RF'],
    'Pooled_R2': [0.807, 0.762, 0.803, 0.780, 0.759], 
    'Median_R2': [np.nan, -0.003, 0.215, 0.235, 0.142]
})
df4.to_csv(os.path.join(path, "tables_csv", "Figure4_22d_성능.csv"), index=False, encoding='utf-8-sig', float_format='%.3f', na_rep='NaN')

# 5. Fig 5 Data
df5 = pd.DataFrame({
    'Feature_Step': ['HAR-3(Base)', '+ GARCH/Vol-of-Vol', '+ HF proxy', '+ IV surface', '+ Alternative Data'],
    'Incremental_Pooled_R2': [0.732, 0.026, 0.021, 0.0259, 0.0067],
    'Cumulative_Pooled_R2': [0.732, 0.758, 0.779, np.nan, 0.803] 
})
df5.to_csv(os.path.join(path, "tables_csv", "Figure5_Ablation_누적기여도.csv"), index=False, encoding='utf-8-sig', float_format='%.4f', na_rep='NaN')

# 6. Fig 6 Data
df6 = pd.DataFrame({
    'Feature': ['Parkinson_5', 'Garch_Daily', 'GarmanKlass_22', 'RogersSatchell_22', 'Garch_Weekly', 'IV_VRP_ma22'],
    '1d': [np.nan]*6, '5d': [np.nan]*6, '22d': [np.nan]*6,
    '60d': [np.nan]*6, '120d': [np.nan]*6, '252d': [np.nan]*6
})
df6.to_csv(os.path.join(path, "tables_csv", "Figure6_변수중요도_기간이동.csv"), index=False, encoding='utf-8-sig', na_rep='NaN')

# 7. Fig 7 Data
df7 = pd.DataFrame({
    'VIX_Regime': ['Low(0-50%)', 'Mid(50-90%)', 'High(90-100%)'],
    'HAR_3_R2': [0.739, 0.698, 0.602],
    'WEns_R2': [0.781, 0.722, 0.765]
})
df7.to_csv(os.path.join(path, "tables_csv", "Figure7_VIX분위수_성능.csv"), index=False, encoding='utf-8-sig', float_format='%.3f')

# 8. Fig 8 Data
df8 = pd.DataFrame({
    'Horizon': [1, 5, 22, 60, 90, 120, 180, 252],
    'Kendalls_W_합의도': [np.nan, 0.577, 0.588, np.nan, np.nan, np.nan, np.nan, np.nan]
})
df8.to_csv(os.path.join(path, "tables_csv", "Figure8_Kendalls_W_합의도.csv"), index=False, encoding='utf-8-sig', float_format='%.3f', na_rep='NaN')

# 10. Fig 10 Data
df10 = pd.DataFrame({
    'Asset_Class': ['Equities']*5 + ['Bonds']*3 + ['Commodities']*3,
    'Ticker': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'AGG', 'GLD', 'SLV', 'USO'],
    'R2_22d': [np.nan]*11
})
df10.to_csv(os.path.join(path, "tables_csv", "Figure10_자산군별_R2.csv"), index=False, encoding='utf-8-sig', na_rep='NaN')

print("확실한 수치만 포함된 CSV 재생성이 완료되었습니다.")
