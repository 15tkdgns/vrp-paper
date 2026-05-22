import pandas as pd
import numpy as np
import os

path = "C:/Users/15tkd/OneDrive/바탕 화면/vrp_paper"
os.makedirs(os.path.join(path, "tables_csv"), exist_ok=True)

# =============================================================
# Figure 3: 기간별 대표 모델 성능 비교 (Pooled R^2)
# 출처: 논문 v25.md 표 7 (515~641행) + 표 9 (WEns 상세, 723~831행)
# =============================================================
df3 = pd.DataFrame({
    'Predict_Horizon': [1, 5, 22, 60, 90, 120, 180, 252],
    # 표 7 데이터 (풀링 R^2)
    'Ridge(37)':    [0.114, 0.625, 0.798, 0.774, 0.752, 0.737, 0.691, 0.516],
    'BiLSTM-A(3)':  [0.092, 0.577, 0.780, 0.796, 0.788, 0.751, 0.708, 0.589],
    'HAR-3(3)':     [0.109, 0.581, 0.761, 0.784, 0.781, 0.758, 0.705, 0.611],
    'LASSO(37)':    [0.116, 0.625, 0.790, 0.782, 0.763, 0.748, 0.707, 0.555],
    'RF(37)':       [0.113, 0.611, 0.770, 0.752, 0.737, 0.730, 0.719, 0.705],
    # 표 9 WEns 상세 (Pooled R^2)
    'WEns(37)':     [0.117, 0.627, 0.797, 0.792, 0.776, 0.759, 0.748, 0.708],
    'Champion':     ['LASSO', 'Ridge', 'Ridge(WEns)', 'BiLSTM-A', 'BiLSTM-A', 'HAR-3', 'RF', 'RF']
})
df3.to_csv(os.path.join(path, "tables_csv", "Figure3_기간별_성능.csv"), index=False, encoding='utf-8-sig', float_format='%.3f')

print("Figure 3 CSV 재생성 완료 (논문 v25.md 표 7 + 표 9 검증 수치)")
