import json
import pandas as pd
import os

json_path = 'data/multi_horizon_benchmark_results_reprod.json'
out_path = 'tables_csv/Figure3_기간별_성능.csv'

with open(json_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

horizons = ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']
models = ['Ridge', 'BiLSTM-A', 'HAR-3', 'LASSO', 'RF', 'WEns']

rows = []
for hz in horizons:
    row = {'기간': hz}
    best_model = None
    best_r2 = -float('inf')
    for m in models:
        # Some JSON keys might differ slightly if we used different names in the script
        m_key = m
        if m_key not in results[hz]:
            pass
            
        r2 = results[hz][m_key]['Pooled_R2']
        row[m] = r2
        if r2 > best_r2:
            best_r2 = r2
            best_model = m
            
    # As per original paper logic, champion is usually WEns if it wins, etc.
    # WEns is actually the proposed model, so we can just highlight the best base model vs WEns,
    # or just label Champion purely by max.
    if best_model == 'WEns':
        # Find best base model
        best_base_r2 = -float('inf')
        best_base = None
        for bm in models:
            if bm != 'WEns':
                bm_r2 = results[hz][bm]['Pooled_R2']
                if bm_r2 > best_base_r2:
                    best_base_r2 = bm_r2
                    best_base = bm
        row['Champion'] = f"{best_base}(WEns)"
    else:
        row['Champion'] = best_model
        
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False, float_format='%.4f', encoding='utf-8-sig', na_rep='NaN')
print(f"Figure 3 CSV 재생성 완료 (재실험 JSON 기반): {out_path}")
print(df)
