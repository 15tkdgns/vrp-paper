"""
실험 3: ElasticNet 피처 선택 안정성 및 궤적(Path) 분석
======================================================
1. 시계열 교차검증(TimeSeriesSplit) 하에서 ElasticNet의 최적 하이퍼파라미터 탐색
2. L1 정규화 경로에 따른 피처 선택 수 감소(Sparsity) 확인
3. 선택된 피처의 중요도 및 도메인 의미 고찰
4. 22d 지평의 풀링 데이터 대상
"""
import sys; sys.path.insert(0, '/root/vrp')
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from src.experiments.creative.v71_model_comparison import build_dataset

print("Loading data...", flush=True)
data, feats = build_dataset()

# Clean feats logic
feats = [f for f in feats if f not in ['Date', 'index', 'Asset', 'Target']]

# Use the full training set (80%) for CV
split_idx = int(len(data) * 0.8)
train_df = data.iloc[:split_idx]

# To do a pooled feature selection, we stack all assets.
# But effectively, ElasticNet with grouped time features is sensitive.
# We will use the pooled cross-sectional dataset directly for feature selection geometry.
X_tr = train_df[feats].fillna(0).values
y_tr = train_df['Target'].values

print(f"Total features: {len(feats)}")
print(f"Training samples: {len(X_tr)}")

# Scale
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)

# Define CV
tscv = TimeSeriesSplit(n_splits=5)

# 1. Broad L1 ratio search to establish regularization limits
# l1_ratio=1.0 is Lasso (sparse), 0.0 is Ridge (dense)
l1_ratios = [0.1, 0.5, 0.9, 1.0]

print("Running ElasticNetCV...", flush=True)
model = ElasticNetCV(l1_ratio=l1_ratios, cv=tscv, alphas=np.logspace(-4, 2, 50), random_state=42, n_jobs=-1, max_iter=2000)
model.fit(X_tr_sc, y_tr)

best_l1 = model.l1_ratio_
best_alpha = model.alpha_

coefs = model.coef_
n_active = np.sum(np.abs(coefs) > 1e-5)

print("\n=== Optimal ElasticNet Configuration ===")
print(f"Best L1_ratio: {best_l1}")
print(f"Best Alpha: {best_alpha:.5f}")
print(f"Active Features (abs coef > 1e-5): {n_active} / {len(feats)}")

# Analyze which features survived
top_indices = np.argsort(np.abs(coefs))[::-1]
print("\nTop 10 Retained Features (by absolute magnitude):")
retained_feat_names = []
for i in range(10):
    idx = top_indices[i]
    val = coefs[idx]
    if abs(val) > 1e-5:
        print(f"  {i+1}. {feats[idx]}: {val:.4f}")
        retained_feat_names.append(feats[idx])

out = {
    'Total_Features': len(feats),
    'Active_Features': int(n_active),
    'Best_L1_Ratio': float(best_l1),
    'Best_Alpha': float(best_alpha),
    'Top_Features': retained_feat_names
}

with open('/root/vrp/paper/csv/elasticnet_feature_selection.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved.")
