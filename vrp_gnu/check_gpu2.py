import numpy as np

from xgboost import XGBRegressor
try:
    m = XGBRegressor(device='cuda', tree_method='hist', n_estimators=10, verbosity=0)
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    m.fit(X, y)
    print("XGBoost GPU OK")
except Exception as e:
    print("XGBoost GPU FAIL:", e)

from catboost import CatBoostRegressor
try:
    m = CatBoostRegressor(iterations=10, task_type='GPU', verbose=0)
    m.fit(X, y)
    print("CatBoost GPU OK")
except Exception as e:
    print("CatBoost GPU FAIL:", e)
