import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    t = torch.tensor([1.0]).cuda()
    print("GPU tensor test OK:", t)
else:
    print("No CUDA")

from xgboost import XGBRegressor
import numpy as np
try:
    m = XGBRegressor(device='cuda', tree_method='hist', n_estimators=5)
    m.fit(np.random.randn(100,5), np.random.randn(100))
    print("XGBoost GPU OK")
except Exception as e:
    print("XGBoost GPU error:", e)

from catboost import CatBoostRegressor
try:
    m = CatBoostRegressor(iterations=5, task_type='GPU', verbose=0)
    m.fit(np.random.randn(100,5), np.random.randn(100))
    print("CatBoost GPU OK")
except Exception as e:
    print("CatBoost GPU error:", e)
