"""WSL 환경 및 데이터 로딩 검증 스크립트"""
import sys
print(f"Python: {sys.version}")

import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import torch
print("패키지 OK: pandas, numpy, sklearn, xgboost, catboost, torch")
print(f"  catboost={CatBoostRegressor.__module__.split('.')[0]}, torch={torch.__version__}")

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

vix_df = pd.read_parquet(os.path.join(DATA_DIR, 'VIX.parquet'))
spy_df = pd.read_parquet(os.path.join(DATA_DIR, 'SPY.parquet'))
print(f"\nVIX.parquet: {vix_df.shape}, columns={vix_df.columns.tolist()}")
print(f"SPY.parquet: {spy_df.shape}, columns={spy_df.columns.tolist()[:3]}...")
print(f"날짜 범위: {vix_df.index[0].date()} ~ {vix_df.index[-1].date()}")

assets = ['SPY','QQQ','IWM','EFA','EEM','TLT','IEF','AGG','GLD','SLV','USO']
missing = [a for a in assets if not os.path.exists(os.path.join(DATA_DIR, f'{a}.parquet'))]
if missing:
    print(f"누락된 자산: {missing}")
else:
    print(f"자산 parquet 11개 모두 존재 ✅")

print("\n✅ WSL 환경 준비 완료 — run_wsl.sh 실행 가능")
