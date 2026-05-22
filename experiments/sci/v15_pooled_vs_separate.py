"""
V15 Exp: Universal Model Utility Verification (SCI Phase 4)
Purpose: To empirically prove the benefit of "Pooling" data across assets vs "Separate" modeling.

Hypothesis: 
Error(Pooled) < Error(Separate)
The Universal Model learns generalized volatility dynamics that transfer across assets.

Methodology:
1. Pooled Model: Train ONE model on ALL train data (Stacking). Predict ALL test data.
2. Separate Models: Train N models (one per asset). Predict respective test data.
3. Compare: Overall R2, Asset-wise R2, and Diebold-Mariano Test on stacked errors.
4. Model Used: HAR-Ridge (since v14 showed it outperforms RF/XGB).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os
import json
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity',
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42

def feature_engineering(data):
    # Same as v14
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1))
        
        # RV calculation
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Extended Features
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        df['RS_Asymmetry'] = df['LogRV_Down'] - df['LogRV_Up']
        
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1']
        
        window = 14
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1) / 100.0
        
        window_bb = 20
        ma = price.rolling(window_bb).mean()
        std = price.rolling(window_bb).std()
        df['BB_Width'] = ((2 * std) / (ma + 1e-6)).shift(1)
        
        df['Skewness'] = ret.rolling(22).skew().shift(1).fillna(0)
        df['Kurtosis'] = ret.rolling(22).kurt().shift(1).fillna(0)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).reset_index(drop=True)

def diebold_mariano_test(real, pred1, pred2, h=1, power=2):
    e1 = real - pred1
    e2 = real - pred2
    d = (np.abs(e1)**power) - (np.abs(e2)**power) 
    d_mean = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    std_error = np.sqrt(gamma0 / len(d))
    dm_stat = d_mean / std_error
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def run_experiment():
    print("="*80)
    print("V15: Universal Model Utility Verification (Pooled vs Separate)")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS + ['^VIX']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    features = [
        'LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 
        'LogRV_Down', 'LogRV_Up', 'RS_Asymmetry',
        'VRP_Proxy', 'RSI', 'BB_Width', 'Skewness', 'Kurtosis'
    ]
    
    # Global Train/Test Split (Time-series)
    # We want to ensure time-alignment. 
    # Since data is stacked, we should split by DATE, not index.
    # But for simplicity in this artifact, we assume data is sorted properly or we respect the same ratio.
    # To be precise, let's split each asset by 80:20 and then stack.
    
    assets_data = {asset: data[data['Asset'] == asset] for asset in data['Asset'].unique()}
    
    train_pooled_list = []
    test_pooled_list = []
    
    for asset, df in assets_data.items():
        split = int(len(df) * 0.8)
        train_pooled_list.append(df.iloc[:split])
        test_pooled_list.append(df.iloc[split:])
        
    train_pooled = pd.concat(train_pooled_list)
    test_pooled = pd.concat(test_pooled_list)
    
    print(f"Pooled Train: {len(train_pooled)}, Pooled Test: {len(test_pooled)}")
    
    # 2. Train Pooled Model (Universal)
    print("\n[Step 2] Training Pooled Model (Universal)...")
    scaler_pool = StandardScaler()
    X_train_pool = scaler_pool.fit_transform(train_pooled[features])
    y_train_pool = train_pooled['Target']
    X_test_pool = scaler_pool.transform(test_pooled[features])
    y_test_pool = test_pooled['Target']
    
    model_pool = Ridge(alpha=1.0)
    model_pool.fit(X_train_pool, y_train_pool)
    pred_pool = model_pool.predict(X_test_pool)
    
    r2_pool_overall = r2_score(y_test_pool, pred_pool)
    print(f"Pooled Model Overall R2: {r2_pool_overall:.4f}")
    
    # 3. Train Separate Models
    print("\n[Step 3] Training Separate Models...")
    pred_separate_list = []
    r2_separate_dict = {}
    r2_pooled_dict = {} # Decomposed pooled performance
    
    # We need to map pooled predictions back to assets to compare pairwise
    # The test_pooled is effectively concatenated asset-by-asset in order
    
    current_idx = 0
    predictions_separate_all = []
    predictions_pooled_all = []
    y_true_all = []
    
    asset_list = list(assets_data.keys())

    for i, asset in enumerate(asset_list):
        df_test_asset = test_pooled_list[i]
        df_train_asset = train_pooled_list[i]
        
        # Prepare Separate Data
        scaler_sep = StandardScaler()
        X_train_sep = scaler_sep.fit_transform(df_train_asset[features])
        y_train_sep = df_train_asset['Target']
        X_test_sep = scaler_sep.transform(df_test_asset[features])
        y_test_sep = df_test_asset['Target']
        
        # Train Separate
        model_sep = Ridge(alpha=1.0)
        model_sep.fit(X_train_sep, y_train_sep)
        pred_sep = model_sep.predict(X_test_sep)
        
        # Get Pooled Prediction for this segment 
        # (Assuming order is preserved: test_pooled was concat of test_pooled_list)
        n_samples = len(df_test_asset)
        pred_pool_asset = pred_pool[current_idx : current_idx + n_samples]
        current_idx += n_samples
        
        # Metrics
        r2_sep = r2_score(y_test_sep, pred_sep)
        r2_pool = r2_score(y_test_sep, pred_pool_asset)
        
        r2_separate_dict[asset] = r2_sep
        r2_pooled_dict[asset] = r2_pool
        
        predictions_separate_all.extend(pred_sep)
        predictions_pooled_all.extend(pred_pool_asset)
        y_true_all.extend(y_test_sep)
        
    predictions_separate_all = np.array(predictions_separate_all)
    predictions_pooled_all = np.array(predictions_pooled_all)
    y_true_all = np.array(y_true_all)
    
    r2_separate_overall = r2_score(y_true_all, predictions_separate_all)
    print(f"Separate Models Overall R2: {r2_separate_overall:.4f}")
    
    # 4. Statistical Test
    print("\n[Step 4] Validation")
    dm_stat, p_val = diebold_mariano_test(y_true_all, predictions_pooled_all, predictions_separate_all)
    # Note: If DM Stat > 0, Pooled has higher error (worse)? 
    # d = e_pool^2 - e_sep^2
    # If d < 0 => e_pool < e_sep => Pooled is better.
    # Stat = mean(d) / se. So negative stat => Pooled Better.
    
    better_model = "Pooled" if dm_stat < 0 else "Separate"
    sig = "Significant" if p_val < 0.05 else "Not Sig"
    
    print(f"DM Statistic: {dm_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Result: {better_model} is better ({sig})")
    
    # 5. Asset-wise breakdown
    print("\n[Step 5] Asset-wise Performance Delta (Pooled - Separate)")
    print(f"{'Asset':<10} {'Pooled R2':<10} {'Sep R2':<10} {'Delta':<10}")
    print("-" * 45)
    
    results_list = []
    for asset in ASSETS:
        if asset not in r2_pooled_dict: continue
        p = r2_pooled_dict[asset]
        s = r2_separate_dict[asset]
        d = p - s
        results_list.append({'Asset': asset, 'Pooled': p, 'Separate': s, 'Delta': d})
        print(f"{asset:<10} {p:.4f}     {s:.4f}     {d:+.4f}")
        
    # Save
    out_data = {
        'overall': {
            'pooled_r2': r2_pool_overall,
            'separate_r2': r2_separate_overall,
            'dm_stat': dm_stat,
            'p_value': p_val,
            'better_model': better_model
        },
        'asset_wise': results_list
    }
    
    out_path = 'src/experiments/sci/v15_pooled_vs_separate.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
