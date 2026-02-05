import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
import torch
import torch.nn as nn

# Import previous model modules (simulated or direct import if structure allows)
# Since we need to re-run or load pre-saved predictions, the best way for this ensemble script
# is to re-generate predictions using the saved weights/models OR re-train quickly if fast enough.
# However, V43/V50 take time. 
# Ideal approach: Ensure previous scripts save their 'Test Predictions' to a CSV. 
# But they saved JSON summaries only.
# Strategy: Re-instantiate and re-train models on the same split (Deterministic) OR
# Modify previous scripts to save predictions. 
# Given constraints, I will re-implement the lightweight V35, V36 here and 
# assume V43/V50 predictions can be loaded if saved, OR simplified versions.

# ACTUALLY, simpler path: 
# This script will assume the existence of 'predictions_v35.csv', 'predictions_v36.csv' etc.
# I need to UPDATED V35, V36, V43, V50 to save their predictions. 
# BUT, I can't interrupt V43/V50. 
# Plan: 
# 1. Re-run lightweight V35, V36 inside here (fast).
# 2. For V43/V50, wait for them to finish, but since I can't easily load their states without file saving,
#    I will create a placeholder in this script that tries to load 'v43_preds.csv'. 
#    I need to EDIT V43/V50 to save csvs. 

# Let's Edit V43 and V50 to save predictions FIRST? 
# V43 is running. I can't edit it.
# Wait, V43 script *already* finished one run (R2 0.81) but failed to save JSON.
# The current run is valid. 
# I will write V42 to be a "wrapper" that calls the others? No, that's messy.

# DECISION: Write V42 to perform the ensemble. 
# It will Re-implement V35 and V36 (Fast).
# For V43/V50, it will hope for a saved file. 
# I WILL APPEND code to the RUNNING V43/V50 scripts? No.
# I will write V42 to be standalone, re-implementing V35/V36. 
# For Deep Learning models, I will skip them if files not found, or re-train if possible.

warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def get_data():
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from local cache: {CACHE_PATH}...", end="", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(" Done.", flush=True)
    else:
        print("Local cache not found! Downloading from yfinance...", end="", flush=True)
        raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
        print(" Done.", flush=True)
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    pooled_data = []
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        pooled_data.append(d)
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80, flush=True)
    print("V42: Ultimate Ensemble Experiment", flush=True)
    print("="*80, flush=True)
    
    print("Loading base data...", end="", flush=True)
    data = get_data()
    print(" Done.", flush=True)
    
    print("Splitting data (80/20)...", flush=True)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    y_test = test_df['Target']
    
    # Placeholder for predictions
    preds = pd.DataFrame(index=test_df.index)
    preds['Target'] = y_test
    preds['Asset'] = test_df['Asset']
    
    # 1. V29 Baseline (Re-run)
    print("Training V29 Baseline (Ridge)...", end="", flush=True)
    from sklearn.linear_model import Ridge
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']])
    X_test = sc.transform(test_df[['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']])
    m29 = Ridge().fit(X_train, train_df['Target'])
    preds['V29'] = m29.predict(X_test)
    print(" Done.", flush=True)
    
    # 2. Load Saved Predictions
    print("Loading predictions from sub-models...", flush=True)
    
    def try_load_and_merge(filepath, col_name, df_target):
        if os.path.exists(filepath):
            print(f"  Loading {col_name} from {filepath}...", end="", flush=True)
            p_df = pd.read_csv(filepath)
            
            # Date handling
            # If first column is Date, use it. In our scripts, index=Date usually.
            if 'Date' in p_df.columns:
                p_df['Date'] = pd.to_datetime(p_df['Date'])
            else:
                p_df.iloc[:,0] = pd.to_datetime(p_df.iloc[:,0])
                p_df.rename(columns={p_df.columns[0]: 'Date'}, inplace=True)
            
            # Select relevant columns
            cols_to_use = ['Date', 'Asset', 'Preds']
            if not all(c in p_df.columns for c in cols_to_use):
                 # Fallback: maybe it only has Date and Preds?
                 if 'Preds' in p_df.columns:
                      cols_to_use = ['Date', 'Preds']
                 else:
                      print(" Missing 'Preds' column.", flush=True)
                      return None
            
            p_df_clean = p_df[cols_to_use].rename(columns={'Preds': col_name})
            
            # Align with df_target
            df_target_reset = df_target.reset_index()
            # If df_target index was named Date, it becomes 'Date'.
            # On Windows/WSL, sometimes it becomes 'index'.
            if 'Date' not in df_target_reset.columns:
                 df_target_reset = df_target_reset.rename(columns={df_target_reset.columns[0]: 'Date'})
            df_target_reset['Date'] = pd.to_datetime(df_target_reset['Date'])

            merge_on = ['Date']
            if 'Asset' in p_df_clean.columns:
                merge_on.append('Asset')
            
            merged = pd.merge(df_target_reset, p_df_clean, on=merge_on, how='left')
            print(f" Merged {merged[col_name].notna().sum()} rows.", flush=True)
            return merged[col_name].values
        else:
            print(f"  Warning: {filepath} not found.", flush=True)
            return None

    # Load All
    for model_ver, path in [('V36', 'src/experiments/creative/v36_preds.csv'),
                            ('V43', 'src/experiments/creative/v43_preds.csv'),
                            ('V50', 'src/experiments/creative/v50_preds.csv')]:
        p = try_load_and_merge(path, model_ver, preds)
        if p is not None:
            preds[model_ver] = p

    # Clean up NaNs
    initial_len = len(preds)
    preds_clean = preds.dropna()
    final_len = len(preds_clean)
    print(f"Final dataset size after intersection: {final_len} (Dropped {initial_len - final_len} samples)", flush=True)

    if final_len == 0:
        print("ERROR: Ensemble intersection is empty! Check alignment logic.", flush=True)
        return

    # Ensemble Calculation
    valid_cols = [c for c in ['V29', 'V36', 'V43', 'V50'] if c in preds_clean.columns]
    print(f"Ensembling models: {valid_cols}", flush=True)
    
    if len(valid_cols) > 1:
        # 1. Simple Average
        preds_clean['Ensemble_Avg'] = preds_clean[valid_cols].mean(axis=1)
        r2_avg = r2_score(preds_clean['Target'], preds_clean['Ensemble_Avg'])
        print(f"Ensemble (Average) R2: {r2_avg:.5f}", flush=True)
        
        # 2. Optimized Weighted Average (V36 priority: 0.6)
        w_map = {'V36': 0.6, 'V29': 0.2, 'V43': 0.1, 'V50': 0.1}
        # Dynamic Weighting (in case some are missing, though all are present now)
        current_weights = {curr: w_map[curr] for curr in valid_cols if curr in w_map}
        total_w = sum(current_weights.values())
        
        preds_clean['Ensemble_Weighted'] = sum(preds_clean[c]*(current_weights[c]/total_w) for c in valid_cols)
        r2_w = r2_score(preds_clean['Target'], preds_clean['Ensemble_Weighted'])
        print(f"Ensemble (Weighted V36 Focus) R2: {r2_w:.5f}", flush=True)
        
        res = {
            'V42_R2_Avg': r2_avg, 
            'V42_R2_Weighted': r2_w,
            'Models': valid_cols
        }
    else:
        print("Not enough sub-models for ensemble.", flush=True)
        base_r2 = r2_score(preds_clean['Target'], preds_clean['V29']) if 'V29' in preds_clean else 0
        res = {'V42_R2': base_r2, 'Note': 'Baseline Only'}
        
    print(f"Writing results to src/experiments/creative/v42_results.json...", end="", flush=True)
    with open('src/experiments/creative/v42_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(" Done.", flush=True)

if __name__ == "__main__":
    run_experiment()
