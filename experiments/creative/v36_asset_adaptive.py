import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Asset Classification
ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'], # Stocks
    'Bond': ['TLT', 'IEF', 'AGG'],                 # Bonds
    'Commodity': ['GLD', 'SLV', 'USO']             # Commodities
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

def run_experiment():
    print("="*80, flush=True)
    print("V36: Asset-Class Adaptive Weights Experiment", flush=True)
    print("="*80, flush=True)
    
    # Data Loading
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from local cache: {CACHE_PATH}...", end="", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(" Done.", flush=True)
    else:
        print("Local cache not found! Downloading from yfinance...", end="", flush=True)
        raw = yf.download(ALL_ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
        print(" Done.", flush=True)
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    pooled_data = []
    print(f"Processing {len(ALL_ASSETS)} assets...", flush=True)
    
    for i, asset in enumerate(ALL_ASSETS):
        if (i+1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(ALL_ASSETS)}] Processing {asset}...", flush=True)
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        
        # Base Features (HAR)
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        
        # Determine Asset Class
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        
        pooled_data.append(d)
        
    print("Concatenating and sorting data...", flush=True)
    data = pd.concat(pooled_data).sort_index()
    
    # Train/Test Split
    print("Splitting data (80/20)...", flush=True)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    # 1. Baseline Global Model (V29-Lite)
    print("Training Global Model (Baseline)...", flush=True)
    feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[feats])
    X_test = sc.transform(test_df[feats])
    y_train = train_df['Target']
    y_test = test_df['Target']
    
    model_global = Ridge(alpha=1.0).fit(X_train, y_train)
    pred_global = model_global.predict(X_test)
    r2_global = r2_score(y_test, pred_global)
    
    # 2. Asset-Class Specific Models
    class_models = {}
    test_predictions_by_class = pd.Series(index=test_df.index, dtype=float)
    
    print("Training Asset-Class Specific Models...", flush=True)
    for cls in ASSET_GROUPS.keys():
        print(f"  Training for class: {cls}...", end="", flush=True)
        # Filter training data for this class
        train_cls = train_df[train_df['Class'] == cls]
        
        if len(train_cls) < 100:
            print(f" Warning: Not enough data for class {cls}", flush=True)
            continue
            
        X_train_cls = sc.transform(train_cls[feats]) # Use same scaler for simplicity/comparability
        y_train_cls = train_cls['Target']
        
        model_cls = Ridge(alpha=1.0).fit(X_train_cls, y_train_cls)
        class_models[cls] = model_cls
        
        # Predict on Test Set (Only for assets in this class)
        test_cls_mask = test_df['Class'] == cls
        if test_cls_mask.sum() > 0:
            X_test_cls = sc.transform(test_df.loc[test_cls_mask, feats])
            pred_cls = model_cls.predict(X_test_cls)
            test_predictions_by_class[test_cls_mask] = pred_cls
        print(" Done.", flush=True)

    # Fill any missing predictions with global model (safety)
    mask_missing = test_predictions_by_class.isna()
    if mask_missing.sum() > 0:
        print(f"Filling {mask_missing.sum()} missing predictions with global model...", flush=True)
        test_predictions_by_class[mask_missing] = pred_global[mask_missing]
        
    r2_adaptive = r2_score(y_test, test_predictions_by_class)
    
    print("\n[Results]", flush=True)
    print(f"Global Model R2: {r2_global:.5f}", flush=True)
    print(f"Adaptive Class R2: {r2_adaptive:.5f}", flush=True)
    print(f"Improvement: {r2_adaptive - r2_global:.5f}", flush=True)
    
    res = {'V36_R2': float(r2_adaptive)}
    with open('src/experiments/creative/v36_results.json', 'w') as f:
        json.dump(res, f)
        
    # Save Predictions for Ensemble
    preds_df = pd.DataFrame({
        'Preds': test_predictions_by_class,
        'Asset': test_df['Asset']
    }, index=y_test.index)
    preds_df.to_csv('src/experiments/creative/v36_preds.csv')
    print("Saved V36 predictions to src/experiments/creative/v36_preds.csv", flush=True)

if __name__ == "__main__":
    run_experiment()
