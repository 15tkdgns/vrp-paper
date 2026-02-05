"""
V28 Exp: Adaptive Regularization (SCI Phase 7)
Purpose: To improve performance by adapting regularization strength (alpha) over time using a rolling window.
Method:
1. Define a rolling window (e.g., 252 days ~ 1 year).
2. For each step t, use data [t-window : t] to find best alpha via TimeSeriesSplit CV.
3. Predict t+1 using the best alpha.
4. Compare with Fixed Alpha model.

Hypothesis: Volatility regimes change, so optimal bias-variance tradeoff (alpha) also changes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond',
    'GLD': 'Commodity'
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42

def feature_engineering(data):
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
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Add Asymmetric Features (Since v27 worked)
        # Actually let's test Adaptive Reg on BASE features first to isolate effect.
        
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        pooled_data.append(df)
        
    return pd.concat(pooled_data).sort_index()

def run_experiment():
    print("="*80)
    print("V28: Adaptive Regularization")
    print("="*80)
    
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    data = feature_engineering(raw)
    
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(train_data[features])
    y_train_full = train_data['Target'].values
    X_test_full = scaler.transform(test_data[features])
    y_test_full = test_data['Target'].values
    
    # Baseline: Fixed Alpha (Optimized on Train)
    print("\n[Step 1] Baseline (Fixed Alpha)...")
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_avg_score = -np.inf
    best_fixed_alpha = 1.0
    
    tscv = TimeSeriesSplit(n_splits=5)
    for a in alphas:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_full):
            m = Ridge(alpha=a)
            m.fit(X_train_full[tr_idx], y_train_full[tr_idx])
            scores.append(m.score(X_train_full[val_idx], y_train_full[val_idx]))
        avg_score = np.mean(scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_fixed_alpha = a
            
    print(f"Best Fixed Alpha: {best_fixed_alpha}")
    model_fixed = Ridge(alpha=best_fixed_alpha)
    model_fixed.fit(X_train_full, y_train_full)
    r2_fixed = r2_score(y_test_full, model_fixed.predict(X_test_full))
    print(f"Fixed Model R2: {r2_fixed:.4f}")
    
    # Adaptive: Rolling Window Update
    print("\n[Step 2] Adaptive (Rolling Alpha)...")
    # Simulation: Walk-forward on Test set
    # We update alpha every N days (e.g. monthly=22) to save compute
    update_freq = 22
    window_size = 500 # Lookback for tuning alpha
    
    # We need to simulate iterating through test set
    y_pred_adaptive = []
    y_true_adaptive = [] # To match indices
    
    # Combine Train+Test to simulate rolling
    # Start predicting from first point of Test
    # History available: Train data
    
    # To implement efficiently:
    # 1. Pre-train models with all alphas on expanding window? expensive.
    # 2. Refit only every Update Freq.
    
    # Let's simple simulation:
    # Test set is 20% ~ 4000 points.
    # Update alpha every 100 points?
    
    current_alpha = best_fixed_alpha
    
    X_combined = np.vstack([X_train_full, X_test_full])
    y_combined = np.concatenate([y_train_full, y_test_full])
    
    test_start_idx = len(X_train_full)
    
    preds = []
    
    print(f"Test size: {len(X_test_full)}")
    
    for i in range(0, len(X_test_full), update_freq):
        # Current index in combined
        curr_idx = test_start_idx + i
        
        # 1. Tune alpha using recent history [curr_idx - window : curr_idx]
        if i % (update_freq * 5) == 0: # Tune less frequently (every ~100 days)
             history_X = X_combined[curr_idx-window_size : curr_idx]
             history_y = y_combined[curr_idx-window_size : curr_idx]
             
             # Micro-CV
             best_a_t = current_alpha
             best_s_t = -np.inf
             
             # split history into train/val (last 20%)
             h_split = int(len(history_X) * 0.8)
             h_X_tr, h_X_val = history_X[:h_split], history_X[h_split:]
             h_y_tr, h_y_val = history_y[:h_split], history_y[h_split:]
             
             for a in alphas:
                 m = Ridge(alpha=a)
                 m.fit(h_X_tr, h_y_tr)
                 s = m.score(h_X_val, h_y_val)
                 if s > best_s_t:
                     best_s_t = s
                     best_a_t = a
             current_alpha = best_a_t
             
        # 2. Train model with current_alpha on ALL history (up to current)
        # Expanding window training
        train_X_exp = X_combined[:curr_idx]
        train_y_exp = y_combined[:curr_idx]
        
        model_t = Ridge(alpha=current_alpha)
        model_t.fit(train_X_exp, train_y_exp)
        
        # 3. Predict next chunk
        chunk_end = min(i + update_freq, len(X_test_full))
        X_chunk = X_test_full[i : chunk_end]
        
        p = model_t.predict(X_chunk)
        preds.extend(p)
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(X_test_full)}... Alpha={current_alpha}")

    y_pred_adaptive = np.array(preds)
    r2_adaptive = r2_score(y_test_full, y_pred_adaptive)
    
    print(f"\nAdaptive Model R2: {r2_adaptive:.4f}")
    
    improv = (r2_adaptive - r2_fixed) / abs(r2_fixed) * 100
    print(f"Improvement: {improv:.2f}%")
    
    out_data = {
        'fixed_r2': r2_fixed,
        'adaptive_r2': r2_adaptive,
        'improvement': improv
    }
    
    out_path = 'src/experiments/sci/v28_adaptive_regularization.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
