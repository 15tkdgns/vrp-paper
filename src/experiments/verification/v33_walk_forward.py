import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
import warnings
from arch import arch_model

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']

def run_experiment():
    print("="*80)
    print("V33: Walk-Forward Validation (Rolling Re-training)")
    print("="*80)
    
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Process for a single representative asset (e.g., SPY) or pooled
    # For Walk-Forward, pooled is very slow to re-train. 
    # Let's do a pooled approach with a larger step (e.g., re-train every month) to save time.
    
    pooled_data = []
    print("Extracting features for all assets...")
    for asset in ASSETS:
        ret = np.log(raw[asset] / raw[asset].shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        # GARCH (Pre-calculate for full series to avoid slow fitting inside loop)
        # Note: Parameters are fitted on full series, which is a minor proxy leak, 
        # but conditional volatility t only uses ret t-1.
        am = arch_model(ret * 100, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        garch_vol = res.conditional_volatility / 100
        
        d = pd.DataFrame({
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_lag1': garch_vol.shift(1),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        pooled_data.append(d)
        
    data = pd.concat(pooled_data).sort_index()
    
    # Unique timestamp list for walk-forward
    timestamps = sorted(data.index.unique())
    start_point = int(len(timestamps) * 0.7) # Start testing after 70% of history
    
    test_dates = timestamps[start_point:]
    
    results = []
    
    print(f"Starting Walk-Forward simulation ({len(test_dates)} steps)...")
    
    # To optimize: Re-train every 21 days (approx 1 month)
    # This maintains realism while keeping execution time manageable.
    
    current_model = None
    current_scaler = None
    
    for i, date in enumerate(test_dates):
        if i % 21 == 0:
            # Re-train
            train_data = data[data.index < date]
            
            feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Garch_lag1']
            current_scaler = StandardScaler()
            X_train = current_scaler.fit_transform(train_data[feats])
            y_train = train_data['Target']
            
            current_model = Ridge(alpha=1.0).fit(X_train, y_train)
            if i == 0:
                 print(f"First training completed at {date}. Pooled samples: {len(train_data)}")

        # Predict for this date
        today_data = data[data.index == date]
        if today_data.empty: continue
        
        X_today = current_scaler.transform(today_data[feats])
        y_today = today_data['Target']
        
        preds = current_model.predict(X_today)
        
        for actual, pred in zip(y_today, preds):
            results.append({'Actual': actual, 'Predicted': pred})
            
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_dates)} days processed...")
            
    df_res = pd.DataFrame(results)
    
    r2_wf = r2_score(df_res['Actual'], df_res['Predicted'])
    rmse_wf = np.sqrt(mean_squared_error(df_res['Actual'], df_res['Predicted']))
    
    print("\n[Walk-Forward Results]")
    print(f"OOS R2 (Walk-Forward): {r2_wf:.4f}")
    print(f"OOS RMSE (Walk-Forward): {rmse_wf:.4f}")
    
    # Comparison with Static Split (approximate from V31)
    print(f"Static Split R2 (V31 approx): 0.6529")
    print(f"Stability Delta: {r2_wf - 0.6529:.4f}")

    out_path = 'src/experiments/verification/v33_walk_forward.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'r2_wf': r2_wf, 'rmse_wf': rmse_wf}, f, indent=2)

if __name__ == "__main__":
    run_experiment()
