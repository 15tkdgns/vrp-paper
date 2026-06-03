import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import json

def run_v60():
    print("Running V60: VIF Analysis for Multicollinearity...")
    
    df = pd.read_csv('src/data/ohlcv_cache.csv', index_col=0)
    spy = df['SPY'].pct_change().dropna()
    log_rv = np.log(spy.rolling(22).std() * np.sqrt(252)).dropna()
    
    # HAR Features
    feat = pd.DataFrame({
        'LogRV_lag1': log_rv.shift(1),
        'LogRV_lag5': log_rv.shift(1).rolling(5).mean(),
        'LogRV_lag22': log_rv.shift(1).rolling(22).mean()
    }).dropna()
    
    # VIF Calculation
    X = add_constant(feat)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    results = vif_data.set_index('feature')['VIF'].to_dict()
    
    with open('src/experiments/verification/v60_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("VIF Results:")
    print(vif_data)

if __name__ == "__main__":
    run_v60()
