"""
V8 Exp: Comprehensive Baseline Comparison
Purpose: Compare various baselines and literature-based models for volatility prediction.

Baselines:
1. Naive (Random Walk): RV_t+h = RV_t
2. Historical Mean: RV_t+h = mean(RV)
3. HAR-Ridge: Corsi (2009) style with lag1/5/22

Literature-Based Models:
4. XGBoost: Christensen et al. (2023), tree-based ensemble
5. LightGBM: Fast gradient boosting
6. MLP: DeepVol (2022), Michael et al. (2025) inspired
7. Ensemble: Ridge + RF + GBM combination
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, skipping...")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed, skipping...")

# Asset Categories
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity',
}
ASSETS = list(ASSET_CATEGORIES.keys())

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(data, window=20):
    ma = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (2 * std) / (ma + 1e-6)

def run_baseline_comparison():
    print("="*70)
    print("V8 Experiment: Comprehensive Baseline Comparison")
    print("="*70)
    
    # 1. Download Data
    tickers = ASSETS + ['^VIX']
    print("Downloading data...")
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    print("\nEngineering Features...")
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1))
        
        # RV calculation
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        # Basic HAR features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Extended features
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        df['RS_Asymmetry'] = df['LogRV_Down'] - df['LogRV_Up']
        
        vix = close['VIX']
        df['LogVIX'] = np.log(vix + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1'] 
        
        df['RSI'] = calculate_rsi(price).shift(1) / 100.0
        df['BB_Width'] = calculate_bollinger_width(price).shift(1)
        df['Skewness'] = ret.rolling(22).skew().shift(1).fillna(0)
        df['Kurtosis'] = ret.rolling(22).kurt().shift(1).fillna(0)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df['Category'] = ASSET_CATEGORIES[asset]
        df = df.dropna()
        if len(df) < 500: continue
        
        pooled_data.append(df)
        
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    # Feature Sets
    features_basic = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    features_expanded = features_basic + [
        'LogRV_Down', 'LogRV_Up', 'RS_Asymmetry',
        'VRP_Proxy', 'RSI', 'BB_Width', 'Skewness', 'Kurtosis'
    ]
    
    print(f"Total Samples: {len(full_df)}")
    
    # Split
    indices = np.arange(len(full_df))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(full_df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_df = full_df.iloc[train_idx].copy()
    test_df = full_df.iloc[test_idx].copy()
    
    # Prepare data
    scaler_basic = StandardScaler()
    X_train_basic = scaler_basic.fit_transform(train_df[features_basic])
    X_test_basic = scaler_basic.transform(test_df[features_basic])
    
    scaler_exp = StandardScaler()
    X_train_exp = scaler_exp.fit_transform(train_df[features_expanded])
    X_test_exp = scaler_exp.transform(test_df[features_expanded])
    
    y_train = train_df['Target'].values
    y_test = test_df['Target'].values
    
    results = {}
    
    # ==================== BASELINES ====================
    print("\n" + "="*50)
    print("BASELINES")
    print("="*50)
    
    # 1. Naive (Random Walk)
    pred_naive = test_df['LogRV_lag1'].values
    r2_naive = r2_score(y_test, pred_naive)
    results['Naive (RW)'] = {'r2': r2_naive, 'type': 'Baseline'}
    print(f"1. Naive (Random Walk): R² = {r2_naive:.4f}")
    
    # 2. Historical Mean
    hist_mean = train_df['Target'].mean()
    pred_mean = np.full_like(y_test, hist_mean)
    r2_mean = r2_score(y_test, pred_mean)
    results['Historical Mean'] = {'r2': r2_mean, 'type': 'Baseline'}
    print(f"2. Historical Mean:     R² = {r2_mean:.4f}")
    
    # 3. HAR-Ridge (Corsi 2009)
    model_har = Ridge(alpha=1.0)
    model_har.fit(X_train_basic, y_train)
    pred_har = model_har.predict(X_test_basic)
    r2_har = r2_score(y_test, pred_har)
    results['HAR-Ridge'] = {'r2': r2_har, 'type': 'Baseline'}
    print(f"3. HAR-Ridge (Corsi):   R² = {r2_har:.4f}")
    
    # ==================== LITERATURE-BASED MODELS ====================
    print("\n" + "="*50)
    print("LITERATURE-BASED MODELS")
    print("="*50)
    
    # 4. Random Forest (Christensen 2023 inspired)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    model_rf.fit(X_train_exp, y_train)
    pred_rf = model_rf.predict(X_test_exp)
    r2_rf = r2_score(y_test, pred_rf)
    results['Random Forest'] = {'r2': r2_rf, 'type': 'ML'}
    print(f"4. Random Forest:       R² = {r2_rf:.4f}")
    
    # 5. GradientBoosting
    model_gbm = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_gbm.fit(X_train_exp, y_train)
    pred_gbm = model_gbm.predict(X_test_exp)
    r2_gbm = r2_score(y_test, pred_gbm)
    results['GradientBoosting'] = {'r2': r2_gbm, 'type': 'ML'}
    print(f"5. GradientBoosting:    R² = {r2_gbm:.4f}")
    
    # 6. XGBoost (if available)
    if HAS_XGB:
        model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_xgb.fit(X_train_exp, y_train)
        pred_xgb = model_xgb.predict(X_test_exp)
        r2_xgb = r2_score(y_test, pred_xgb)
        results['XGBoost'] = {'r2': r2_xgb, 'type': 'ML'}
        print(f"6. XGBoost:             R² = {r2_xgb:.4f}")
    
    # 7. LightGBM (if available)
    if HAS_LGB:
        model_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
        model_lgb.fit(X_train_exp, y_train)
        pred_lgb = model_lgb.predict(X_test_exp)
        r2_lgb = r2_score(y_test, pred_lgb)
        results['LightGBM'] = {'r2': r2_lgb, 'type': 'ML'}
        print(f"7. LightGBM:            R² = {r2_lgb:.4f}")
    
    # 8. MLP (DeepVol / Michael 2025 inspired)
    model_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
    model_mlp.fit(X_train_exp, y_train)
    pred_mlp = model_mlp.predict(X_test_exp)
    r2_mlp = r2_score(y_test, pred_mlp)
    results['MLP (DeepVol)'] = {'r2': r2_mlp, 'type': 'DL'}
    print(f"8. MLP (DeepVol):       R² = {r2_mlp:.4f}")
    
    # 9. Ensemble (Ridge + RF + GBM average)
    pred_ensemble = (pred_har + pred_rf + pred_gbm) / 3
    r2_ensemble = r2_score(y_test, pred_ensemble)
    results['Ensemble'] = {'r2': r2_ensemble, 'type': 'Ensemble'}
    print(f"9. Ensemble (Avg):      R² = {r2_ensemble:.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("SUMMARY: Model Comparison")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    print(f"{'Rank':<5} {'Model':<20} {'Type':<12} {'R²':<10}")
    print("-"*50)
    for i, (model, data) in enumerate(sorted_results, 1):
        print(f"{i:<5} {model:<20} {data['type']:<12} {data['r2']:.4f}")
    
    # Save Results
    out_data = {
        'model_comparison': {k: v['r2'] for k, v in results.items()},
        'model_types': {k: v['type'] for k, v in results.items()},
        'best_model': sorted_results[0][0],
        'best_r2': sorted_results[0][1]['r2']
    }
    
    out_path = 'src/experiments/universal/v8_baseline_comparison.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_baseline_comparison()
