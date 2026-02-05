"""
V6 Exp: Feature Expansion for Universal Model
Purpose: Add Literature-based features (Downside RV, VRP, Technicals) and evaluate performance.

Features added:
1. Downside/Upside RV (Feunou & Okou 2017)
2. VRP Proxy (Carr & Wu 2009)
3. Technical Indicators (RSI, Bollinger Band)
4. Higher Moments (Skewness, Kurtosis) as Jump Proxies
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import r2_score
import json
import os
import warnings

warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'EFA', 'EEM', 'IOO',
    'TLT', 'IEF', 'SHY', 'TIP', 'ZROZ',
    'GLD', 'USO', 'SLV', 'DBC',
]

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(data, window=20):
    ma = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / (ma + 1e-6)

def run_feature_expansion():
    print("="*70)
    print("V6 Experiment: Feature Expansion")
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
        
        # --- Basic Features (HAR) ---
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000 # Scaling
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # --- 1. Downside / Upside RV (Feunou & Okou 2017) ---
        # Using daily returns sign over rolling window
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        
        # Downside RV: Sum of squared negative returns
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        df['RS_Asymmetry'] = df['LogRV_Down'] - df['LogRV_Up'] # Signed asymmetry
        
        # --- 2. VRP Proxy (Carr & Wu 2009) ---
        # VIX is market-wide, but we map it to asset specific if possible.
        # Here we use Global VIX as a common factor
        vix = close['VIX']
        df['LogVIX'] = np.log(vix + 1e-6).shift(1)
        # VRP = VIX^2 - RV (Approx)
        # We assume VIX predicts SPY RV, but acts as risk factor for others.
        # Asset Specific VRP Proxy: LogVIX - LogRV_Asset
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1'] 
        
        # --- 3. Technical Indicators ---
        df['RSI'] = calculate_rsi(price).shift(1) / 100.0 # Normalize 0-1
        df['BB_Width'] = calculate_bollinger_width(price).shift(1)
        
        # --- 4. Higher Moments (Jump Proxies) ---
        df['Skewness'] = ret.rolling(22).skew().shift(1).fillna(0)
        df['Kurtosis'] = ret.rolling(22).kurt().shift(1).fillna(0)
        
        # --- Target ---
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        
        pooled_data.append(df)
        
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    # Defining Feature Sets
    features_basic = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    features_expanded = features_basic + [
        'LogRV_Down', 'LogRV_Up', 'RS_Asymmetry',
        'VRP_Proxy', 
        'RSI', 'BB_Width',
        'Skewness', 'Kurtosis'
    ]
    
    print(f"Total Samples: {len(full_df)}")
    
    # Split
    indices = np.arange(len(full_df))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(full_df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_df = full_df.iloc[train_idx]
    test_df = full_df.iloc[test_idx]
    
    # Model Training & Comparison
    results = {}
    
    # 1. Baseline Model (Basic Params)
    print("\nTraining Baseline (Ridge)...")
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(train_df[features_basic])
    X_test_base = scaler_base.transform(test_df[features_basic])
    
    model_base = Ridge(alpha=1.0)
    model_base.fit(X_train_base, train_df['Target'])
    pred_base = model_base.predict(X_test_base)
    r2_base = r2_score(test_df['Target'], pred_base)
    print(f"Baseline R²: {r2_base:.4f}")
    
    # 2. Expanded Model (Ridge)
    print("Training Expanded (Ridge)...")
    scaler_exp = StandardScaler()
    X_train_exp = scaler_exp.fit_transform(train_df[features_expanded])
    X_test_exp = scaler_exp.transform(test_df[features_expanded])
    
    model_exp_linear = Ridge(alpha=1.0)
    model_exp_linear.fit(X_train_exp, train_df['Target'])
    pred_exp_linear = model_exp_linear.predict(X_test_exp)
    r2_exp_linear = r2_score(test_df['Target'], pred_exp_linear)
    print(f"Expanded (Ridge) R²: {r2_exp_linear:.4f}")
    
    # 3. Expanded Model (Random Forest) - To capture non-linearities in technicals
    print("Training Expanded (Random Forest) - This may take a moment...")
    model_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model_rf.fit(X_train_exp, train_df['Target'])
    pred_rf = model_rf.predict(X_test_exp)
    r2_rf = r2_score(test_df['Target'], pred_rf)
    print(f"Expanded (RF) R²: {r2_rf:.4f}")

    # Feature Importance (RF)
    importances = model_rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features_expanded, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    
    print("\nTop 5 Features:")
    print(feat_imp.head(5))
    
    # Save Results
    out_data = {
        "baseline_r2": r2_base,
        "expanded_linear_r2": r2_exp_linear,
        "expanded_rf_r2": r2_rf,
        "feature_importance": feat_imp.to_dict(orient='records')
    }
    
    out_path = 'src/experiments/universal/v6_feature_expansion.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_feature_expansion()
