"""
V7 Exp: Feature Expansion with Asset Category Analysis
Purpose: Add Literature-based features and evaluate performance BY ASSET CATEGORY.

Asset Categories:
- Equity: SPY, QQQ, IWM, DIA, MDY, XLF, XLE, XLK, XLV, XLI, EFA, EEM, IOO
- Bond: TLT, IEF, SHY, TIP, ZROZ
- Commodity: GLD, USO, SLV, DBC
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Asset Category Mapping
ASSET_CATEGORIES = {
    # Equity
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    # Bond
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    # Commodity
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
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / (ma + 1e-6)

def run_feature_expansion_v7():
    print("="*70)
    print("V7 Experiment: Feature Expansion with Category Analysis")
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
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Downside / Upside RV
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        df['RS_Asymmetry'] = df['LogRV_Down'] - df['LogRV_Up']
        
        # VRP Proxy
        vix = close['VIX']
        df['LogVIX'] = np.log(vix + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1'] 
        
        # Technical Indicators
        df['RSI'] = calculate_rsi(price).shift(1) / 100.0
        df['BB_Width'] = calculate_bollinger_width(price).shift(1)
        
        # Higher Moments
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
        'VRP_Proxy', 
        'RSI', 'BB_Width',
        'Skewness', 'Kurtosis'
    ]
    
    print(f"Total Samples: {len(full_df)}")
    print(f"Category Distribution:\n{full_df['Category'].value_counts()}")
    
    # Split
    indices = np.arange(len(full_df))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(full_df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_df = full_df.iloc[train_idx]
    test_df = full_df.iloc[test_idx].copy()
    
    # Train Expanded RF
    print("\nTraining Expanded (Random Forest)...")
    scaler_exp = StandardScaler()
    X_train_exp = scaler_exp.fit_transform(train_df[features_expanded])
    X_test_exp = scaler_exp.transform(test_df[features_expanded])
    
    model_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model_rf.fit(X_train_exp, train_df['Target'])
    pred_rf = model_rf.predict(X_test_exp)
    
    test_df['Pred'] = pred_rf
    
    # Overall R2
    r2_overall = r2_score(test_df['Target'], test_df['Pred'])
    print(f"\nOverall R²: {r2_overall:.4f}")
    
    # Per-Category R2
    print("\n" + "="*50)
    print("Per-Category Performance")
    print("="*50)
    
    category_results = {}
    for cat in test_df['Category'].unique():
        mask = test_df['Category'] == cat
        sub = test_df[mask]
        r2_cat = r2_score(sub['Target'], sub['Pred'])
        n_samples = len(sub)
        category_results[cat] = {'r2': r2_cat, 'n_samples': n_samples}
        print(f"  {cat:12}: R² = {r2_cat:.4f} (n={n_samples})")
    
    # Per-Asset R2
    print("\n" + "="*50)
    print("Per-Asset Performance (Top 10)")
    print("="*50)
    
    asset_results = {}
    for asset in test_df['Asset'].unique():
        mask = test_df['Asset'] == asset
        sub = test_df[mask]
        if len(sub) < 50: continue
        r2_asset = r2_score(sub['Target'], sub['Pred'])
        asset_results[asset] = r2_asset
    
    sorted_assets = sorted(asset_results.items(), key=lambda x: x[1], reverse=True)
    for i, (asset, r2) in enumerate(sorted_assets[:10], 1):
        cat = ASSET_CATEGORIES[asset]
        print(f"  {i:2}. {asset:5} ({cat:10}): R² = {r2:.4f}")
    
    # Feature Importance
    importances = model_rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features_expanded, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    
    print("\nTop 5 Features:")
    print(feat_imp.head(5))
    
    # Save Results
    out_data = {
        "overall_r2": r2_overall,
        "category_r2": category_results,
        "per_asset_r2": asset_results,
        "feature_importance": feat_imp.to_dict(orient='records')
    }
    
    out_path = 'src/experiments/universal/v7_category_analysis.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_feature_expansion_v7()
