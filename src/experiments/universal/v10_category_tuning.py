"""
V10 Exp: Per-Category Model Optimization
Purpose: Tune Top3 models (RF, XGBoost, Deep MLP) for each asset category.

Categories:
- Equity: 13 assets
- Bond: 5 assets
- Commodity: 4 assets

Models to Tune:
1. Random Forest
2. XGBoost
3. Deep MLP (PyTorch)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, make_scorer
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Asset Categories
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity',
}
ASSETS = list(ASSET_CATEGORIES.keys())

class DeepMLP(nn.Module):
    def __init__(self, n_features, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(3)
        ])
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x).squeeze(-1)

def train_mlp(X_train, y_train, X_test, y_test, hidden_dim=128, dropout=0.3, lr=0.001, epochs=50):
    device = torch.device('cpu') # Use CPU to avoid OOM
    n_features = X_train.shape[1]
    
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model = DeepMLP(n_features, hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(X_te).cpu().numpy()
    return r2_score(y_test, pred)

def run_category_tuning():
    print("="*70)
    print("V10 Experiment: Per-Category Model Optimization")
    print("="*70)
    
    # 1. Download & Prepare Data
    tickers = ASSETS + ['^VIX']
    print("Downloading data...")
    # Increase timeout and add random user agent if possible (yfinance handles UA internally mostly)
    # Using a loop to retry if empty
    raw = pd.DataFrame()
    for _ in range(3):
        try:
            raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False, timeout=60)
            if not raw.empty:
                break
        except Exception as e:
            print(f"Download failed: {e}, retrying...")
    
    if raw.empty:
        print("Failed to download data.")
        return
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1))
        
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_lag1'] = np.log(rv + 1e-6).shift(1)
        df['LogRV_lag5'] = np.log(rv + 1e-6).shift(5)
        df['LogRV_lag22'] = np.log(rv + 1e-6).shift(22)
        
        ret_neg = ret.where(ret < 0, 0)
        df['LogRV_Down'] = np.log((ret_neg**2).rolling(22).mean() * 252 * 10000 + 1e-6).shift(1)
        
        vix = close['VIX']
        df['LogVIX'] = np.log(vix + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1']
        
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Category'] = ASSET_CATEGORIES[asset]
        df = df.dropna()
        if len(df) < 500: continue
        
        pooled_data.append(df)
        
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Down', 'VRP_Proxy']
    
    results = {}
    
    for cat in ['Equity', 'Bond', 'Commodity']:
        print(f"\n{'='*50}")
        print(f"Category: {cat}")
        print("="*50)
        
        cat_df = full_df[full_df['Category'] == cat].reset_index(drop=True)
        print(f"Samples: {len(cat_df)}")
        
        # Split
        indices = np.arange(len(cat_df))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(len(cat_df) * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        
        train_df = cat_df.iloc[train_idx]
        test_df = cat_df.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        y_train = train_df['Target'].values
        y_test = test_df['Target'].values
        
        cat_results = {}
        
        # 1. Random Forest Tuning
        print("\n1. Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10]
        }
        rf_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_params, cv=3, scoring='r2', n_jobs=-1
        )
        rf_search.fit(X_train, y_train)
        rf_pred = rf_search.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        cat_results['RandomForest'] = {
            'r2': rf_r2,
            'best_params': rf_search.best_params_
        }
        print(f"   Best R²: {rf_r2:.4f}")
        print(f"   Params: {rf_search.best_params_}")
        
        # 2. XGBoost Tuning
        print("\n2. Tuning XGBoost...")
        xgb_params = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        }
        xgb_search = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params, cv=3, scoring='r2', n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        xgb_pred = xgb_search.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        cat_results['XGBoost'] = {
            'r2': xgb_r2,
            'best_params': xgb_search.best_params_
        }
        print(f"   Best R²: {xgb_r2:.4f}")
        print(f"   Params: {xgb_search.best_params_}")
        
        # 3. Deep MLP Tuning (Simple Grid)
        print("\n3. Tuning Deep MLP...")
        best_mlp_r2 = -float('inf')
        best_mlp_params = {}
        
        for hidden in [64, 128]:
            for dropout in [0.2, 0.3]:
                for lr in [0.001, 0.0005]:
                    r2 = train_mlp(X_train, y_train, X_test, y_test, 
                                   hidden_dim=hidden, dropout=dropout, lr=lr, epochs=30)
                    if r2 > best_mlp_r2:
                        best_mlp_r2 = r2
                        best_mlp_params = {'hidden_dim': hidden, 'dropout': dropout, 'lr': lr}
        
        cat_results['DeepMLP'] = {
            'r2': best_mlp_r2,
            'best_params': best_mlp_params
        }
        print(f"   Best R²: {best_mlp_r2:.4f}")
        print(f"   Params: {best_mlp_params}")
        
        # Summary for this category
        print(f"\n--- {cat} Summary ---")
        sorted_models = sorted(cat_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        for i, (model, data) in enumerate(sorted_models, 1):
            print(f"   {i}. {model}: R² = {data['r2']:.4f}")
        
        results[cat] = cat_results
    
    # Save Results
    out_data = {'per_category': results}
    out_path = 'src/experiments/universal/v10_category_tuning.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_category_tuning()
