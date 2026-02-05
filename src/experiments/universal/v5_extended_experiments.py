"""
V5 Extended: Alternative Improvement Strategies
================================================
Since complex architectures didn't help, try:
1. More diverse features (macro, cross-asset, technicals)
2. Ensemble of simple models
3. Larger asset universe

Baseline to beat: V4 Universal MLP (Avg Asset R² = 0.43 in log-space)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Extended asset universe (30+ assets)
ASSETS = [
    # US Equity Sectors
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLB', 'XLP', 'XLU', 'XLY', 'XLRE',
    # Global
    'EFA', 'EEM', 'VGK', 'EWJ', 'FXI',
    # Bonds
    'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG', 'AGG',
    # Commodities
    'GLD', 'SLV', 'USO', 'DBC', 'UNG',
    # Volatility
    'VXX',
]

class UniversalMLP(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=4, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_features + embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, asset_ids):
        embed = self.asset_embedding(asset_ids)
        x = torch.cat([x, embed], dim=1)
        return self.net(x).squeeze()

def run_extended_experiments():
    print("="*70)
    print("V5 Extended: Alternative Improvement Strategies")
    print("="*70)
    
    # 1. Download Extended Data
    tickers = ASSETS + ['^VIX', '^VIX3M', '^SKEW', '^TNX', '^DJI', 'DX-Y.NYB']
    print(f"Downloading {len(tickers)} tickers...")
    
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '').replace('-Y.NYB', '') for c in close.columns]
    close = close.ffill()
    
    valid_assets = [a for a in ASSETS if a in close.columns]
    print(f"\nValid assets: {len(valid_assets)}")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_assets)
    
    # ============================================================
    # EXPERIMENT 1: EXTENDED FEATURE SET
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Extended Feature Engineering")
    print("="*70)
    
    pooled_data = []
    
    # Extended features
    basic_features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom']
    global_features = ['LogVIX', 'VIX_Term', 'SKEW', 'TNX', 'DXY']
    technical_features = ['RSI', 'MACD', 'BB_Width']
    cross_asset_features = ['SPY_RV', 'TLT_RV', 'GLD_RV']
    
    all_features = basic_features + global_features + technical_features + cross_asset_features
    
    # Compute cross-asset RVs
    cross_rvs = {}
    for ca in ['SPY', 'TLT', 'GLD']:
        if ca in close.columns:
            ret = np.log(close[ca] / close[ca].shift(1))
            cross_rvs[ca] = np.log(ret.rolling(22).std() * np.sqrt(252) * 100 + 1e-6)
    
    for asset in valid_assets:
        df = pd.DataFrame(index=close.index)
        
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        # Basic features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        
        # Global features
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1) if 'VIX' in close.columns else 0
        df['VIX_Term'] = (close.get('VIX3M', close['VIX']) - close['VIX']).shift(1) if 'VIX' in close.columns else 0
        df['SKEW'] = (close['SKEW'] / 100 - 1).shift(1) if 'SKEW' in close.columns else 0
        df['TNX'] = close['TNX'].shift(1) / 10 if 'TNX' in close.columns else 0
        df['DXY'] = (close.get('DX', 100) / 100 - 1).shift(1)
        
        # Technical indicators
        price = close[asset]
        df['RSI'] = self_compute_rsi(price, 14).shift(1) / 100
        df['MACD'] = (price.ewm(span=12).mean() - price.ewm(span=26).mean()).shift(1) / price.shift(1)
        bb_mid = price.rolling(20).mean()
        bb_std = price.rolling(20).std()
        df['BB_Width'] = ((bb_std * 2) / bb_mid).shift(1)
        
        # Cross-asset features
        df['SPY_RV'] = cross_rvs.get('SPY', pd.Series(0, index=df.index)).shift(1)
        df['TLT_RV'] = cross_rvs.get('TLT', pd.Series(0, index=df.index)).shift(1)
        df['GLD_RV'] = cross_rvs.get('GLD', pd.Series(0, index=df.index)).shift(1)
        
        # Target
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        df['Asset'] = asset
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) > 100:
            pooled_data.append(df)
    
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    print(f"Total samples with extended features: {len(full_df)}")
    
    # Normalize
    scaler = StandardScaler()
    full_df[all_features] = scaler.fit_transform(full_df[all_features])
    full_df['Target_Scaled'] = StandardScaler().fit_transform(full_df[['Target']])
    
    # Split
    split = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split]
    test_df = full_df.iloc[split:]
    
    # ============================================================
    # EXPERIMENT 2: MODEL ENSEMBLE
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Model Ensemble")
    print("="*70)
    
    X_train = train_df[all_features].values
    y_train = train_df['Target_Scaled'].values
    X_test = test_df[all_features].values
    y_test = test_df['Target_Scaled'].values
    
    # Individual models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Huber': HuberRegressor(epsilon=1.35),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    
    predictions = {}
    individual_r2 = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        r2 = r2_score(y_test, pred)
        individual_r2[name] = r2
        print(f"    {name} R²: {r2:.4f}")
    
    # Ensemble: Simple Average
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    print(f"\n  Ensemble (Average) R²: {ensemble_r2:.4f}")
    
    # ============================================================
    # EXPERIMENT 3: NEURAL NETWORK WITH EXTENDED FEATURES
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: MLP with Extended Features")
    print("="*70)
    
    device = torch.device('cpu')
    
    X_train_t = torch.tensor(train_df[all_features].values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(train_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_train = torch.tensor(train_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    X_test_t = torch.tensor(test_df[all_features].values, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(test_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_test = torch.tensor(test_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    model = UniversalMLP(n_features=len(all_features), n_assets=len(valid_assets), embed_dim=8, hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(train_df))[:10000]
        optimizer.zero_grad()
        pred = model(X_train_t[perm], a_train[perm])
        loss = criterion(pred, y_train_t[perm])
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        nn_pred = model(X_test_t, a_test).numpy()
    
    nn_r2 = r2_score(y_test, nn_pred)
    print(f"\n  MLP Extended R²: {nn_r2:.4f}")
    
    # Per-asset analysis
    test_df_copy = test_df.copy()
    test_df_copy['Pred_Ensemble'] = ensemble_pred
    test_df_copy['Pred_NN'] = nn_pred
    
    print("\n" + "="*70)
    print("PER-ASSET PERFORMANCE (Top 10)")
    print("="*70)
    
    asset_results = []
    for asset in valid_assets:
        mask = test_df_copy['Asset'] == asset
        if mask.sum() < 50:
            continue
        
        r2_ens = r2_score(test_df_copy.loc[mask, 'Target_Scaled'], test_df_copy.loc[mask, 'Pred_Ensemble'])
        r2_nn = r2_score(test_df_copy.loc[mask, 'Target_Scaled'], test_df_copy.loc[mask, 'Pred_NN'])
        
        asset_results.append({'Asset': asset, 'Ensemble': r2_ens, 'NN_Extended': r2_nn})
    
    asset_df = pd.DataFrame(asset_results).sort_values('NN_Extended', ascending=False)
    print(asset_df.head(10).to_string())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Approach':<30} | {'Overall R²':>12}")
    print("-"*50)
    for name, r2 in individual_r2.items():
        print(f"{name:<30} | {r2:>12.4f}")
    print(f"{'Ensemble (Average)':<30} | {ensemble_r2:>12.4f}")
    print(f"{'MLP Extended Features':<30} | {nn_r2:>12.4f}")
    
    best_approach = max([('Ensemble', ensemble_r2), ('MLP Extended', nn_r2)] + list(individual_r2.items()), key=lambda x: x[1])
    print(f"\n★ Best Approach: {best_approach[0]} (R² = {best_approach[1]:.4f})")
    
    # Avg per-asset
    avg_ens = asset_df['Ensemble'].mean()
    avg_nn = asset_df['NN_Extended'].mean()
    print(f"\nAvg Per-Asset R²:")
    print(f"  Ensemble: {avg_ens:.4f}")
    print(f"  NN Extended: {avg_nn:.4f}")
    
    # Save
    results = {
        'individual_models': individual_r2,
        'ensemble': ensemble_r2,
        'nn_extended': nn_r2,
        'avg_per_asset_ensemble': avg_ens,
        'avg_per_asset_nn': avg_nn,
        'per_asset': asset_df.to_dict(orient='records')
    }
    
    out_path = 'experiments/07_v2_methodology/results/v5_extended.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

def self_compute_rsi(prices, period=14):
    """Compute RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    run_extended_experiments()
