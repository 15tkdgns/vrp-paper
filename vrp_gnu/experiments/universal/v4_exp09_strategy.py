"""
V4 Exp 09: Strategy Backtest (Vol Timing)
Purpose: Test if the same vol-timing rule works universally across asset classes

Strategy:
- Use Universal Model predictions to adjust exposure
- Rule: If predicted vol > current vol * 1.1 → Reduce exposure (risk-off)
- Compare Sharpe ratios across asset classes
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'USO', 'EFA', 'EEM']

ASSET_CLASSES = {
    'US_Equity': ['SPY', 'QQQ'],
    'Bonds': ['TLT', 'IEF'],
    'Commodities': ['GLD', 'USO'],
    'EM': ['EFA', 'EEM'],
}

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

def compute_sharpe(returns):
    """Annualized Sharpe Ratio"""
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def run_strategy_backtest():
    print("="*70)
    print("V4 Exp 09: Strategy Backtest (Universal Vol Timing)")
    print("="*70)
    
    # 1. Download Data
    raw = yf.download(ASSETS + ['^VIX'], start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # 2. Train Universal Model
    label_encoder = LabelEncoder()
    label_encoder.fit(ASSETS)
    
    pooled_data = []
    feature_cols = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom', 'LogVIX']
    
    for asset in ASSETS:
        if asset not in close.columns:
            continue
        df = pd.DataFrame(index=close.index)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['RV'] = rv
        df['Return'] = ret
        df['Price'] = close[asset]
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset'] = asset
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        df = df.dropna()
        if len(df) > 100:
            pooled_data.append(df)
    
    full_df = pd.concat(pooled_data).reset_index()
    full_df = full_df.rename(columns={'index': 'Date'})
    
    # Normalize features
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])
    
    scaler_y = StandardScaler()
    full_df['Target_Scaled'] = scaler_y.fit_transform(full_df[['Target']])
    
    # Train on 80%
    train_end = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:train_end]
    test_df = full_df.iloc[train_end:].copy()
    
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Train model
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Target_Scaled'].values, dtype=torch.float32)
    a_train = torch.tensor(train_df['Asset_ID'].values, dtype=torch.long)
    
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(train_df))[:10000]
        optimizer.zero_grad()
        pred = model(X_train[perm], a_train[perm])
        loss = criterion(pred, y_train[perm])
        loss.backward()
        optimizer.step()
    
    # 3. Generate Predictions on Test Set
    model.eval()
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    a_test = torch.tensor(test_df['Asset_ID'].values, dtype=torch.long)
    
    with torch.no_grad():
        pred_scaled = model(X_test, a_test).numpy()
    
    # Convert back to log-RV
    pred_log_rv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    test_df['Pred_LogRV'] = pred_log_rv
    test_df['Pred_RV'] = np.exp(pred_log_rv)
    
    # 4. Trading Strategy
    print("\n" + "="*70)
    print("STRATEGY BACKTEST RESULTS")
    print("="*70)
    
    results = {}
    
    for asset in ASSETS:
        asset_df = test_df[test_df['Asset'] == asset].copy()
        
        if len(asset_df) < 100:
            continue
        
        # Strategy: Reduce exposure when vol is expected to rise
        # Signal: pred_RV / current_RV > 1.1 → risk off (weight = 0.5)
        # Otherwise: full exposure (weight = 1.0)
        
        asset_df['Vol_Signal'] = asset_df['Pred_RV'] / asset_df['RV']
        asset_df['Weight'] = np.where(asset_df['Vol_Signal'] > 1.1, 0.5, 1.0)
        
        # Buy and hold returns
        asset_df['BH_Return'] = asset_df['Return']
        
        # Strategy returns
        asset_df['Strat_Return'] = asset_df['Return'] * asset_df['Weight'].shift(1)
        
        # Metrics
        sharpe_bh = compute_sharpe(asset_df['BH_Return'].dropna())
        sharpe_strat = compute_sharpe(asset_df['Strat_Return'].dropna())
        
        cum_bh = (1 + asset_df['BH_Return'].fillna(0)).cumprod().iloc[-1] - 1
        cum_strat = (1 + asset_df['Strat_Return'].fillna(0)).cumprod().iloc[-1] - 1
        
        max_dd_bh = ((1 + asset_df['BH_Return'].fillna(0)).cumprod().cummax() - 
                    (1 + asset_df['BH_Return'].fillna(0)).cumprod()).max()
        max_dd_strat = ((1 + asset_df['Strat_Return'].fillna(0)).cumprod().cummax() - 
                       (1 + asset_df['Strat_Return'].fillna(0)).cumprod()).max()
        
        # Get asset class
        asset_class = 'Unknown'
        for cls, assets in ASSET_CLASSES.items():
            if asset in assets:
                asset_class = cls
                break
        
        results[asset] = {
            'class': asset_class,
            'sharpe_bh': sharpe_bh,
            'sharpe_strat': sharpe_strat,
            'sharpe_improve': sharpe_strat - sharpe_bh,
            'total_return_bh': cum_bh,
            'total_return_strat': cum_strat,
            'max_dd_bh': max_dd_bh,
            'max_dd_strat': max_dd_strat
        }
        
        print(f"  {asset:6} | BH Sharpe: {sharpe_bh:6.3f} | Strat Sharpe: {sharpe_strat:6.3f} | Δ: {sharpe_strat - sharpe_bh:+6.3f}")
    
    # 5. Summary by Asset Class
    print("\n" + "="*70)
    print("SUMMARY BY ASSET CLASS")
    print("="*70)
    
    for cls in ASSET_CLASSES.keys():
        class_assets = [a for a, r in results.items() if r['class'] == cls]
        if not class_assets:
            continue
        
        avg_improve = np.mean([results[a]['sharpe_improve'] for a in class_assets])
        win_rate = np.mean([1 if results[a]['sharpe_improve'] > 0 else 0 for a in class_assets])
        
        print(f"  {cls:15}: Avg Sharpe Improvement = {avg_improve:+.3f}, Win Rate = {win_rate*100:.0f}%")
    
    # 6. Overall
    all_improve = [r['sharpe_improve'] for r in results.values()]
    overall_avg = np.mean(all_improve)
    overall_win = np.mean([1 if x > 0 else 0 for x in all_improve])
    
    print(f"\n  Overall: Avg Improvement = {overall_avg:+.3f}, Win Rate = {overall_win*100:.0f}%")
    
    # 7. Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if overall_win >= 0.7:
        print("✅ STRONG STRATEGY UNIVERSALITY: Vol timing improves Sharpe in 70%+ of assets")
    elif overall_win >= 0.5:
        print("⚠️ MODERATE UNIVERSALITY: Vol timing helps more often than not")
    else:
        print("❌ NO UNIVERSALITY: Vol timing does not consistently improve performance")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp09_strategy.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_strategy_backtest()
