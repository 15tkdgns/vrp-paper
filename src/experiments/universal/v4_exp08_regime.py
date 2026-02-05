"""
V4 Exp 08: Regime Invariance Test
Purpose: Test if Universal Model works equally well in different VIX regimes

Regimes:
- Low Vol: VIX < 15
- Normal: 15 <= VIX < 25
- High Vol: VIX >= 25
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'IEF', 'GLD', 'USO', 'EFA', 'EEM']

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

def run_regime_test():
    print("="*70)
    print("V4 Exp 08: Regime Invariance Test")
    print("="*70)
    
    # 1. Download Data
    raw = yf.download(ASSETS + ['^VIX'], start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # 2. Prepare Data
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
        df['VIX'] = close['VIX']
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        df = df.dropna()
        if len(df) > 100:
            pooled_data.append(df)
    
    full_df = pd.concat(pooled_data).reset_index()
    full_df = full_df.rename(columns={'index': 'Date'})
    
    # Normalize
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])
    full_df['Target_Scaled'] = StandardScaler().fit_transform(full_df[['Target']])
    
    # 3. Define Regimes
    full_df['Regime'] = pd.cut(full_df['VIX'], 
                               bins=[0, 15, 25, 100], 
                               labels=['Low', 'Normal', 'High'])
    
    print("\nRegime Distribution:")
    print(full_df['Regime'].value_counts())
    
    # 4. Train Single Model on All Data
    print("\nTraining Universal Model on all regimes...")
    X = torch.tensor(full_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(full_df['Target_Scaled'].values, dtype=torch.float32)
    a = torch.tensor(full_df['Asset_ID'].values, dtype=torch.long)
    
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(full_df))[:10000]
        optimizer.zero_grad()
        pred = model(X[perm], a[perm])
        loss = criterion(pred, y[perm])
        loss.backward()
        optimizer.step()
    
    # 5. Evaluate by Regime
    print("\n" + "="*70)
    print("REGIME-SPECIFIC PERFORMANCE")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X, a).numpy()
    
    full_df['Pred'] = predictions
    
    results = {}
    
    for regime in ['Low', 'Normal', 'High']:
        mask = full_df['Regime'] == regime
        if mask.sum() < 100:
            continue
        
        y_true = full_df.loc[mask, 'Target_Scaled'].values
        y_pred = full_df.loc[mask, 'Pred'].values
        
        r2 = r2_score(y_true, y_pred)
        n_samples = mask.sum()
        
        results[regime] = {'r2': r2, 'n_samples': int(n_samples)}
        print(f"  {regime:8} Regime: R² = {r2:.4f} (N = {n_samples})")
    
    # 6. Regime Transition Analysis
    print("\n" + "="*70)
    print("REGIME TRANSITION ANALYSIS")
    print("="*70)
    
    # Find regime transitions
    full_df['Prev_Regime'] = full_df['Regime'].shift(1)
    transitions = full_df[full_df['Regime'] != full_df['Prev_Regime']]
    
    print(f"  Total regime transitions: {len(transitions)}")
    
    # Performance around transitions (±5 days window)
    trans_indices = transitions.index.tolist()
    
    if len(trans_indices) > 10:
        window_indices = []
        for idx in trans_indices[:100]:  # Limit for speed
            pos = full_df.index.get_loc(idx)
            window_indices.extend(range(max(0, pos-5), min(len(full_df), pos+6)))
        
        window_indices = list(set(window_indices))
        window_df = full_df.iloc[window_indices]
        
        r2_transition = r2_score(window_df['Target_Scaled'], window_df['Pred'])
        r2_stable = r2_score(full_df.loc[~full_df.index.isin(window_indices), 'Target_Scaled'],
                            full_df.loc[~full_df.index.isin(window_indices), 'Pred'])
        
        print(f"  R² during transitions (±5d): {r2_transition:.4f}")
        print(f"  R² during stable periods: {r2_stable:.4f}")
        
        results['transition'] = r2_transition
        results['stable'] = r2_stable
    
    # 7. Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    r2_values = [results[r]['r2'] for r in ['Low', 'Normal', 'High'] if r in results]
    r2_std = np.std(r2_values)
    
    if r2_std < 0.05:
        print("✅ STRONG INVARIANCE: Model performance is stable across regimes")
    elif r2_std < 0.1:
        print("⚠️ MODERATE INVARIANCE: Some regime-dependence exists")
    else:
        print("❌ NO INVARIANCE: Model performance differs significantly by regime")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp08_regime.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_regime_test()
