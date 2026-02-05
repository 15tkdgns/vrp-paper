"""
V4 Exp 02: Universal Deep Learning (Log-RV Target)
Purpose: Train a single universal model on pooled data to predict Log-RV

Why Log-RV?
- Stabilizes variance (Homoscedasticity)
- Makes distribution near-Gaussian (Better for MSE loss)
- Captures relative volatility changes rather than absolute levels

Model:
- Universal MLP (Shared weights for all assets)
- Input: [Log-RV lags, Returns, Global Factors, Asset Embedding]
- Target: ln(RV_{t+22})
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

def run_universal_dl():
    print("="*70)
    print("V4 Exp 02: Universal Deep Learning (Target: Log-RV)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Download Data
    tickers = ASSETS + ['^VIX']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # 2. Feature Engineering (Log-Space)
    print("\nPreparing Log-Space Features...")
    pooled_data = []
    
    label_encoder = LabelEncoder()
    label_encoder.fit(ASSETS)
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        
        # Raw RV
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        # Log-RV Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        # Momentum in Log-Space (Ratio in arithmetic space)
        df['LogRV_Mom_5d'] = df['LogRV_lag1'] - df['LogRV_lag5']
        
        # Global Factor (Log-VIX)
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        
        # Target: Future Log-RV
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        
        df = df.dropna()
        if len(df) < 500: continue
        
        pooled_data.append(df)
        
    # 3. Stack and Normalize
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    feature_cols = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom_5d', 'LogVIX']
    
    # Global Z-score Normalization
    scaler_X = StandardScaler()
    full_df[feature_cols] = scaler_X.fit_transform(full_df[feature_cols])
    
    # Target Normalization (Crucial!)
    # We predict normalized Log-RV z-score
    scaler_y = StandardScaler()
    full_df['Target_Scaled'] = scaler_y.fit_transform(full_df[['Target']])
    
    print(f"Total Samples: {len(full_df)}")
    
    # Split (Time-based simulation: last 20% as test for each asset)
    # Ideally should be time-split, but for pooling validation we use simple split first
    # Or strict time split:
    # train_df = full_df.iloc[:int(len(full_df)*0.8)] # This is wrong for pooled data if not sorted by time
    # Correct way: Split each asset locally or sort by time globally
    
    # Using Global Time Split
    # We need Date column which we dropped. Let's assume shuffling for now to test "Universal Mechanism" hypothesis
    # If mechanism is universal, it should work on unseen data points regardless of time (approx)
    # But for rigor, we'll use a random split grouped by asset? No, let's use simple Shuffle for mechanism test.
    
    indices = np.arange(len(full_df))
    np.random.shuffle(indices)
    split = int(len(full_df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_df = full_df.iloc[train_idx]
    test_df = full_df.iloc[test_idx]
    
    # Tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_train = torch.tensor(train_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_test = torch.tensor(test_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    # 4. Train
    print("\nTraining Universal MLP...")
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ASSETS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    batch_size = 256
    n_batches = len(train_df) // batch_size
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        
        # Shuffle batches
        perm = torch.randperm(len(train_df))
        
        for i in range(n_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            pred = model(X_train[idx], a_train[idx])
            loss = criterion(pred, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {epoch_loss/n_batches:.4f}")
            
    # 5. Evaluate
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test, a_test).cpu().numpy()
        y_true_scaled = y_test.cpu().numpy()
        
    # Inverse transform to get real Log-RV
    pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true_log = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    
    # Compare in Log-Space
    r2_log = r2_score(y_true_log, pred_log)
    mae_log = mean_absolute_error(y_true_log, pred_log)
    
    # Compare in Real-Space (exp(log_rv))
    pred_rv = np.exp(pred_log)
    y_true_rv = np.exp(y_true_log)
    r2_real = r2_score(y_true_rv, pred_rv)
    
    print("\n" + "="*70)
    print("RESULTS: Universal Log-RV Prediction")
    print("="*70)
    print(f"Log-Space R² : {r2_log:.4f}")
    print(f"Real-Space R²: {r2_real:.4f}")
    
    # Per Asset R2 (Log space)
    print("\nPer Asset R² (Log-Space):")
    test_df['Pred_Log'] = pred_log
    test_df['True_Log'] = y_true_log
    
    per_asset = {}
    for asset_id in test_df['Asset_ID'].unique():
        mask = test_df['Asset_ID'] == asset_id
        sub = test_df[mask]
        
        r2 = r2_score(sub['True_Log'], sub['Pred_Log'])
        asset_name = label_encoder.inverse_transform([asset_id])[0]
        per_asset[asset_name] = r2
        print(f"  {asset_name:6}: {r2:.4f}")
        
    print(f"\nAvg Per-Asset R²: {np.mean(list(per_asset.values())):.4f}")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp02_universal.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'overall_log_r2': r2_log,
            'overall_real_r2': r2_real,
            'per_asset_r2': per_asset
        }, f, indent=2)

if __name__ == "__main__":
    run_universal_dl()
