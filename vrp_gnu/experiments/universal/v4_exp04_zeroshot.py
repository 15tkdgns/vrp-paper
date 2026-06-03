"""
V4 Exp 04: Zero-shot / Out-of-Domain Transfer Test
Purpose: Test if Universal Model generalizes to unseen assets

Design:
- Train on 12 assets (US Equity only)
- Test on 5 unseen assets (Bonds, Commodities, EM)
- Compare: Zero-shot vs Few-shot (10 epochs) vs Full training
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Training: US Equity only
TRAIN_ASSETS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'MDY', 'XLI', 'XLB', 'XLP']

# Test-Unseen: Never seen during training  
TEST_UNSEEN = ['TLT', 'IEF', 'GLD', 'USO', 'EEM']

# All assets for embedding
ALL_ASSETS = TRAIN_ASSETS + TEST_UNSEEN

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

def prepare_data(close, assets, label_encoder, scaler_X=None, scaler_y=None, fit_scalers=False):
    """Prepare Log-RV features for given assets"""
    pooled_data = []
    feature_cols = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom', 'LogVIX']
    
    for asset in assets:
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
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1) if 'VIX' in close.columns else 0
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        
        df = df.dropna()
        if len(df) > 100:
            pooled_data.append(df)
    
    if not pooled_data:
        return None, None, None, None, None
        
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    # Scaling
    if fit_scalers:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        full_df[feature_cols] = scaler_X.fit_transform(full_df[feature_cols])
        full_df['Target_Scaled'] = scaler_y.fit_transform(full_df[['Target']])
    else:
        full_df[feature_cols] = scaler_X.transform(full_df[feature_cols])
        full_df['Target_Scaled'] = scaler_y.transform(full_df[['Target']])
    
    return full_df, feature_cols, scaler_X, scaler_y, label_encoder

def run_zeroshot_experiment():
    print("="*70)
    print("V4 Exp 04: Zero-shot / Out-of-Domain Transfer Test")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Download Data
    tickers = ALL_ASSETS + ['^VIX']
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # 2. Prepare Label Encoder (for all assets)
    label_encoder = LabelEncoder()
    label_encoder.fit(ALL_ASSETS)
    
    # 3. Prepare Training Data (US Equity only)
    print("\n[1] Preparing Training Data (US Equity only)...")
    train_df, feature_cols, scaler_X, scaler_y, _ = prepare_data(
        close, TRAIN_ASSETS, label_encoder, fit_scalers=True
    )
    print(f"  Training samples: {len(train_df)}")
    
    # 4. Train Universal Model
    print("\n[2] Training Universal Model on US Equity...")
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_train = torch.tensor(train_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ALL_ASSETS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    batch_size = 256
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(train_df))
        for i in range(len(train_df) // batch_size):
            idx = perm[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            pred = model(X_train[idx], a_train[idx])
            loss = criterion(pred, y_train[idx])
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # 5. Evaluate on Training Assets (Sanity Check)
    print("\n[3] Evaluating on SEEN assets (US Equity)...")
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train, a_train).cpu().numpy()
    
    r2_seen = r2_score(y_train.cpu().numpy(), pred_train)
    print(f"  R² (Seen): {r2_seen:.4f}")
    
    # 6. Zero-shot Test on Unseen Assets
    print("\n[4] Zero-shot Test on UNSEEN assets...")
    test_df, _, _, _, _ = prepare_data(
        close, TEST_UNSEEN, label_encoder, scaler_X, scaler_y, fit_scalers=False
    )
    
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_test = torch.tensor(test_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    with torch.no_grad():
        pred_zeroshot = model(X_test, a_test).cpu().numpy()
    
    r2_zeroshot = r2_score(y_test.cpu().numpy(), pred_zeroshot)
    print(f"  R² (Zero-shot): {r2_zeroshot:.4f}")
    
    # Per-asset zero-shot
    print("\n  Per-Asset Zero-shot R²:")
    zeroshot_results = {}
    for asset in TEST_UNSEEN:
        asset_id = label_encoder.transform([asset])[0]
        mask = test_df['Asset_ID'] == asset_id
        if mask.sum() < 10:
            continue
        
        y_asset = test_df.loc[mask, 'Target_Scaled'].values
        pred_asset = pred_zeroshot[mask.values]
        r2 = r2_score(y_asset, pred_asset)
        zeroshot_results[asset] = r2
        print(f"    {asset}: {r2:.4f}")
    
    # 7. Few-shot Fine-tuning (10 epochs)
    print("\n[5] Few-shot Fine-tuning (10 epochs per unseen asset)...")
    
    # Copy model for fine-tuning
    model_fewshot = UniversalMLP(n_features=len(feature_cols), n_assets=len(ALL_ASSETS)).to(device)
    model_fewshot.load_state_dict(model.state_dict())
    
    # Freeze backbone, only train embedding for new assets
    for param in model_fewshot.net.parameters():
        param.requires_grad = False
    
    optimizer_ft = torch.optim.Adam(model_fewshot.asset_embedding.parameters(), lr=0.01)
    
    for epoch in range(10):
        model_fewshot.train()
        optimizer_ft.zero_grad()
        pred_ft = model_fewshot(X_test, a_test)
        loss_ft = criterion(pred_ft, y_test)
        loss_ft.backward()
        optimizer_ft.step()
    
    model_fewshot.eval()
    with torch.no_grad():
        pred_fewshot = model_fewshot(X_test, a_test).cpu().numpy()
    
    r2_fewshot = r2_score(y_test.cpu().numpy(), pred_fewshot)
    print(f"  R² (Few-shot): {r2_fewshot:.4f}")
    
    # Per-asset few-shot
    print("\n  Per-Asset Few-shot R²:")
    fewshot_results = {}
    for asset in TEST_UNSEEN:
        asset_id = label_encoder.transform([asset])[0]
        mask = test_df['Asset_ID'] == asset_id
        if mask.sum() < 10:
            continue
        
        y_asset = test_df.loc[mask, 'Target_Scaled'].values
        pred_asset = pred_fewshot[mask.values]
        r2 = r2_score(y_asset, pred_asset)
        fewshot_results[asset] = r2
        print(f"    {asset}: {r2:.4f}")
    
    # 8. Summary
    print("\n" + "="*70)
    print("SUMMARY: Transfer Learning Results")
    print("="*70)
    print(f"{'Condition':<20} | {'R²':>10}")
    print("-"*35)
    print(f"{'Seen (US Equity)':<20} | {r2_seen:>10.4f}")
    print(f"{'Zero-shot (Unseen)':<20} | {r2_zeroshot:>10.4f}")
    print(f"{'Few-shot (10 epochs)':<20} | {r2_fewshot:>10.4f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if r2_zeroshot > 0.2:
        print("✅ STRONG UNIVERSALITY: Zero-shot R² > 0.2 indicates backbone learned universal dynamics")
    elif r2_zeroshot > 0:
        print("⚠️ WEAK UNIVERSALITY: Positive zero-shot R², but limited transfer")
    else:
        print("❌ NO UNIVERSALITY: Negative zero-shot R², model does not transfer")
    
    if r2_fewshot > r2_zeroshot + 0.1:
        print("✅ STRONG ADAPTABILITY: Few-shot significantly improves performance")
    
    # Save
    results = {
        'r2_seen': r2_seen,
        'r2_zeroshot': r2_zeroshot,
        'r2_fewshot': r2_fewshot,
        'per_asset_zeroshot': zeroshot_results,
        'per_asset_fewshot': fewshot_results
    }
    
    out_path = 'experiments/07_v2_methodology/results/v4_exp04_zeroshot.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_zeroshot_experiment()
