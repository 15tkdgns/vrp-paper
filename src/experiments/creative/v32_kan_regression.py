import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import time
import os
import json
import warnings
from src.models.kan import KAN

warnings.filterwarnings('ignore')

# --- Configuration ---
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading (Reuse logic from v31) ---
def get_data(tickers):
    print(f"Downloading data for {tickers}...")
    try:
        raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close']
        else:
            close = raw
        close.columns = [c.replace('^', '') for c in close.columns]
        close = close.ffill()
        return close
    except Exception as e:
        print(f"Data download failed: {e}")
        return pd.DataFrame()

class VolatilityDataset(Dataset):
    def __init__(self, data, seq_len=60, scaler=None, is_train=True):
        samples = []
        targets = []
        asset_data = []
        all_features = []
        
        for asset in data.columns:
            price = data[asset]
            ret = np.log(price / price.shift(1)).fillna(0).values
            s_ret = pd.Series(ret)
            rv = s_ret.rolling(22).mean() * 252 * 10000
            log_rv = np.log(rv + 1e-6).fillna(0).values
            forward_vol = pd.Series(log_rv).shift(-22).fillna(0).values
            
            features = np.stack([ret, log_rv], axis=1)
            
            asset_data.append({
                'features': features,
                'target': forward_vol
            })
            
            if is_train:
                all_features.append(features[22:])
                
        if is_train:
            self.scaler = StandardScaler()
            full_feat = np.concatenate(all_features, axis=0)
            self.scaler.fit(full_feat)
        else:
            self.scaler = scaler
            
        for item in asset_data:
            feats = item['features']
            targs = item['target']
            feats_scaled = self.scaler.transform(feats)
            
            mean_rv = self.scaler.mean_[1]
            scale_rv = self.scaler.scale_[1]
            targs_scaled = (targs - mean_rv) / scale_rv
            
            valid_len = len(feats) - 22
            for i in range(seq_len + 22, valid_len):
                x = feats_scaled[i-seq_len:i].flatten() # (SeqLen * 2) = 120
                y = targs_scaled[i]
                samples.append(x)
                targets.append(y)
                
        self.X = torch.tensor(np.array(samples), dtype=torch.float32)
        self.y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Training Loop ---
def train_model(model, train_loader, test_loader, name):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Eval
        model.eval()
        preds = []
        actuals = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                p = model(X)
                preds.append(p.cpu())
                actuals.append(y.cpu())
        
        preds = torch.cat(preds).numpy()
        actuals = torch.cat(actuals).numpy()
        
        if np.isnan(preds).any():
            print(f"[{name}] Epoch {epoch+1}: NaNs detected!")
            return -999, 0
            
        val_r2 = r2_score(actuals, preds)
        print(f"[{name}] Epoch {epoch+1}: Loss {avg_loss:.4f}, Test R2 {val_r2:.4f}")
        
    train_time = time.time() - start_time
    return val_r2, train_time

def main():
    print(f"Using Device: {DEVICE}")
    close_df = get_data(ASSETS)
    if close_df.empty: return

    split_idx = int(len(close_df) * 0.8)
    train_df = close_df.iloc[:split_idx]
    test_df = close_df.iloc[split_idx:]
    
    print("Building Datasets...")
    train_ds = VolatilityDataset(train_df, SEQ_LEN, is_train=True)
    test_ds = VolatilityDataset(test_df, SEQ_LEN, scaler=train_ds.scaler, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # KAN Model
    # Input: 120, Hidden: [64, 32], Out: 1
    print("\n--- Training KAN Regression ---")
    model = KAN([120, 64, 32, 1]).to(DEVICE)
    kan_r2, kan_time = train_model(model, train_loader, test_loader, "KAN")
    
    print("\n=== KAN Experiment Results ===")
    print(f"KAN: R2={kan_r2:.4f}, Time={kan_time:.2f}s")
    
    res = {'KAN': {'R2': kan_r2, 'Time': kan_time}}
    with open('src/experiments/creative/v32_kan_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
