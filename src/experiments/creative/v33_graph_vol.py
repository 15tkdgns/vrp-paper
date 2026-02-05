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
from src.models.graph_vol import AdaptiveGraphVol

warnings.filterwarnings('ignore')

# --- Configuration ---
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 60
BATCH_SIZE = 32 # Smaller batch because each sample is larger
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading (Graph Dataset) ---
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

class GraphVolatilityDataset(Dataset):
    def __init__(self, data, seq_len=60, scaler=None, is_train=True):
        # We need all assets aligned by date
        asset_list = data.columns.tolist()
        
        all_feats = [] # List of (T, 2) arrays per asset
        all_targs = [] # List of (T,) arrays per asset
        
        for asset in asset_list:
            price = data[asset]
            ret = np.log(price / price.shift(1)).fillna(0).values
            s_ret = pd.Series(ret)
            rv = s_ret.rolling(22).mean() * 252 * 10000
            log_rv = np.log(rv + 1e-6).fillna(0).values
            forward_vol = pd.Series(log_rv).shift(-22).fillna(0).values
            
            all_feats.append(np.stack([ret, log_rv], axis=1))
            all_targs.append(forward_vol)
            
        # Fit Scaler on all data combined (shared scaling for simplicity)
        if is_train:
            self.scaler = StandardScaler()
            combined_feats = np.concatenate([f[22:] for f in all_feats], axis=0)
            self.scaler.fit(combined_feats)
        else:
            self.scaler = scaler
            
        # Transform and Scale
        scaled_feats = [self.scaler.transform(f) for f in all_feats]
        
        # Scale Target using LogRV scaler
        mean_rv = self.scaler.mean_[1]
        scale_rv = self.scaler.scale_[1]
        scaled_targs = [(t - mean_rv) / scale_rv for t in all_targs]
        
        # Create Multi-Asset Samples
        T = len(data)
        self.X = []
        self.y = []
        
        for i in range(seq_len + 22, T - 22):
            # One sample contains all nodes
            node_feats = []
            node_targs = []
            for n in range(len(asset_list)):
                node_feats.append(scaled_feats[n][i-seq_len:i].flatten())
                node_targs.append(scaled_targs[n][i])
            
            self.X.append(np.stack(node_feats)) # (N, 120)
            self.y.append(np.array(node_targs)) # (N,)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Training Loop ---
def train_graph_model(model, train_loader, test_loader):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X) # (B, N)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        all_preds = []
        all_actuals = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                p = model(X)
                all_preds.append(p.cpu())
                all_actuals.append(y.cpu())
        
        preds = torch.cat(all_preds).numpy().flatten()
        actuals = torch.cat(all_actuals).numpy().flatten()
        val_r2 = r2_score(actuals, preds)
        
        print(f"[Graph] Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Test R2 {val_r2:.4f}")
        
    return val_r2, time.time() - start_time

def main():
    print(f"Using Device: {DEVICE}")
    close_df = get_data(ASSETS)
    if close_df.empty: return

    split_idx = int(len(close_df) * 0.8)
    train_df = close_df.iloc[:split_idx]
    test_df = close_df.iloc[split_idx:]
    
    print("Building Graph Datasets...")
    train_ds = GraphVolatilityDataset(train_df, SEQ_LEN, is_train=True)
    test_ds = GraphVolatilityDataset(test_df, SEQ_LEN, scaler=train_ds.scaler, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model: Node 6, In: 120, Hidden: 64
    model = AdaptiveGraphVol(num_nodes=len(ASSETS), in_channels=120, hidden_channels=64).to(DEVICE)
    
    print("\n--- Training Graph-Vol ---")
    r2, duration = train_graph_model(model, train_loader, test_loader)
    
    print(f"\nFinal Graph R2: {r2:.4f}, Time: {duration:.2f}s")
    
    res = {'GraphVol': {'R2': r2, 'Time': duration}}
    with open('src/experiments/creative/v33_graph_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
