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
from src.models.minimal_mamba import MinimalMambaBlock

warnings.filterwarnings('ignore')

# --- Configuration ---
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.0005 # Lower LR slightly
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading (Reuse logic) ---
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
        
        # Prepare data first to fit scaler
        all_features = []
        all_targets = []
        
        # Intermediate storage
        asset_data = [] # List of (features, target) per asset
        
        for asset in data.columns:
            price = data[asset]
            ret = np.log(price / price.shift(1)).fillna(0).values
            
            s_ret = pd.Series(ret)
            rv = s_ret.rolling(22).mean() * 252 * 10000
            log_rv = np.log(rv + 1e-6).fillna(0).values
            
            forward_vol = pd.Series(log_rv).shift(-22).fillna(0).values
            
            # (T, 2)
            features = np.stack([ret, log_rv], axis=1)
            
            # Store valid parts
            valid_mask = (features[:, 1] != 0) & (features[:, 1] > -10) # Filter outliers/zeros
            # More robust check: LogRV should be reliable. 
            # -10 corresponds to RV ~ e^-10 ~= 4e-5, which is tiny but possible.
            # Just drop initial NaN/Zero zone.
            
            asset_data.append({
                'features': features,
                'target': forward_vol,
                'valid_start': 22 # Skip initial rolling window
            })
            
            if is_train:
                # Collect for scaler fitting
                # Skip first 22
                all_features.append(features[22:])
                
        # Fit or Load Scaler
        if is_train:
            self.scaler = StandardScaler()
            full_feat = np.concatenate(all_features, axis=0)
            self.scaler.fit(full_feat)
        else:
            self.scaler = scaler
            
        # Create Windows with Scaled Data
        for item in asset_data:
            feats = item['features']
            targs = item['target']
            
            # Transform
            feats_scaled = self.scaler.transform(feats)
            # Target is LogRV, which is feature index 1. 
            # We should probably scale target too?
            # Ideally predict Scaled Target, then inverse transform.
            # But let's just predict Raw LogRV to compare with v29?
            # v29 predicted Raw LogRV? No, v29 used Ridge on Scaled Features but target was LogRV.
            # Deep Learning works better with scaled target.
            # Let's Scale Target using the LogRV scaler (index 1).
            
            mean_rv = self.scaler.mean_[1]
            scale_rv = self.scaler.scale_[1]
            targs_scaled = (targs - mean_rv) / scale_rv
            
            valid_len = len(feats) - 22
            for i in range(seq_len + 22, valid_len):
                x = feats_scaled[i-seq_len:i] # (Seq, 2)
                y = targs_scaled[i]         # Scalar (Scaled)
                
                samples.append(x)
                targets.append(y)
                
        self.X = torch.tensor(np.array(samples), dtype=torch.float32)
        self.y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Models ---
class MambaVol(nn.Module):
    def __init__(self, input_dim=2, d_model=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MinimalMambaBlock(d_model=d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1) 
        
    def forward(self, x):
        x = self.embedding(x) 
        for layer in self.layers:
            x = x + layer(x) # Residual
        x = self.norm(x)
        out = x[:, -1, :] 
        return self.head(out)

class LSTMVol(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)

# --- Training Loop ---
def train_model(model, train_loader, test_loader, name):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
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
            
            # Clip Gradients
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
        
        # Check NaNs
        if np.isnan(preds).any():
            print(f"[{name}] Epoch {epoch+1}: Predictions contain NaNs!")
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
    
    print("Building Datasets (Normalizing)...")
    train_ds = VolatilityDataset(train_df, SEQ_LEN, is_train=True)
    test_ds = VolatilityDataset(test_df, SEQ_LEN, scaler=train_ds.scaler, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train/Test Samples: {len(train_ds)} / {len(test_ds)}")
    
    # 1. Mamba
    print("\n--- Training Mamba ---")
    mamba = MambaVol().to(DEVICE)
    mamba_r2, mamba_time = train_model(mamba, train_loader, test_loader, "Mamba")
    
    # 2. LSTM
    print("\n--- Training LSTM ---")
    lstm = LSTMVol().to(DEVICE)
    lstm_r2, lstm_time = train_model(lstm, train_loader, test_loader, "LSTM")
    
    print("\n=== Experiment Results (Scaled R2) ===")
    print(f"Mamba: R2={mamba_r2:.4f}, Time={mamba_time:.2f}s")
    print(f"LSTM : R2={lstm_r2:.4f}, Time={lstm_time:.2f}s")
    
    res = {
        'Mamba': {'R2': mamba_r2, 'Time': mamba_time},
        'LSTM': {'R2': lstm_r2, 'Time': lstm_time}
    }
    with open('src/experiments/creative/v31_mamba_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
