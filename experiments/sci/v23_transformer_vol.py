"""
V23 Exp: Deep Learning with Temporal Attention (Transformer) (SCI Phase 5)
Purpose: To overcome fixed lags (1, 5, 22) by letting Attention Mechanism decide "Which past day matters?".
Architecture:
    - Input: Sequence of past 22 days LogRV (Batch, 22, 1).
    - Model: Single Layer Transformer Encoder + MLP Head.
    - Output: Next 22-day average LogRV.

Hypothesis: Attention can capture dynamic dependencies (e.g., FOMC dynamics) that fixed lags miss.
Limitations: Transformer needs HUGE data. We have ~3000 daily points per asset. Might overfit.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
import warnings
import random

warnings.filterwarnings('ignore')

# Configuration
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond',
    'GLD': 'Commodity'
}
ASSETS = list(ASSET_CATEGORIES.keys())
SEED = 42
SEQ_LEN = 22 # Lookback window

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class VolTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=1, dropout=0.1):
        super(VolTransformer, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, src):
        # src: [Batch, Seq, 1]
        x = self.input_proj(src) + self.pos_encoder # [Batch, Seq, d_model]
        x = self.transformer_encoder(x) # [Batch, Seq, d_model]
        x = x.flatten(start_dim=1) # [Batch, Seq * d_model]
        output = self.decoder(x)
        return output

def feature_engineering_sequence(data):
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    X_list = []
    y_list = []
    
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        price = close[asset]
        ret = np.log(price / price.shift(1))
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        # Create Sequences
        values = log_rv.values
        targets = log_rv.shift(-22).dropna().values # Target is next 22 days avg
        
        # Align lengths
        # valid inputs are up to len(values) - 22
        # valid targets are aligned.
        
        # Efficient sequence creation
        # X: [t-21, ..., t]
        # y: t+22 avg (which is the value at index t for the target series? No target is future avg)
        # Let's align carefully.
        # df index t. LogRV[t] is realized variance of past 22 days.
        # We want to predict LogRV[t+22].
        
        # Prepare
        series = pd.DataFrame({'val': log_rv})
        series['target'] = series['val'].shift(-22)
        series = series.dropna()
        
        raw_x = series['val'].values
        raw_y = series['target'].values
        
        for i in range(len(raw_x) - SEQ_LEN):
            X_list.append(raw_x[i : i+SEQ_LEN])
            y_list.append(raw_y[i+SEQ_LEN - 1]) # Target corresponding to the last step of sequence? 
            # No. At step t (end of sequence), we want target of t.
            # raw_y is already shifted. So raw_y[i+SEQ_LEN-1] corresponds to target for time at end of seq.
            
    return np.array(X_list), np.array(y_list)

def run_experiment():
    print("="*80)
    print("V23: Transformer Volatility Prediction")
    print("="*80)
    
    # 1. Data Prep
    print("\n[Step 1] Preparing Data...")
    tickers = ASSETS
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    X, y = feature_engineering_sequence(raw)
    
    # Check shapes
    print(f"Total Sequences: {X.shape}")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    # Flatten for scaling fit
    scaler.fit(X_train_raw.reshape(-1, 1))
    X_train = scaler.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, 1)).reshape(X_test_raw.shape)
    
    # Tensor conversion
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1) # [Batch, Seq, 1]
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 2. Train Transformer
    print("\n[Step 2] Training Transformer...")
    model = VolTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    # 3. Evaluate Transformer
    model.eval()
    with torch.no_grad():
        pred_trans = model(X_test_t).squeeze().numpy()
        
    r2_trans = r2_score(y_test, pred_trans)
    rmse_trans = np.sqrt(mean_squared_error(y_test, pred_trans))
    
    # 4. Compare with Baseline (HAR-Ridge)
    # Re-train HAR-Ridge on SAME data/split logic for fairness
    # HAR features from sequence: last, mean of last 5, mean of last 22
    print("\n[Step 3] Comparisons")
    
    # Construct HAR features from X matrix manually
    # X index: 0 is t-21, ..., 21 is t
    har_train = []
    for seq in X_train:
        lag1 = seq[-1]
        lag5 = np.mean(seq[-5:])
        lag22 = np.mean(seq)
        har_train.append([lag1, lag5, lag22])
        
    har_test = []
    for seq in X_test:
        lag1 = seq[-1]
        lag5 = np.mean(seq[-5:])
        lag22 = np.mean(seq)
        har_test.append([lag1, lag5, lag22])
        
    model_har = Ridge(alpha=1.0)
    model_har.fit(har_train, y_train)
    pred_har = model_har.predict(har_test)
    
    r2_har = r2_score(y_test, pred_har)
    rmse_har = np.sqrt(mean_squared_error(y_test, pred_har))
    
    print(f"HAR-Ridge   R2: {r2_har:.4f}")
    print(f"Transformer R2: {r2_trans:.4f}")
    
    improv = (r2_trans - r2_har) / abs(r2_har) * 100
    print(f"Improvement: {improv:.2f}%")
    
    out_data = {
        'har_r2': r2_har,
        'transformer_r2': r2_trans,
        'improvement': improv
    }
    
    out_path = 'src/experiments/sci/v23_transformer_vol.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
