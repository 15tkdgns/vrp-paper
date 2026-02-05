import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import warnings
from itertools import product
import time

warnings.filterwarnings('ignore')

# Core Model Definition (Same as V50)
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        output = self.fc(context)
        return output

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_and_eval(train_x, train_y, test_x, test_y, hidden_dim, lr, batch_size, epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualAttentionLSTM(input_dim=2, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_x = torch.FloatTensor(train_x).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    
    print(f"    Train shape: {train_x.shape}, Device: {device}", flush=True)
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(train_x.size(0))
        epoch_loss = 0
        batches = 0
        for i in range(0, train_x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_x[indices], train_y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        # Verbose per epoch
        print(f"    Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/batches:.6f}", flush=True)
            
    model.eval()
    with torch.no_grad():
        preds = model(test_x).cpu().numpy().flatten()
        
    return r2_score(test_y, preds)

def run_sensitivity_analysis():
    print("="*80, flush=True)
    print("V54: Hyperparameter Sensitivity Analysis (Dry Run Mode)", flush=True)
    print("="*80, flush=True)
    
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if not os.path.exists(CACHE_PATH):
        print("Error: Local cache not found! Run data_prep first.")
        return
        
    raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # 1. Ultra Light Settings for Verification
    ASSETS = ['SPY'] 
    print(f"Assets: {ASSETS} (Reduced for speed)", flush=True)

    # 2. Minimal Grid
    param_grid = {
        'hidden_dim': [64],
        'seq_len': [22],
        'lr': [0.001],
        'batch_size': [128] # Larger batch for speed
    }
    
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))
    results = []
    
    print(f"Starting analysis for {len(combinations)} combinations...", flush=True)
    
    for i, combo in enumerate(combinations):
        start_time = time.time()
        params = dict(zip(keys, combo))
        print(f" [{i+1}/{len(combinations)}] Testing: {params}", flush=True)
        
        # Data Prep
        pooled_xs, pooled_ys = [], []
        for asset in ASSETS:
            price = raw[asset]
            ret = np.log(price / price.shift(1)).dropna()
            rv = (ret**2).rolling(22).mean() * 252 * 10000
            log_rv = np.log(rv + 1e-6)
            
            asset_df = pd.DataFrame({'LogRV': log_rv, 'Ret': ret}).dropna()
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(asset_df)
            
            x, y = create_sequences(data_scaled, params['seq_len'])
            pooled_xs.append(x)
            pooled_ys.append(y)
            
        X = np.concatenate(pooled_xs)
        Y = np.concatenate(pooled_ys)
        
        # Limit data size for dry run speed
        X = X[:1000] 
        Y = Y[:1000]
        print("    (Data truncated to 1000 samples for Dry Run)", flush=True)
        
        split = int(len(X) * 0.8)
        train_x, test_x = X[:split], X[split:]
        train_y, test_y = Y[:split], Y[split:]
        
        try:
            # Run for just 1 Epoch
            r2 = train_and_eval(train_x, train_y, test_x, test_y, 
                                params['hidden_dim'], params['lr'], params['batch_size'],
                                epochs=1)
            
            elapsed = time.time() - start_time
            print(f"   -> R2: {r2:.5f} (Time: {elapsed:.2f}s)", flush=True)
            
            params['r2'] = float(r2)
            results.append(params)
        except Exception as e:
            print(f"   -> Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
    # Save Results
    output_path = 'src/experiments/verification/v54_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDry Run analysis complete. Results saved to {output_path}", flush=True)

if __name__ == "__main__":
    run_sensitivity_analysis()
