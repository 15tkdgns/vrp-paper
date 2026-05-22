import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional
        
    def forward(self, lstm_output):
        # lstm_output: (Batch, Seq_Len, Hidden_Dim * 2)
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1) # (Batch, Seq_Len)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1) # (Batch, Hidden_Dim * 2)
        return context, attn_weights

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x) # (Batch, Seq_Len, Hidden*2)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        
        if return_attention:
            return output, attn_weights
        return output

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_experiment():
    print("="*80, flush=True)
    print("V50: Dual Attention LSTM Experiment", flush=True)
    print("="*80, flush=True)
    
    # Data Loading
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from local cache: {CACHE_PATH}...", end="", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(" Done.", flush=True)
    else:
        print("Local cache not found! Downloading from yfinance...", end="", flush=True)
        raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
        print(" Done.", flush=True)
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Collect data with dates
    all_data = []
    print(f"Processing {len(ASSETS)} assets and creating sequences...", flush=True)
    
    for i, asset in enumerate(ASSETS):
        if (i+1) % 2 == 0 or i == 0:
            print(f"  [{i+1}/{len(ASSETS)}] Processing {asset}...", flush=True)
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        # Standard Horizon: 22-day rolling mean
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        df_asset = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_asset)
        m_y, s_y = scaler.mean_[0], scaler.scale_[0]
        
        vals = data_scaled
        dates = df_asset.index
        
        # Target: 22 days ahead
        for j in range(len(vals) - SEQ_LEN - 22):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j+SEQ_LEN + 22 - 1, 0] # Target is LogRV at t+22
            target_date = dates[j+SEQ_LEN + 22 - 1]
            
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x, # Array (22, 2)
                'y_scaled': y_scaled,
                'm_y': m_y,
                's_y': s_y
            })
            
    # Sort by Time
    print("Sorting all records by Date...", flush=True)
    df_all = pd.DataFrame(all_data).sort_values('Date')
    
    # Split
    print("Splitting data (80/20)...", flush=True)
    split = int(len(df_all) * 0.8)
    train_slice = df_all.iloc[:split]
    test_slice = df_all.iloc[split:]
    
    # Prepare Tensors
    X_train = np.stack(train_slice['X'].values)
    y_train = np.stack(train_slice['y_scaled'].values)
    X_test = np.stack(test_slice['X'].values)
    y_test = np.stack(test_slice['y_scaled'].values)
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Model Training
    model = DualAttentionLSTM(input_dim=2, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training Dual Attention LSTM (Time-Sorted) for {EPOCHS} epochs...", flush=True)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}", flush=True)
            
    # Evaluation
    print("Evaluating on test set...", flush=True)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).squeeze().numpy()
        
    # Inverse Transform
    m_y = test_slice['m_y'].values
    s_y = test_slice['s_y'].values
    preds_orig = preds_scaled * s_y + m_y
    y_test_orig = y_test * s_y + m_y
    
    r2 = r2_score(y_test_orig, preds_orig)
    print(f"\n[Results]", flush=True)
    print(f"Dual Attn LSTM (V50) R2 (Original Scale): {r2:.5f}", flush=True)
    
    res = {'V50_R2': float(r2)}
    with open('src/experiments/creative/v50_results.json', 'w') as f:
        json.dump(res, f)
        
    # Save Preds with Index (Date, Asset)
    preds_df = pd.DataFrame({
        'Preds': preds_orig,
        'Asset': test_slice['Asset'].values
    }, index=test_slice['Date'])
    preds_df.to_csv('src/experiments/creative/v50_preds.csv')
    print("Saved V50 predictions to src/experiments/creative/v50_preds.csv", flush=True)

if __name__ == "__main__":
    run_experiment()
