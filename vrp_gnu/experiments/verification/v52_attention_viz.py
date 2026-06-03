import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (Batch, Seq_Len, Hidden_Dim * 2)
        # raw_scores: (Batch, Seq_Len)
        raw_scores = self.attention(lstm_output).squeeze(-1)
        
        # Tanh activation for scores before softmax is common in additive attention
        attn_weights = torch.softmax(torch.tanh(raw_scores), dim=1)
        
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
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_experiment():
    print("="*80)
    print("V52: Attention Weight Analysis (Explainability)")
    print("="*80)
    
    # Data Prep (Pooled like V50)
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    pooled_X, pooled_y = [], []
    
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(5).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        df = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        
        X, y = create_sequences(data_scaled, SEQ_LEN)
        pooled_X.append(X)
        pooled_y.append(y)
        
    X_all = np.concatenate(pooled_X)
    y_all = np.concatenate(pooled_y)
    
    # Train is 80%, Test 20%
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]
    
    # Train Model
    model = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training V50 Model for Visualization...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in train_dl:
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out.squeeze(), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_dl):.4f}")
            
    # Extract Attention Weights from Test Set
    model.eval()
    X_test_t = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        _, attn_weights = model(X_test_t)
        
    attn_np = attn_weights.numpy() # (N_samples, SEQ_LEN)
    
    # 1. Average Attention Profile
    avg_attn = np.mean(attn_np, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(SEQ_LEN), avg_attn, color='skyblue', edgecolor='navy')
    plt.title('Average Attention Weights by Lag (Test Set)')
    plt.xlabel('Lag (Days: 0=Oldest, 21=Newest)')
    plt.ylabel('Attention Weight')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('src/experiments/verification/v52_avg_attention.png')
    plt.close()
    
    print("Saved Average Attention Plot.")
    
    # 2. Heatmap (First 100 samples of Test)
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn_np[:100, :], cmap='viridis', cbar_kws={'label': 'Attention Weight'})
    plt.title('Attention Weights Heatmap (First 100 Test Samples)')
    plt.xlabel('Lag (Days)')
    plt.ylabel('Time Step')
    plt.savefig('src/experiments/verification/v52_attn_heatmap.png')
    plt.close()
    
    print("Saved Attention Heatmap.")
    
    # 3. Save Summary stats
    res = {
        'Max_Lag': int(np.argmax(avg_attn)),
        'Max_Weight': float(np.max(avg_attn)),
        'Min_Weight': float(np.min(avg_attn))
    }
    with open('src/experiments/verification/v52_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_experiment()
