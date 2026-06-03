"""
V60: Ablation Study
- V50 모델의 핵심 구성요소 제거 실험
- Ablation targets: Attention, Bi-directional, Feature subsets
"""

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
import time

warnings.filterwarnings('ignore')

# Use CPU for stability if hangs occur
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# ============================================================
# Model Variants for Ablation Study
# ============================================================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

# Full Model (V50 Original)
class FullModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        output = self.fc(context)
        return output

# Ablation 1: No Attention (use last hidden state)
class NoAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NoAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time step instead of attention
        output = self.fc(lstm_out[:, -1, :])
        return output

# Ablation 2: No Bidirectional (unidirectional LSTM)
class UnidirectionalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UnidirectionalModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        output = self.fc(context)
        return output

# Ablation 3: Simple MLP (No LSTM, No Attention)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output

# ============================================================
# Training and Evaluation
# ============================================================

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_and_eval(model, train_x, train_y, test_x, test_y, epochs=10, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_x = torch.FloatTensor(train_x).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(train_x.size(0))
        for i in range(0, train_x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_x[indices], train_y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(test_x).cpu().numpy().flatten()
        
    return r2_score(test_y, preds)

def run_ablation_study():
    print("="*80)
    print("V60: Ablation Study")
    print("="*80)
    
    # Load cached data
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if not os.path.exists(CACHE_PATH):
        print("Error: Local cache not found!")
        return
        
    raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Use subset for speed
    ASSETS = ['SPY', 'TLT']
    SEQ_LEN = 22
    HIDDEN_DIM = 64
    SEQ_LEN = 22
    HIDDEN_DIM = 64
    INPUT_DIM = 2
    MAX_SAMPLES = 10000 # 학습 속도를 위해 샘플 수 제한
    
    print(f"Assets: {ASSETS}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Hidden Dimension: {HIDDEN_DIM}")
    print()
    
    # Prepare data
    pooled_xs, pooled_ys = [], []
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        asset_df = pd.DataFrame({'LogRV': log_rv, 'Ret': ret}).dropna()
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(asset_df)
        
        x, y = create_sequences(data_scaled, SEQ_LEN)
        pooled_xs.append(x)
        pooled_ys.append(y)
        
    X = np.concatenate(pooled_xs)[:MAX_SAMPLES]
    Y = np.concatenate(pooled_ys)[:MAX_SAMPLES]
    
    # Train/Test split
    split = int(len(X) * 0.8)
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = Y[:split], Y[split:]
    
    print(f"Train samples: {len(train_x)}, Test samples: {len(test_x)}")
    print()
    
    # Run ablation experiments
    results = {}
    
    # 1. Full Model (Baseline)
    print("[1/4] Full Model (Bi-LSTM + Attention)...", flush=True)
    model_full = FullModel(INPUT_DIM, HIDDEN_DIM)
    r2_full = train_and_eval(model_full, train_x, train_y, test_x, test_y)
    results['Full Model'] = r2_full
    print(f"      R2: {r2_full:.4f}")
    
    # 2. No Attention
    print("[2/4] No Attention Model...", flush=True)
    model_no_attn = NoAttentionModel(INPUT_DIM, HIDDEN_DIM)
    r2_no_attn = train_and_eval(model_no_attn, train_x, train_y, test_x, test_y)
    results['No Attention'] = r2_no_attn
    print(f"      R2: {r2_no_attn:.4f} (Drop: {(r2_full - r2_no_attn)*100:.2f}%p)")
    
    # 3. Unidirectional LSTM
    print("[3/4] Unidirectional LSTM...", flush=True)
    model_uni = UnidirectionalModel(INPUT_DIM, HIDDEN_DIM)
    r2_uni = train_and_eval(model_uni, train_x, train_y, test_x, test_y)
    results['Unidirectional'] = r2_uni
    print(f"      R2: {r2_uni:.4f} (Drop: {(r2_full - r2_uni)*100:.2f}%p)")
    
    # 4. Simple MLP
    print("[4/4] Simple MLP (No LSTM)...", flush=True)
    model_mlp = SimpleMLP(INPUT_DIM, HIDDEN_DIM, SEQ_LEN)
    r2_mlp = train_and_eval(model_mlp, train_x, train_y, test_x, test_y)
    results['Simple MLP'] = r2_mlp
    print(f"      R2: {r2_mlp:.4f} (Drop: {(r2_full - r2_mlp)*100:.2f}%p)")
    
    # Summary
    print()
    print("="*80)
    print("Ablation Study Summary")
    print("="*80)
    print(f"{'Model':<20} {'R2 Score':<12} {'Drop (%p)':<12} {'Contribution'}")
    print("-"*60)
    for name, r2 in results.items():
        drop = (r2_full - r2) * 100 if name != 'Full Model' else 0
        contrib = f"+{drop:.2f}%p" if drop > 0 else f"{drop:.2f}%p"
        print(f"{name:<20} {r2:.4f}       {drop:>6.2f}        {contrib}")
    
    # Save results
    output_path = 'src/experiments/verification/v60_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_ablation_study()
