"""
V61: Feature Importance Analysis
- Permutation Feature Importance for V50 model
- Measures contribution of each input variable
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import warnings

warnings.filterwarnings('ignore')

# Use same model architecture as V50
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

def permutation_importance(model, X, y, n_repeats=10):
    """Calculate permutation feature importance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Baseline score
    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy().flatten()
    baseline_score = r2_score(y, baseline_preds)
    
    n_features = X.shape[2]
    importances = {}
    feature_names = ['LogRV', 'Return']
    
    for feat_idx in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Permute feature across all time steps
            np.random.shuffle(X_permuted[:, :, feat_idx])
            
            X_perm_tensor = torch.FloatTensor(X_permuted).to(device)
            with torch.no_grad():
                perm_preds = model(X_perm_tensor).cpu().numpy().flatten()
            perm_score = r2_score(y, perm_preds)
            scores.append(baseline_score - perm_score)
        
        importances[feature_names[feat_idx]] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }
    
    return baseline_score, importances

def run_feature_importance():
    print("="*80)
    print("V61: Feature Importance Analysis")
    print("="*80)
    
    # Load cached data
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if not os.path.exists(CACHE_PATH):
        print("Error: Local cache not found!")
        return
        
    raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    ASSETS = ['SPY', 'TLT']
    SEQ_LEN = 22
    HIDDEN_DIM = 64
    INPUT_DIM = 2
    
    print(f"Assets: {ASSETS}")
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
        
    X = np.concatenate(pooled_xs)
    Y = np.concatenate(pooled_ys)
    
    split = int(len(X) * 0.8)
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = Y[:split], Y[split:]
    
    # Train model
    print("Training model...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualAttentionLSTM(INPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_x_tensor = torch.FloatTensor(train_x).to(device)
    train_y_tensor = torch.FloatTensor(train_y).to(device)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(train_x_tensor).squeeze()
        loss = criterion(outputs, train_y_tensor)
        loss.backward()
        optimizer.step()
    
    # Calculate feature importance
    print("Calculating permutation importance (10 repeats)...", flush=True)
    baseline_score, importances = permutation_importance(model, test_x, test_y, n_repeats=10)
    
    # Results
    print()
    print("="*80)
    print("Feature Importance Results")
    print("="*80)
    print(f"Baseline R²: {baseline_score:.4f}")
    print()
    print(f"{'Feature':<15} {'Importance':<15} {'Std':<15} {'Contribution'}")
    print("-"*60)
    
    total_imp = sum(v['mean'] for v in importances.values())
    for feat, data in sorted(importances.items(), key=lambda x: -x[1]['mean']):
        pct = data['mean'] / total_imp * 100 if total_imp > 0 else 0
        print(f"{feat:<15} {data['mean']:.4f}         {data['std']:.4f}         {pct:.1f}%")
    
    # Save results
    results = {
        'baseline_r2': baseline_score,
        'importances': importances
    }
    output_path = 'src/experiments/verification/v61_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_feature_importance()
