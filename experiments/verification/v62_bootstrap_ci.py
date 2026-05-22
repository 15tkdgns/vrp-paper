"""
V62: Bootstrap Confidence Interval
- Calculate 95% CI for R² using bootstrap resampling
- Verify statistical robustness of performance metrics
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

def bootstrap_r2(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for R²"""
    n_samples = len(y_true)
    r2_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resampling
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        r2 = r2_score(y_true_boot, y_pred_boot)
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    
    # Calculate confidence interval
    alpha = 1 - ci
    lower = np.percentile(r2_scores, alpha/2 * 100)
    upper = np.percentile(r2_scores, (1 - alpha/2) * 100)
    
    return {
        'mean': float(np.mean(r2_scores)),
        'std': float(np.std(r2_scores)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'ci_level': ci
    }

def run_bootstrap_ci():
    print("="*80)
    print("V62: Bootstrap Confidence Interval")
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
    N_BOOTSTRAP = 1000
    
    print(f"Assets: {ASSETS}")
    print(f"Bootstrap samples: {N_BOOTSTRAP}")
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
    
    # Get predictions
    model.eval()
    test_x_tensor = torch.FloatTensor(test_x).to(device)
    with torch.no_grad():
        predictions = model(test_x_tensor).cpu().numpy().flatten()
    
    # Calculate bootstrap CI
    print(f"Calculating {N_BOOTSTRAP} bootstrap samples...", flush=True)
    ci_results = bootstrap_r2(test_y, predictions, n_bootstrap=N_BOOTSTRAP)
    
    # Results
    print()
    print("="*80)
    print("Bootstrap Confidence Interval Results")
    print("="*80)
    print(f"Point Estimate R²: {r2_score(test_y, predictions):.4f}")
    print(f"Bootstrap Mean R²: {ci_results['mean']:.4f}")
    print(f"Bootstrap Std:     {ci_results['std']:.4f}")
    print(f"95% CI:            [{ci_results['ci_lower']:.4f}, {ci_results['ci_upper']:.4f}]")
    print()
    
    # Statistical significance test
    if ci_results['ci_lower'] > 0:
        print("Result: CI does NOT include 0 -> Model is statistically significant!")
    else:
        print("Result: CI includes 0 -> Model is NOT statistically significant")
    
    # Save results
    output_path = 'src/experiments/verification/v62_results.json'
    with open(output_path, 'w') as f:
        json.dump(ci_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_bootstrap_ci()
