import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import t
import json
import os

# --- Model Definitions ---

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
    def forward(self, lstm_output):
        attn_weights = torch.softmax(torch.tanh(self.attention(lstm_output)).squeeze(-1), dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.fc(context)

def har_features(series):
    # HAR(1, 5, 22)
    day = series.shift(1)
    week = series.shift(1).rolling(5).mean()
    month = series.shift(1).rolling(22).mean()
    return pd.concat([day, week, month], axis=1).dropna()

def diebold_mariano_test(target, p1, p2, h=1):
    e1 = (target - p1)**2
    e2 = (target - p2)**2
    d = e1 - e2
    d_bar = np.mean(d)
    
    # Autocovariance for h-step ahead (h=1 for daily)
    def autocov(x, k):
        n = len(x)
        x_mean = np.mean(x)
        return np.sum((x[:n-k] - x_mean) * (x[k:] - x_mean)) / n

    gamma_0 = np.var(d)
    var_d = gamma_0
    for k in range(1, h):
        var_d += 2 * autocov(d, k)
    
    n = len(d)
    # Harvey et al. (1997) correction for small samples & h-step
    harvey_adj = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat = d_bar / (np.sqrt(var_d / n)) * harvey_adj
    p_val = 2 * (1 - t.cdf(np.abs(dm_stat), df=n-1))
    
    return dm_stat, p_val

from sklearn.preprocessing import StandardScaler

def run_v57():
    print("Running V57: Refined DM Test & MCS...")
    
    # Load data
    df = pd.read_csv('src/data/ohlcv_cache.csv', index_col=0)
    spy = df['SPY'].pct_change().dropna()
    log_rv = np.log(spy.rolling(22).std() * np.sqrt(252)).dropna()
    
    # Alignment
    features = har_features(log_rv)
    target = log_rv.loc[features.index]
    
    # Scaling for LSTM
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    feat_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    # Train/Test Split
    split_idx = int(len(target) * 0.8)
    X_train_s, X_test_s = feat_scaled[:split_idx], feat_scaled[split_idx:]
    y_train_s, y_test_s = target_scaled[:split_idx], target_scaled[split_idx:]
    
    # V29: HAR-Ridge (on original scale for fair comparison with history)
    har_model = Ridge(alpha=1.0)
    har_model.fit(features.iloc[:split_idx], target.iloc[:split_idx])
    
    # V50: LSTM-Attention
    seq_len = 22
    def to_seq(data, target_vals, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i+seq_len])
            ys.append(target_vals[i+seq_len])
        return np.array(xs), np.array(ys)
    
    X_seq, y_seq = to_seq(feat_scaled, target_scaled, seq_len)
    s_split = int(len(X_seq) * 0.8)
    
    X_tr_t = torch.FloatTensor(X_seq[:s_split])
    y_tr_t = torch.FloatTensor(y_seq[:s_split]).view(-1, 1)
    X_te_t = torch.FloatTensor(X_seq[s_split:])
    
    model = DualAttentionLSTM(input_dim=3, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
        
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_te_t).squeeze().numpy()
        pred_v50 = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        y_test_aligned = target.values[seq_len + s_split:]
        pred_v29_aligned = har_model.predict(features.values[seq_len + s_split:])
        
    # DM Test
    dm_stat, p_val = diebold_mariano_test(y_test_aligned, pred_v29_aligned, pred_v50)
    
    results = {
        'V29_RMSE': float(np.sqrt(mean_squared_error(y_test_aligned, pred_v29_aligned))),
        'V50_RMSE': float(np.sqrt(mean_squared_error(y_test_aligned, pred_v50))),
        'DM_Stat': float(dm_stat),
        'P_Value': float(p_val),
        'Significant': bool(p_val < 0.05)
    }
    
    os.makedirs('src/experiments/verification', exist_ok=True)
    with open('src/experiments/verification/v57_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved. DM P-Value: {p_val:.6f}")

if __name__ == "__main__":
    run_v57()
