import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os

# --- Model & Helper Reuse ---
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
    day = series.shift(1)
    week = series.shift(1).rolling(5).mean()
    month = series.shift(1).rolling(22).mean()
    return pd.concat([day, week, month], axis=1).dropna()

def run_v59():
    print("Running V59: Extreme Market Condition Testing...")
    
    df = pd.read_csv('src/data/ohlcv_cache.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    spy = df['SPY'].pct_change().dropna()
    log_rv = np.log(spy.rolling(22).std() * np.sqrt(252)).dropna()
    
    features = har_features(log_rv)
    target = log_rv.loc[features.index]
    
    # Scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    feat_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    # Intervals for testing
    regimes = {
        'Financial_Crisis_2008': ('2008-01-01', '2009-12-31'),
        'Normal_2012_2019': ('2012-01-01', '2019-12-31'),
        'COVID_2020': ('2020-01-01', '2020-12-31'),
        'Inflation_2022': ('2022-01-01', '2022-12-31')
    }
    
    # Train on "Normal" (prior to 2018 for conservative OOD)
    train_mask = (target.index < '2018-01-01')
    X_train_s = feat_scaled[train_mask]
    y_train_s = target_scaled[train_mask]
    
    seq_len = 22
    def to_seq(data, target_vals, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i+seq_len])
            ys.append(target_vals[i+seq_len])
        return np.array(xs), np.array(ys)
    
    X_seq, y_seq = to_seq(feat_scaled, target_scaled, seq_len)
    # Re-apply training mask for sequences
    train_mask_seq = (target.index[seq_len:] < '2018-01-01')
    X_tr_t = torch.FloatTensor(X_seq[train_mask_seq])
    y_tr_t = torch.FloatTensor(y_seq[train_mask_seq]).view(-1, 1)
    
    # Train Model
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
        
    model.eval()
    results = {}
    
    full_target_dates = target.index[seq_len:]
    
    with torch.no_grad():
        for name, (start, end) in regimes.items():
            regime_mask = (full_target_dates >= start) & (full_target_dates <= end)
            if not regime_mask.any():
                continue
                
            X_reg_t = torch.FloatTensor(X_seq[regime_mask])
            y_reg_true = target.values[seq_len:][regime_mask]
            
            pred_scaled = model(X_reg_t).squeeze().numpy()
            pred_inv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_reg_true, pred_inv))
            r2 = r2_score(y_reg_true, pred_inv)
            
            results[name] = {
                'RMSE': float(rmse),
                'R2': float(r2)
            }
            print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")
            
    with open('src/experiments/verification/v59_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_v59()
