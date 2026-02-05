import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
import json
import os
import sys

# Add src to path just in case, though executing as module handles it usually
sys.path.append(os.getcwd())

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion="MSE"):
    """
    Diebold-Mariano test for predictive accuracy.
    H0: Two forecasts have the same accuracy.
    H1: Two forecasts have different accuracy.
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    T = len(y_true)
    
    if criterion == "MSE":
        e1 = (y_true - y_pred1)**2
        e2 = (y_true - y_pred2)**2
    elif criterion == "MAE":
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)
        
    d = e1 - e2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(d_var / T)
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def load_predictions(model_name):
    # This assumes previous scripts saved full predictions.
    # CAUTION: Previous scripts (V29, V43, V50) mainly saved JSON summaries or generic csvs.
    # We need to ensure we have aligned predictions.
    # V51 (Walk-Forward) saved JSON summary but printed R2. 
    # V35/V36/V43/V50 scripts in 'creative' folder did NOT explicitly save full prediction CSVs in their main block,
    # except V42 which tried to load them.
    
    # To do this rigorously without re-running everything:
    # We will use the V51 (Walk-Forward) logic to generate V50 predictions again? No, too slow.
    # We will re-run V29 (Fast) and V50 (Fast inference if model saved? No model saved).
    # We will Re-Run V29 and V50 on the SAME test split for consistency.
    pass

# Since we need aligned arrays for DM test, and we didn't save them in CSVs previously:
# We will implement a lightweight "Re-Run and Compare" in this script.
# We will compare V29 (Baseline) vs V50 (Champion).

import torch
import torch.nn as nn
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64

# Define V50 Model Class again
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
    print("V53: Diebold-Mariano Test (V29 vs V50)")
    print("="*80)
    
    # Data Prep
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Prepare Aligned Data
    pooled_X_seq = []
    pooled_y_seq = []
    pooled_X_flat = [] # For V29
    
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(5).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        # 1. Sequence Data for V50
        df = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        sc = StandardScaler()
        data_scaled = sc.fit_transform(df)
        X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN)
        
        # 2. Flat Data for V29 (Aligned indices)
        # Sequence input X[i] uses data from i to i+22. Target is at i+22.
        # V29 features need to be at i+22 (lag1, lag5 from that perspective?)
        # Actually, V29 uses lags FROM the prediction point.
        # If we predict y_{t+1}, we use info up to t.
        # Sequence data: Input=[t-21...t], Target=t+1.
        # V29 features at t: LogRV_lag1 (t), LogRV_lag5 (mean t-4..t), etc.
        
        # Reconstruct V29 features aligned with sequences
        # X_seq[j] ends at index j+SEQ_LEN-1. corresp date is df.index[j+SEQ_LEN-1].
        # Target is at j+SEQ_LEN.
        # So we need features at j+SEQ_LEN-1.
        
        # Let's extract features from dataframe directly
        lags = df['LogRV'] # shifted?
        # Construct V29 DataFrame
        d_v29 = pd.DataFrame({
            'LogRV_lag1': df['LogRV'].shift(1),
            'LogRV_lag5': df['LogRV'].shift(5),
            'LogRV_lag22': df['LogRV'].shift(22),
            'Target': df['LogRV'] # This is y_t
        })
        # We need to align with y_seq. 
        # y_seq corresponds to d_v29['Target'] at specific indices.
        # create_sequences cuts off start.
        
        # To ensure perfect alignment: use the exact same targets.
        valid_indices = range(SEQ_LEN, len(df)-1) # corresponding to y_seq logic?
        # y_seq length is len(data) - SEQ_LEN - 1
        # y_seq[0] is data[SEQ_LEN]
        
        v29_feats = d_v29.iloc[SEQ_LEN:-1][['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']]
        v29_targets = d_v29.iloc[SEQ_LEN:-1]['Target']
        
        if len(v29_feats) != len(y_seq):
             # Adjustment
             min_len = min(len(v29_feats), len(y_seq))
             v29_feats = v29_feats.iloc[:min_len]
             y_seq = y_seq[:min_len]
             X_seq = X_seq[:min_len]
             
        pooled_X_flat.append(v29_feats.values)
        pooled_X_seq.append(X_seq)
        pooled_y_seq.append(y_seq)
        
    X_flat_all = np.concatenate(pooled_X_flat)
    X_seq_all = np.concatenate(pooled_X_seq)
    y_all = np.concatenate(pooled_y_seq)
    
    # Split
    split = int(len(y_all) * 0.8)
    
    # --- Train/Test V29 ---
    X_train_flat = X_flat_all[:split]
    X_test_flat = X_flat_all[split:]
    y_train = y_all[:split]
    y_test = y_all[split:]
    
    sc_v29 = StandardScaler()
    X_train_flat_sc = sc_v29.fit_transform(X_train_flat)
    X_test_flat_sc = sc_v29.transform(X_test_flat)
    
    m_v29 = Ridge()
    m_v29.fit(X_train_flat_sc, y_train)
    pred_v29 = m_v29.predict(X_test_flat_sc)
    
    # --- Train/Test V50 ---
    X_train_seq = X_seq_all[:split]
    X_test_seq = X_seq_all[split:]
    
    X_train_t = torch.FloatTensor(X_train_seq)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_seq)
    
    model = DualAttentionLSTM(2, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    
    print("Training V50 for DM Test...")
    model.train()
    for e in range(5): # Fast retrain
        for bx, by in dl:
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        pred_v50 = model(X_test_t).squeeze().numpy()
        
    # --- Diebold-Mariano Test ---
    print("\n[Comparison]")
    print(f"V29 R2: {r2_score(y_test, pred_v29):.5f}")
    print(f"V50 R2: {r2_score(y_test, pred_v50):.5f}")
    
    dm_stat, p_val = diebold_mariano_test(y_test, pred_v29, pred_v50)
    
    print(f"\n[DM Test Results]")
    print(f"DM Statistic: {dm_stat:.5f}")
    print(f"P-Value: {p_val:.10f}")
    
    sig = "Significant" if p_val < 0.05 else "Not Significant"
    print(f"Conclusion: V50 improvement over V29 is {sig} (alpha=0.05)")
    
    res = {
        'V29_R2': r2_score(y_test, pred_v29),
        'V50_R2': r2_score(y_test, pred_v50),
        'DM_Stat': dm_stat,
        'P_Value': p_val,
        'Significant': p_val < 0.05
    }
    
    with open('src/experiments/verification/v53_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    run_experiment()
