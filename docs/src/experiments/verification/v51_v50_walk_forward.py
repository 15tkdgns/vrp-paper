import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 10 # Reduced for WF speed, assuming incremental learning or sufficient data
BATCH_SIZE = 64
LR = 0.001
REFIT_STEP = 22 # Refit every 22 days (1 month)

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

def run_experiment():
    print("="*80)
    print("V51: Walk-Forward Validation (V50 Model)")
    print("="*80)
    
    # Data Prep
    raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # We will perform WF on pooled data logic to mimic the main experiment, 
    # but strictly separating train/test in time.
    # Actually, main experiment pooled assets. 
    # To keep it manageable, let's do Asset-Specific WF or Pooled WF?
    # Pooled WF is better for general model.
    # Time index alignment is crucial.
    
    # 1. Create a common time index and pool data per day? 
    # Easier: Just collect all sequences sorted by time.
    
    pooled_records = []
    
    print("Preprocessing & Pooling Data...")
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(5).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        # DataFrame with Date index
        df = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        
        # Scaling (fit on initial period? or rolling scale?)
        # For strict OOS, scaler should effectively be refit or use expanding window.
        # But global scaling is often acceptable if 'mean/std' don't drift massively.
        # Let's use Expanding Window Scaling or just Fixed Initial Scaling (riskier).
        # Best: Fit Scaler on Training portion at each step.
        
        # Store raw values for now
        for date, row in df.iterrows():
            pooled_records.append({
                'Date': date,
                'Asset': asset,
                'LogRV': row['LogRV'],
                'Ret': row['Ret']
            })
            
    df_all = pd.DataFrame(pooled_records).sort_values('Date')
    dates = df_all['Date'].unique()
    
    # Split Time: Train (2010-2018), Test (2019-2025)
    split_date = pd.Timestamp('2019-01-01')
    test_dates = [d for d in dates if d >= split_date]
    
    # We will step through test_dates with stride REFIT_STEP
    # But wait, pooling mixes assets. 
    # Better approach for WF: 
    # Train on all data < T. Predict for all assets in [T, T+Step].
    
    predictions = []
    truths = []
    
    # Initial Training Window
    current_date = split_date
    end_date = dates[-1]
    
    print(f"Starting Walk-Forward Loop... ({len(test_dates)} days to cover)")
    step_count = 0
    
    while current_date < end_date:
        next_date = current_date + pd.Timedelta(days=REFIT_STEP * 1.5) # search buffer
        # Find index in unique dates
        try:
            curr_idx = np.where(dates == current_date)[0][0]
            # Next refit date (approx)
            target_next_idx = min(len(dates)-1, curr_idx + REFIT_STEP)
            next_limit_date = dates[target_next_idx]
        except:
             # Fallback if specific date not found match
             # Just increment by fixed days
             next_limit_date = current_date + pd.Timedelta(days=30)
             if next_limit_date > end_date: next_limit_date = end_date

        print(f"Step {step_count}: Train up to {current_date.date()}, Predict -> {next_limit_date.date()}")
        
        # 1. Prepare Train Data (Expanding Window)
        train_mask = df_all['Date'] < current_date
        train_data = df_all[train_mask]
        
        # Quick Scaler (Fit on Train)
        # Note: We need to scale sequences. 
        # Efficient way: Group by asset, create sequences.
        # But creating sequences every loop is slow. 
        # Optimization: Pre-create all sequences, just filter by index?
        # Yes.
        
        # Let's do a simplified approach: 
        # Retrain model every loop? Yes.
        # To speed up, we can load state_dict from previous step? (Warm start) -> YES.
        
        pass # Actual logic below
        
        # Logic is getting complex for a single script file without pre-calc.
        # Let's implement correctly:
        current_date = next_limit_date # Advance
        step_count += 1
        if step_count > 5: break # SAFETY BREAK for dry run validation
        
    print("Dry Run Complete. Generating hypothetical output for verification logic check.")
    # Actually, I should write the REAL logic.
    
    # REAL LOGIC START
    # 1. Pre-process ALL data into sequences (Date, Asset, X, y)
    all_seqs = []
    
    # Asset-wise scaling and sequencing
    scaler_map = {} # Fit on 2010-2018 base
    
    # Train scaler on base period
    base_mask = df_all['Date'] < split_date
    base_df = df_all[base_mask]
    
    # Global Scaler? Or Asset-specific?
    # V50 used asset-specific.
    for asset in ASSETS:
        asset_df = df_all[df_all['Asset'] == asset].set_index('Date')
        
        # Fit scaler on pre-2019 data
        train_part = asset_df[asset_df.index < split_date]
        sc = StandardScaler()
        if len(train_part) > 22:
            sc.fit(train_part[['LogRV', 'Ret']])
        else:
            sc.fit(asset_df[['LogRV', 'Ret']]) # Fallback
            
        # Transform ALL data
        data_scaled = sc.transform(asset_df[['LogRV', 'Ret']])
        
        # Create sequences
        Xs, ys = create_sequences(data_scaled, SEQ_LEN)
        
        # We need dates for these sequences.
        # Sequence ends at index i+seq_len. Target is at i+seq_len.
        # Date of target is asset_df.index[i+seq_len]
        target_dates = asset_df.index[SEQ_LEN+1:] # Adjust for potential alignment
        # create_sequences len is (N - seq_len - 1)
        # indices: 0..(N-seq_len-2)
        # target index in data: seq_len .. (N-1)
        
        valid_dates = asset_df.index[SEQ_LEN:-1] # Wait, let's be precise
        # create_sequences loop: i from 0 to len-seq-1
        # y is data[i+seq_len]. Date is index[i+seq_len]
        # range is roughly correct.
        
        # Store sequences with date
        for j in range(len(Xs)):
            t_date = asset_df.index[j + SEQ_LEN]
            all_seqs.append({
                'Date': t_date,
                'X': Xs[j],
                'y': ys[j]
            })
            
    seq_df = pd.DataFrame(all_seqs).sort_values('Date')
    
    # Now Walk-Forward Loop
    model = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Dates for WF
    unique_dates = seq_df['Date'].unique()
    test_start_idx = np.searchsorted(unique_dates, split_date)
    
    preds_all = []
    actuals_all = []
    
    # Iterate through test period in chunks
    curr_idx = test_start_idx
    
    while curr_idx < len(unique_dates):
        # 1. Define Train/Test window
        # Train: All history up to curr_idx
        # Test: curr_idx to curr_idx + REFIT_STEP
        
        curr_date = unique_dates[curr_idx]
        next_idx = min(len(unique_dates), curr_idx + REFIT_STEP)
        
        print(f"Training on data < {curr_date} (approx {curr_idx} trading days)...")
        
        # Prepare Train Tensor
        train_slice = seq_df[seq_df['Date'] < curr_date]
        # To save memory/time, maybe limit lookback? (e.g., last 5 years)
        # For now, full history.
        
        X_train = np.stack(train_slice['X'].values)
        y_train = np.stack(train_slice['y'].values)
        
        train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train (Warm start from previous state)
        model.train()
        for epoch in range(3): # Short refit epochs
            for bx, by in train_dl:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out.squeeze(), by)
                loss.backward()
                optimizer.step()
                
        # Prepare Test Tensor (Next window)
        if curr_idx >= len(unique_dates): break
        
        test_end_date = unique_dates[next_idx-1]
        test_slice = seq_df[(seq_df['Date'] >= curr_date) & (seq_df['Date'] <= test_end_date)]
        
        if len(test_slice) == 0: break
        
        X_test = np.stack(test_slice['X'].values)
        y_test = np.stack(test_slice['y'].values)
        X_test_t = torch.from_numpy(X_test).float()
        
        model.eval()
        with torch.no_grad():
            p = model(X_test_t).squeeze().numpy()
            
        preds_all.extend(p)
        actuals_all.extend(y_test)
        
        print(f"Predicted {len(p)} samples. Moving window.")
        curr_idx = next_idx
        
    # Calculate OOS R2
    r2 = r2_score(actuals_all, preds_all)
    print(f"\n[Walk-Forward Results]")
    print(f"OOS R2: {r2:.5f}")
    
    res = {'WF_R2': float(r2), 'Samples': len(preds_all)}
    with open('src/experiments/verification/v51_results.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_experiment()
