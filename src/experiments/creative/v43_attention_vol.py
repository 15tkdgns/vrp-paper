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
SEQ_LEN = 22  # Lookback window (1 month)
HIDDEN_DIM = 64
NUM_HEADS = 4
LAYERS = 2
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001

class VolatilityTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(VolatilityTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim * SEQ_LEN, 1) # Flatten and project
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        output = self.fc_out(x)
        return output

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, 0] # Predicting next step LogRV (index 0)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_experiment():
    print("="*80, flush=True)
    print("V43: Multi-Head Self-Attention Experiment", flush=True)
    print("="*80, flush=True)
    
    # 1. Data Prep
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
    
    all_data = []
    print(f"Processing {len(ASSETS)} assets and creating sequences...", flush=True)
    
    for i, asset in enumerate(ASSETS):
        if (i+1) % 2 == 0 or i == 0:
            print(f"  [{i+1}/{len(ASSETS)}] Processing {asset}...", flush=True)
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000 
        log_rv = np.log(rv + 1e-6).dropna()
        
        df_asset = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        
        # Scale per asset
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_asset)
        m_y, s_y = scaler.mean_[0], scaler.scale_[0] # Mean and Scale for LogRV
        
        vals = data_scaled
        dates = df_asset.index
        
        for j in range(len(vals) - SEQ_LEN - 22):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j+SEQ_LEN + 22 - 1, 0]
            target_date = dates[j+SEQ_LEN + 22 - 1]
            
            # Save original y for evaluation later if needed, but for now we follow the pattern
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y,
                's_y': s_y
            })
            
    # Sort by Time for proper cross-asset temporal split
    print("Sorting all records by Date...", flush=True)
    df_all = pd.DataFrame(all_data).sort_values('Date')
    
    # Time-based Split
    print("Splitting data (80/20)...", flush=True)
    split = int(len(df_all) * 0.8)
    train_slice = df_all.iloc[:split]
    test_slice = df_all.iloc[split:]
    
    X_train = np.stack(train_slice['X'].values)
    y_train = np.stack(train_slice['y_scaled'].values)
    X_test = np.stack(test_slice['X'].values)
    y_test = np.stack(test_slice['y_scaled'].values)
    
    # Convert to Tensor
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 2. Model Init
    model = VolatilityTransformer(input_dim=2, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=LAYERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training Transformer for {EPOCHS} epochs...", flush=True)
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
            
    # 3. Evaluation
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
    print(f"Transformer (V43) R2 (Original Scale): {r2:.5f}", flush=True)
    
    # Save Results
    res = {'V43_R2': float(r2)}
    with open('src/experiments/creative/v43_results.json', 'w') as f:
        json.dump(res, f)
        
    # Save Predictions for Ensemble
    preds_df = pd.DataFrame({
        'Preds': preds_orig,
        'Asset': test_slice['Asset'].values
    }, index=test_slice['Date'])
    preds_df.to_csv('src/experiments/creative/v43_preds.csv')
    print("Saved V43 predictions to src/experiments/creative/v43_preds.csv", flush=True)

if __name__ == "__main__":
    run_experiment()
