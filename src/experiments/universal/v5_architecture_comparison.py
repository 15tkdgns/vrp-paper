"""
V5 Model Architecture Improvements
===================================
Sequential testing of model improvements:
1. Level 1: Residual + LR Scheduler
2. Level 2: LSTM Sequence Model  
3. Level 3: Cross-Asset Attention
4. Level 4: Transformer Encoder

Baseline: V4 Universal MLP (R² = 0.85 log-space, 0.43 per-asset avg)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV',
    'EFA', 'EEM',
    'TLT', 'IEF', 'TIP',
    'GLD', 'USO', 'SLV', 'DBC',
]

SEQ_LEN = 22  # Lookback window for sequence models

# ============================================================
# MODEL DEFINITIONS
# ============================================================

# Baseline: V4 Universal MLP
class BaselineMLP(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=4, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_features + embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, asset_ids, x_seq=None):
        embed = self.asset_embedding(asset_ids)
        x = torch.cat([x, embed], dim=1)
        return self.net(x).squeeze()

# Level 1: Residual MLP
class ResidualMLP(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        
        self.input_proj = nn.Linear(n_features + embed_dim, hidden_dim)
        
        # Residual Blocks
        self.res1 = self._make_residual_block(hidden_dim, dropout)
        self.res2 = self._make_residual_block(hidden_dim, dropout)
        self.res3 = self._make_residual_block(hidden_dim, dropout)
        
        self.output = nn.Linear(hidden_dim, 1)
        
    def _make_residual_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        
    def forward(self, x, asset_ids, x_seq=None):
        embed = self.asset_embedding(asset_ids)
        x = torch.cat([x, embed], dim=1)
        x = self.input_proj(x)
        
        # Residual connections
        x = F.relu(x + self.res1(x))
        x = F.relu(x + self.res2(x))
        x = F.relu(x + self.res3(x))
        
        return self.output(x).squeeze()

# Level 2: LSTM Sequence Model
class UniversalLSTM(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, asset_ids, x_seq):
        # x_seq: (batch, seq_len, features)
        embed = self.asset_embedding(asset_ids)
        _, (h_n, _) = self.lstm(x_seq)
        # Concat forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.fc(torch.cat([h, embed], dim=1))
        return out.squeeze()

# Level 3: Cross-Asset Attention
class CrossAssetAttentionModel(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.n_assets = n_assets
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        
        self.feature_proj = nn.Linear(n_features, hidden_dim)
        
        # Cross-asset attention
        self.query = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, asset_ids, x_seq=None, all_asset_states=None):
        embed = self.asset_embedding(asset_ids)
        x_proj = self.feature_proj(x)
        x_combined = torch.cat([x_proj, embed], dim=1)
        
        # Self representation
        q = self.query(x_combined)
        
        # For simplicity, use self-attention on batch (each sample attends to others)
        # In production, would want proper cross-asset attention with all assets
        k = self.key(x_combined)
        v = self.value(x_combined)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) / np.sqrt(k.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v.unsqueeze(1)).squeeze(1)
        
        out = self.output(torch.cat([x_proj, context], dim=1))
        return out.squeeze()

# Level 4: Transformer Encoder
class UniversalTransformer(nn.Module):
    def __init__(self, n_features, n_assets, embed_dim=8, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, d_model)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x, asset_ids, x_seq):
        # x_seq: (batch, seq_len, features)
        batch_size = x_seq.size(0)
        
        # Asset token
        asset_token = self.asset_embedding(asset_ids).unsqueeze(1)  # (B, 1, d_model)
        
        # Project sequence
        x_proj = self.input_proj(x_seq)  # (B, seq_len, d_model)
        
        # Concat: [ASSET_TOKEN, seq...]
        x = torch.cat([asset_token, x_proj], dim=1)  # (B, seq_len+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # Use [ASSET] token and last token for prediction
        out = self.output(torch.cat([x[:, 0], x[:, -1]], dim=1))
        return out.squeeze()


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data_with_sequences():
    print("Downloading data...")
    raw = yf.download(ASSETS + ['^VIX'], start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    label_encoder = LabelEncoder()
    label_encoder.fit(ASSETS)
    
    # Prepare data with sequences
    all_data = []
    feature_cols = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom', 'LogVIX']
    
    for asset in ASSETS:
        if asset not in close.columns:
            continue
            
        df = pd.DataFrame(index=close.index)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        # Features
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        df['Asset'] = asset
        
        # For sequence: store raw LogRV for each timestep
        for lag in range(SEQ_LEN):
            df[f'LogRV_t{lag}'] = df['LogRV'].shift(lag)
            df[f'LogVIX_t{lag}'] = df['LogVIX'].shift(lag - 1)  # VIX lagged
        
        df = df.dropna()
        if len(df) > 100:
            all_data.append(df)
    
    full_df = pd.concat(all_data).reset_index(drop=True)
    
    # Normalize
    scaler_X = StandardScaler()
    full_df[feature_cols] = scaler_X.fit_transform(full_df[feature_cols])
    
    scaler_y = StandardScaler()
    full_df['Target_Scaled'] = scaler_y.fit_transform(full_df[['Target']])
    
    return full_df, feature_cols, scaler_y, label_encoder


def get_sequence_features(df, idx, seq_len=SEQ_LEN):
    """Extract sequence features for LSTM/Transformer"""
    seq_cols = [f'LogRV_t{i}' for i in range(seq_len)]
    vix_cols = [f'LogVIX_t{i}' for i in range(seq_len)]
    
    seq = df.loc[idx, seq_cols].values.reshape(-1, seq_len, 1)
    vix_seq = df.loc[idx, vix_cols].values.reshape(-1, seq_len, 1)
    
    # Combine: (batch, seq_len, 2)
    return np.concatenate([seq, vix_seq], axis=2).astype(np.float32)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate(model, train_df, test_df, feature_cols, 
                       model_name, use_sequence=False, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_train = torch.tensor(train_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_df['Target_Scaled'].values, dtype=torch.float32).to(device)
    a_test = torch.tensor(test_df['Asset_ID'].values, dtype=torch.long).to(device)
    
    # Sequence data
    if use_sequence:
        seq_train = torch.tensor(get_sequence_features(train_df, train_df.index), dtype=torch.float32).to(device)
        seq_test = torch.tensor(get_sequence_features(test_df, test_df.index), dtype=torch.float32).to(device)
    else:
        seq_train = seq_test = None
    
    # Optimizer with scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    batch_size = 256
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_df))[:min(10000, len(train_df))]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            
            optimizer.zero_grad()
            
            if use_sequence:
                pred = model(X_train[idx], a_train[idx], seq_train[idx])
            else:
                pred = model(X_train[idx], a_train[idx])
            
            loss = criterion(pred, y_train[idx])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        if use_sequence:
            pred_test = model(X_test, a_test, seq_test).cpu().numpy()
        else:
            pred_test = model(X_test, a_test).cpu().numpy()
    
    r2_overall = r2_score(y_test.cpu().numpy(), pred_test)
    
    # Per-asset R2
    test_df = test_df.copy()
    test_df['Pred'] = pred_test
    
    asset_r2s = {}
    for asset in ASSETS:
        mask = test_df['Asset'] == asset
        if mask.sum() < 50:
            continue
        r2 = r2_score(test_df.loc[mask, 'Target_Scaled'], test_df.loc[mask, 'Pred'])
        asset_r2s[asset] = r2
    
    avg_asset_r2 = np.mean(list(asset_r2s.values()))
    
    return r2_overall, avg_asset_r2, asset_r2s


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_architecture_comparison():
    print("="*70)
    print("V5: Model Architecture Comparison")
    print("="*70)
    
    # Prepare data
    full_df, feature_cols, scaler_y, label_encoder = prepare_data_with_sequences()
    
    # Split
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    
    n_features = len(feature_cols)
    n_assets = len(ASSETS)
    
    # Models to compare
    models = [
        ("Baseline MLP", BaselineMLP(n_features, n_assets), False),
        ("Level 1: Residual MLP", ResidualMLP(n_features, n_assets), False),
        ("Level 2: LSTM", UniversalLSTM(2, n_assets), True),  # 2 features in sequence
        ("Level 3: Cross-Attention", CrossAssetAttentionModel(n_features, n_assets), False),
        ("Level 4: Transformer", UniversalTransformer(2, n_assets), True),
    ]
    
    results = {}
    
    for name, model, use_seq in models:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        r2_overall, avg_r2, asset_r2s = train_and_evaluate(
            model, train_df, test_df, feature_cols, 
            name, use_sequence=use_seq, epochs=50
        )
        
        results[name] = {
            'r2_overall': r2_overall,
            'avg_asset_r2': avg_r2,
            'per_asset': asset_r2s
        }
        
        print(f"\n  Overall R²: {r2_overall:.4f}")
        print(f"  Avg Asset R²: {avg_r2:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Architecture Comparison")
    print("="*70)
    print(f"{'Model':<30} | {'Overall R²':>12} | {'Avg Asset R²':>12}")
    print("-"*60)
    
    for name, res in results.items():
        print(f"{name:<30} | {res['r2_overall']:>12.4f} | {res['avg_asset_r2']:>12.4f}")
    
    # Best model
    best = max(results.items(), key=lambda x: x[1]['avg_asset_r2'])
    print(f"\n★ Best Model: {best[0]} (Avg Asset R² = {best[1]['avg_asset_r2']:.4f})")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v5_architecture_comparison.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    
    return results

if __name__ == "__main__":
    run_architecture_comparison()
