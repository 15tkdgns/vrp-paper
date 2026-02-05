"""
V9 Exp: Neural Network Architecture Comparison
Purpose: Compare various neural network architectures for volatility prediction.

Models:
1. MLP (Multi-Layer Perceptron) - Baseline NN
2. Deep MLP - Deeper architecture with residual connections
3. LSTM - DeepVol (2022) inspired
4. Transformer - Michael et al. (2025) inspired
5. TCN (Temporal Convolutional Network) - Alternative to LSTM
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Asset Categories
ASSET_CATEGORIES = {
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity', 'DIA': 'Equity', 'MDY': 'Equity',
    'XLF': 'Equity', 'XLE': 'Equity', 'XLK': 'Equity', 'XLV': 'Equity', 'XLI': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'IOO': 'Equity',
    'TLT': 'Bond', 'IEF': 'Bond', 'SHY': 'Bond', 'TIP': 'Bond', 'ZROZ': 'Bond',
    'GLD': 'Commodity', 'USO': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity',
}
ASSETS = list(ASSET_CATEGORIES.keys())

# ==================== MODEL DEFINITIONS ====================

class SimpleMLP(nn.Module):
    """Basic MLP"""
    def __init__(self, n_features, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class DeepMLP(nn.Module):
    """Deeper MLP with Residual Connections"""
    def __init__(self, n_features, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(3)
        ])
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)  # Residual
        return self.output(x).squeeze(-1)

class LSTMModel(nn.Module):
    """LSTM for sequence modeling - DeepVol (2022) inspired"""
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len dim
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)

class TransformerModel(nn.Module):
    """Transformer Encoder - Michael et al. (2025) inspired"""
    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len dim
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(-1)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, n_features, hidden_dim=64, kernel_size=3, n_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_features if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, features, 1)
        x = x.transpose(1, 2)  # (batch, features, seq) -> (batch, seq, features)
        if x.size(2) == 1:
            x = x.repeat(1, 1, 3)  # Minimum sequence length for conv
        x = x.transpose(1, 2)  # Back to (batch, features, seq)
        x = self.conv(x)
        return self.fc(x[:, :, -1]).squeeze(-1)

# ==================== TRAINING FUNCTION ====================

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=256, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_r2 = -float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test.to(device)).cpu().numpy()
            r2 = r2_score(y_test.numpy(), pred_test)
            
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_r2

# ==================== MAIN ====================

def run_nn_comparison():
    print("="*70)
    print("V9 Experiment: Neural Network Architecture Comparison")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Download Data
    tickers = ASSETS + ['^VIX']
    print("Downloading data...")
    raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    pooled_data = []
    
    print("\nEngineering Features...")
    for asset in ASSETS:
        if asset not in close.columns: continue
        
        df = pd.DataFrame(index=close.index)
        price = close[asset]
        ret = np.log(price / price.shift(1))
        
        rv_daily = ret**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        
        ret_neg = ret.where(ret < 0, 0)
        ret_pos = ret.where(ret > 0, 0)
        rv_down = (ret_neg**2).rolling(22).mean() * 252 * 10000
        rv_up = (ret_pos**2).rolling(22).mean() * 252 * 10000
        
        df['LogRV_Down'] = np.log(rv_down + 1e-6).shift(1)
        df['LogRV_Up'] = np.log(rv_up + 1e-6).shift(1)
        
        vix = close['VIX']
        df['LogVIX'] = np.log(vix + 1e-6).shift(1)
        df['VRP_Proxy'] = df['LogVIX'] - df['LogRV_lag1']
        
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        
        df['Asset'] = asset
        df = df.dropna()
        if len(df) < 500: continue
        
        pooled_data.append(df)
        
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    features = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Down', 'LogRV_Up', 'VRP_Proxy']
    n_features = len(features)
    
    print(f"Total Samples: {len(full_df)}")
    
    # Split
    indices = np.arange(len(full_df))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(full_df) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_df = full_df.iloc[train_idx]
    test_df = full_df.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(train_df[features]), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(test_df[features]), dtype=torch.float32)
    y_train = torch.tensor(train_df['Target'].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Target'].values, dtype=torch.float32)
    
    results = {}
    
    # ==================== TEST MODELS ====================
    print("\n" + "="*50)
    print("NEURAL NETWORK MODELS")
    print("="*50)
    
    # 1. Simple MLP
    print("1. Training Simple MLP...")
    model_mlp = SimpleMLP(n_features)
    r2_mlp = train_model(model_mlp, X_train, y_train, X_test, y_test, device=device)
    results['Simple MLP'] = r2_mlp
    print(f"   R² = {r2_mlp:.4f}")
    
    # 2. Deep MLP with Residual
    print("2. Training Deep MLP (Residual)...")
    model_deep = DeepMLP(n_features)
    r2_deep = train_model(model_deep, X_train, y_train, X_test, y_test, device=device)
    results['Deep MLP (Residual)'] = r2_deep
    print(f"   R² = {r2_deep:.4f}")
    
    # 3. LSTM
    print("3. Training LSTM (DeepVol)...")
    model_lstm = LSTMModel(n_features)
    r2_lstm = train_model(model_lstm, X_train, y_train, X_test, y_test, device=device)
    results['LSTM (DeepVol)'] = r2_lstm
    print(f"   R² = {r2_lstm:.4f}")
    
    # 4. Transformer
    print("4. Training Transformer...")
    model_tf = TransformerModel(n_features)
    r2_tf = train_model(model_tf, X_train, y_train, X_test, y_test, device=device)
    results['Transformer'] = r2_tf
    print(f"   R² = {r2_tf:.4f}")
    
    # 5. TCN
    print("5. Training TCN...")
    model_tcn = TCN(n_features)
    r2_tcn = train_model(model_tcn, X_train, y_train, X_test, y_test, device=device)
    results['TCN'] = r2_tcn
    print(f"   R² = {r2_tcn:.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("SUMMARY: Neural Network Comparison")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Rank':<5} {'Model':<25} {'R²':<10}")
    print("-"*45)
    for i, (model, r2) in enumerate(sorted_results, 1):
        print(f"{i:<5} {model:<25} {r2:.4f}")
    
    # Save Results
    out_data = {
        'nn_comparison': results,
        'best_model': sorted_results[0][0],
        'best_r2': sorted_results[0][1]
    }
    
    out_path = 'src/experiments/universal/v9_nn_comparison.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_nn_comparison()
