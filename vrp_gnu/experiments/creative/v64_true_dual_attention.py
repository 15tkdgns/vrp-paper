import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001

# ==========================================
# Model Definitions
# ==========================================

class Attention(nn.Module):
    """Temporal Attention mechanism"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (Batch, Seq_Len, Hidden_Dim * 2)
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class TrueDualAttentionLSTM(nn.Module):
    """
    True Dual Attention LSTM with:
    1. Temporal Attention: 시간 축 (22 lags)에서 중요한 시점 선택
    2. Feature Attention: 특성 축 (LogRV, Return)에서 중요한 변수 선택
    """
    def __init__(self, input_dim, hidden_dim):
        super(TrueDualAttentionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Temporal attention (기존)
        self.temporal_attention = Attention(hidden_dim)
        
        # Feature attention (신규)
        # Context vector에서 각 input feature의 중요도를 학습
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        # x: (Batch, Seq_Len, Input_Dim)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (Batch, Seq_Len, Hidden*2)
        
        # Temporal attention
        context, temporal_attn = self.temporal_attention(lstm_out)  # context: (Batch, Hidden*2)
        
        # Feature attention
        feature_weights = self.feature_attention(context)  # (Batch, Input_Dim)
        
        # Final prediction
        output = self.fc(context)
        
        if return_attention:
            return output, temporal_attn, feature_weights
        return output


class DualAttentionLSTM(nn.Module):
    """Original model (only temporal attention) for comparison"""
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        
        if return_attention:
            return output, attn_weights
        return output


# ==========================================
# Training and Evaluation
# ==========================================

def run_experiment():
    print("="*80)
    print("V64: True Dual Attention LSTM (Temporal + Feature Attention)")
    print("="*80)
    
    # Data Loading
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from cache: {CACHE_PATH}...", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(" Done.", flush=True)
    else:
        print("Downloading from yfinance...", flush=True)
        raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
        print(" Done.", flush=True)
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # Prepare Data
    print("Processing assets and creating sequences...", flush=True)
    all_data = []
    
    for i, asset in enumerate(ASSETS):
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        df_asset = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_asset)
        
        m_y, s_y = scaler.mean_[0], scaler.scale_[0]
        vals = data_scaled
        dates = df_asset.index
        
        for j in range(len(vals) - SEQ_LEN - 22):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j+SEQ_LEN + 22 - 1, 0]
            target_date = dates[j+SEQ_LEN + 22 - 1]
            
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y,
                's_y': s_y
            })
    
    # Sort and split
    df_all = pd.DataFrame(all_data).sort_values('Date')
    split = int(len(df_all) * 0.8)
    train_slice = df_all.iloc[:split]
    test_slice = df_all.iloc[split:].reset_index(drop=True)
    
    X_train = np.stack(train_slice['X'].values)
    y_train = np.stack(train_slice['y_scaled'].values)
    X_test = np.stack(test_slice['X'].values)
    y_test = np.stack(test_slice['y_scaled'].values)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    X_test_tensor = torch.FloatTensor(X_test)
    
    # ==========================================
    # Train Original Model (v50)
    # ==========================================
    print("\n" + "-"*80)
    print("Training Original DualAttentionLSTM (only temporal attention)...")
    print("-"*80)
    
    model_v50 = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer_v50 = optim.Adam(model_v50.parameters(), lr=LR)
    
    model_v50.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer_v50.zero_grad()
            outputs = model_v50(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer_v50.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}", flush=True)
    
    # Evaluate v50
    model_v50.eval()
    with torch.no_grad():
        preds_v50 = model_v50(X_test_tensor).squeeze().numpy()
    
    m_y = test_slice['m_y'].values
    s_y = test_slice['s_y'].values
    preds_v50_orig = preds_v50 * s_y + m_y
    y_test_orig = y_test * s_y + m_y
    r2_v50 = r2_score(y_test_orig, preds_v50_orig)
    
    print(f"\nOriginal Model R²: {r2_v50:.5f}")
    
    # ==========================================
    # Train True Dual Attention Model (v64)
    # ==========================================
    print("\n" + "-"*80)
    print("Training True Dual Attention LSTM (temporal + feature attention)...")
    print("-"*80)
    
    model_v64 = TrueDualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    optimizer_v64 = optim.Adam(model_v64.parameters(), lr=LR)
    
    model_v64.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer_v64.zero_grad()
            outputs = model_v64(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer_v64.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}", flush=True)
    
    # Evaluate v64
    model_v64.eval()
    with torch.no_grad():
        preds_v64, temporal_attn, feature_attn = model_v64(X_test_tensor, return_attention=True)
    
    preds_v64 = preds_v64.squeeze().numpy()
    temporal_attn = temporal_attn.numpy()
    feature_attn = feature_attn.numpy()
    
    preds_v64_orig = preds_v64 * s_y + m_y
    r2_v64 = r2_score(y_test_orig, preds_v64_orig)
    
    print(f"\nTrue Dual Attention Model R²: {r2_v64:.5f}")
    print(f"Improvement: {(r2_v64 - r2_v50):.5f} ({(r2_v64/r2_v50 - 1)*100:.2f}%)")
    
    # ==========================================
    # Analyze Feature Attention
    # ==========================================
    print("\n" + "="*80)
    print("Feature Attention Analysis")
    print("="*80)
    
    # Average feature importance
    avg_feature_attn = feature_attn.mean(axis=0)
    print(f"\nAverage Feature Weights:")
    print(f"  LogRV: {avg_feature_attn[0]:.4f}")
    print(f"  Return: {avg_feature_attn[1]:.4f}")
    
    # Feature attention by asset
    feature_by_asset = {}
    for asset in test_slice['Asset'].unique():
        mask = test_slice['Asset'] == asset
        avg_attn = feature_attn[mask].mean(axis=0)
        feature_by_asset[asset] = {'LogRV': float(avg_attn[0]), 'Return': float(avg_attn[1])}
        print(f"\n{asset}:")
        print(f"  LogRV: {avg_attn[0]:.4f}, Return: {avg_attn[1]:.4f}")
    
    # ==========================================
    # Visualization
    # ==========================================
    print("\nCreating visualizations...")
    
    # 1. Feature importance comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(test_slice['Asset'].unique()):
        mask = test_slice['Asset'] == asset
        avg_feat = feature_attn[mask].mean(axis=0)
        
        axes[idx].bar(['LogRV', 'Return'], avg_feat, color=['skyblue', 'coral'], edgecolor='black')
        axes[idx].set_title(f'{asset} - Feature Importance', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Attention Weight')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(avg_feat):
            axes[idx].text(i, v + 0.02, f'{v:.3f}\n({v*100:.1f}%)', 
                          ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('src/experiments/creative/v64_feature_importance_by_asset.png', dpi=150)
    plt.close()
    print("  Saved: v64_feature_importance_by_asset.png")
    
    # 2. Model comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    models = ['v50\n(Temporal Only)', 'v64\n(Dual Attention)']
    r2_scores = [r2_v50, r2_v64]
    colors = ['skyblue', 'green']
    
    bars = ax.bar(models, r2_scores, color=colors, edgecolor='black', width=0.5)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_ylim([0, max(r2_scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('src/experiments/creative/v64_model_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v64_model_comparison.png")
    
    # 3. Overall feature importance
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.bar(['LogRV', 'Return'], avg_feature_attn, 
           color=['#3498db', '#e74c3c'], edgecolor='black', width=0.6)
    ax.set_title('Overall Feature Importance (Average)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attention Weight')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_feature_attn):
        ax.text(i, v + 0.02, f'{v:.4f}\n({v*100:.1f}%)', 
                ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('src/experiments/creative/v64_overall_feature_importance.png', dpi=150)
    plt.close()
    print("  Saved: v64_overall_feature_importance.png")
    
    # ==========================================
    # Save Results
    # ==========================================
    results = {
        'model_comparison': {
            'v50_r2': float(r2_v50),
            'v64_r2': float(r2_v64),
            'improvement': float(r2_v64 - r2_v50),
            'improvement_pct': float((r2_v64/r2_v50 - 1) * 100)
        },
        'overall_feature_importance': {
            'LogRV': float(avg_feature_attn[0]),
            'Return': float(avg_feature_attn[1])
        },
        'feature_importance_by_asset': feature_by_asset
    }
    
    with open('src/experiments/creative/v64_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: v64_results.json")
    
    # Save feature attention data
    feature_df = pd.DataFrame(feature_attn, columns=['LogRV_Weight', 'Return_Weight'])
    feature_df['Asset'] = test_slice['Asset'].values
    feature_df['Date'] = test_slice['Date'].values
    feature_df.to_csv('src/experiments/creative/v64_feature_attention.csv', index=False)
    print("  Saved: v64_feature_attention.csv")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - True Dual Attention improves R² by {(r2_v64/r2_v50 - 1)*100:.2f}%")
    print(f"  - LogRV is {avg_feature_attn[0]/avg_feature_attn[1]:.2f}x more important than Return")

if __name__ == "__main__":
    run_experiment()
