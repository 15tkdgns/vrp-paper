import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001

# Model Definitions (필수: v50과 동일한 구조)
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
        
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        
        if return_attention:
            return output, attn_weights
        return output

def run_analysis():
    print("="*80)
    print("V63: Advanced Attention Analysis")
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
    
    # Prepare Data by Asset
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
        
        # Pre-calculate market regime features (much faster)
        ret_aligned = ret.reindex(df_asset.index)
        recent_ret_rolling = ret_aligned.rolling(5).mean()
        recent_vol_rolling = ret_aligned.rolling(22).std() * np.sqrt(252)
        
        # Create sequences with metadata
        for j in range(len(vals) - SEQ_LEN - 22):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j+SEQ_LEN + 22 - 1, 0]
            target_date = dates[j+SEQ_LEN + 22 - 1]
            
            # Use pre-calculated values
            recent_ret = recent_ret_rolling.iloc[j+SEQ_LEN-1]
            recent_vol = recent_vol_rolling.iloc[j+SEQ_LEN-1]
            
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y,
                's_y': s_y,
                'recent_ret': recent_ret if not pd.isna(recent_ret) else 0.0,
                'recent_vol': recent_vol if not pd.isna(recent_vol) else 0.2
            })
    
    # Sort by date
    df_all = pd.DataFrame(all_data).sort_values('Date')
    
    # Split data
    split = int(len(df_all) * 0.8)
    train_slice = df_all.iloc[:split]
    test_slice = df_all.iloc[split:].reset_index(drop=True)
    
    # Prepare tensors
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
    
    # Train Model
    print(f"Training model for {EPOCHS} epochs...", flush=True)
    model = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
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
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}", flush=True)
    
    # Extract Attention Weights
    print("\nExtracting attention weights from test set...", flush=True)
    model.eval()
    
    with torch.no_grad():
        preds, attn_weights = model(X_test_tensor, return_attention=True)
    
    preds = preds.squeeze().numpy()
    attn_weights = attn_weights.numpy()  # Shape: (N_samples, SEQ_LEN)
    
    # Add predictions and attention to test_slice
    test_slice['pred_scaled'] = preds
    
    # Inverse transform
    m_y = test_slice['m_y'].values
    s_y = test_slice['s_y'].values
    test_slice['pred_orig'] = preds * s_y + m_y
    test_slice['y_orig'] = y_test * s_y + m_y
    test_slice['error'] = np.abs(test_slice['pred_orig'] - test_slice['y_orig'])
    
    # ========================================
    # Analysis 1: Asset-wise Attention
    # ========================================
    print("\n1. Analyzing attention by asset...", flush=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(ASSETS):
        asset_mask = test_slice['Asset'] == asset
        asset_attn = attn_weights[asset_mask]
        avg_attn = np.mean(asset_attn, axis=0)
        
        axes[idx].bar(range(SEQ_LEN), avg_attn, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[idx].set_title(f'{asset} - Avg Attention', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Lag (0=oldest, 21=newest)')
        axes[idx].set_ylabel('Weight')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, max(avg_attn) * 1.2])
    
    plt.tight_layout()
    plt.savefig('src/experiments/verification/v63_asset_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v63_asset_comparison.png")
    
    # ========================================
    # Analysis 2: Market Regime Analysis
    # ========================================
    print("\n2. Analyzing attention by market regime...", flush=True)
    
    # Define regimes
    vol_median = test_slice['recent_vol'].median()
    ret_median = test_slice['recent_ret'].median()
    
    high_vol_mask = test_slice['recent_vol'] > vol_median
    low_vol_mask = test_slice['recent_vol'] <= vol_median
    bull_mask = test_slice['recent_ret'] > ret_median
    bear_mask = test_slice['recent_ret'] <= ret_median
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # High Vol
    avg_attn_high_vol = np.mean(attn_weights[high_vol_mask], axis=0)
    axes[0, 0].bar(range(SEQ_LEN), avg_attn_high_vol, color='red', alpha=0.6, edgecolor='darkred')
    axes[0, 0].set_title('High Volatility Period', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Attention Weight')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Low Vol
    avg_attn_low_vol = np.mean(attn_weights[low_vol_mask], axis=0)
    axes[0, 1].bar(range(SEQ_LEN), avg_attn_low_vol, color='green', alpha=0.6, edgecolor='darkgreen')
    axes[0, 1].set_title('Low Volatility Period', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Attention Weight')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Bull
    avg_attn_bull = np.mean(attn_weights[bull_mask], axis=0)
    axes[1, 0].bar(range(SEQ_LEN), avg_attn_bull, color='blue', alpha=0.6, edgecolor='darkblue')
    axes[1, 0].set_title('Bull Market (Positive Recent Return)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Lag (0=oldest, 21=newest)')
    axes[1, 0].set_ylabel('Attention Weight')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Bear
    avg_attn_bear = np.mean(attn_weights[bear_mask], axis=0)
    axes[1, 1].bar(range(SEQ_LEN), avg_attn_bear, color='orange', alpha=0.6, edgecolor='darkorange')
    axes[1, 1].set_title('Bear Market (Negative Recent Return)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lag (0=oldest, 21=newest)')
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/experiments/verification/v63_market_regime.png', dpi=150)
    plt.close()
    print("  Saved: v63_market_regime.png")
    
    # ========================================
    # Analysis 3: Lag Importance
    # ========================================
    print("\n3. Analyzing lag importance...", flush=True)
    
    avg_attn_all = np.mean(attn_weights, axis=0)
    
    # Split into early vs recent lags
    early_lags = avg_attn_all[:11]  # 0-10
    recent_lags = avg_attn_all[11:]  # 11-21
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall lag profile
    axes[0].plot(range(SEQ_LEN), avg_attn_all, marker='o', color='navy', linewidth=2)
    axes[0].axvline(x=10.5, color='red', linestyle='--', label='Early/Recent Split')
    axes[0].set_title('Overall Attention Profile', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag (0=oldest, 21=newest)')
    axes[0].set_ylabel('Average Attention Weight')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Early vs Recent
    labels = ['Early Lags (0-10)', 'Recent Lags (11-21)']
    values = [np.sum(early_lags), np.sum(recent_lags)]
    colors = ['skyblue', 'coral']
    
    axes[1].bar(labels, values, color=colors, edgecolor='black', width=0.6)
    axes[1].set_title('Cumulative Attention: Early vs Recent', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Cumulative Attention Weight')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.01, f'{v:.3f}\n({v/sum(values)*100:.1f}%)', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('src/experiments/verification/v63_lag_importance.png', dpi=150)
    plt.close()
    print("  Saved: v63_lag_importance.png")
    
    # ========================================
    # Analysis 4: Performance vs Attention
    # ========================================
    print("\n4. Analyzing relationship between attention and performance...", flush=True)
    
    # Calculate attention entropy for each sample
    def entropy(p):
        """Calculate Shannon entropy"""
        p = p + 1e-10  # Avoid log(0)
        return -np.sum(p * np.log(p))
    
    attention_entropy = np.array([entropy(attn) for attn in attn_weights])
    test_slice['attn_entropy'] = attention_entropy
    
    # Split by performance (top 25% vs bottom 25% by error)
    error_q25 = test_slice['error'].quantile(0.25)
    error_q75 = test_slice['error'].quantile(0.75)
    
    good_pred_mask = test_slice['error'] <= error_q25
    bad_pred_mask = test_slice['error'] >= error_q75
    
    avg_attn_good = np.mean(attn_weights[good_pred_mask], axis=0)
    avg_attn_bad = np.mean(attn_weights[bad_pred_mask], axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Good vs Bad predictions
    x_pos = np.arange(SEQ_LEN)
    width = 0.35
    
    axes[0].bar(x_pos - width/2, avg_attn_good, width, label='Good Predictions (Top 25%)', 
                color='green', alpha=0.7, edgecolor='darkgreen')
    axes[0].bar(x_pos + width/2, avg_attn_bad, width, label='Bad Predictions (Bottom 25%)', 
                color='red', alpha=0.7, edgecolor='darkred')
    axes[0].set_title('Attention Pattern by Prediction Quality', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Attention Weight')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Entropy vs Error scatter
    axes[1].scatter(test_slice['attn_entropy'], test_slice['error'], 
                    alpha=0.3, s=10, color='purple')
    axes[1].set_title('Attention Entropy vs Prediction Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Attention Entropy')
    axes[1].set_ylabel('Prediction Error (Absolute)')
    axes[1].grid(alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(test_slice['attn_entropy'], test_slice['error'])[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                 transform=axes[1].transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('src/experiments/verification/v63_performance_attention.png', dpi=150)
    plt.close()
    print("  Saved: v63_performance_attention.png")
    
    # ========================================
    # Save Results
    # ========================================
    print("\n5. Saving analysis results...", flush=True)
    
    # Summary statistics
    results = {
        'overall_stats': {
            'max_lag': int(np.argmax(avg_attn_all)),
            'max_weight': float(np.max(avg_attn_all)),
            'early_lags_total': float(np.sum(early_lags)),
            'recent_lags_total': float(np.sum(recent_lags)),
            'avg_entropy': float(np.mean(attention_entropy))
        },
        'asset_stats': {},
        'regime_stats': {
            'high_vol_focus_lag': int(np.argmax(avg_attn_high_vol)),
            'low_vol_focus_lag': int(np.argmax(avg_attn_low_vol)),
            'bull_focus_lag': int(np.argmax(avg_attn_bull)),
            'bear_focus_lag': int(np.argmax(avg_attn_bear))
        },
        'performance_correlation': {
            'entropy_error_corr': float(corr)
        }
    }
    
    # Asset-specific stats
    for asset in ASSETS:
        asset_mask = test_slice['Asset'] == asset
        asset_attn = attn_weights[asset_mask]
        avg_attn = np.mean(asset_attn, axis=0)
        
        results['asset_stats'][asset] = {
            'focus_lag': int(np.argmax(avg_attn)),
            'max_weight': float(np.max(avg_attn)),
            'avg_entropy': float(np.mean([entropy(a) for a in asset_attn]))
        }
    
    # Save JSON
    with open('src/experiments/verification/v63_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: v63_results.json")
    
    # Save attention weights to CSV
    attn_df = pd.DataFrame(
        attn_weights, 
        columns=[f'lag_{i}' for i in range(SEQ_LEN)]
    )
    attn_df['Asset'] = test_slice['Asset'].values
    attn_df['Date'] = test_slice['Date'].values
    attn_df['Error'] = test_slice['error'].values
    attn_df['Entropy'] = attention_entropy
    
    attn_df.to_csv('src/experiments/verification/v63_attention_maps.csv', index=False)
    print("  Saved: v63_attention_maps.csv")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - v63_asset_comparison.png")
    print("  - v63_market_regime.png")
    print("  - v63_lag_importance.png")
    print("  - v63_performance_attention.png")
    print("  - v63_results.json")
    print("  - v63_attention_maps.csv")

if __name__ == "__main__":
    run_analysis()
