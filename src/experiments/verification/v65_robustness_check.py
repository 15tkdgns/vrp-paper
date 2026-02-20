import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================

ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
SEEDS = [42, 123, 456, 789, 2024]  # 5 different random seeds

# ==========================================
# Model Definitions
# ==========================================

class Attention(nn.Module):
    """Temporal Attention mechanism"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class DualAttentionLSTM(nn.Module):
    """v50 Model - Temporal attention only"""
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


class TrueDualAttentionLSTM(nn.Module):
    """v64 Model - Temporal + Feature attention"""
    def __init__(self, input_dim, hidden_dim):
        super(TrueDualAttentionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_attention = Attention(hidden_dim)
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        context, temporal_attn = self.temporal_attention(lstm_out)
        feature_weights = self.feature_attention(context)
        output = self.fc(context)
        
        if return_attention:
            return output, temporal_attn, feature_weights
        return output


# ==========================================
# Training Function
# ==========================================

def train_and_evaluate(model_class, model_name, seed, X_train, y_train, X_test, y_test, 
                       test_slice, verbose=True):
    """Train model with specific seed and evaluate"""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    model = model_class(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Prepare data
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    # Training
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
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}", flush=True)
    
    # Evaluation
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        preds = model(X_test_tensor).squeeze().numpy()
    
    # Inverse transform
    m_y = test_slice['m_y'].values
    s_y = test_slice['s_y'].values
    preds_orig = preds * s_y + m_y
    y_test_orig = y_test * s_y + m_y
    
    # Metrics
    r2 = r2_score(y_test_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae = mean_absolute_error(y_test_orig, preds_orig)
    
    return {
        'seed': seed,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': preds_orig,
        'actuals': y_test_orig
    }


# ==========================================
# Main Experiment
# ==========================================

def run_robustness_experiment():
    print("="*80)
    print("모델 강건성 검증: 5-Seed Experiment")
    print("="*80)
    print(f"Seeds: {SEEDS}")
    print()
    
    # Load Data
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
    
    df_all = pd.DataFrame(all_data).sort_values('Date')
    split = int(len(df_all) * 0.8)
    train_slice = df_all.iloc[:split]
    test_slice = df_all.iloc[split:].reset_index(drop=True)
    
    X_train = np.stack(train_slice['X'].values)
    y_train = np.stack(train_slice['y_scaled'].values)
    X_test = np.stack(test_slice['X'].values)
    y_test = np.stack(test_slice['y_scaled'].values)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()
    
    # ==========================================
    # Experiment 1: v50 (Temporal Attention Only)
    # ==========================================
    print("="*80)
    print("Experiment 1: v50 DualAttentionLSTM (Temporal Only)")
    print("="*80)
    
    v50_results = []
    
    for i, seed in enumerate(SEEDS, 1):
        print(f"\nRun {i}/5 - Seed: {seed}")
        print("-"*40)
        
        result = train_and_evaluate(
            DualAttentionLSTM, 'v50', seed,
            X_train, y_train, X_test, y_test, test_slice,
            verbose=True
        )
        
        v50_results.append(result)
        print(f"  R²: {result['r2']:.5f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # ==========================================
    # Experiment 2: v64 (True Dual Attention)
    # ==========================================
    print("\n" + "="*80)
    print("Experiment 2: v64 TrueDualAttentionLSTM (Temporal + Feature)")
    print("="*80)
    
    v64_results = []
    
    for i, seed in enumerate(SEEDS, 1):
        print(f"\nRun {i}/5 - Seed: {seed}")
        print("-"*40)
        
        result = train_and_evaluate(
            TrueDualAttentionLSTM, 'v64', seed,
            X_train, y_train, X_test, y_test, test_slice,
            verbose=True
        )
        
        v64_results.append(result)
        print(f"  R²: {result['r2']:.5f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # ==========================================
    # Statistical Analysis
    # ==========================================
    print("\n" + "="*80)
    print("Statistical Analysis")
    print("="*80)
    
    # v50 statistics
    v50_r2 = [r['r2'] for r in v50_results]
    v50_rmse = [r['rmse'] for r in v50_results]
    v50_mae = [r['mae'] for r in v50_results]
    
    print("\nv50 (Temporal Only):")
    print(f"  R² - Mean: {np.mean(v50_r2):.5f}, Std: {np.std(v50_r2):.5f}")
    print(f"  RMSE - Mean: {np.mean(v50_rmse):.4f}, Std: {np.std(v50_rmse):.4f}")
    print(f"  MAE - Mean: {np.mean(v50_mae):.4f}, Std: {np.std(v50_mae):.4f}")
    
    # v64 statistics
    v64_r2 = [r['r2'] for r in v64_results]
    v64_rmse = [r['rmse'] for r in v64_results]
    v64_mae = [r['mae'] for r in v64_results]
    
    print("\nv64 (Dual Attention):")
    print(f"  R² - Mean: {np.mean(v64_r2):.5f}, Std: {np.std(v64_r2):.5f}")
    print(f"  RMSE - Mean: {np.mean(v64_rmse):.4f}, Std: {np.std(v64_rmse):.4f}")
    print(f"  MAE - Mean: {np.mean(v64_mae):.4f}, Std: {np.std(v64_mae):.4f}")
    
    # ==========================================
    # Visualization
    # ==========================================
    print("\nCreating visualizations...")
    
    output_dir = Path('src/experiments/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Box plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² boxplot
    data_r2 = pd.DataFrame({
        'v50': v50_r2,
        'v64': v64_r2
    })
    axes[0].boxplot([v50_r2, v64_r2], labels=['v50\n(Temporal)', 'v64\n(Dual)'])
    axes[0].set_title('R² Score Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R² Score')
    axes[0].grid(axis='y', alpha=0.3)
    
    # RMSE boxplot
    axes[1].boxplot([v50_rmse, v64_rmse], labels=['v50\n(Temporal)', 'v64\n(Dual)'])
    axes[1].set_title('RMSE Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE')
    axes[1].grid(axis='y', alpha=0.3)
    
    # MAE boxplot
    axes[2].boxplot([v50_mae, v64_mae], labels=['v50\n(Temporal)', 'v64\n(Dual)'])
    axes[2].set_title('MAE Distribution', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MAE')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v65_robustness_boxplot.png', dpi=150)
    plt.close()
    print("  Saved: v65_robustness_boxplot.png")
    
    # 2. Performance across seeds
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(SEEDS))
    width = 0.35
    
    ax.bar(x - width/2, v50_r2, width, label='v50 (Temporal)', color='skyblue', edgecolor='black')
    ax.bar(x + width/2, v64_r2, width, label='v64 (Dual)', color='coral', edgecolor='black')
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('R² Score')
    ax.set_title('Performance Consistency Across Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SEEDS)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v65_seed_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v65_seed_comparison.png")
    
    # 3. Mean ± Std comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    models = ['v50\n(Temporal Only)', 'v64\n(Dual Attention)']
    means = [np.mean(v50_r2), np.mean(v64_r2)]
    stds = [np.std(v50_r2), np.std(v64_r2)]
    
    bars = ax.bar(models, means, yerr=stds, capsize=10, 
                   color=['skyblue', 'coral'], edgecolor='black', width=0.6)
    
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance: Mean ± Std (5 seeds)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{mean:.5f}\n±{std:.5f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v65_mean_std_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v65_mean_std_comparison.png")
    
    # ==========================================
    # Save Results
    # ==========================================
    results_summary = {
        'v50': {
            'seeds': SEEDS,
            'r2_scores': v50_r2,
            'rmse_scores': v50_rmse,
            'mae_scores': v50_mae,
            'statistics': {
                'r2_mean': float(np.mean(v50_r2)),
                'r2_std': float(np.std(v50_r2)),
                'r2_min': float(np.min(v50_r2)),
                'r2_max': float(np.max(v50_r2)),
                'rmse_mean': float(np.mean(v50_rmse)),
                'rmse_std': float(np.std(v50_rmse)),
                'mae_mean': float(np.mean(v50_mae)),
                'mae_std': float(np.std(v50_mae))
            }
        },
        'v64': {
            'seeds': SEEDS,
            'r2_scores': v64_r2,
            'rmse_scores': v64_rmse,
            'mae_scores': v64_mae,
            'statistics': {
                'r2_mean': float(np.mean(v64_r2)),
                'r2_std': float(np.std(v64_r2)),
                'r2_min': float(np.min(v64_r2)),
                'r2_max': float(np.max(v64_r2)),
                'rmse_mean': float(np.mean(v64_rmse)),
                'rmse_std': float(np.std(v64_rmse)),
                'mae_mean': float(np.mean(v64_mae)),
                'mae_std': float(np.std(v64_mae))
            }
        }
    }
    
    with open(output_dir / 'v65_robustness_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("  Saved: v65_robustness_results.json")
    
    print("\n" + "="*80)
    print("Robustness Experiment Complete!")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  v50 is {'more' if np.std(v50_r2) < np.std(v64_r2) else 'less'} robust (Std: {np.std(v50_r2):.5f} vs {np.std(v64_r2):.5f})")
    print(f"  v50 average performance: R² = {np.mean(v50_r2):.5f} ± {np.std(v50_r2):.5f}")
    print(f"  v64 average performance: R² = {np.mean(v64_r2):.5f} ± {np.std(v64_r2):.5f}")


if __name__ == "__main__":
    run_robustness_experiment()
