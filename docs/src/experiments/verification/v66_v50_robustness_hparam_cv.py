import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from pathlib import Path
from itertools import product
import time

warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN_DEFAULT = 22
EPOCHS = 20
BATCH_SIZE_DEFAULT = 64
SEEDS = [42, 123, 456, 789, 2024]

# Hyperparameter Grid
HPARAM_GRID = {
    'hidden_dim': [32, 64, 128],
    'lr': [0.0001, 0.001, 0.01],
    'dropout': [0.0, 0.3, 0.5],
    'seq_len': [10, 22, 44]
}

# ==========================================
# Model Definitions  
# ==========================================

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
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        
        if return_attention:
            return output, attn_weights
        return output


# ==========================================
# Data Preparation
# ==========================================

def load_and_prepare_data(seq_len=22):
    """Load data and create sequences"""
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from cache: {CACHE_PATH}...", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    else:
        print("Downloading from yfinance...", flush=True)
        raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)[' Close']
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    all_data = []
    print(f"Processing {len(ASSETS)} assets with seq_len={seq_len}...", flush=True)
    
    for asset in ASSETS:
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
        
        for j in range(len(vals) - seq_len - 1):
            x = vals[j:j+seq_len]
            y_scaled = vals[j+seq_len, 0]
            target_date = dates[j+seq_len]
            
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y,
                's_y': s_y
            })
    
    df_all = pd.DataFrame(all_data).sort_values('Date')
    return df_all


# ==========================================
# Training and Evaluation
# ==========================================

def train_and_evaluate(config, train_data, val_data, seed, verbose=False):
    """Train model with specific config and evaluate"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Model
    model = DualAttentionLSTM(
        input_dim=2, 
        hidden_dim=int(config['hidden_dim']),
        dropout=config['dropout']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Prepare data
    X_train = torch.FloatTensor(np.stack(train_data['X'].values))
    y_train = torch.FloatTensor(train_data['y_scaled'].values)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=int(config.get('batch_size', BATCH_SIZE_DEFAULT)), 
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
    X_val = torch.FloatTensor(np.stack(val_data['X'].values))
    
    with torch.no_grad():
        preds = model(X_val).squeeze().numpy()
    
    # Inverse transform
    m_y = val_data['m_y'].values
    s_y = val_data['s_y'].values
    y_val = val_data['y_scaled'].values
    
    preds_orig = preds * s_y + m_y
    y_val_orig = y_val * s_y + m_y
    
    # Metrics
    r2 = r2_score(y_val_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_val_orig, preds_orig))
    mae = mean_absolute_error(y_val_orig, preds_orig)
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


# ==========================================
# Phase 1: Hyperparameter Grid Search
# ==========================================

def grid_search_stage1(df_all):
    """Stage 1: hidden_dim x lr grid search"""
    print("\n" + "="*80)
    print("STAGE 1: Grid Search (hidden_dim × lr)")
    print("="*80)
    
    # Use simple 80/20 split for grid search
    split = int(len(df_all) * 0.8)
    train_data = df_all.iloc[:split].reset_index(drop=True)
    val_data = df_all.iloc[split:].reset_index(drop=True)
    
    results = []
    total_combinations = len(HPARAM_GRID['hidden_dim']) * len(HPARAM_GRID['lr'])
    
    for i, (hidden_dim, lr) in enumerate(product(HPARAM_GRID['hidden_dim'], HPARAM_GRID['lr']), 1):
        config = {
            'hidden_dim': hidden_dim,
            'lr': lr,
            'dropout': 0.0,  # Default
            'seq_len': SEQ_LEN_DEFAULT,
            'batch_size': BATCH_SIZE_DEFAULT
        }
        
        print(f"\n[{i}/{total_combinations}] Testing hidden_dim={hidden_dim}, lr={lr}")
        print("-" * 40)
        
        start_time = time.time()
        metrics = train_and_evaluate(config, train_data, val_data, seed=42, verbose=True)
        elapsed = time.time() - start_time
        
        results.append({
            **config,
            **metrics,
            'time': elapsed
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, Time: {elapsed:.1f}s")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['r2'].idxmax()
    best_config = results_df.loc[best_idx].to_dict()
    
    print(f"\n[Stage 1 Best Config]")
    print(f"  hidden_dim: {best_config['hidden_dim']}")
    print(f"  lr: {best_config['lr']}")
    print(f"  R²: {best_config['r2']:.5f}")
    
    return results_df, best_config


def grid_search_stage2(df_all, best_stage1):
    """Stage 2: dropout x seq_len grid search"""
    print("\n" + "="*80)
    print("STAGE 2: Grid Search (dropout × seq_len)")
    print("="*80)
    
    results = []
    total_combinations = len(HPARAM_GRID['dropout']) * len(HPARAM_GRID['seq_len'])
    
    for i, (dropout, seq_len) in enumerate(product(HPARAM_GRID['dropout'], HPARAM_GRID['seq_len']), 1):
        # Reload data with different seq_len
        df_all_new = load_and_prepare_data(seq_len=seq_len)
        split = int(len(df_all_new) * 0.8)
        train_data = df_all_new.iloc[:split].reset_index(drop=True)
        val_data = df_all_new.iloc[split:].reset_index(drop=True)
        
        config = {
            'hidden_dim': int(best_stage1['hidden_dim']),
            'lr': best_stage1['lr'],
            'dropout': dropout,
            'seq_len': seq_len,
            'batch_size': BATCH_SIZE_DEFAULT
        }
        
        print(f"\n[{i}/{total_combinations}] Testing dropout={dropout}, seq_len={seq_len}")
        print("-" * 40)
        
        start_time = time.time()
        metrics = train_and_evaluate(config, train_data, val_data, seed=42, verbose=True)
        elapsed = time.time() - start_time
        
        results.append({
            **config,
            **metrics,
            'time': elapsed
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, Time: {elapsed:.1f}s")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['r2'].idxmax()
    best_config = results_df.loc[best_idx].to_dict()
    
    print(f"\n[Stage 2 Best Config]")
    print(f"  dropout: {best_config['dropout']}")
    print(f"  seq_len: {best_config['seq_len']}")
    print(f"  R²: {best_config['r2']:.5f}")
    
    return results_df, best_config


# ==========================================
# Phase 2: 5-Fold Cross-Validation
# ==========================================

def cv_evaluation(best_config, df_all, n_splits=5):
    """5-fold TimeSeriesSplit CV"""
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_all), 1):
        print(f"\n[Fold {fold}/{n_splits}]")
        print("-" * 40)
        
        train_data = df_all.iloc[train_idx].reset_index(drop=True)
        val_data = df_all.iloc[val_idx].reset_index(drop=True)
        
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
        
        metrics = train_and_evaluate(best_config, train_data, val_data, seed=42, verbose=True)
        
        fold_results.append({
            'fold': fold,
            **metrics
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    cv_df = pd.DataFrame(fold_results)
    
    print(f"\n[Cross-Validation Summary]")
    print(f"  R² - Mean: {cv_df['r2'].mean():.5f}, Std: {cv_df['r2'].std():.5f}")
    print(f"  RMSE - Mean: {cv_df['rmse'].mean():.4f}, Std: {cv_df['rmse'].std():.4f}")
    print(f"  MAE - Mean: {cv_df['mae'].mean():.4f}, Std: {cv_df['mae'].std():.4f}")
    
    return cv_df


# ==========================================
# Phase 3: 5-Seed Robustness
# ==========================================

def seed_robustness(best_config, df_all):
    """5-seed experiments"""
    print("\n" + "="*80)
    print("5-SEED ROBUSTNESS TEST")
    print("="*80)
    
    split = int(len(df_all) * 0.8)
    train_data = df_all.iloc[:split].reset_index(drop=True)
    val_data = df_all.iloc[split:].reset_index(drop=True)
    
    seed_results = []
    
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n[Seed {i}/{len(SEEDS)}: {seed}]")
        print("-" * 40)
        
        metrics = train_and_evaluate(best_config, train_data, val_data, seed=seed, verbose=True)
        
        seed_results.append({
            'seed': seed,
            **metrics
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    seed_df = pd.DataFrame(seed_results)
    
    print(f"\n[Seed Robustness Summary]")
    print(f"  R² - Mean: {seed_df['r2'].mean():.5f}, Std: {seed_df['r2'].std():.5f}")
    print(f"  RMSE - Mean: {seed_df['rmse'].mean():.4f}, Std: {seed_df['rmse'].std():.4f}")
    print(f"  MAE - Mean: {seed_df['mae'].mean():.4f}, Std: {seed_df['mae'].std():.4f}")
    
    return seed_df


# ==========================================
# Visualization
# ==========================================

def create_visualizations(stage1_df, stage2_df, cv_df, seed_df, best_config):
    """Create all visualizations"""
    output_dir = Path('src/experiments/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. Stage 1 Heatmap: hidden_dim x lr
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = stage1_df.pivot(index='hidden_dim', columns='lr', values='r2')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'R² Score'})
    ax.set_title('Stage 1: Hidden Dim × Learning Rate Grid Search', fontsize=14, fontweight='bold')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Hidden Dimension')
    plt.tight_layout()
    plt.savefig(output_dir / 'v66_stage1_heatmap.png', dpi=150)
    plt.close()
    print("  Saved: v66_stage1_heatmap.png")
    
    # 2. Stage 2 Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dropout
    dropout_grouped = stage2_df.groupby('dropout')['r2'].mean()
    axes[0].bar(dropout_grouped.index.astype(str), dropout_grouped.values, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Dropout Rate')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Dropout Impact', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Seq Len
    seqlen_grouped = stage2_df.groupby('seq_len')['r2'].mean()
    axes[1].bar(seqlen_grouped.index.astype(str), seqlen_grouped.values, color='coral', edgecolor='black')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Sequence Length Impact', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v66_stage2_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v66_stage2_comparison.png")
    
    # 3. CV Fold Performance
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cv_df))
    width = 0.25
    
    ax.bar(x - width, cv_df['r2'], width, label='R²', color='skyblue', edgecolor='black')
    ax.bar(x, cv_df['rmse'], width, label='RMSE', color='coral', edgecolor='black')
    ax.bar(x + width, cv_df['mae'], width, label='MAE', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('5-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in cv_df['fold']])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v66_cv_performance.png', dpi=150)
    plt.close()
    print("  Saved: v66_cv_performance.png")
    
    # 4. Seed Robustness
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R²
    axes[0].bar(seed_df['seed'].astype(str), seed_df['r2'], color='skyblue', edgecolor='black')
    axes[0].axhline(y=seed_df['r2'].mean(), color='red', linestyle='--', label=f"Mean: {seed_df['r2'].mean():.4f}")
    axes[0].set_xlabel('Random Seed')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² across Seeds', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[1].bar(seed_df['seed'].astype(str), seed_df['rmse'], color='coral', edgecolor='black')
    axes[1].axhline(y=seed_df['rmse'].mean(), color='red', linestyle='--', label=f"Mean: {seed_df['rmse'].mean():.4f}")
    axes[1].set_xlabel('Random Seed')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE across Seeds', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[2].bar(seed_df['seed'].astype(str), seed_df['mae'], color='lightgreen', edgecolor='black')
    axes[2].axhline(y=seed_df['mae'].mean(), color='red', linestyle='--', label=f"Mean: {seed_df['mae'].mean():.4f}")
    axes[2].set_xlabel('Random Seed')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('MAE across Seeds', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v66_seed_robustness.png', dpi=150)
    plt.close()
    print("  Saved: v66_seed_robustness.png")
    
    # 5. Summary
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics_names = ['Best Config\n(Single Run)', 'CV Mean\n(5 Folds)', 'Seed Mean\n(5 Seeds)']
    r2_values = [best_config['r2'], cv_df['r2'].mean(), seed_df['r2'].mean()]
    r2_stds = [0, cv_df['r2'].std(), seed_df['r2'].std()]
    
    bars = ax.bar(metrics_names, r2_values, yerr=r2_stds, capsize=10, 
                   color=['#38A169', '#3182CE', '#E53E3E'], edgecolor='black', width=0.6)
    
    ax.set_ylabel('R² Score')
    ax.set_title('v50 Robustness Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(r2_values) * 1.2)
    
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v66_summary.png', dpi=150)
    plt.close()
    print("  Saved: v66_summary.png")


# ==========================================
# Main Execution
# ==========================================

def main():
    print("="*80)
    print("V66: v50 Hyperparameter + CV + 5-Seed Robustness")
    print("="*80)
    
    # Load data with default seq_len
    df_all = load_and_prepare_data(seq_len=SEQ_LEN_DEFAULT)
    print(f"\nTotal samples: {len(df_all)}\n")
    
    # Stage 1: hidden_dim x lr
    stage1_df, best_stage1 = grid_search_stage1(df_all)
    
    # Stage 2: dropout x seq_len
    stage2_df, best_config = grid_search_stage2(df_all, best_stage1)
    
    # Reload with best seq_len
    print(f"\nReloading data with best seq_len={int(best_config['seq_len'])}...")
    df_all_final = load_and_prepare_data(seq_len=int(best_config['seq_len']))
    
    # 5-Fold CV
    cv_df = cv_evaluation(best_config, df_all_final)
    
    # 5-Seed Robustness
    seed_df = seed_robustness(best_config, df_all_final)
    
    # Save results
    output_dir = Path('src/experiments/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'best_config': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                       for k, v in best_config.items()},
        'stage1_results': stage1_df.to_dict('records'),
        'stage2_results': stage2_df.to_dict('records'),
        'cv_results': {
            'folds': cv_df.to_dict('records'),
            'mean_r2': float(cv_df['r2'].mean()),
            'std_r2': float(cv_df['r2'].std()),
            'mean_rmse': float(cv_df['rmse'].mean()),
            'std_rmse': float(cv_df['rmse'].std())
        },
        'seed_results': {
            'seeds': seed_df.to_dict('records'),
            'mean_r2': float(seed_df['r2'].mean()),
            'std_r2': float(seed_df['r2'].std()),
            'mean_rmse': float(seed_df['rmse'].mean()),
            'std_rmse': float(seed_df['rmse'].std())
        }
    }
    
    with open(output_dir / 'v66_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NpEncoder)
    print("\n  Saved: v66_results.json")
    
    # Visualization
    create_visualizations(stage1_df, stage2_df, cv_df, seed_df, best_config)
    
    print("\n" + "="*80)
    print("V66 COMPLETE!")
    print("="*80)
    print(f"\nBest Config:")
    print(f"  hidden_dim: {best_config['hidden_dim']}")
    print(f"  lr: {best_config['lr']}")
    print(f"  dropout: {best_config['dropout']}")
    print(f"  seq_len: {best_config['seq_len']}")
    print(f"\nPerformance:")
    print(f"  Best Single Run R²: {best_config['r2']:.5f}")
    print(f"  CV Mean R²: {cv_df['r2'].mean():.5f} ± {cv_df['r2'].std():.5f}")
    print(f"  Seed Mean R²: {seed_df['r2'].mean():.5f} ± {seed_df['r2'].std():.5f}")


if __name__ == "__main__":
    main()
