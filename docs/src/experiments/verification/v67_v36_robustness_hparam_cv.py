import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
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

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]

# Hyperparameter Grid
ALPHA_GRID = {
    'Equity': [0.1, 1.0, 10.0, 100.0],
    'Bond': [0.1, 1.0, 10.0],
    'Commodity': [0.1, 1.0, 10.0, 100.0]
}

WINDOW_GRID = [
    (1, 5, 22),   # Standard HAR
    (1, 5, 10),   # Shorter long-term
    (1, 10, 22)   # Different medium-term
]

N_BOOTSTRAP = 5  # Bootstrap samples instead of seeds (Ridge is deterministic)

# ==========================================
# Data Preparation
# ==========================================

def load_and_prepare_data(window=(1, 5, 22)):
    """Load data and prepare HAR features"""
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from cache: {CACHE_PATH}...", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    else:
        print("Downloading from yfinance...", flush=True)
        raw = yf.download(ALL_ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    pooled_data = []
    lag1, lag5, lag22 = window
    print(f"Processing {len(ALL_ASSETS)} assets with window={window}...", flush=True)
    
    for asset in ALL_ASSETS:
        price = raw[asset]
        ret_daily = np.log(price / price.shift(1)).dropna()
        
        rv_daily = ret_daily**2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        d = pd.DataFrame({
            f'LogRV_lag{lag1}': log_rv.shift(lag1),
            f'LogRV_lag{lag5}': log_rv.shift(lag5),
            f'LogRV_lag{lag22}': log_rv.shift(lag22),
            'Target': log_rv.shift(-22),
            'Asset': asset
        }).dropna()
        
        # Asset Class
        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        
        pooled_data.append(d)
    
    data = pd.concat(pooled_data).sort_index()
    return data


# ==========================================
# Model Training and Evaluation
# ==========================================

def train_and_evaluate(alpha_config, train_df, test_df, window):
    """Train asset-class specific Ridge models"""
    lag1, lag5, lag22 = window
    feats = [f'LogRV_lag{lag1}', f'LogRV_lag{lag5}', f'LogRV_lag{lag22}']
    
    # Standardize
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[feats])
    X_test = sc.transform(test_df[feats])
    y_train = train_df['Target']
    y_test = test_df['Target']
    
    # Train asset-class specific models
    class_models = {}
    test_predictions = pd.Series(index=test_df.index, dtype=float)
    
    for cls in ASSET_GROUPS.keys():
        # Filter training data
        train_cls = train_df[train_df['Class'] == cls]
        
        if len(train_cls) < 100:
            continue
        
        X_train_cls = sc.transform(train_cls[feats])
        y_train_cls = train_cls['Target']
        
        alpha = alpha_config.get(cls, 1.0)
        model_cls = Ridge(alpha=alpha).fit(X_train_cls, y_train_cls)
        class_models[cls] = model_cls
        
        # Predict on test set
        test_cls_mask = test_df['Class'] == cls
        if test_cls_mask.sum() > 0:
            X_test_cls = sc.transform(test_df.loc[test_cls_mask, feats])
            pred_cls = model_cls.predict(X_test_cls)
            test_predictions[test_cls_mask] = pred_cls
    
    # Fill missing with global model (safety)
    mask_missing = test_predictions.isna()
    if mask_missing.sum() > 0:
        global_model = Ridge(alpha=1.0).fit(X_train, y_train)
        pred_global = global_model.predict(X_test)
        test_predictions[mask_missing] = pred_global[mask_missing]
    
    # Metrics
    r2 = r2_score(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae = mean_absolute_error(y_test, test_predictions)
    
    # Asset-class specific metrics
    class_metrics = {}
    for cls in ASSET_GROUPS.keys():
        mask = test_df['Class'] == cls
        if mask.sum() > 0:
            r2_cls = r2_score(y_test[mask], test_predictions[mask])
            class_metrics[cls] = {'r2': r2_cls, 'count': mask.sum()}
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'class_metrics': class_metrics
    }


# ==========================================
# Phase 1: Alpha Grid Search
# ==========================================

def grid_search_alpha(data, window=(1, 5, 22)):
    """Grid search for asset-class specific alphas"""
    print("\n" + "="*80)
    print("ALPHA GRID SEARCH (Asset-Class Specific)")
    print("="*80)
    
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split]
    test_df = data.iloc[split:].reset_index(drop=True)
    
    results = []
    
    # Grid search for each asset class independently
    best_alphas = {}
    
    for cls in ['Equity', 'Bond', 'Commodity']:
        print(f"\n[Optimizing {cls} Alpha]")
        print("-" * 40)
        
        cls_results = []
        
        for alpha in ALPHA_GRID[cls]:
            # Set temporary alpha config
            temp_config = {c: 1.0 for c in ASSET_GROUPS.keys()}
            temp_config[cls] = alpha
            
            metrics = train_and_evaluate(temp_config, train_df, test_df, window)
            
            # Get class-specific R²
            r2_cls = metrics['class_metrics'].get(cls, {}).get('r2', 0)
            
            cls_results.append({
                'class': cls,
                'alpha': alpha,
                'r2_class': r2_cls,
                'r2_overall': metrics['r2']
            })
            
            print(f"  Alpha={alpha:6.1f}: Class R²={r2_cls:.5f}, Overall R²={metrics['r2']:.5f}")
        
        # Find best alpha for this class
        cls_df = pd.DataFrame(cls_results)
        best_idx = cls_df['r2_class'].idxmax()
        best_alpha = cls_df.loc[best_idx, 'alpha']
        best_alphas[cls] = best_alpha
        
        print(f"  Best {cls} alpha: {best_alpha}")
        results.extend(cls_results)
    
    print(f"\n[Best Alpha Configuration]")
    for cls, alpha in best_alphas.items():
        print(f"  {cls}: {alpha}")
    
    # Evaluate with best alphas
    final_metrics = train_and_evaluate(best_alphas, train_df, test_df, window)
    print(f"\nFinal R² with optimized alphas: {final_metrics['r2']:.5f}")
    
    results_df = pd.DataFrame(results)
    return results_df, best_alphas, final_metrics


# ==========================================
# Phase 2: Window Size Search
# ==========================================

def grid_search_window(data, best_alphas):
    """Search for optimal HAR window"""
    print("\n" + "="*80)
    print("WINDOW SIZE GRID SEARCH")
    print("="*80)
    
    split = int(len(data) * 0.8)
    
    results = []
    
    for i, window in enumerate(WINDOW_GRID, 1):
        print(f"\n[{i}/{len(WINDOW_GRID)}] Testing window={window}")
        print("-" * 40)
        
        # Reload data with this window
        data_window = load_and_prepare_data(window=window)
        train_df = data_window.iloc[:split]
        test_df = data_window.iloc[split:].reset_index(drop=True)
        
        metrics = train_and_evaluate(best_alphas, train_df, test_df, window)
        
        results.append({
            'window': str(window),
            **metrics
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['r2'].idxmax()
    best_window = results_df.loc[best_idx, 'window']
    
    print(f"\n[Best Window]: {best_window}")
    
    return results_df, eval(best_window)


# ==========================================
# Phase 3: 5-Fold Cross-Validation
# ==========================================

def cv_evaluation(best_alphas, data, window, n_splits=5):
    """5-fold TimeSeriesSplit CV"""
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data), 1):
        print(f"\n[Fold {fold}/{n_splits}]")
        print("-" * 40)
        
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx].reset_index(drop=True)
        
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
        
        metrics = train_and_evaluate(best_alphas, train_df, val_df, window)
        
        fold_results.append({
            'fold': fold,
            **metrics
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        # Print class-specific results
        for cls, cls_metrics in metrics.get('class_metrics', {}).items():
            print(f"    {cls}: R²={cls_metrics['r2']:.5f}")
    
    cv_df = pd.DataFrame(fold_results)
    
    print(f"\n[Cross-Validation Summary]")
    print(f"  R² - Mean: {cv_df['r2'].mean():.5f}, Std: {cv_df['r2'].std():.5f}")
    print(f"  RMSE - Mean: {cv_df['rmse'].mean():.4f}, Std: {cv_df['rmse'].std():.4f}")
    print(f"  MAE - Mean: {cv_df['mae'].mean():.4f}, Std: {cv_df['mae'].std():.4f}")
    
    return cv_df


# ==========================================
# Phase 4: Bootstrap Robustness
# ==========================================

def bootstrap_robustness(best_alphas, data, window, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap resampling for stability (Ridge is deterministic)"""
    print("\n" + "="*80)
    print(f"{n_bootstrap}-BOOTSTRAP ROBUSTNESS TEST")
    print("="*80)
    
    split = int(len(data) * 0.8)
    train_df_orig = data.iloc[:split]
    test_df = data.iloc[split:].reset_index(drop=True)
    
    bootstrap_results = []
    
    for i in range(1, n_bootstrap + 1):
        print(f"\n[Bootstrap {i}/{n_bootstrap}]")
        print("-" * 40)
        
        # Resample training data
        train_df = resample(train_df_orig, replace=True, n_samples=len(train_df_orig), random_state=i*42)
        
        metrics = train_and_evaluate(best_alphas, train_df, test_df, window)
        
        bootstrap_results.append({
            'bootstrap': i,
            **metrics
        })
        
        print(f"  R²: {metrics['r2']:.5f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    boot_df = pd.DataFrame(bootstrap_results)
    
    print(f"\n[Bootstrap Robustness Summary]")
    print(f"  R² - Mean: {boot_df['r2'].mean():.5f}, Std: {boot_df['r2'].std():.5f}")
    print(f"  RMSE - Mean: {boot_df['rmse'].mean():.4f}, Std: {boot_df['rmse'].std():.4f}")
    print(f"  MAE - Mean: {boot_df['mae'].mean():.4f}, Std: {boot_df['mae'].std():.4f}")
    
    return boot_df


# ==========================================
# Visualization
# ==========================================

def create_visualizations(alpha_df, window_df, cv_df, boot_df, best_alphas, best_window):
    """Create all visualizations"""
    output_dir = Path('src/experiments/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. Alpha Sensitivity by Asset Class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, cls in enumerate(['Equity', 'Bond', 'Commodity']):
        cls_data = alpha_df[alpha_df['class'] == cls]
        axes[i].plot(cls_data['alpha'], cls_data['r2_class'], 'o-', linewidth=2, markersize=8, color='#3182CE')
        axes[i].axvline(x=best_alphas[cls], color='red', linestyle='--', label=f'Optimal: {best_alphas[cls]}')
        axes[i].set_xlabel('Ridge Alpha')
        axes[i].set_ylabel('R² Score')
        axes[i].set_title(f'{cls} Alpha Sensitivity', fontsize=12, fontweight='bold')
        axes[i].set_xscale('log')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v67_alpha_sensitivity.png', dpi=150)
    plt.close()
    print("  Saved: v67_alpha_sensitivity.png")
    
    # 2. Window Size Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    window_df_clean = window_df.copy()
    window_df_clean['r2'] = pd.to_numeric(window_df_clean['r2'], errors='coerce')
    
    bars = ax.bar(range(len(window_df_clean)), window_df_clean['r2'], color='skyblue', edgecolor='black')
    ax.set_xlabel('HAR Window')
    ax.set_ylabel('R² Score')
    ax.set_title('Window Size Impact', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(window_df_clean)))
    ax.set_xticklabels(window_df_clean['window'])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, window_df_clean['r2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v67_window_comparison.png', dpi=150)
    plt.close()
    print("  Saved: v67_window_comparison.png")
    
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
    plt.savefig(output_dir / 'v67_cv_performance.png', dpi=150)
    plt.close()
    print("  Saved: v67_cv_performance.png")
    
    # 4. Bootstrap Robustness
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R²
    axes[0].bar(boot_df['bootstrap'].astype(str), boot_df['r2'], color='skyblue', edgecolor='black')
    axes[0].axhline(y=boot_df['r2'].mean(), color='red', linestyle='--', label=f"Mean: {boot_df['r2'].mean():.4f}")
    axes[0].set_xlabel('Bootstrap Sample')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² across Bootstraps', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[1].bar(boot_df['bootstrap'].astype(str), boot_df['rmse'], color='coral', edgecolor='black')
    axes[1].axhline(y=boot_df['rmse'].mean(), color='red', linestyle='--', label=f"Mean: {boot_df['rmse'].mean():.4f}")
    axes[1].set_xlabel('Bootstrap Sample')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE across Bootstraps', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[2].bar(boot_df['bootstrap'].astype(str), boot_df['mae'], color='lightgreen', edgecolor='black')
    axes[2].axhline(y=boot_df['mae'].mean(), color='red', linestyle='--', label=f"Mean: {boot_df['mae'].mean():.4f}")
    axes[2].set_xlabel('Bootstrap Sample')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('MAE across Bootstraps', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v67_bootstrap_robustness.png', dpi=150)
    plt.close()
    print("  Saved: v67_bootstrap_robustness.png")
    
    # 5. Summary
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics_names = ['CV Mean\n(5 Folds)', 'Bootstrap Mean\n(5 Samples)']
    r2_values = [cv_df['r2'].mean(), boot_df['r2'].mean()]
    r2_stds = [cv_df['r2'].std(), boot_df['r2'].std()]
    
    bars = ax.bar(metrics_names, r2_values, yerr=r2_stds, capsize=10,
                   color=['#3182CE', '#E53E3E'], edgecolor='black', width=0.6)
    
    ax.set_ylabel('R² Score')
    ax.set_title('v36 Robustness Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(r2_values) * 1.2)
    
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'v67_summary.png', dpi=150)
    plt.close()
    print("  Saved: v67_summary.png")


# ==========================================
# Main Execution
# ==========================================

def main():
    print("="*80)
    print("V67: v36 Hyperparameter + CV + Bootstrap Robustness")
    print("="*80)
    
    # Default window
    default_window = (1, 5, 22)
    data = load_and_prepare_data(window=default_window)
    print(f"\nTotal samples: {len(data)}\n")
    
    # Phase 1: Alpha Grid Search
    alpha_df, best_alphas, alpha_metrics = grid_search_alpha(data, window=default_window)
    
    # Phase 2: Window Size Search
    window_df, best_window = grid_search_window(data, best_alphas)
    
    # Reload with best window
    print(f"\nReloading data with best window={best_window}...")
    data_final = load_and_prepare_data(window=best_window)
    
    # Phase 3: 5-Fold CV
    cv_df = cv_evaluation(best_alphas, data_final, best_window)
    
    # Phase 4: Bootstrap Robustness
    boot_df = bootstrap_robustness(best_alphas, data_final, best_window)
    
    # Save results
    output_dir = Path('src/experiments/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'best_alphas': {k: float(v) for k, v in best_alphas.items()},
        'best_window': best_window,
        'alpha_grid_results': alpha_df.to_dict('records'),
        'window_results': window_df.to_dict('records'),
        'cv_results': {
            'folds': cv_df.to_dict('records'),
            'mean_r2': float(cv_df['r2'].mean()),
            'std_r2': float(cv_df['r2'].std()),
            'mean_rmse': float(cv_df['rmse'].mean()),
            'std_rmse': float(cv_df['rmse'].std())
        },
        'bootstrap_results': {
            'bootstraps': boot_df.to_dict('records'),
            'mean_r2': float(boot_df['r2'].mean()),
            'std_r2': float(boot_df['r2'].std()),
            'mean_rmse': float(boot_df['rmse'].mean()),
            'std_rmse': float(boot_df['rmse'].std())
        }
    }
    
    with open(output_dir / 'v67_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NpEncoder)
    print("\n  Saved: v67_results.json")
    
    # Visualization
    create_visualizations(alpha_df, window_df, cv_df, boot_df, best_alphas, best_window)
    
    print("\n" + "="*80)
    print("V67 COMPLETE!")
    print("="*80)
    print(f"\nBest Config:")
    for cls, alpha in best_alphas.items():
        print(f"  {cls} alpha: {alpha}")
    print(f"  Window: {best_window}")
    print(f"\nPerformance:")
    print(f"  CV Mean R²: {cv_df['r2'].mean():.5f} ± {cv_df['r2'].std():.5f}")
    print(f"  Bootstrap Mean R²: {boot_df['r2'].mean():.5f} ± {boot_df['r2'].std():.5f}")


if __name__ == "__main__":
    main()
