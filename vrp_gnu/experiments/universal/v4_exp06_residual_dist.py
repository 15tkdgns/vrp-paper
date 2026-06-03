"""
V4 Exp 06: Residual Distribution Analysis
Purpose: Test if prediction residuals have universal distributional properties

Analysis:
1. Basic stats: Mean, Std, Skew, Kurtosis per asset
2. Tail index (Hill estimator)
3. KS test for distribution equality across assets
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import skew, kurtosis, kstest, norm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import json
import os
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK',
    'EFA', 'EEM',
    'TLT', 'IEF', 'TIP',
    'GLD', 'USO', 'SLV',
]

class UniversalMLP(nn.Module):
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
        
    def forward(self, x, asset_ids):
        embed = self.asset_embedding(asset_ids)
        x = torch.cat([x, embed], dim=1)
        return self.net(x).squeeze()

def hill_estimator(data, k=None):
    """Hill estimator for tail index"""
    data = np.abs(data)
    data = data[data > 0]
    data = np.sort(data)[::-1]
    
    if k is None:
        k = int(len(data) * 0.1)  # Top 10%
    
    if k < 2:
        return np.nan
    
    log_data = np.log(data[:k])
    alpha = 1.0 / (np.mean(log_data) - log_data[-1])
    return alpha

def run_residual_analysis():
    print("="*70)
    print("V4 Exp 06: Residual Distribution Analysis")
    print("="*70)
    
    # 1. Download and Train Model
    raw = yf.download(ASSETS + ['^VIX'], start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    label_encoder = LabelEncoder()
    label_encoder.fit(ASSETS)
    
    pooled_data = []
    feature_cols = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'LogRV_Mom', 'LogVIX']
    
    for asset in ASSETS:
        if asset not in close.columns:
            continue
        df = pd.DataFrame(index=close.index)
        ret = np.log(close[asset] / close[asset].shift(1))
        rv = ret.rolling(22).std() * np.sqrt(252) * 100
        
        df['LogRV'] = np.log(rv + 1e-6)
        df['LogRV_lag1'] = df['LogRV'].shift(1)
        df['LogRV_lag5'] = df['LogRV'].shift(5)
        df['LogRV_lag22'] = df['LogRV'].shift(22)
        df['LogRV_Mom'] = df['LogRV_lag1'] - df['LogRV_lag5']
        df['LogVIX'] = np.log(close['VIX'] + 1e-6).shift(1)
        df['Target'] = np.log(rv.shift(-22) + 1e-6)
        df['Asset_ID'] = label_encoder.transform([asset])[0]
        df = df.dropna()
        if len(df) > 100:
            pooled_data.append(df)
    
    full_df = pd.concat(pooled_data).reset_index(drop=True)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    full_df[feature_cols] = scaler_X.fit_transform(full_df[feature_cols])
    full_df['Target_Scaled'] = scaler_y.fit_transform(full_df[['Target']])
    
    # Train
    print("\nTraining model...")
    X = torch.tensor(full_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(full_df['Target_Scaled'].values, dtype=torch.float32)
    a = torch.tensor(full_df['Asset_ID'].values, dtype=torch.long)
    
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(full_df))[:10000]
        optimizer.zero_grad()
        pred = model(X[perm], a[perm])
        loss = criterion(pred, y[perm])
        loss.backward()
        optimizer.step()
    
    # 2. Compute Residuals
    print("\nComputing residuals...")
    model.eval()
    with torch.no_grad():
        predictions = model(X, a).numpy()
    
    full_df['Pred'] = predictions
    full_df['Residual'] = full_df['Target_Scaled'] - full_df['Pred']
    
    # Standardize residuals per asset
    residual_stats = []
    all_residuals = []
    
    for asset in ASSETS:
        asset_id = label_encoder.transform([asset])[0]
        mask = full_df['Asset_ID'] == asset_id
        
        if mask.sum() < 50:
            continue
        
        resid = full_df.loc[mask, 'Residual'].values
        
        # Standardize
        std_resid = (resid - resid.mean()) / resid.std()
        all_residuals.append(std_resid)
        
        stats = {
            'Asset': asset,
            'N': len(resid),
            'Mean': resid.mean(),
            'Std': resid.std(),
            'Skew': skew(resid),
            'Kurtosis': kurtosis(resid),
            'Hill_Alpha': hill_estimator(resid),
            'KS_pvalue': kstest(std_resid, 'norm')[1]
        }
        residual_stats.append(stats)
    
    df_stats = pd.DataFrame(residual_stats)
    
    # 3. Results
    print("\n" + "="*70)
    print("RESIDUAL DISTRIBUTION STATISTICS")
    print("="*70)
    print(df_stats[['Asset', 'Mean', 'Std', 'Skew', 'Kurtosis', 'Hill_Alpha']].round(4).to_string())
    
    # 4. Universality Test
    print("\n" + "="*70)
    print("UNIVERSALITY TEST")
    print("="*70)
    
    # Skewness consistency
    skew_std = df_stats['Skew'].std()
    kurt_std = df_stats['Kurtosis'].std()
    hill_std = df_stats['Hill_Alpha'].std()
    
    print(f"Skewness Std across assets: {skew_std:.4f}")
    print(f"Kurtosis Std across assets: {kurt_std:.4f}")
    print(f"Hill Alpha Std across assets: {hill_std:.4f}")
    
    # KS test: are all residuals from the same distribution?
    pooled_residuals = np.concatenate(all_residuals)
    ks_stat, ks_pval = kstest(pooled_residuals, 'norm')
    print(f"\nPooled Residuals KS test (vs Normal): stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
    
    # 5. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pooled histogram
    axes[0, 0].hist(pooled_residuals, bins=50, density=True, alpha=0.7, label='Pooled Residuals')
    x = np.linspace(-4, 4, 100)
    axes[0, 0].plot(x, norm.pdf(x), 'r--', label='Normal')
    axes[0, 0].set_title('Pooled Standardized Residuals')
    axes[0, 0].legend()
    
    # Per-asset overlaid
    for i, resid in enumerate(all_residuals[:5]):
        sns.kdeplot(resid, ax=axes[0, 1], label=ASSETS[i])
    axes[0, 1].set_title('Residual Distributions by Asset (Overlay)')
    axes[0, 1].legend()
    
    # Skewness comparison
    axes[1, 0].bar(df_stats['Asset'], df_stats['Skew'])
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title('Skewness by Asset')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Kurtosis comparison
    axes[1, 1].bar(df_stats['Asset'], df_stats['Kurtosis'])
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Excess Kurtosis by Asset')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v4_exp06_residuals.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    
    # 6. Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if skew_std < 0.3 and kurt_std < 1.0:
        print("✅ STRONG UNIVERSALITY: Residual distributions are very similar across assets")
    elif skew_std < 0.5:
        print("⚠️ MODERATE UNIVERSALITY: Residual shapes are roughly similar")
    else:
        print("❌ NO UNIVERSALITY: Residual distributions differ significantly")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp06_residuals.json'
    df_stats.to_json(out_path, orient='records', indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_residual_analysis()
