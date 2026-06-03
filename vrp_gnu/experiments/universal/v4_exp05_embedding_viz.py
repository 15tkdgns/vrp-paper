"""
V4 Exp 05: Asset Embedding Visualization & Analysis
Purpose: Visualize learned embeddings and test if they encode financial characteristics

Analysis:
1. t-SNE / PCA visualization of asset embeddings
2. Cluster analysis (do embeddings separate by asset class?)
3. Correlation with traditional risk characteristics
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

ASSET_CLASSES = {
    'US_Equity': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV'],
    'Global_Equity': ['EFA', 'EEM'],
    'Bonds': ['TLT', 'IEF', 'TIP'],
    'Commodities': ['GLD', 'USO', 'SLV', 'DBC'],
}

CLASS_COLORS = {
    'US_Equity': 'blue',
    'Global_Equity': 'green', 
    'Bonds': 'red',
    'Commodities': 'orange'
}

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

def get_asset_class(asset):
    for cls, assets in ASSET_CLASSES.items():
        if asset in assets:
            return cls
    return 'Unknown'

def run_embedding_analysis():
    print("="*70)
    print("V4 Exp 05: Asset Embedding Visualization")
    print("="*70)
    
    device = torch.device('cpu')
    
    # 1. Download Data and Train Model
    raw = yf.download(ASSETS + ['^VIX'], start='2010-01-01', end='2025-01-01', progress=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close.columns = [c.replace('^', '') for c in close.columns]
    close = close.ffill()
    
    # Prepare data
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
    
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])
    full_df['Target_Scaled'] = StandardScaler().fit_transform(full_df[['Target']])
    
    # Train model
    print("\nTraining model to learn embeddings...")
    X = torch.tensor(full_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(full_df['Target_Scaled'].values, dtype=torch.float32)
    a = torch.tensor(full_df['Asset_ID'].values, dtype=torch.long)
    
    model = UniversalMLP(n_features=len(feature_cols), n_assets=len(ASSETS), embed_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(full_df))[:10000]  # Sample for speed
        optimizer.zero_grad()
        pred = model(X[perm], a[perm])
        loss = criterion(pred, y[perm])
        loss.backward()
        optimizer.step()
    
    # 2. Extract Embeddings
    print("\nExtracting learned embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model.asset_embedding.weight.cpu().numpy()
    
    embed_df = pd.DataFrame(embeddings, columns=['E1', 'E2', 'E3', 'E4'])
    embed_df['Asset'] = label_encoder.inverse_transform(range(len(ASSETS)))
    embed_df['Class'] = embed_df['Asset'].apply(get_asset_class)
    
    print("\nLearned Embeddings (4D):")
    print(embed_df.round(3))
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 3.1 PCA Projection
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    
    for cls, color in CLASS_COLORS.items():
        mask = embed_df['Class'] == cls
        axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                       c=color, label=cls, s=100)
        for i, asset in enumerate(embed_df.loc[mask, 'Asset']):
            axes[0].annotate(asset, (pca_coords[mask.values][sum(mask[:i+1])-1, 0], 
                                     pca_coords[mask.values][sum(mask[:i+1])-1, 1]))
    
    axes[0].set_title('PCA Projection of Asset Embeddings')
    axes[0].legend()
    
    # 3.2 t-SNE (if enough assets)
    if len(embeddings) >= 5:
        tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42)
        tsne_coords = tsne.fit_transform(embeddings)
        
        for cls, color in CLASS_COLORS.items():
            mask = embed_df['Class'] == cls
            axes[1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                           c=color, label=cls, s=100)
        
        axes[1].set_title('t-SNE Projection of Asset Embeddings')
        axes[1].legend()
    
    # 3.3 Embedding Heatmap
    im = axes[2].imshow(embeddings, cmap='RdBu', aspect='auto')
    axes[2].set_yticks(range(len(ASSETS)))
    axes[2].set_yticklabels(ASSETS)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(['E1', 'E2', 'E3', 'E4'])
    axes[2].set_title('Embedding Weights Heatmap')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plot_path = 'experiments/07_v2_methodology/results/v4_exp05_embeddings.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")
    
    # 4. Cluster Analysis
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    embed_df['Cluster'] = kmeans.fit_predict(embeddings)
    
    print("\nClustering Results (K=4):")
    cluster_summary = embed_df.groupby('Cluster')['Asset'].apply(list)
    for cluster, assets in cluster_summary.items():
        # Get dominant class in cluster
        classes = [get_asset_class(a) for a in assets]
        dominant = max(set(classes), key=classes.count)
        print(f"  Cluster {cluster}: {assets} → Dominant: {dominant}")
    
    # Check if clusters match asset classes
    from sklearn.metrics import adjusted_rand_score
    class_labels = [list(ASSET_CLASSES.keys()).index(get_asset_class(a)) for a in ASSETS]
    ari = adjusted_rand_score(class_labels, embed_df['Cluster'].values)
    print(f"\nAdjusted Rand Index (vs True Classes): {ari:.4f}")
    
    if ari > 0.5:
        print("✅ STRONG: Embeddings naturally separate asset classes")
    elif ari > 0.2:
        print("⚠️ MODERATE: Partial separation of asset classes")
    else:
        print("❌ WEAK: Embeddings do not align with asset classes")
    
    # Save
    out_path = 'experiments/07_v2_methodology/results/v4_exp05_embeddings.json'
    embed_df.to_json(out_path, orient='records', indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run_embedding_analysis()
