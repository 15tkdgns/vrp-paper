import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

INPUT_JSON = '/root/vrp/data/processed/feature_importance/feature_importance_results.json'
FIG_DIR = '/root/vrp/experiments/figs/feature_importance'
TAB_DIR = '/root/vrp/experiments/tables/feature_importance'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

horizons = ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']
groups = data['feature_groups']
feature_to_group = {f: g for g, fs in groups.items() for f in fs}

def get_group(feat):
    return feature_to_group.get(feat, 'Other')

# ==========================================
# O1: PFI Heatmap (Ridge, top 20 overall)
# ==========================================
pfi_data = []
for h in horizons:
    pfi_h = data[h]['layer2']['pfi_ridge']
    for f, imp in pfi_h.items():
        pfi_data.append({'Horizon': h, 'Feature': f, 'PFI': imp})

df_pfi = pd.DataFrame(pfi_data)
# Find top 20 features by average PFI across horizons
top20 = df_pfi.groupby('Feature')['PFI'].mean().nlargest(20).index
df_pfi_top = df_pfi[df_pfi['Feature'].isin(top20)]

pivot_pfi = df_pfi_top.pivot(index='Feature', columns='Horizon', values='PFI')
pivot_pfi = pivot_pfi[horizons] # order columns

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_pfi, cmap='YlGnBu', annot=False, fmt='.3f', cbar_kws={'label': 'R² drop'})
plt.title('Permutation Feature Importance (Ridge) Across Horizons')
plt.ylabel('Feature')
plt.xlabel('Prediction Horizon')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'O1_PFI_Heatmap.png'))
plt.close()

# ==========================================
# O2: Model-specific Top-10 at 22d
# ==========================================
h22 = data['22d']['layer1']
ridge_imp = h22['ridge_coeff']
xgb_imp = h22['xgb_gain']
rf_imp = h22['rf_mdi']

def get_top10(imp_dict):
    return sorted(imp_dict.keys(), key=lambda k: imp_dict[k], reverse=True)[:10]

top10_ridge = get_top10(ridge_imp)
top10_xgb = get_top10(xgb_imp)
top10_rf = get_top10(rf_imp)

df_models = pd.DataFrame({
    'Rank': range(1, 11),
    'Ridge (Std Coeff)': top10_ridge,
    'XGBoost (Gain)': top10_xgb,
    'Random Forest (MDI)': top10_rf
})
with open(os.path.join(TAB_DIR, 'O2_Top10_22d.tex'), 'w') as f:
    f.write(df_models.to_latex(index=False, column_format='lccc'))

# ==========================================
# O3: Feature Group Contribution (1/MeanRank for proportion)
# ==========================================
group_ranks = {g: [] for g in groups.keys()}
for h in horizons:
    mr = data[h]['layer3']['mean_ranks']
    
    # Calculate average (1/rank) for each group as importance
    h_group_imp = {g: 0.0 for g in groups.keys()}
    for f, r in mr.items():
        g = get_group(f)
        if g in groups:
            h_group_imp[g] +=(1.0 / r)
            
    # Normalize to 100% per horizon
    total = sum(h_group_imp.values())
    for g in groups.keys():
        group_ranks[g].append(h_group_imp[g] / total * 100)

plt.figure(figsize=(10, 6))
# Area chart
x = range(len(horizons))
y = [group_ranks[g] for g in groups.keys()]
plt.stackplot(x, y, labels=groups.keys(), alpha=0.8, colors=sns.color_palette("Set2"))
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(x, horizons)
plt.title('Feature Group Contribution Across Horizons (based on Rank Inverse)')
plt.ylabel('Relative Contribution (%)')
plt.xlabel('Prediction Horizon')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'O3_Group_Contribution.png'))
plt.close()

# ==========================================
# O4: Kendall's W Table
# ==========================================
kendall_data = []
for h in horizons:
    l3 = data[h]['layer3']
    kendall_data.append({
        'Horizon': h,
        "Kendall's W": l3['kendall_w'],
        'Chi2': l3['kendall_chi2'],
        'p-value': l3['kendall_p']
    })
df_k = pd.DataFrame(kendall_data)
with open(os.path.join(TAB_DIR, 'O4_Kendalls_W.tex'), 'w') as f:
    f.write(df_k.to_latex(index=False, float_format="%.3f"))

# ==========================================
# O5: Spearman Cross-Horizon Heatmap
# ==========================================
cross_sp = data['cross_horizon_spearman']
sp_matrix = pd.DataFrame(index=horizons, columns=horizons, dtype=float)
for k, v in cross_sp.items():
    h1, h2 = k.split('_vs_')
    sp_matrix.loc[h1, h2] = v

plt.figure(figsize=(8, 6))
sns.heatmap(sp_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=0.5, vmax=1.0)
plt.title('Cross-Horizon Rank Stability (Spearman ρ)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'O5_Cross_Horizon_Spearman.png'))
plt.close()

# ==========================================
# O6: Bootstrap CI Tables (1d, 5d, 22d, 60d, 120d, 252d)
# ==========================================
for h in ['1d', '22d', '60d', '252d']:
    mr = data[h]['layer3']['mean_ranks']
    ci = data[h]['layer3']['bootstrap_ci']
    
    # Sort features by mean rank
    sorted_feats = sorted(mr.keys(), key=lambda k: mr[k])[:15]
    
    t_data = []
    for rank, f in enumerate(sorted_feats):
        t_data.append({
            'Consensus Rank': rank + 1,
            'Feature': f.replace('_', '\\_'),
            'Group': get_group(f).replace('_', '\\_'),
            'Mean Rank': round(mr[f], 1),
            '95\\% CI Lower': round(ci[f]['ci_lower'], 1),
            '95\\% CI Upper': round(ci[f]['ci_upper'], 1),
        })
    df_ci = pd.DataFrame(t_data)
    with open(os.path.join(TAB_DIR, f'O6_BootstrapCI_{h}.tex'), 'w') as f_out:
        f_out.write(df_ci.to_latex(index=False, escape=False))

print("Feature importance figures and tables generated successfully")
