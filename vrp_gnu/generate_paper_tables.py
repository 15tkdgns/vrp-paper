"""
Paper tables and figures generation script (English labels)
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/root/vrp/paper/figs'
CSV_DIR = '/root/vrp/paper/csv'
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']
MODELS_ORDER = ['HAR-3', 'Ridge', 'ENet', 'LASSO', 'RF', 'XGBoost', 'LightGBM', 'BiLSTM-A', 'MLP', 'WEns']

# ─────────────────────────────────────────────
# Fig. A: Model × Horizon Pooled R² Heatmap
# ─────────────────────────────────────────────
perf = pd.read_csv(f'{CSV_DIR}/main_benchmark_v6_performance.csv')

pivot = perf.pivot(index='Model', columns='Horizon', values='Pooled_R2')
pivot = pivot.reindex(index=[m for m in MODELS_ORDER if m in pivot.index],
                      columns=HORIZONS)

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(pivot.values.astype(float), cmap='RdYlGn', aspect='auto',
               vmin=0.0, vmax=0.85)

ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels(HORIZONS, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index.tolist(), fontsize=11)

for i in range(len(pivot.index)):
    for j in range(len(HORIZONS)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.25 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8.5, color=color)

plt.colorbar(im, ax=ax, label='Pooled R²')
ax.set_title('Fig. 1 — Model × Horizon Pooled R² Heatmap', fontsize=12, pad=10)
ax.set_xlabel('Forecast Horizon', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_A_pooled_r2_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig A saved.")

# ─────────────────────────────────────────────
# Table B: 22d Model Performance Summary
# ─────────────────────────────────────────────
df22 = perf[perf['Horizon'] == '22d'].copy()
df22 = df22.set_index('Model').reindex(MODELS_ORDER)
df22 = df22[['Pooled_R2', 'Median_R2', 'RMSE']]

lines_B = []
lines_B.append('[Table 7] Model Performance Comparison — 22-day Horizon')
lines_B.append(f"{'Model':<12} | {'Pooled R2':>10} | {'Median R2':>10} | {'RMSE':>8}")
lines_B.append('-' * 50)
for model, row in df22.iterrows():
    if pd.notna(row['Pooled_R2']):
        lines_B.append(f"{model:<12} | {row['Pooled_R2']:>10.4f} | {row['Median_R2']:>10.4f} | {row['RMSE']:>8.4f}")

table_B = '\n'.join(lines_B)
print("\n" + table_B)
with open(f'{OUT_DIR}/table_B_22d_performance.txt', 'w') as f:
    f.write(table_B)

# ─────────────────────────────────────────────
# Table C: Feature Group Subset Experiment
# ─────────────────────────────────────────────
with open(f'{CSV_DIR}/v71_model_comparison_results.json') as f:
    mc = json.load(f)

with open(f'{CSV_DIR}/feature_subset_v6.json') as f:
    subset_v6 = json.load(f)

subset_order = ['HAR (3)', 'Range only (5)', 'IV Surface (10)', 'HAR + Range (8)', 'Full (35)']
subset_labels = {
    'HAR (3)':          'HAR (3 features)',
    'Range only (5)':   'Range only (5)',
    'IV Surface (10)':  'IV Surface only (10)',
    'HAR + Range (8)':  'HAR + Range (8)',
    'Full (35)':        'Full (35 features)',
}

lines_C = []
lines_C.append('[Table 8] Feature Group Subset Experiment — 22-day Horizon (Pooled R²)')
lines_C.append(f"{'Feature Group':<22} | {'N':>4} | {'Pooled R2':>10} | {'Median R2':>10}")
lines_C.append('-' * 56)
for key in subset_order:
    if key in subset_v6:
        d = subset_v6[key]
        label = subset_labels.get(key, key)
        lines_C.append(f"{label:<22} | {d['n_features']:>4} | {d['pooled_r2']:>10.4f} | {d['median_r2']:>10.4f}")

table_C = '\n'.join(lines_C)
print("\n" + table_C)
with open(f'{OUT_DIR}/table_C_feature_subset.txt', 'w') as f:
    f.write(table_C)

# ─────────────────────────────────────────────
# Table D: Group Removal Experiment ΔR²
# ─────────────────────────────────────────────
removal_data = [
    ('IV Surface (10 feat.)', 0.8036, 0.8036 - 0.0164, 0.0164),
    ('HF Proxy (7 feat.)',    0.8036, 0.8036 - 0.0153, 0.0153),
    ('Alternative (6 feat.)', 0.8036, 0.8036 - 0.0016, 0.0016),
]

lines_D = []
lines_D.append('[Table 9] WEns 22d — Feature Group Removal Experiment (Delta Pooled R²)')
lines_D.append(f"{'Removed Group':<22} | {'Full R2':>8} | {'w/o Group':>10} | {'Delta R2':>10}")
lines_D.append('-' * 58)
for label, full, removed, delta in removal_data:
    lines_D.append(f"{label:<22} | {full:>8.4f} | {removed:>10.4f} | {delta:>+10.4f}")

table_D = '\n'.join(lines_D)
print("\n" + table_D)
with open(f'{OUT_DIR}/table_D_group_removal.txt', 'w') as f:
    f.write(table_D)

# ─────────────────────────────────────────────
# Fig. E: Feature Importance Consensus Top-10 (22d)
# ─────────────────────────────────────────────
with open(f'{CSV_DIR}/feature_analysis_v6.json') as f:
    fi_data = json.load(f)

ranked = fi_data['hz_importance']['22d']  # list of [feat, mean_rank]
feat_names = [r[0] for r in ranked]
mean_ranks = [r[1] for r in ranked]

def get_color(feat):
    if any(x in feat for x in ['RogersSatchell', 'Parkinson', 'GarmanKlass', 'Range', 'Overnight']):
        return '#2196F3'
    elif any(x in feat for x in ['IV_', 'VIX']):
        return '#FF9800'
    else:
        return '#4CAF50'

# Keep ascending order: high rank (less important) at top, low rank (most important) at bottom
colors = [get_color(f) for f in feat_names]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(feat_names)), mean_ranks, color=colors, height=0.6)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=10)
ax.set_xlabel('Mean Rank (lower = more important)', fontsize=10)
ax.set_title('Fig. 3 — Feature Importance Consensus Top-10\n(22d, avg. of 3 methods)', fontsize=11)
ax.axvline(x=0, color='black', linewidth=0.5)

# Add value labels on each bar
for i, (bar, val) in enumerate(zip(bars, mean_ranks)):
    ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', ha='left', fontsize=9, color='#333333')

ax.set_xlim(0, max(mean_ranks) * 1.15)

legend_elements = [
    Patch(facecolor='#2196F3', label='HF Proxy (Range-based)'),
    Patch(facecolor='#FF9800', label='IV Surface'),
    Patch(facecolor='#4CAF50', label='HAR/Base'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_E_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig E saved.")

# ─────────────────────────────────────────────
# Table F: DM Test Results (WEns vs baselines)
# ─────────────────────────────────────────────
# Fig. I: Feature Importance Heatmap (all horizons × all features, score-based)
# ─────────────────────────────────────────────
hz_imp = fi_data['hz_importance']

# Load score dicts for each horizon (consensus importance share × 100)
score_dicts = {hk: fi_data['hz_importance'][f'{hk}_scores'] for hk in HORIZONS}

# Sort features by 22d score descending, show top 20
feat_22d_scores = score_dicts['22d']
feat_order = sorted(feat_22d_scores, key=lambda f: feat_22d_scores[f], reverse=True)[:20]

# Build matrix: rows = features, cols = horizons
matrix = np.array([[score_dicts[hk].get(f, 0.0) for hk in HORIZONS]
                   for f in feat_order])

fig, ax = plt.subplots(figsize=(11, len(feat_order) * 0.42 + 1.2))
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix.max())

ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels(HORIZONS, fontsize=10)
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order, fontsize=9)

for i in range(len(feat_order)):
    for j in range(len(HORIZONS)):
        val = matrix[i, j]
        color = 'white' if val > matrix.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Importance Share (%, avg. of 3 methods)', fontsize=9)

ax.set_title('Fig. 4 — Feature Importance Consensus Heatmap\n(top 20 by 22d score, 3 methods avg.)', fontsize=11, pad=10)
ax.set_xlabel('Forecast Horizon', fontsize=10)
ax.set_ylabel('Feature', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_I_feature_importance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig I saved.")

# ─────────────────────────────────────────────
dm = pd.read_csv(f'{CSV_DIR}/dm_corrected.csv')
dm_wens = dm[dm['Challenger'] == 'WEns'].copy()
dm_wens = dm_wens[dm_wens['Reference'].isin(['HAR-3', 'NaiveRV', 'GARCH', 'IV-only'])]

horizon_sort = {'1d':1,'5d':2,'22d':3,'60d':4,'90d':5,'120d':6,'180d':7,'252d':8}
dm_wens = dm_wens.copy()
dm_wens['h_order'] = dm_wens['Horizon'].map(horizon_sort)
dm_wens = dm_wens.sort_values(['h_order', 'Reference'])

lines_F = []
lines_F.append('[Table 11] DM Test Results — WEns vs. Baselines (BH correction)')
lines_F.append(f"{'Horizon':<8} | {'Baseline':<10} | {'DM stat':>8} | {'p(raw)':>8} | {'p(BH)':>8} | {'Sig.(BH)':>9}")
lines_F.append('-' * 62)
for _, row in dm_wens.iterrows():
    p_bh_str  = '<0.001' if row['p_BH']  < 0.001 else f"{row['p_BH']:.3f}"
    p_raw_str = '<0.001' if row['p_raw'] < 0.001 else f"{row['p_raw']:.3f}"
    sig = 'Yes' if row['Sig_BH'] else 'No'
    lines_F.append(f"{row['Horizon']:<8} | {row['Reference']:<10} | {row['DM_stat']:>8.3f} | {p_raw_str:>8} | {p_bh_str:>8} | {sig:>9}")

table_F = '\n'.join(lines_F)
print("\n" + table_F)
with open(f'{OUT_DIR}/table_F_dm_test.txt', 'w') as f:
    f.write(table_F)

# ─────────────────────────────────────────────
# Table G: High-Volatility Regime Performance
# ─────────────────────────────────────────────
high_vol_data = [
    ('VIX Bottom 50%', 0.739, 0.781),
    ('VIX 50-90%',     0.698, 0.722),
    ('VIX Top 10%',    0.602, 0.765),
]

lines_G = []
lines_G.append('[Table 10] Pooled R² by VIX Regime — HAR-3 vs. WEns (22d)')
lines_G.append(f"{'VIX Regime':<16} | {'HAR-3':>8} | {'WEns':>8} | {'Improvement':>12}")
lines_G.append('-' * 52)
for label, har, wens in high_vol_data:
    lines_G.append(f"{label:<16} | {har:>8.3f} | {wens:>8.3f} | {wens-har:>+12.3f}")

table_G = '\n'.join(lines_G)
print("\n" + table_G)
with open(f'{OUT_DIR}/table_G_high_vol.txt', 'w') as f:
    f.write(table_G)

# ─────────────────────────────────────────────
# Fig. H: R²_Within Trend by Horizon
# ─────────────────────────────────────────────
r2w = pd.read_csv(f'{CSV_DIR}/r2_within.csv')
wens_r2w = r2w[r2w['Model'] == 'WEns'].copy()
wens_r2w = wens_r2w.set_index('Horizon').reindex(HORIZONS)

pooled_vals = wens_r2w['Pooled_R2'].values
within_vals = wens_r2w['Within_R2'].values
x_idx = range(len(HORIZONS))

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x_idx, pooled_vals, 'o-', color='#2196F3',
        linewidth=2, markersize=7, label='Pooled R²')
ax.plot(x_idx, within_vals, 's--', color='#F44336',
        linewidth=2, markersize=7, label='R²_Within')
ax.fill_between(x_idx,
                pooled_vals,
                within_vals,
                alpha=0.12, color='gray', label='Cross-sectional contribution')
ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')

# Add value labels on each data point
for i, (p, w) in enumerate(zip(pooled_vals, within_vals)):
    if not np.isnan(p):
        ax.text(i, p + 0.018, f'{p:.3f}', ha='center', va='bottom',
                fontsize=8, color='#1565C0')
    if not np.isnan(w):
        offset = -0.025 if w >= 0 else 0.018
        va = 'top' if w >= 0 else 'bottom'
        ax.text(i, w + offset, f'{w:.3f}', ha='center', va=va,
                fontsize=8, color='#C62828')

ax.set_xticks(list(x_idx))
ax.set_xticklabels(HORIZONS, fontsize=11)
ax.set_ylabel('R²', fontsize=11)
ax.set_xlabel('Forecast Horizon', fontsize=11)
ax.set_title('Fig. 2 — WEns: Pooled R² vs. R²_Within by Horizon', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_H_r2_within_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig H saved.")

print("\nAll tables and figures generated.")
print(f"Output: {OUT_DIR}")
