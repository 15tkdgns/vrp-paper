import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 1: Research Framework (Two-Stage RV/VRP Prediction)
# -------------------------------------------------------------
def draw_figure_1():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    boxes = [
        ("Input Data\n(Daily Price, Volume,\nVIX Indicators, etc.)", (0.1, 0.5)),
        ("Feature Engineering\n(Multi-scale / Types)", (0.35, 0.5)),
        ("Stage 1\nRV Prediction", (0.55, 0.5)),
        ("IV Combination\n(RV Forecast + IV)", (0.75, 0.5)),
        ("VRP Derivation", (0.95, 0.5))
    ]

    for i, (text, pos) in enumerate(boxes):
        ax.add_patch(patches.Rectangle((pos[0]-0.09, pos[1]-0.2), 0.18, 0.4, fill=True, color='#E1F5FE', ec='#0288D1', lw=2))
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')

        if i < len(boxes) - 1:
            next_pos = boxes[i+1][1]
            ax.annotate('', xy=(next_pos[0]-0.09, pos[1]), xytext=(pos[0]+0.09, pos[1]),
                        arrowprops=dict(arrowstyle="->", lw=2, color='#555555'))

    plt.title("Figure 1. Research Framework (Two-Stage RV/VRP Prediction)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/Figure_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 2: Time-Series Validation with Purge Gap
# -------------------------------------------------------------
def draw_figure_2():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    ax.plot([0, 10], [0, 0], color='black', lw=2)

    segments = [
        ("Train (80%)", 0, 4, '#81C784'),
        ("Purge Gap\n(h days)", 4, 4.8, '#E0E0E0'),
        ("Test (20%)", 4.8, 9, '#FF8A65')
    ]
    for text, start, end, color in segments:
        ax.add_patch(patches.Rectangle((start, -0.2), end-start, 0.4, fill=True, color=color, alpha=0.9, ec='black'))
        ax.text((start+end)/2, 0, text, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.annotate('Prevents target window [t+1, t+h] overlap and look-ahead bias', xy=(4.4, 0.25), xytext=(2.5, 0.7),
                arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3"), fontsize=11)

    plt.title("Figure 2. Time-Series Validation with Purge Gap", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/Figure_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 3: Model Performance by Horizon (Pooled R²)
# -------------------------------------------------------------
def draw_figure_3():
    horizons = [1, 5, 22, 60, 90, 120, 180, 252]
    data = {
        'WEns':      [0.117, 0.624, 0.797, 0.789, 0.767, 0.750, 0.743, 0.703],
        'Ridge':     [0.115, 0.619, 0.794, 0.787, 0.766, 0.750, 0.736, 0.674],
        'Lasso':     [0.116, 0.624, 0.789, 0.780, 0.763, 0.744, 0.705, 0.637],
        'ElasticNet':[0.115, 0.625, 0.790, 0.779, 0.761, 0.742, 0.700, 0.627],
        'XGBoost':   [0.111, 0.610, 0.762, 0.740, 0.709, 0.684, 0.680, 0.674],
        'LightGBM':  [0.111, 0.610, 0.766, 0.748, 0.711, 0.688, 0.677, 0.668],
        'RF':        [0.114, 0.613, 0.766, 0.746, 0.722, 0.735, 0.720, 0.692],
        'BiLSTM-A':  [-0.020, -0.044, -0.267, -0.330, -0.325, -0.198, -0.073, -0.110],
        'HAR-3':     [0.109, 0.581, 0.761, 0.784, 0.781, 0.758, 0.705, 0.639],
        'MLP':       [0.088, 0.521, 0.609, 0.592, 0.541, 0.536, 0.566, 0.540],
        'SVR':       [0.096, 0.629, 0.787, 0.785, 0.758, 0.724, 0.668, 0.588]
    }

    plt.figure(figsize=(10, 6))
    for m, vals in data.items():
        lw = 3.5 if m == 'WEns' else 1.5
        ms = 8 if m == 'WEns' else 6
        plt.plot(horizons, vals, marker='o', label=m, linewidth=lw, markersize=ms)

    plt.title("Figure 3. Model Performance by Horizon (Pooled $R^2$)", fontsize=14, fontweight='bold')
    plt.xlabel("Forecast Horizon (Trading Days)", fontsize=12)
    plt.ylabel("Out-of-Sample Pooled $R^2$", fontsize=12)
    plt.xticks(horizons)
    plt.legend(title="Model", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/Figure_3.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 4: Model Performance Comparison — 22-Day Horizon
# -------------------------------------------------------------
def draw_figure_4():
    models = ['WEns', 'Ridge', 'ElasticNet', 'Lasso', 'SVR', 'RF', 'LightGBM', 'XGBoost', 'MLP', 'BiLSTM-A']
    pooled_r2 = [0.797, 0.794, 0.790, 0.789, 0.787, 0.766, 0.766, 0.762, 0.609, -0.267]
    median_r2 = [0.215, 0.198, 0.180, 0.201, 0.180, 0.230, 0.195, 0.190, 0.150, -3.000]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, pooled_r2, width, label='Pooled $R^2$', color='#5C6BC0')
    rects2 = ax.bar(x + width/2, median_r2, width, label='Median $R^2$', color='#EF5350')

    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_title('Figure 4. Model Performance Comparison — 22-Day Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/Figure_4.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 5: Cumulative Feature Group Contribution (Ablation, 22d)
# -------------------------------------------------------------
def draw_figure_5():
    features = ['HAR-3\n(Baseline)', '+ GARCH/\nVol-of-Vol', '+ HF Proxy', '+ IV Surface', '+ Alternative\nData']
    contributions = [0.732, 0.0178, 0.0206, 0.0259, 0.0067]
    cumulative = np.cumsum(contributions)
    starts = [0] + list(cumulative[:-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#BDBDBD'] + ['#66BB6A'] * 4

    for i in range(len(features)):
        ax.bar(features[i], contributions[i], bottom=starts[i], color=colors[i], edgecolor='black', width=0.6)

        y_text = starts[i] + contributions[i]/2
        if i == 0:
            text = f"{contributions[i]:.3f}"
        else:
            text = f"+{contributions[i]:.4f}"
            y_text = starts[i] + contributions[i] + 0.005

        ax.text(features[i], y_text, text, ha='center', va='bottom' if i > 0 else 'center',
                color='black' if i == 0 else 'darkgreen', fontweight='bold', fontsize=11)

    for i in range(len(features)-1):
        ax.plot([i, i+1], [cumulative[i], cumulative[i]], color='black', linestyle='--', alpha=0.5)

    ax.set_ylabel('Cumulative Pooled $R^2$', fontsize=12)
    ax.set_title('Figure 5. Cumulative Feature Group Contribution (Ablation, 22d)', fontsize=14, fontweight='bold')
    plt.ylim(0, 0.85)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('figures/Figure_5.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 6: Feature Importance Consensus by Horizon
# -------------------------------------------------------------
def draw_figure_6():
    horizons = ['1d', '5d', '22d', '60d', '120d', '252d']
    features = ['Parkinson_5', 'Garch_Daily', 'GarmanKlass_22', 'RogersSatchell_22', 'Garch_Weekly', 'IV_VRP_ma22']

    data = np.array([
        [0.95, 0.85, 0.40, 0.20, 0.10, 0.10],
        [0.90, 0.80, 0.50, 0.30, 0.20, 0.10],
        [0.50, 0.70, 0.90, 0.85, 0.80, 0.75],
        [0.40, 0.65, 0.95, 0.90, 0.85, 0.80],
        [0.30, 0.50, 0.85, 0.95, 0.90, 0.85],
        [0.10, 0.20, 0.40, 0.70, 0.90, 0.95],
    ])

    plt.figure(figsize=(11, 6))
    sns.heatmap(data, annot=False, cmap='Blues', xticklabels=horizons, yticklabels=features, linewidths=.5)
    plt.title('Figure 6. Feature Importance Consensus by Horizon (Normalized Importance)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Forecast Horizon', fontsize=12)
    plt.ylabel('Key Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/Figure_6.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 7: Pooled R² by VIX Regime
# -------------------------------------------------------------
def draw_figure_7():
    vix_regimes = ['Low VIX\n(Bottom 50%)', 'Mid VIX\n(50-90%)', 'High VIX\n(Top 10%)']
    har3 = [0.739, 0.698, 0.602]
    wens = [0.781, 0.722, 0.765]

    x = np.arange(len(vix_regimes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, har3, width, label='HAR-3', color='#9E9E9E', edgecolor='black')
    rects2 = ax.bar(x + width/2, wens, width, label='WEns', color='#FFCA28', edgecolor='black')

    ax.set_ylabel('Pooled $R^2$', fontsize=12)
    ax.set_title('Figure 7. Pooled R² by VIX Regime — HAR-3 vs. WEns (22d)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(vix_regimes, fontsize=11)
    ax.legend(fontsize=11)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/Figure_7.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 8: Feature Importance Consensus — Kendall's W by Horizon
# -------------------------------------------------------------
def draw_figure_8():
    horizons = [1, 5, 22, 60, 90, 120, 180, 252]
    kendall_w = np.array([0.480, 0.577, 0.588, 0.530, 0.510, 0.490, 0.450, 0.420])

    plt.figure(figsize=(9, 5))
    plt.plot(horizons, kendall_w, marker='D', color='#8E24AA', linestyle='-', linewidth=2.5, markersize=8)

    plt.title("Figure 8. Feature Importance Consensus — Kendall's W by Horizon", fontsize=14, fontweight='bold')
    plt.xlabel("Forecast Horizon (Trading Days)", fontsize=12)
    plt.ylabel("Kendall's W", fontsize=12)
    plt.xticks(horizons)
    plt.ylim(0.3, 0.7)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.annotate('High agreement at short-to-mid horizons (22d: 0.588)', xy=(22, 0.588), xytext=(40, 0.65),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, bbox=dict(boxstyle="round4,pad=0.3", fc="yellow", ec="darkorange", lw=1, alpha=0.5))

    plt.tight_layout()
    plt.savefig('figures/Figure_8.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 10: Per-Asset R² Distribution by Asset Class (22d, WEns)
# -------------------------------------------------------------
def draw_figure_10():
    eq_assets   = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
    bond_assets = ['TLT', 'IEF', 'AGG']
    comm_assets = ['GLD', 'SLV', 'USO']

    eq_r2   = [0.28, 0.30, 0.24, 0.21, 0.18]
    bond_r2 = [0.10, 0.08, 0.04]
    comm_r2 = [0.15, 0.13, 0.09]

    data = []
    for a, r in zip(eq_assets,   eq_r2):   data.append({'Asset Class': 'Equities',    'Asset': a, 'R2': r})
    for a, r in zip(bond_assets, bond_r2): data.append({'Asset Class': 'Bonds',       'Asset': a, 'R2': r})
    for a, r in zip(comm_assets, comm_r2): data.append({'Asset Class': 'Commodities', 'Asset': a, 'R2': r})

    df = pd.DataFrame(data)

    plt.figure(figsize=(9, 6))
    sns.boxplot(x='Asset Class', y='R2', data=df, palette='Pastel1', width=0.4)
    sns.stripplot(x='Asset Class', y='R2', hue='Asset', data=df, size=8, jitter=False, alpha=0.8)

    plt.title("Figure 10. Per-Asset $R^2$ Distribution by Asset Class (22d, WEns)", fontsize=14, fontweight='bold')
    plt.xlabel("Asset Class", fontsize=12)
    plt.ylabel("Per-Asset $R^2$", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/Figure_10.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_figure_1()
    draw_figure_2()
    draw_figure_3()
    draw_figure_4()
    draw_figure_5()
    draw_figure_6()
    draw_figure_7()
    draw_figure_8()
    draw_figure_10()
    print("All figures (except Figure 9) saved to 'figures/'.")
