"""
그림 2. 전체 모델×기간 Pooled R² 히트맵 — 흑백 논문용
- 흑백 그레이스케일 (높을수록 진함)
- 셀 내 숫자 폰트 확대 (가독성 개선)
- 22d 열 강조 (두꺼운 테두리)
- 데이터: //wsl.localhost/Ubuntu-24.04/root/vrp/paper/csv/main_benchmark_v6_performance.csv
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. 데이터 로드
csv_path = '//wsl.localhost/Ubuntu-24.04/root/vrp/paper/csv/main_benchmark_v6_performance.csv'
raw = pd.read_csv(csv_path)

horizons = ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']

# 모델 순서: 22d Pooled_R2 기준 내림차순
order_22d = (raw[raw['Horizon'] == '22d']
             .sort_values('Pooled_R2', ascending=False)['Model'].tolist())

# ── 2. 피벗 (models × horizons)
df = raw.pivot(index='Model', columns='Horizon', values='Pooled_R2')
df = df[horizons]           # 열 순서 고정
df = df.reindex(order_22d)  # 행 순서: 22d 성능 내림차순
models_ordered = df.index.tolist()

# ── 3. 흑백 컬러맵 설정
# Greys_r: 낮을수록 진함 → 반전: 높을수록 진함 (화이트→블랙)
cmap = matplotlib.cm.get_cmap('Greys')  # 낮=흰, 높=검

vmin, vmax = -0.4, 0.85

# ── 4. 그래프 생성
fig, ax = plt.subplots(figsize=(13, 5.5))

# 히트맵 (imshow)
im = ax.imshow(df.values.astype(float),
               cmap=cmap, vmin=vmin, vmax=vmax,
               aspect='auto')

# ── 5. 축 설정
ax.set_xticks(range(len(horizons)))
ax.set_xticklabels(horizons, fontsize=13, fontweight='bold')
ax.set_yticks(range(len(models_ordered)))
ax.set_yticklabels(models_ordered, fontsize=13)

ax.set_xlabel('Forecast Horizon', fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel('Model', fontsize=13, fontweight='bold', labelpad=8)

# ── 6. 셀 내 숫자 표기 (폰트 확대)
for i, m in enumerate(models_ordered):
    for j, h in enumerate(horizons):
        val = df.loc[m, h]
        if np.isnan(val):
            txt = '—'
            color = '#999999'
            fontsize = 11
        elif val < 0:
            txt = f'{val:.3f}'
            color = '#FFFFFF'  # 진한 배경에 흰 텍스트
            fontsize = 12
        else:
            txt = f'{val:.3f}'
            # 배경이 진하면 흰 글씨, 밝으면 검은 글씨
            norm_val = (val - vmin) / (vmax - vmin)
            color = '#FFFFFF' if norm_val > 0.6 else '#000000'
            fontsize = 12

        ax.text(j, i, txt, ha='center', va='center',
                fontsize=fontsize, color=color, fontweight='bold')

# ── 7. 22d 열 강조 (두꺼운 테두리)
col_22d = horizons.index('22d')
rect = mpatches.FancyBboxPatch(
    (col_22d - 0.5, -0.5),
    1, len(models_ordered),
    boxstyle="square,pad=0",
    linewidth=2.5, edgecolor='black', facecolor='none',
    zorder=5
)
ax.add_patch(rect)

# ── 8. 그리드 (얇은 흰 선으로 셀 구분)
ax.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(models_ordered), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.2)
ax.tick_params(which='minor', bottom=False, left=False)

# ── 9. 컬러바
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Pooled R²', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

# ── 10. WEns 행 강조 (옅은 회색 배경)
wens_row = models_ordered.index('WEns')
ax.axhspan(wens_row - 0.5, wens_row + 0.5, color='#000000', alpha=0.07, zorder=0)

# ── 11. HAR-3 구분선 제거

# ── 12. 제목 및 주석
ax.set_title('Figure 2.  Model × Horizon Pooled R² Heatmap\n'
             '(bold column: 22d primary horizon)',
             fontsize=12, pad=12)

plt.tight_layout()
plt.savefig('kiit_paper/그림2_heatmap_bw.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('kiit_paper/그림2_heatmap_bw.pdf', bbox_inches='tight',
            facecolor='white')
print("저장 완료: 그림2_heatmap_bw.png / .pdf")
plt.show()
