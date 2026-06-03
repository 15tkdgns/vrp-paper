"""
RepeatRV baseline computation (Windows-compatible, no xgboost needed).
Loads per-asset parquet files from vrp_gnu/data/ and computes:
  - NaiveRV   : horizon-matched trailing rolling mean (already in baseline_comparison_results.json)
  - RepeatRV  : fixed 22d trailing rolling mean for all horizons (new)

Outputs:
  results/repeat_rv_results.json
  paper/table/baseline_comparison_pooled_r2.csv  (updated with RepeatRV row)
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
OUTER_TRAIN_RATIO = 0.8

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
RES_DIR   = os.path.join(BASE_DIR, 'results')
TABLE_DIR = os.path.join(BASE_DIR, '..', 'paper', 'table')


def load_asset(asset):
    p = os.path.join(DATA_DIR, f'{asset}.parquet')
    df = pd.read_parquet(p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
    return df[cols].ffill()


def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)


def pooled_r2(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(r2_score(y_true[valid], y_pred[valid]))


def per_asset_r2(df_test, y_true, y_pred):
    scores = []
    for a in df_test['Asset'].unique():
        m = (df_test['Asset'] == a).values
        if m.sum() < 2:
            continue
        scores.append(float(r2_score(y_true[m], y_pred[m])))
    return float(np.median(scores)), float(np.mean(scores))


print("Loading assets...")
asset_frames = {}
for asset in ALL_ASSETS:
    df = load_asset(asset)
    close = df.get('Adj Close', df.get('Close')).squeeze()
    ret   = np.log(close / close.shift(1))
    ret_sq = ret ** 2
    d = pd.DataFrame({
        'ret_sq': ret_sq,
        'Asset':  asset,
        'Class':  next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets),
    })
    asset_frames[asset] = d
print(f"Loaded {len(asset_frames)} assets")

results = {}

for hz in HORIZONS:
    hz_key = f'{hz}d'
    print(f"\n--- Horizon: {hz_key} ---")

    frames = []
    for asset, df in asset_frames.items():
        d = df.copy()
        d['Target']     = forward_rv(d['ret_sq'], hz)
        # horizon-matched
        d['NaiveRV']    = np.log(d['ret_sq'].rolling(hz).mean().shift(1) * 252 + 1e-6)
        # fixed 22d (horizon-agnostic)
        d['RepeatRV']   = np.log(d['ret_sq'].rolling(22).mean().shift(1) * 252 + 1e-6)
        frames.append(d)

    panel = pd.concat(frames).sort_index().reset_index(drop=True)
    panel = panel.dropna(subset=['Target', 'NaiveRV', 'RepeatRV'])

    # Outer split (time-based on index order — panel is already sorted by date via sort_index)
    # Use the same positional split logic as v6
    split     = int(len(panel) * OUTER_TRAIN_RATIO)
    test_df   = panel.iloc[split:].copy().reset_index(drop=True)
    y_te      = test_df['Target'].values
    p_naive   = test_df['NaiveRV'].values
    p_repeat  = test_df['RepeatRV'].values

    naive_r2  = pooled_r2(y_te, p_naive)
    repeat_r2 = pooled_r2(y_te, p_repeat)

    naive_med,  naive_mean  = per_asset_r2(test_df, y_te, p_naive)
    repeat_med, repeat_mean = per_asset_r2(test_df, y_te, p_repeat)

    print(f"  NaiveRV  Pooled_R2={naive_r2:+.4f}  Median={naive_med:+.4f}  Mean={naive_mean:+.4f}")
    print(f"  RepeatRV Pooled_R2={repeat_r2:+.4f}  Median={repeat_med:+.4f}  Mean={repeat_mean:+.4f}")

    results[hz_key] = {
        'NaiveRV':  {'Pooled_R2': round(naive_r2, 4),  'Median_R2': round(naive_med, 4),  'Mean_R2': round(naive_mean, 4)},
        'RepeatRV': {'Pooled_R2': round(repeat_r2, 4), 'Median_R2': round(repeat_med, 4), 'Mean_R2': round(repeat_mean, 4)},
    }

# Save JSON
os.makedirs(RES_DIR, exist_ok=True)
out_json = os.path.join(RES_DIR, 'repeat_rv_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# Update baseline_comparison_pooled_r2.csv
table_path = os.path.join(TABLE_DIR, 'baseline_comparison_pooled_r2.csv')
if os.path.exists(table_path):
    df_table = pd.read_csv(table_path)
    # Remove existing RepeatRV row if any, then insert after NaiveRV
    df_table = df_table[df_table['Model'] != 'RepeatRV']
    repeat_row = {'Model': 'RepeatRV'}
    for hz in HORIZONS:
        repeat_row[f'{hz}d'] = results[f'{hz}d']['RepeatRV']['Pooled_R2']
    repeat_df  = pd.DataFrame([repeat_row])
    naive_idx  = df_table[df_table['Model'] == 'NaiveRV'].index[0]
    df_table = pd.concat([
        df_table.iloc[:naive_idx + 1],
        repeat_df,
        df_table.iloc[naive_idx + 1:],
    ], ignore_index=True)
    df_table.to_csv(table_path, index=False)
    print(f"Updated: {table_path}")
else:
    print(f"Table not found: {table_path} — skipping CSV update")

print("\nDone.")
print("\nSummary: NaiveRV vs RepeatRV Pooled R²")
print(f"{'Horizon':<8}", end='')
for hz in HORIZONS:
    print(f"  {hz}d", end='')
print()
for model in ['NaiveRV', 'RepeatRV']:
    print(f"{model:<8}", end='')
    for hz in HORIZONS:
        v = results[f'{hz}d'][model]['Pooled_R2']
        print(f"  {v:+.4f}"[1:6] if v >= 0 else f"{v:.4f}", end='')
    print()
