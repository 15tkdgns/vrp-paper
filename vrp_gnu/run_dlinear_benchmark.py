"""
DLinear Benchmark: Sequential Per-Asset RV Forecasting
=======================================================
Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.

Design:
  - Input: L-day lookback of daily 1-day instantaneous log RV per asset
  - Model: moving-average decomposition + two linear layers (DLinear)
  - Per-asset fitting (sequential, not cross-sectional panel)
  - Same outer 80/20 split + purge gap = h as run_main_benchmark_v6.py
  - Inner holdout on last 20% of train for HP search

Output:
  results/dlinear_benchmark_results.json
  paper/csv/dlinear_benchmark_performance.csv
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS         = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE     = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

PARAM_GRID = [
    {'lookback': L, 'kernel_size': k}
    for L in [22, 63, 126, 252]
    for k in [5, 11, 25]
]  # 12 configs per asset

# ── Data Loading ─────────────────────────────────────────────────────────────
print("=" * 70)
print("DLinear Benchmark: Per-asset sequential RV forecasting (AAAI 2023)")
print("=" * 70)
print("\nLoading data...", flush=True)

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKL_PATH   = '/root/vrp/src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = _os.path.join(_SCRIPT_DIR, 'data')
_RES_DIR    = _os.path.join(_SCRIPT_DIR, 'results')
_CSV_DIR    = _os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv')

def _load_from_parquet():
    vix_df = pd.read_parquet(_os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames = {}
    for asset in ALL_ASSETS:
        p = _os.path.join(_DATA_DIR, f'{asset}.parquet')
        if not _os.path.exists(p):
            continue
        frames[asset] = pd.read_parquet(p)
    combined = pd.concat(frames.values(), axis=1)
    combined[('Close', 'VIX')]   = vix_df['Close']
    combined[('Close', 'VIX3M')] = vix_df['Close_3M']
    combined[('Close', 'VIX9D')] = vix_df['Close_9D']
    return combined

try:
    raw = pd.read_pickle(_PKL_PATH)
except Exception:
    raw = _load_from_parquet()

def forward_rv(ret_sq, horizon):
    cs      = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

# Build per-asset daily instantaneous log RV series
asset_rv = {}
for asset in ALL_ASSETS:
    c      = raw[('Close', asset)]
    ret    = np.log(c / c.shift(1)).dropna()
    ret_sq = ret ** 2
    lrv1d  = np.log(ret_sq * 252 + 1e-12)   # 1-day instantaneous log RV
    asset_rv[asset] = pd.DataFrame({
        'lrv1d':  lrv1d,
        'ret_sq': ret_sq,
        'Class':  next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets),
    })

print(f"Assets loaded: {len(asset_rv)}")

# ── DLinear Model ─────────────────────────────────────────────────────────────
class DLinearModel:
    """Moving-average decomposition + two linear layers."""
    def __init__(self, lookback=126, kernel_size=25):
        self.L = lookback
        self.k = kernel_size
        self.w_trend   = None
        self.w_seasonal = None

    def _decompose(self, seqs):
        # seqs: (N, L)
        half = self.k // 2
        trend = np.zeros_like(seqs, dtype=float)
        for i in range(len(seqs)):
            padded = np.pad(seqs[i], (half, half), mode='edge')
            trend[i] = np.convolve(padded, np.ones(self.k) / self.k, mode='valid')[:self.L]
        seasonal = seqs - trend
        return trend, seasonal

    def fit(self, seqs, y):
        t, s = self._decompose(seqs)
        self.w_trend    = LinearRegression(fit_intercept=True).fit(t, y)
        self.w_seasonal = LinearRegression(fit_intercept=True).fit(s, y)

    def predict(self, seqs):
        t, s = self._decompose(seqs)
        return self.w_trend.predict(t) + self.w_seasonal.predict(s)


def make_sequences(series, horizon, lookback):
    """Sliding-window sequences from a 1D series.
    Returns (seqs, targets) where:
      seqs[i]   = series[i : i+lookback]  (lookback-dim input)
      targets[i] = forward h-day log RV starting at i+lookback
    """
    seqs, targets, dates = [], [], []
    n = len(series)
    for i in range(lookback, n - horizon):
        seqs.append(series[i - lookback:i])
        fwd = np.mean(series[i:i + horizon])   # using lrv1d as proxy for target
        targets.append(fwd)
        dates.append(i)
    return np.array(seqs, dtype=float), np.array(targets, dtype=float), np.array(dates)


def make_sequences_rv(rv1d_series, fwd_rv_series, horizon, lookback):
    """Sliding windows on rv1d input, with forward RV as target."""
    seqs, targets, idx = [], [], []
    n = len(rv1d_series)
    valid_fwd = ~np.isnan(fwd_rv_series)
    for i in range(lookback, n):
        if i >= n or not valid_fwd[i]:
            continue
        seq_vals = rv1d_series[i - lookback:i]
        if np.any(np.isnan(seq_vals)):
            continue
        seqs.append(seq_vals)
        targets.append(fwd_rv_series[i])
        idx.append(i)
    return np.array(seqs, dtype=float), np.array(targets, dtype=float), np.array(idx)


# ── Evaluation ──────────────────────────────────────────────────────────────
def pooled_r2(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float('nan')
    return float(r2_score(y_true[valid], y_pred[valid]))


def pooled_rmse(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return float('nan')
    return float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))


# ── Main Loop ────────────────────────────────────────────────────────────────
results = {}

for hz in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*70}")

    all_y_true  = []
    all_y_pred  = []
    all_assets  = []
    asset_r2    = {}

    for asset in ALL_ASSETS:
        df = asset_rv[asset].copy()

        # Forward target for this asset
        df['target'] = forward_rv(df['ret_sq'], hz)

        # Drop NaN from ret_sq and target
        df = df.dropna(subset=['lrv1d', 'ret_sq'])
        lrv1d_arr = df['lrv1d'].values
        target_arr = df['target'].values

        n = len(df)
        outer_split = int(n * OUTER_TRAIN_RATIO)
        inner_split = int(outer_split * INNER_TRAIN_RATIO)

        # Per-asset HP search: inner holdout
        best_cfg    = PARAM_GRID[0]
        best_inner_r2 = -np.inf

        for cfg in PARAM_GRID:
            L = cfg['lookback']
            if inner_split - hz < L + 1:
                continue   # not enough data

            # Inner train sequences
            itr_seqs, itr_tgts, _ = make_sequences_rv(
                lrv1d_arr[:inner_split - hz],
                target_arr[:inner_split - hz],
                hz, L)
            # Inner val sequences (starting from inner_split)
            ival_seqs, ival_tgts, _ = make_sequences_rv(
                lrv1d_arr[:outer_split],
                target_arr[:outer_split],
                hz, L)
            ival_mask = _ >= inner_split if len(_) > 0 else np.array([], dtype=bool)
            # Recompute correctly: val sequences where index falls in [inner_split, outer_split]
            ival_seqs2, ival_tgts2, ival_idx2 = [], [], []
            for ii in range(L, outer_split):
                if ii < inner_split:
                    continue
                if not (0 <= ii < len(target_arr)) or np.isnan(target_arr[ii]):
                    continue
                seq = lrv1d_arr[ii - L:ii]
                if np.any(np.isnan(seq)):
                    continue
                ival_seqs2.append(seq)
                ival_tgts2.append(target_arr[ii])
                ival_idx2.append(ii)

            if len(itr_seqs) < 10 or len(ival_seqs2) < 5:
                continue

            ival_seqs2  = np.array(ival_seqs2, dtype=float)
            ival_tgts2  = np.array(ival_tgts2, dtype=float)

            m = DLinearModel(lookback=L, kernel_size=cfg['kernel_size'])
            m.fit(itr_seqs, itr_tgts)
            preds = m.predict(ival_seqs2)
            r2 = pooled_r2(ival_tgts2, preds)
            if r2 > best_inner_r2:
                best_inner_r2 = r2
                best_cfg = cfg

        # Refit on full outer train, predict on test
        L = best_cfg['lookback']
        # Collect train sequences: index in [L, outer_split - hz)
        tr_seqs, tr_tgts, tr_idx = [], [], []
        for ii in range(L, outer_split - hz):
            if np.isnan(target_arr[ii]):
                continue
            seq = lrv1d_arr[ii - L:ii]
            if np.any(np.isnan(seq)):
                continue
            tr_seqs.append(seq)
            tr_tgts.append(target_arr[ii])
            tr_idx.append(ii)

        # Test sequences: index in [outer_split, n)
        te_seqs, te_tgts, te_idx = [], [], []
        for ii in range(max(L, outer_split), n):
            if np.isnan(target_arr[ii]):
                continue
            seq = lrv1d_arr[ii - L:ii]
            if np.any(np.isnan(seq)):
                continue
            te_seqs.append(seq)
            te_tgts.append(target_arr[ii])
            te_idx.append(ii)

        if len(tr_seqs) < 10 or len(te_seqs) < 5:
            print(f"    {asset}: insufficient data, skipping")
            continue

        tr_seqs = np.array(tr_seqs, dtype=float)
        tr_tgts = np.array(tr_tgts, dtype=float)
        te_seqs = np.array(te_seqs, dtype=float)
        te_tgts = np.array(te_tgts, dtype=float)

        m_final = DLinearModel(lookback=best_cfg['lookback'],
                               kernel_size=best_cfg['kernel_size'])
        m_final.fit(tr_seqs, tr_tgts)
        te_preds = m_final.predict(te_seqs)

        a_r2 = pooled_r2(te_tgts, te_preds)
        asset_r2[asset] = round(a_r2, 4)
        all_y_true.extend(te_tgts.tolist())
        all_y_pred.extend(te_preds.tolist())
        all_assets.extend([asset] * len(te_tgts))

        print(f"    {asset}: L={best_cfg['lookback']}, k={best_cfg['kernel_size']}, "
              f"inner_R²={best_inner_r2:.4f}, test_R²={a_r2:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    pooled = pooled_r2(all_y_true, all_y_pred)
    rmse   = pooled_rmse(all_y_true, all_y_pred)

    per_r2 = list(asset_r2.values())
    median_r2 = float(np.median(per_r2)) if per_r2 else float('nan')
    mean_r2   = float(np.mean(per_r2))   if per_r2 else float('nan')

    hz_res = {
        'Pooled_R2':  round(pooled, 4),
        'Median_R2':  round(median_r2, 4),
        'Mean_R2':    round(mean_r2, 4),
        'RMSE':       round(rmse, 4),
        'per_asset':  asset_r2,
    }
    results[f'{hz}d'] = hz_res
    print(f"  → DLinear {hz}d: Pooled_R²={pooled:.4f}  Median_R²={median_r2:.4f}  "
          f"Mean_R²={mean_r2:.4f}  RMSE={rmse:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = _os.path.join(_RES_DIR, 'dlinear_benchmark_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# Summary CSV
rows = []
for hz_key, res in results.items():
    rows.append({
        'Model': 'DLinear',
        'Horizon': hz_key,
        'Pooled_R2': res['Pooled_R2'],
        'Median_R2': res['Median_R2'],
        'Mean_R2':   res['Mean_R2'],
        'RMSE':      res['RMSE'],
    })
df_out = pd.DataFrame(rows)
csv_path = _os.path.join(_CSV_DIR, 'dlinear_benchmark_performance.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df_out.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

print("\n" + "=" * 70)
print("DLinear Summary (Pooled R²)")
print("=" * 70)
for hz_key, res in results.items():
    print(f"  {hz_key:>5}: {res['Pooled_R2']:>7.4f}")
