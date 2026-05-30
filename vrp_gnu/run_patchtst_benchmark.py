"""
PatchTST Benchmark: Patch-based Transformer for RV Forecasting
==============================================================
Nie et al. (2023) "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers", ICLR 2023.

Design:
  - Input: L-day lookback of daily 1-day instantaneous log RV per asset
  - Model: patch-based Transformer encoder → linear prediction head
  - Per-asset fitting (sequential, consistent with DLinear benchmark)
  - Same outer 80/20 split + purge gap = h as run_main_benchmark_v6.py
  - Inner holdout on last 20% of train for HP search

Output:
  results/patchtst_benchmark_results.json
  paper/csv/patchtst_benchmark_performance.csv
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE      = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8
SEQ_LEN           = 126   # fixed lookback

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]

# HP search: 8 configs × per-asset (reduced to avoid excessive training time)
PARAM_GRID = [
    {'patch_len': p, 'd_model': d, 'n_heads': h, 'n_layers': nl, 'dropout': dr}
    for p   in [8, 16]
    for d   in [64]
    for h   in [4]
    for nl  in [2]
    for dr  in [0.1, 0.2]
]  # 4 configs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS_INNER = 20
EPOCHS_FINAL = 30
BATCH_SIZE   = 64
SEEDS        = [42]

# ── Data Loading ─────────────────────────────────────────────────────────────
print("=" * 70)
print("PatchTST Benchmark: Patch-based Transformer (ICLR 2023)")
print(f"Device: {DEVICE}")
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
    cs       = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

asset_rv = {}
for asset in ALL_ASSETS:
    c      = raw[('Close', asset)]
    ret    = np.log(c / c.shift(1)).dropna()
    ret_sq = ret ** 2
    lrv1d  = np.log(ret_sq * 252 + 1e-12)
    asset_rv[asset] = pd.DataFrame({
        'lrv1d':  lrv1d,
        'ret_sq': ret_sq,
        'Class':  next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets),
    })

print(f"Assets loaded: {len(asset_rv)}")

# ── PatchTST Model ────────────────────────────────────────────────────────────
class PatchTSTModel(nn.Module):
    """Lightweight PatchTST: patch embedding + Transformer encoder + linear head."""
    def __init__(self, seq_len, patch_len, d_model, n_heads, n_layers, dropout):
        super().__init__()
        stride     = patch_len // 2
        n_patches  = (seq_len - patch_len) // stride + 1
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = n_patches

        self.patch_embed  = nn.Linear(patch_len, d_model)
        self.pos_embed    = nn.Parameter(torch.zeros(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm         = nn.LayerNorm(d_model)
        self.head         = nn.Linear(n_patches * d_model, 1)
        self.drop         = nn.Dropout(dropout)

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, seq_len)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            patches.append(x[:, start:start + self.patch_len])
        patches = torch.stack(patches, dim=1)            # (B, n_patches, patch_len)
        emb = self.patch_embed(patches) + self.pos_embed  # (B, n_patches, d_model)
        out = self.transformer(emb)                        # (B, n_patches, d_model)
        out = self.norm(out)
        out = self.drop(out.reshape(out.size(0), -1))      # (B, n_patches * d_model)
        return self.head(out).squeeze(-1)                  # (B,)


# ── Sequence Building ─────────────────────────────────────────────────────────
def make_sequences_rv(lrv1d_arr, target_arr, lookback, outer_split=None, inner_split=None):
    """
    Build (seqs, targets) for indices in [lookback, len).
    If outer_split given: sequences up to outer_split - 1 only (train region).
    """
    seqs, tgts, idx = [], [], []
    end = len(lrv1d_arr) if outer_split is None else outer_split
    for i in range(lookback, end):
        if np.isnan(target_arr[i]):
            continue
        seq = lrv1d_arr[i - lookback:i]
        if np.any(np.isnan(seq)):
            continue
        seqs.append(seq)
        tgts.append(target_arr[i])
        idx.append(i)
    return (np.array(seqs, dtype=np.float32),
            np.array(tgts,  dtype=np.float32),
            np.array(idx))


# ── Training Helpers ──────────────────────────────────────────────────────────
def train_patchtst(X_tr, y_tr, cfg, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m   = PatchTSTModel(SEQ_LEN, cfg['patch_len'], cfg['d_model'],
                        cfg['n_heads'], cfg['n_layers'], cfg['dropout']).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf  = nn.MSELoss()
    Xt  = torch.FloatTensor(X_tr).to(DEVICE)
    yt  = torch.FloatTensor(y_tr).to(DEVICE)
    m.train()
    for _ in range(epochs):
        perm = np.random.permutation(len(Xt))
        for s in range(0, len(perm), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            loss = lf(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    return m


def predict_patchtst(m, X_te):
    Xt = torch.FloatTensor(X_te).to(DEVICE)
    with torch.no_grad():
        return m(Xt).cpu().numpy()


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

    all_y_true = []
    all_y_pred = []
    asset_r2   = {}

    for asset in ALL_ASSETS:
        df = asset_rv[asset].copy()
        df['target'] = forward_rv(df['ret_sq'], hz)
        df = df.dropna(subset=['lrv1d', 'ret_sq'])

        lrv1d_arr  = df['lrv1d'].values.astype(np.float32)
        target_arr = df['target'].values.astype(np.float32)

        n           = len(df)
        outer_split = int(n * OUTER_TRAIN_RATIO)
        inner_split = int(outer_split * INNER_TRAIN_RATIO)

        if outer_split - hz < SEQ_LEN + 10:
            print(f"    {asset}: insufficient data for SEQ_LEN={SEQ_LEN}, skipping")
            continue

        # Inner train / inner val sequences
        itr_seqs, itr_tgts, itr_idx = make_sequences_rv(
            lrv1d_arr, target_arr, SEQ_LEN, outer_split=inner_split - hz)
        ival_seqs, ival_tgts, _ = make_sequences_rv(
            lrv1d_arr, target_arr, SEQ_LEN, outer_split=outer_split)
        # Keep only validation portion
        ival_mask = _ >= inner_split
        ival_seqs = ival_seqs[ival_mask]
        ival_tgts = ival_tgts[ival_mask]

        # HP search on inner holdout
        best_cfg      = PARAM_GRID[0]
        best_inner_r2 = -np.inf

        if len(itr_seqs) >= 10 and len(ival_seqs) >= 5:
            for cfg in PARAM_GRID:
                try:
                    m_inner = train_patchtst(itr_seqs, itr_tgts, cfg,
                                             epochs=EPOCHS_INNER, seed=RANDOM_STATE)
                    preds_val = predict_patchtst(m_inner, ival_seqs)
                    r2 = pooled_r2(ival_tgts, preds_val)
                    if r2 > best_inner_r2:
                        best_inner_r2 = r2
                        best_cfg = cfg
                except Exception as e:
                    print(f"      cfg={cfg} failed: {e}")
                    continue

        print(f"    {asset}: best_cfg=patch{best_cfg['patch_len']},"
              f"d{best_cfg['d_model']}  inner_R²={best_inner_r2:.4f}", flush=True)

        # Full train sequences
        tr_seqs, tr_tgts, _ = make_sequences_rv(
            lrv1d_arr, target_arr, SEQ_LEN, outer_split=outer_split - hz)
        # Test sequences
        te_seqs, te_tgts, _ = make_sequences_rv(
            lrv1d_arr, target_arr, SEQ_LEN)
        te_mask = _ >= outer_split
        te_seqs = te_seqs[te_mask]
        te_tgts = te_tgts[te_mask]

        if len(tr_seqs) < 10 or len(te_seqs) < 5:
            print(f"    {asset}: insufficient test data, skipping")
            continue

        # Final fit (averaged over seeds)
        seed_preds = []
        for seed in SEEDS:
            try:
                m_final = train_patchtst(tr_seqs, tr_tgts, best_cfg,
                                          epochs=EPOCHS_FINAL, seed=seed)
                seed_preds.append(predict_patchtst(m_final, te_seqs))
            except Exception as e:
                print(f"      final seed={seed} failed: {e}")
        if not seed_preds:
            continue
        te_preds = np.mean(seed_preds, axis=0)

        a_r2 = pooled_r2(te_tgts, te_preds)
        asset_r2[asset] = round(a_r2, 4)
        all_y_true.extend(te_tgts.tolist())
        all_y_pred.extend(te_preds.tolist())
        print(f"      test_R²={a_r2:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    pooled = pooled_r2(all_y_true, all_y_pred)
    rmse   = pooled_rmse(all_y_true, all_y_pred)
    per_r2 = list(asset_r2.values())
    median_r2 = float(np.median(per_r2)) if per_r2 else float('nan')
    mean_r2   = float(np.mean(per_r2))   if per_r2 else float('nan')

    hz_res = {
        'Pooled_R2': round(pooled, 4),
        'Median_R2': round(median_r2, 4),
        'Mean_R2':   round(mean_r2, 4),
        'RMSE':      round(rmse, 4),
        'per_asset': asset_r2,
    }
    results[f'{hz}d'] = hz_res
    print(f"  → PatchTST {hz}d: Pooled_R²={pooled:.4f}  Median_R²={median_r2:.4f}  "
          f"Mean_R²={mean_r2:.4f}  RMSE={rmse:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(_RES_DIR, exist_ok=True)
os.makedirs(_CSV_DIR, exist_ok=True)

out_json = _os.path.join(_RES_DIR, 'patchtst_benchmark_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

rows = []
for hz_key, res in results.items():
    rows.append({
        'Model': 'PatchTST',
        'Horizon': hz_key,
        'Pooled_R2': res['Pooled_R2'],
        'Median_R2': res['Median_R2'],
        'Mean_R2':   res['Mean_R2'],
        'RMSE':      res['RMSE'],
    })
df_out = pd.DataFrame(rows)
csv_path = _os.path.join(_CSV_DIR, 'patchtst_benchmark_performance.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df_out.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

print("\n" + "=" * 70)
print("PatchTST Summary (Pooled R²)")
print("=" * 70)
for hz_key, res in results.items():
    print(f"  {hz_key:>5}: {res['Pooled_R2']:>7.4f}")
