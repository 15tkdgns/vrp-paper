"""
Main Benchmark v4: Fair Per-Class Comparison (per-horizon tuning)
=================================================================
Changes from v3:
  - WEns weight tuning fixed: train Ridge/XGB directly on itr_df using
    already-found best_params (no nested inner-inner split of itr_df),
    predict on ival_df, tune weight. Matches verification script approach.
  - BiLSTM-A final seeds reduced 3→1 for speed (inner tuning unchanged).

Expected: WEns 22d >= 0.803

Outputs:
  results/main_benchmark_v4_results.json
  paper/csv/main_benchmark_v4_performance.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed, will be skipped.")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
RANDOM_STATE = 42
OUTER_TRAIN_RATIO = 0.8
INNER_TRAIN_RATIO = 0.8   # last 20% of outer-train → inner val
N_SEEDS_INNER = 1
N_SEEDS_FINAL = 1
SEQ_LEN = 22

ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

# 6 configs per model (× 3 classes = 18 inner fits each)
PARAM_GRIDS = {
    'Ridge':    [{'alpha': a} for a in [10, 50, 100, 500, 1000, 2000]],
    'LASSO':    [{'alpha': a} for a in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]],
    'ENet':     [{'alpha': a, 'l1_ratio': r}
                 for a in [0.005, 0.01, 0.1] for r in [0.1, 0.9]],
    'RF':       [{'max_depth': d, 'min_samples_leaf': l}
                 for d in [5, 8, 10] for l in [5, 20]],
    'XGBoost':  [{'max_depth': d, 'learning_rate': lr}
                 for d in [3, 4] for lr in [0.03, 0.05, 0.1]],
    'LightGBM': [{'max_depth': d, 'learning_rate': lr, 'num_leaves': nl}
                 for d in [3, 5] for lr in [0.05, 0.1] for nl in [15, 31]],
    'MLP':      [{'hidden_layer_sizes': h, 'alpha': a}
                 for h in [(64, 32), (128, 64), (64,)] for a in [0.0001, 0.01]],
    'BiLSTM-A': [{'hidden': h, 'dropout': d}
                 for h in [32, 64] for d in [0.1, 0.3]],          # 4 configs × 3 classes
    'WEns':     [{'pw': w} for w in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
}

# ── Feature Engineering Helpers ─────────────────────────────────────────────
def fit_garch(r):
    if not HAS_ARCH:
        return r.rolling(22).std().fillna(0)
    try:
        am = arch_model(r * 100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return pd.Series(res.conditional_volatility.values.flatten() / 100, index=r.index)
    except:
        return r.rolling(22).std().fillna(0)

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h / l) ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h / l); co = np.log(c / o)
    return np.sqrt((0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(w).mean().clip(0) * 252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.rolling(w).mean().clip(0) * 252)

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

# ── Evaluation ──────────────────────────────────────────────────────────────
def calc_stats(df_eval, y_true, y_pred):
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()
    if len(y_t) != len(y_p): return {}
    valid = ~np.isnan(y_p)
    if valid.sum() < 2: return {}
    y_t, y_p = y_t[valid], y_p[valid]
    df_v = df_eval.iloc[valid].reset_index(drop=True)
    pooled = float(r2_score(y_t, y_p))
    rmse   = float(np.sqrt(mean_squared_error(y_t, y_p)))
    per_asset = []
    for a in df_v['Asset'].unique():
        m = (df_v['Asset'] == a).values
        if m.sum() < 2: continue
        per_asset.append(float(r2_score(y_t[m], y_p[m])))
    if not per_asset:
        return {'Pooled_R2': round(pooled, 4), 'RMSE': round(rmse, 4)}
    return {
        'Pooled_R2': round(pooled, 4),
        'Median_R2': round(float(np.median(per_asset)), 4),
        'Mean_R2':   round(float(np.mean(per_asset)), 4),
        'RMSE':      round(rmse, 4),
    }

# ── Model Factory ────────────────────────────────────────────────────────────
def make_model(name, cfg):
    if name == 'Ridge':
        return Ridge(alpha=cfg['alpha'])
    elif name == 'LASSO':
        return Lasso(alpha=cfg['alpha'], max_iter=5000)
    elif name == 'ENet':
        return ElasticNet(alpha=cfg['alpha'], l1_ratio=cfg['l1_ratio'], max_iter=5000)
    elif name == 'RF':
        return RandomForestRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], max_features='sqrt',
            min_samples_leaf=cfg['min_samples_leaf'], n_jobs=-1, random_state=RANDOM_STATE)
    elif name == 'XGBoost':
        return XGBRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            min_child_weight=5, random_state=RANDOM_STATE, verbosity=0, n_jobs=1, tree_method='hist')
    elif name == 'LightGBM':
        return LGBMRegressor(
            n_estimators=200, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
            num_leaves=cfg['num_leaves'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, min_child_samples=5,
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    elif name == 'MLP':
        return MLPRegressor(
            hidden_layer_sizes=cfg['hidden_layer_sizes'], alpha=cfg['alpha'],
            max_iter=500, early_stopping=True, n_iter_no_change=20,
            random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")

# ── Per-class Sklearn Tuning & Fitting ───────────────────────────────────────
def fit_perclass_sklearn(name, grid, train_df, itr_df, ival_df, sc, feats):
    """
    Tune hyperparams per class on inner holdout, refit on full train.
    Returns (pred_array aligned to train_df.index, best_params dict by class).
    Also returns per-class final models for later test prediction.
    """
    best_params = {}
    final_models = {}

    for cls in ASSET_GROUPS:
        tr_c  = train_df[train_df['Class'] == cls]
        itr_c = itr_df[itr_df['Class'] == cls]
        ival_c = ival_df[ival_df['Class'] == cls]

        if len(itr_c) < 30 or len(ival_c) < 10:
            # fallback: use first config
            best_params[cls] = grid[0]
        else:
            X_itr_c  = sc.transform(itr_c[feats]);  y_itr_c  = itr_c['Target'].values
            X_ival_c = sc.transform(ival_c[feats]); y_ival_c = ival_c['Target'].values
            best_r2, best_cfg = -np.inf, grid[0]
            for cfg in grid:
                m = make_model(name, cfg)
                m.fit(X_itr_c, y_itr_c)
                pred = m.predict(X_ival_c)
                r2 = float(r2_score(y_ival_c, pred)) if len(pred) > 1 else -np.inf
                if r2 > best_r2:
                    best_r2, best_cfg = r2, cfg
            best_params[cls] = best_cfg

        # Refit on full train for this class
        X_tr_c = sc.transform(tr_c[feats]); y_tr_c = tr_c['Target'].values
        if len(X_tr_c) < 10:
            final_models[cls] = None
            continue
        m_final = make_model(name, best_params[cls])
        m_final.fit(X_tr_c, y_tr_c)
        final_models[cls] = m_final

    return best_params, final_models


def predict_perclass_sklearn(final_models, test_df, sc, feats):
    """Predict on test set using per-class models."""
    preds = np.full(len(test_df), np.nan)
    for cls, m in final_models.items():
        if m is None: continue
        mask = (test_df['Class'] == cls).values
        if mask.sum() == 0: continue
        preds[mask] = m.predict(sc.transform(test_df.loc[mask, feats]))
    return preds

# ── BiLSTM-A ─────────────────────────────────────────────────────────────────
class BiLSTMAttn(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc   = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        w   = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(-1)


def build_sequences(df, X_scaled, y, seq_len=SEQ_LEN):
    Xs, ys, ids = [], [], []
    for asset in df['Asset'].unique():
        mask = df['Asset'] == asset
        Xa = X_scaled[mask]; ya = y[mask]; idx = df.index[mask]
        if len(Xa) <= seq_len: continue
        for i in range(seq_len, len(Xa)):
            Xs.append(Xa[i - seq_len:i]); ys.append(ya[i]); ids.append(idx[i])
    if not Xs:
        return np.empty((0, seq_len, X_scaled.shape[1])), np.array([]), []
    return np.array(Xs), np.array(ys), ids


def train_bilstm_one(X_tr, y_tr, X_te, train_df, test_df, cfg, epochs, seeds):
    Xtr_s, ytr_s, _      = build_sequences(train_df, X_tr, y_tr)
    Xte_s, _,     te_ids = build_sequences(test_df,  X_te, np.zeros(len(test_df)))
    if len(Xtr_s) == 0 or len(Xte_s) == 0:
        return np.array([]), []
    in_dim = X_tr.shape[1]
    all_preds = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m   = BiLSTMAttn(in_dim, hidden=cfg['hidden'], dropout=cfg['dropout'])
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        lf  = nn.MSELoss()
        Xt  = torch.FloatTensor(Xtr_s); yt = torch.FloatTensor(ytr_s)
        m.train()
        for _ in range(epochs):
            perm = np.random.permutation(len(Xt))
            for s in range(0, len(perm), 64):
                b = perm[s:s + 64]
                loss = lf(m(Xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            all_preds.append(m(torch.FloatTensor(Xte_s)).numpy())
    return np.mean(all_preds, axis=0), te_ids


def fit_perclass_bilstm(grid, train_df, itr_df, ival_df, sc, feats, hz):
    """Tune BiLSTM per class on inner holdout; return best config per class."""
    best_params = {}
    for cls in ASSET_GROUPS:
        itr_c  = itr_df[itr_df['Class'] == cls]
        ival_c = ival_df[ival_df['Class'] == cls]
        if len(itr_c) < SEQ_LEN * 2 or len(ival_c) < SEQ_LEN:
            best_params[cls] = grid[0]; continue

        X_itr_c  = sc.transform(itr_c[feats])
        X_ival_c = sc.transform(ival_c[feats])
        y_itr_c  = itr_c['Target'].values

        best_r2, best_cfg = -np.inf, grid[0]
        for cfg in grid:
            preds, te_ids = train_bilstm_one(
                X_itr_c, y_itr_c, X_ival_c, itr_c, ival_c,
                cfg, epochs=15, seeds=[0])
            if len(preds) < 2: continue
            y_al = ival_c.loc[te_ids, 'Target'].values
            r2   = float(r2_score(y_al, preds))
            if r2 > best_r2: best_r2, best_cfg = r2, cfg
        best_params[cls] = best_cfg
        print(f"      {cls}: best={best_cfg}  inner_R²={best_r2:.4f}", flush=True)
    return best_params


def predict_perclass_bilstm(best_params, train_df, test_df, sc, feats):
    """Final BiLSTM prediction per class (3 seeds)."""
    all_preds = np.full(len(test_df), np.nan)
    all_ids   = []
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_c = test_df[test_df['Class'] == cls]
        if len(tr_c) < SEQ_LEN * 2 or len(te_c) == 0: continue
        X_tr_c = sc.transform(tr_c[feats])
        X_te_c = sc.transform(te_c[feats])
        preds, te_ids = train_bilstm_one(
            X_tr_c, tr_c['Target'].values, X_te_c, tr_c, te_c,
            best_params[cls], epochs=20, seeds=list(range(N_SEEDS_FINAL)))
        if len(preds) == 0: continue
        # Map te_ids (original index) back to position in test_df
        pos = [test_df.index.get_loc(i) for i in te_ids]
        for p_idx, pred_val in zip(pos, preds):
            all_preds[p_idx] = pred_val
        all_ids.extend(te_ids)
    return all_preds, all_ids

# ── Data Loading ─────────────────────────────────────────────────────────────
print("=" * 70)
print("Main Benchmark v4: Per-class fair comparison (per-horizon tuning)")
print("=" * 70)
print("\nLoading data & building 37 features...", flush=True)

raw     = pd.read_pickle('/root/vrp/src/data/v71_ohlcv_cache.pkl')
vix     = raw[('Close', 'VIX')]
spy_c   = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

iv_features = {
    'VIX':           np.log(vix + 1e-6),
    'VIX_chg':       np.log(vix + 1e-6).diff(),
    'VIX_ma5':       np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5':      np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M':         np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D':         np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope':np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp_val = (vix ** 2 / 100) - spy_rv / 10000
iv_features['VRP']      = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

asset_frames = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]; o = raw[('Open', asset)]
    h = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
    ret    = np.log(c / c.shift(1)).dropna()
    ret_sq = ret ** 2
    rv     = ret_sq.rolling(22).mean() * 252 * 10000
    lrv    = np.log(rv + 1e-6)
    gd = fit_garch(ret)
    rw = ret.resample('W').sum()
    gw = fit_garch(rw).reindex(ret.index, method='ffill')
    feat = {
        'LogRV_lag1':  lrv.shift(1),  'LogRV_lag5':  lrv.shift(5),
        'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
        'Garch_Daily': gd.shift(1),   'Garch_Weekly': gw.shift(1),
        'LogRV_Std5':  lrv.rolling(5).std().shift(1),
        'LogRV_Std22': lrv.rolling(22).std().shift(1),
        'RV_Mom5':     (lrv - lrv.shift(5)).shift(1),
        'RV_Mom22':    (lrv - lrv.shift(22)).shift(1),
        'SPY_LogRV':   spy_lrv.shift(1),
        'Ret_lag1':    ret.shift(1),
        'Ret_abs_lag1':ret.abs().shift(1),
        'Corr_SPY':    (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                        if asset != 'SPY' else pd.Series(1.0, index=ret.index)),
    }
    p5  = compute_parkinson(h, l, 5);  p22 = compute_parkinson(h, l, 22)
    gk22 = compute_gk(o, h, l, c, 22); rs22 = compute_rs(o, h, l, c, 22)
    feat['Parkinson_5']       = np.log(p5  + 1e-6).shift(1)
    feat['Parkinson_22']      = np.log(p22 + 1e-6).shift(1)
    feat['GarmanKlass_22']    = np.log(gk22 + 1e-6).shift(1)
    feat['RogersSatchell_22'] = np.log(rs22 + 1e-6).shift(1)
    feat['Range_Close_Ratio'] = (np.log(p22 + 1e-6) - lrv).shift(1)
    on = np.log(o / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    for k, val in iv_features.items():
        feat[f'IV_{k}'] = val.shift(1)
    dv = v * c
    feat['AltVol_Amihud']          = (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']       = (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1)
    feat['AltVol_PV_Corr']         = ret.rolling(22).corr(np.log(v + 1)).shift(1)
    feat['AltVol_Vol_Surprise']    = ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1)
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    feat['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']     = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset']  = asset
    d['Class']  = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    asset_frames[asset] = d

print(f"Assets: {len(asset_frames)}, Features: 37")

# ── Main Loop ────────────────────────────────────────────────────────────────
results = {}

for hz in HORIZONS:
    print(f"\n{'='*70}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*70}")

    # Build pooled dataset
    pooled = []
    for asset in ALL_ASSETS:
        df = asset_frames[asset].copy()
        df['Target'] = forward_rv(df['ret_sq'], hz)
        df = df.drop(columns=['ret_sq']).dropna()
        pooled.append(df)
    data  = pd.concat(pooled).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    assert len(feats) == 37

    # Outer split with purge
    split    = int(len(data) * OUTER_TRAIN_RATIO)
    train_df = data.iloc[:split - hz].copy()
    test_df  = data.iloc[split:].copy()
    y_te     = test_df['Target'].values

    sc   = StandardScaler().fit(train_df[feats])
    X_tr = sc.transform(train_df[feats])
    X_te = sc.transform(test_df[feats])
    y_tr = train_df['Target'].values

    # Inner holdout with purge
    n_tr    = len(train_df)
    v_split = int(n_tr * INNER_TRAIN_RATIO)
    itr_df  = train_df.iloc[:v_split - hz].copy()
    ival_df = train_df.iloc[v_split:].copy()

    print(f"  Outer  train: {len(train_df):,}  test: {len(test_df):,}")
    print(f"  Inner  train: {len(itr_df):,}   val:  {len(ival_df):,}")

    hz_res = {}

    # ── HAR-3 (fixed, per-class) ──
    print("  [HAR-3] fixed α=1.0, per-class...")
    har_idx = [feats.index(f) for f in HAR_FEATS]
    p_har   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        m = Ridge(alpha=1.0).fit(sc.transform(tr_c[feats])[:, har_idx], tr_c['Target'].values)
        p_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, feats])[:, har_idx])
    hz_res['HAR-3'] = calc_stats(test_df, y_te, p_har)
    print(f"    → Pooled_R2={hz_res['HAR-3']['Pooled_R2']:.4f}")

    # ── Sklearn models (per-class inner-holdout tuning) ──
    SKLEARN_MODELS = ['Ridge', 'LASSO', 'ENet', 'RF', 'XGBoost']
    if HAS_LGBM: SKLEARN_MODELS.append('LightGBM')
    SKLEARN_MODELS.append('MLP')

    best_params_hz = {}
    p_ridge_final  = None
    p_xgb_final    = None
    fm_ridge        = None
    fm_xgb          = None

    for model_name in SKLEARN_MODELS:
        n_cfg = len(PARAM_GRIDS[model_name])
        print(f"  [{model_name}] {n_cfg} configs × 3 classes...", flush=True)
        best_params, final_models = fit_perclass_sklearn(
            model_name, PARAM_GRIDS[model_name],
            train_df, itr_df, ival_df, sc, feats)
        preds = predict_perclass_sklearn(final_models, test_df, sc, feats)

        hz_res[model_name] = calc_stats(test_df, y_te, preds)
        hz_res[model_name]['best_params'] = best_params
        best_params_hz[model_name] = best_params
        print(f"    params={best_params}  "
              f"test_Pooled_R2={hz_res[model_name]['Pooled_R2']:.4f}")

        if model_name == 'Ridge':
            p_ridge_final = preds; fm_ridge = final_models
        if model_name == 'XGBoost':
            p_xgb_final = preds; fm_xgb = final_models

    # ── BiLSTM-A (per-class, 37 features) ──
    print("  [BiLSTM-A] 4 configs × 3 classes, inner holdout...", flush=True)
    bl_best_params = fit_perclass_bilstm(
        PARAM_GRIDS['BiLSTM-A'], train_df, itr_df, ival_df, sc, feats, hz)
    print(f"  [BiLSTM-A] final fit ({N_SEEDS_FINAL} seeds per class)...", flush=True)
    bl_preds, bl_ids = predict_perclass_bilstm(bl_best_params, train_df, test_df, sc, feats)
    valid_bl = ~np.isnan(bl_preds)
    if valid_bl.sum() > 1:
        bl_df = test_df.reset_index(drop=True)
        hz_res['BiLSTM-A'] = calc_stats(bl_df, y_te, bl_preds)
        hz_res['BiLSTM-A']['best_params'] = bl_best_params
        print(f"    → Pooled_R2={hz_res['BiLSTM-A']['Pooled_R2']:.4f}")

    # ── WEns (per-class Ridge + XGBoost, weight from inner holdout) ──
    if p_ridge_final is not None and p_xgb_final is not None:
        print("  [WEns] weight search on inner holdout...", flush=True)

        # Train Ridge/XGB on itr_df using already-found best_params,
        # then predict on ival_df — no nested inner-inner split.
        fm_r_in = {}
        fm_x_in = {}
        for cls in ASSET_GROUPS:
            itr_c = itr_df[itr_df['Class'] == cls]
            if len(itr_c) < 5:
                fm_r_in[cls] = None; fm_x_in[cls] = None; continue
            X_itr_c = sc.transform(itr_c[feats]); y_itr_c = itr_c['Target'].values
            mr = make_model('Ridge', best_params_hz['Ridge'][cls])
            mr.fit(X_itr_c, y_itr_c)
            fm_r_in[cls] = mr
            mx = make_model('XGBoost', best_params_hz['XGBoost'][cls])
            mx.fit(X_itr_c, y_itr_c)
            fm_x_in[cls] = mx
        p_r_ival = predict_perclass_sklearn(fm_r_in, ival_df, sc, feats)
        p_x_ival = predict_perclass_sklearn(fm_x_in, ival_df, sc, feats)
        y_ival   = ival_df['Target'].values

        best_pw, best_pw_r2 = 0.7, -np.inf
        for cfg in PARAM_GRIDS['WEns']:
            pw  = cfg['pw']
            valid = ~(np.isnan(p_r_ival) | np.isnan(p_x_ival))
            if valid.sum() < 2: continue
            ens = pw * p_r_ival[valid] + (1 - pw) * p_x_ival[valid]
            r2  = float(r2_score(y_ival[valid], ens))
            if r2 > best_pw_r2: best_pw_r2, best_pw = r2, pw

        valid = ~(np.isnan(p_ridge_final) | np.isnan(p_xgb_final))
        p_wens = np.full(len(test_df), np.nan)
        p_wens[valid] = best_pw * p_ridge_final[valid] + (1 - best_pw) * p_xgb_final[valid]
        hz_res['WEns'] = calc_stats(test_df, y_te, p_wens)
        hz_res['WEns']['best_params'] = {
            'pw': best_pw,
            'Ridge':   best_params_hz['Ridge'],
            'XGBoost': best_params_hz['XGBoost'],
        }
        print(f"    best pw={best_pw}  inner_R²={best_pw_r2:.4f}  "
              f"test_Pooled_R2={hz_res['WEns']['Pooled_R2']:.4f}")

    results[f'{hz}d'] = hz_res

    # Horizon summary
    print(f"\n  {'Model':<12} {'Pooled_R2':>10} {'Median_R2':>10} {'Mean_R2':>10} {'RMSE':>8}")
    print(f"  {'-'*52}")
    for name, res in hz_res.items():
        pr  = res.get('Pooled_R2',  float('nan'))
        mr  = res.get('Median_R2',  float('nan'))
        mnr = res.get('Mean_R2',    float('nan'))
        rm  = res.get('RMSE',       float('nan'))
        print(f"  {name:<12} {pr:>10.4f} {mr:>10.4f} {mnr:>10.4f} {rm:>8.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
import os
os.makedirs('/root/vrp/results', exist_ok=True)

out_json = '/root/vrp/results/main_benchmark_v4_results.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

rows = []
for hz_key, hz_data in results.items():
    for model_name, res in hz_data.items():
        rows.append({
            'Model':       model_name,
            'Horizon':     hz_key,
            'Pooled_R2':   res.get('Pooled_R2'),
            'Median_R2':   res.get('Median_R2'),
            'Mean_R2':     res.get('Mean_R2'),
            'RMSE':        res.get('RMSE'),
            'Best_Params': str(res.get('best_params', 'fixed')),
        })
df_out = pd.DataFrame(rows)
out_csv = '/root/vrp/paper/csv/main_benchmark_v4_performance.csv'
df_out.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

pivot = df_out.pivot_table(index='Horizon', columns='Model', values='Pooled_R2')
pivot = pivot.reindex([f'{h}d' for h in HORIZONS])
print("\nPooled_R2 Summary (v3 — per-class):")
print(pivot.round(4).to_string())
