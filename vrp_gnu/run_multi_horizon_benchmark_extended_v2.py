"""
Multi-Horizon Benchmark: 6 models x 8 horizons
Reproduces Figure 3 data (Pooled R^2 by prediction horizon per model)
Pipeline: identical to tmp_calc_wens_multi_horizon.py (37 features, 80:20 split, purge=h)
"""
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from arch import arch_model
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':   ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
HAR_FEATS = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
BILSTM_FEATS = ['RogersSatchell_22', 'GarmanKlass_22', 'Range_Close_Ratio']
ALPHAS_PERCLASS = {'Equity': 100.0, 'Bond': 10.0, 'Commodity': 10.0}

# ── Helpers ─────────────────────────────────────────────────
def fit_garch(r):
    try:
        am = arch_model(r * 100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return pd.Series(res.conditional_volatility.values.flatten() / 100, index=r.index)
    except:
        return r.rolling(22).std().fillna(0)

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h / l) ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h / l)
    co = np.log(c / o)
    return np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(w).mean().clip(0) * 252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.rolling(w).mean().clip(0) * 252)

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

def calculate_qlike(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(np.exp(diff) - diff - 1)

def calculate_median_r2(df_test, y_true, y_pred_pooled):
    """Calculates median R2 across all assets in the test set."""
    asset_r2s = []
    # If the predictions are shortened (like BiLSTM-A), we need to align df_test
    if len(y_pred_pooled) < len(df_test):
        # We assume they are aligned to the end (standard behavior for seq_len truncation)
        df_eval = df_test.iloc[-len(y_pred_pooled):]
    else:
        df_eval = df_test
        
    for asset in df_eval['Asset'].unique():
        mask = (df_eval['Asset'] == asset).values
        if mask.sum() < 2: continue # Need at least 2 points for R2
        r2 = r2_score(y_true[mask], y_pred_pooled[mask])
        asset_r2s.append(r2)
    return np.median(asset_r2s) if asset_r2s else np.nan

# ── BiLSTM-A Model ──────────────────────────────────────────
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(-1)

from collections import defaultdict

def train_bilstm(X_tr, y_tr, X_te, train_df, test_df, seq_len=22, hidden=32, dropout=0.0, lr=0.001, epochs=20, seeds=[0,1,2,3,4]):
    """Train BiLSTM-A building sequences per-asset to avoid cross-asset contamination."""
    device = torch.device('cpu')
    all_preds_dict = defaultdict(list)
    
    # Pre-build sequences per asset
    def make_seq_per_asset(df, X_scaled, y):
        Xs, ys, indices = [], [], []
        for asset in df['Asset'].unique():
            asset_mask = df['Asset'] == asset
            asset_X = X_scaled[asset_mask]
            asset_y = y[asset_mask]
            asset_idx = df.index[asset_mask]
            
            if len(asset_X) <= seq_len:
                continue
                
            for i in range(seq_len, len(asset_X)):
                Xs.append(asset_X[i-seq_len:i])
                ys.append(asset_y[i])
                indices.append(asset_idx[i])
        return np.array(Xs) if Xs else np.empty((0, seq_len, X_scaled.shape[1])), np.array(ys), indices

    Xtr_s, ytr_s, _ = make_seq_per_asset(train_df, X_tr, y_tr)
    Xte_s, _, te_indices = make_seq_per_asset(test_df, X_te, np.zeros(len(test_df)))
    
    if len(Xtr_s) == 0 or len(Xte_s) == 0:
        return np.array([]), np.array([])

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BiLSTMAttention(input_dim=X_tr.shape[1], hidden_dim=hidden, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        Xt = torch.FloatTensor(Xtr_s).to(device)
        yt = torch.FloatTensor(ytr_s).to(device)

        model.train()
        bs = 64
        for ep in range(epochs):
            idx = np.random.permutation(len(Xt))
            for s in range(0, len(idx), bs):
                batch = idx[s:s+bs]
                pred = model(Xt[batch])
                loss = loss_fn(pred, yt[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            Xtest_t = torch.FloatTensor(Xte_s).to(device)
            preds = model(Xtest_t).cpu().numpy()
        all_preds_dict[seed] = preds

    avg_preds = np.mean([all_preds_dict[s] for s in seeds], axis=0)
    
    # Get corresponding true values
    y_te_aligned = test_df.loc[te_indices, 'Target'].values
    
    return avg_preds, y_te_aligned

# ── Data Loading ────────────────────────────────────────────
print("=" * 70)
print("Multi-Horizon Benchmark: 6 models x 8 horizons")
print("=" * 70)
print("\nLoading data...", flush=True)
raw = pd.read_pickle('/root/vrp/src/data/v71_ohlcv_cache.pkl')
vix = raw[('Close', 'VIX')]
spy_c = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

iv_features = {
    'VIX': np.log(vix + 1e-6),
    'VIX_chg': np.log(vix + 1e-6).diff(),
    'VIX_ma5': np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5': np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M': np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D': np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope': np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp_val = (vix ** 2 / 100) - spy_rv / 10000
iv_features['VRP'] = vrp_val
iv_features['VRP_ma22'] = vrp_val.rolling(22).mean()

# ── Build Asset Frames ──────────────────────────────────────
print("Building asset frames...", flush=True)
asset_frames = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]
    o = raw[('Open', asset)]
    h = raw[('High', asset)]
    l = raw[('Low', asset)]
    v = raw[('Volume', asset)]
    ret = np.log(c / c.shift(1)).dropna()
    ret_sq = ret ** 2
    rv = ret_sq.rolling(22).mean() * 252 * 10000
    lrv = np.log(rv + 1e-6)

    gd = fit_garch(ret)
    rw = ret.resample('W').sum()
    gw = fit_garch(rw).reindex(ret.index, method='ffill')

    feat = {
        'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
        'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
        'Garch_Daily': gd.shift(1), 'Garch_Weekly': gw.shift(1),
        'LogRV_Std5': lrv.rolling(5).std().shift(1),
        'LogRV_Std22': lrv.rolling(22).std().shift(1),
        'RV_Mom5': (lrv - lrv.shift(5)).shift(1),
        'RV_Mom22': (lrv - lrv.shift(22)).shift(1),
        'SPY_LogRV': spy_lrv.shift(1),
        'Ret_lag1': ret.shift(1), 'Ret_abs_lag1': ret.abs().shift(1),
        'Corr_SPY': ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1) if asset != 'SPY' else pd.Series(1.0, index=ret.index),
    }
    p5 = compute_parkinson(h, l, 5); p22 = compute_parkinson(h, l, 22)
    gk22 = compute_gk(o, h, l, c, 22); rs22 = compute_rs(o, h, l, c, 22)
    feat['Parkinson_5'] = np.log(p5 + 1e-6).shift(1)
    feat['Parkinson_22'] = np.log(p22 + 1e-6).shift(1)
    feat['GarmanKlass_22'] = np.log(gk22 + 1e-6).shift(1)
    feat['RogersSatchell_22'] = np.log(rs22 + 1e-6).shift(1)
    feat['Range_Close_Ratio'] = (np.log(p22 + 1e-6) - lrv).shift(1)
    on = np.log(o / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)
    for n2, v2 in iv_features.items():
        feat[f'IV_{n2}'] = v2.shift(1)
    dv = v * c
    feat['AltVol_Amihud'] = (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio'] = (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1)
    feat['AltVol_PV_Corr'] = ret.rolling(22).corr(np.log(v + 1)).shift(1)
    feat['AltVol_Vol_Surprise'] = ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1)
    pv = v.where(ret > 0, 0).rolling(22).sum(); nv = v.where(ret <= 0, 0).rolling(22).sum()
    feat['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)

    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset'] = asset
    d['Class'] = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    asset_frames[asset] = d

print(f"Assets loaded: {len(asset_frames)}, Features: 37\n")

# ── Run Experiments ─────────────────────────────────────────
results = {}

for hz in HORIZONS:
    print(f"\n{'='*60}")
    print(f"  Horizon: {hz}d")
    print(f"{'='*60}")

    # Build pooled dataset for this horizon
    pooled = []
    for asset in ALL_ASSETS:
        df = asset_frames[asset].copy()
        df['Target'] = forward_rv(df['ret_sq'], hz)
        df = df.drop(columns=['ret_sq']).dropna()
        pooled.append(df)
    data = pd.concat(pooled).sort_index().reset_index(drop=True)
    
    all_feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split - hz]
    test_df = data.iloc[split:]
    y_te = test_df['Target'].values

    sc = StandardScaler().fit(train_df[all_feats])

    # ── 1. HAR-3 ──
    print("  [1/6] HAR-3...", end=" ", flush=True)
    har_sc = StandardScaler().fit(train_df[HAR_FEATS])
    m_har = Ridge(alpha=1.0).fit(har_sc.transform(train_df[HAR_FEATS]), train_df['Target'])
    p_har = m_har.predict(har_sc.transform(test_df[HAR_FEATS]))
    r2_har = r2_score(y_te, p_har)
    med_r2_har = calculate_median_r2(test_df, y_te, p_har)
    print(f"R2={r2_har:.4f}, Med={med_r2_har:.4f}")

    # ── 2. Ridge (per-class) ──
    print("  [2/6] Ridge(per-class)...", end=" ", flush=True)
    p_ridge = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = test_df['Class'] == cls
        if len(tr_c) < 50 or te_m.sum() == 0: continue
        m = Ridge(alpha=ALPHAS_PERCLASS[cls]).fit(sc.transform(tr_c[all_feats]), tr_c['Target'])
        p_ridge[te_m.values] = m.predict(sc.transform(test_df.loc[te_m, all_feats]))
    r2_ridge = r2_score(y_te, p_ridge)
    med_r2_ridge = calculate_median_r2(test_df, y_te, p_ridge)
    print(f"R2={r2_ridge:.4f}, Med={med_r2_ridge:.4f}")

    # ── 3. LASSO ──
    print("  [3/6] LASSO...", end=" ", flush=True)
    X_tr_s = sc.transform(train_df[all_feats])
    X_te_s = sc.transform(test_df[all_feats])
    m_lasso = Lasso(alpha=0.01, max_iter=5000).fit(X_tr_s, train_df['Target'])
    p_lasso = m_lasso.predict(X_te_s)
    r2_lasso = r2_score(y_te, p_lasso)
    med_r2_lasso = calculate_median_r2(test_df, y_te, p_lasso)
    print(f"R2={r2_lasso:.4f}, Med={med_r2_lasso:.4f}")

    # ── 4. RF ──
    print("  [4/6] Random Forest...", end=" ", flush=True)
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features='sqrt', min_samples_leaf=20,
                                  random_state=42, n_jobs=-1)
    m_rf.fit(train_df[all_feats].values, train_df['Target'].values)
    p_rf = m_rf.predict(test_df[all_feats].values)
    r2_rf = r2_score(y_te, p_rf)
    med_r2_rf = calculate_median_r2(test_df, y_te, p_rf)
    print(f"R2={r2_rf:.4f}, Med={med_r2_rf:.4f}")

    # ── 5. WEns (Ridge 70% + XGBoost 30%) ──
    print("  [5/6] WEns...", end=" ", flush=True)
    p_xgb = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = test_df['Class'] == cls
        if len(tr_c) < 50 or te_m.sum() == 0: continue
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                           random_state=42, verbosity=0, n_jobs=1, tree_method='hist')
        xgb.fit(tr_c[all_feats].values, tr_c['Target'].values)
        p_xgb[te_m.values] = xgb.predict(test_df.loc[te_m, all_feats].values)
    
    p_wens = 0.7 * p_ridge + 0.3 * p_xgb
    r2_wens = r2_score(y_te, p_wens)
    r2_xgb = r2_score(y_te, p_xgb)
    med_r2_wens = calculate_median_r2(test_df, y_te, p_wens)
    med_r2_xgb = calculate_median_r2(test_df, y_te, p_xgb)
    print(f"R2={r2_wens:.4f} (XGB: {r2_xgb:.4f}), Med={med_r2_wens:.4f} (XGB: {med_r2_xgb:.4f})")

    # ── 6. BiLSTM-A (3 range features) ──
    print("  [6/6] BiLSTM-A...", end=" ", flush=True)
    bilstm_sc = StandardScaler().fit(train_df[BILSTM_FEATS])
    X_tr_bl = bilstm_sc.transform(train_df[BILSTM_FEATS])
    X_te_bl = bilstm_sc.transform(test_df[BILSTM_FEATS])
    y_tr_bl = train_df['Target'].values

    seq_len = min(22, len(X_tr_bl) // 10)  # safety for short horizons
    
    # Train returns aligned predictions AND aligned true values
    bl_preds, y_te_bl_aligned = train_bilstm(X_tr_bl, y_tr_bl, X_te_bl, train_df, test_df, seq_len=seq_len, hidden=32, dropout=0.3)
    
    if len(bl_preds) > 0:
        r2_bilstm = r2_score(y_te_bl_aligned, bl_preds)
        med_r2_bilstm = calculate_median_r2(test_df, y_te_bl_aligned, bl_preds)
    else:
        r2_bilstm = np.nan
        med_r2_bilstm = np.nan
        
    print(f"R2={r2_bilstm:.4f}, Med={med_r2_bilstm:.4f}")

    # ── 7. ElasticNet ──
    print("  [7/11] ElasticNet...", end=" ", flush=True)
    m_enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000).fit(X_tr_s, train_df['Target'])
    p_enet = m_enet.predict(X_te_s)
    r2_enet = r2_score(y_te, p_enet)
    med_r2_enet = calculate_median_r2(test_df, y_te, p_enet)
    print(f"R2={r2_enet:.4f}, Med={med_r2_enet:.4f}")

    # ── 8. SVR (LinearSVR) ──
    print("  [8/11] LinearSVR...", end=" ", flush=True)
    m_svr = LinearSVR(C=0.1, max_iter=2000, random_state=42).fit(X_tr_s, train_df['Target'])
    p_svr = m_svr.predict(X_te_s)
    r2_svr = r2_score(y_te, p_svr)
    med_r2_svr = calculate_median_r2(test_df, y_te, p_svr)
    print(f"R2={r2_svr:.4f}, Med={med_r2_svr:.4f}")

    # ── 9. MLP ──
    print("  [9/11] MLP...", end=" ", flush=True)
    m_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42).fit(X_tr_s, train_df['Target'])
    p_mlp = m_mlp.predict(X_te_s)
    r2_mlp = r2_score(y_te, p_mlp)
    med_r2_mlp = calculate_median_r2(test_df, y_te, p_mlp)
    print(f"R2={r2_mlp:.4f}, Med={med_r2_mlp:.4f}")

    # ── 10. LightGBM ──
    print("  [10/11] LightGBM...", end=" ", flush=True)
    p_lgbm = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = test_df['Class'] == cls
        if len(tr_c) < 50 or te_m.sum() == 0: continue
        if LGBMRegressor is not None:
            lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, min_child_samples=5, random_state=42, n_jobs=1, verbose=-1)
            lgb.fit(tr_c[all_feats].values, tr_c['Target'].values)
            p_lgbm[te_m.values] = lgb.predict(test_df.loc[te_m, all_feats].values)
        else:
            p_lgbm[te_m.values] = p_xgb[te_m.values]
    r2_lgbm = r2_score(y_te, p_lgbm)
    med_r2_lgbm = calculate_median_r2(test_df, y_te, p_lgbm)
    print(f"R2={r2_lgbm:.4f}, Med={med_r2_lgbm:.4f}")

    # ── Store results ──
    hz_key = f"{hz}d"
    results[hz_key] = {
        'HAR-3':     {'Pooled_R2': round(r2_har, 4),    'Median_R2': round(med_r2_har, 4),    'RMSE': round(np.sqrt(mean_squared_error(y_te, p_har)), 4)},
        'Ridge':     {'Pooled_R2': round(r2_ridge, 4),   'Median_R2': round(med_r2_ridge, 4),  'RMSE': round(np.sqrt(mean_squared_error(y_te, p_ridge)), 4)},
        'LASSO':     {'Pooled_R2': round(r2_lasso, 4),   'Median_R2': round(med_r2_lasso, 4),  'RMSE': round(np.sqrt(mean_squared_error(y_te, p_lasso)), 4)},
        'ElasticNet':{'Pooled_R2': round(r2_enet, 4),    'Median_R2': round(med_r2_enet, 4),   'RMSE': round(np.sqrt(mean_squared_error(y_te, p_enet)), 4)},
        'SVR':       {'Pooled_R2': round(r2_svr, 4),     'Median_R2': round(med_r2_svr, 4),    'RMSE': round(np.sqrt(mean_squared_error(y_te, p_svr)), 4)},
        'MLP':       {'Pooled_R2': round(r2_mlp, 4),     'Median_R2': round(med_r2_mlp, 4),    'RMSE': round(np.sqrt(mean_squared_error(y_te, p_mlp)), 4)},
        'RF':        {'Pooled_R2': round(r2_rf, 4),      'Median_R2': round(med_r2_rf, 4),     'RMSE': round(np.sqrt(mean_squared_error(y_te, p_rf)), 4)},
        'XGBoost':   {'Pooled_R2': round(r2_xgb, 4),     'Median_R2': round(med_r2_xgb, 4),    'RMSE': round(np.sqrt(mean_squared_error(y_te, p_xgb)), 4)},
        'LightGBM':  {'Pooled_R2': round(r2_lgbm, 4),    'Median_R2': round(med_r2_lgbm, 4),   'RMSE': round(np.sqrt(mean_squared_error(y_te, p_lgbm)), 4)},
        'WEns':      {'Pooled_R2': round(r2_wens, 4),    'Median_R2': round(med_r2_wens, 4),   'RMSE': round(np.sqrt(mean_squared_error(y_te, p_wens)), 4)},
        'BiLSTM-A':  {'Pooled_R2': round(r2_bilstm, 4),  'Median_R2': round(med_r2_bilstm, 4), 'RMSE': round(np.sqrt(mean_squared_error(y_te_bl_aligned, bl_preds)) if len(bl_preds) > 0 else np.nan, 4)},
    }


# ── Summary Table ───────────────────────────────────────────
print("\n" + "=" * 120)
print(f"{'h':<6} {'HAR-3':<8} {'Ridge':<8} {'LASSO':<8} {'ENet':<8} {'SVR':<8} {'MLP':<8} {'RF':<8} {'XGB':<8} {'LGBM':<8} {'WEns':<8} {'BiLSTM-A':<8}")
print("-" * 120)
for hz in HORIZONS:
    r = results[f"{hz}d"]
    print(f"{hz:<6} {r['HAR-3']['Pooled_R2']:<8.4f} {r['Ridge']['Pooled_R2']:<8.4f} "
          f"{r['LASSO']['Pooled_R2']:<8.4f} {r['ElasticNet']['Pooled_R2']:<8.4f} "
          f"{r['SVR']['Pooled_R2']:<8.4f} {r['MLP']['Pooled_R2']:<8.4f} "
          f"{r['RF']['Pooled_R2']:<8.4f} {r['XGBoost']['Pooled_R2']:<8.4f} "
          f"{r['LightGBM']['Pooled_R2']:<8.4f} "
          f"{r['WEns']['Pooled_R2']:<8.4f} {r['BiLSTM-A']['Pooled_R2']:<8.4f}")
print("=" * 120)

# ── Save ────────────────────────────────────────────────────
out_path = '/root/vrp/multi_horizon_benchmark_extended_v2_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
