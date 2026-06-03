"""
WEns Weight Combination Experiment
===================================
Compare Ridge:XGBoost weight combinations for the 22d horizon WEns ensemble.
Tests weights from 0:100 to 100:0 in 10% increments.
Reports Pooled R2, Median R2, and RMSE for each combination.
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from arch import arch_model

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HORIZON = 22

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

print("Loading data...", flush=True)
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

# Build pooled dataset for h=22
print(f"\nBuilding dataset for h={HORIZON}...", flush=True)
pooled = []
for asset in ALL_ASSETS:
    df = asset_frames[asset].copy()
    df['Target'] = forward_rv(df['ret_sq'], HORIZON)
    df = df.drop(columns=['ret_sq']).dropna()
    pooled.append(df)

data = pd.concat(pooled).sort_index().reset_index(drop=True)
feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]

split = int(len(data) * 0.8)
train_df = data.iloc[:split - HORIZON]
test_df = data.iloc[split:]

sc = StandardScaler().fit(train_df[feats])
y_te = test_df['Target'].values

alphas_perclass = {'Equity': 100.0, 'Bond': 10.0, 'Commodity': 10.0}

# Train Ridge per-class
print("Training Ridge per-class...", flush=True)
preds_ridge = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_m = test_df['Class'] == cls
    if len(tr_c) < 50 or te_m.sum() == 0: continue
    m = Ridge(alpha=alphas_perclass[cls]).fit(sc.transform(tr_c[feats]), tr_c['Target'])
    preds_ridge[te_m.values] = m.predict(sc.transform(test_df.loc[te_m, feats]))

# Train XGBoost per-class
print("Training XGBoost per-class...", flush=True)
preds_xgb = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_m = test_df['Class'] == cls
    if len(tr_c) < 50 or te_m.sum() == 0: continue
    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                       random_state=42, verbosity=0, n_jobs=1, tree_method='hist')
    xgb.fit(tr_c[feats].values, tr_c['Target'].values)
    preds_xgb[te_m.values] = xgb.predict(test_df.loc[te_m, feats].values)

# Test weight combinations
print("\n" + "="*80)
print("WEns Weight Combination Experiment (22d horizon)")
print("="*80)

weights = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]

results = []
print(f"\n{'Weight (Ridge:XGB)':<22} {'Pooled R2':>10} {'Median R2':>10} {'RMSE':>10} {'MAE':>10}")
print("-" * 66)

best_pooled = {'w': None, 'r2': -999}
best_median = {'w': None, 'r2': -999}

for w_ridge, w_xgb in weights:
    preds = w_ridge * preds_ridge + w_xgb * preds_xgb
    pooled_r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae = mean_absolute_error(y_te, preds)

    asset_r2s = []
    for asset in ALL_ASSETS:
        am = (test_df['Asset'] == asset).values
        if am.sum() > 10:
            asset_r2s.append(r2_score(y_te[am], preds[am]))
    med_r2 = np.median(asset_r2s) if asset_r2s else np.nan

    marker = " <-- CURRENT" if (w_ridge == 0.7 and w_xgb == 0.3) else ""

    print(f"  {int(w_ridge*100):>3}:{int(w_xgb*100):<3}              {pooled_r2:>10.4f} {med_r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}{marker}")

    if pooled_r2 > best_pooled['r2']:
        best_pooled = {'w': f"{int(w_ridge*100)}:{int(w_xgb*100)}", 'r2': pooled_r2}
    if med_r2 > best_median['r2']:
        best_median = {'w': f"{int(w_ridge*100)}:{int(w_xgb*100)}", 'r2': med_r2}

    results.append({
        'ridge_weight': w_ridge,
        'xgb_weight': w_xgb,
        'label': f"{int(w_ridge*100)}:{int(w_xgb*100)}",
        'pooled_r2': round(pooled_r2, 4),
        'median_r2': round(med_r2, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
    })

print("-" * 66)
print(f"\nBest by Pooled R2:  {best_pooled['w']}  (R2 = {best_pooled['r2']:.4f})")
print(f"Best by Median R2:  {best_median['w']}  (R2 = {best_median['r2']:.4f})")

# Fine-grained search
print("\n\n" + "="*80)
print("Fine-grained Search around 70:30 (5% increments)")
print("="*80)

fine_weights = [(w/100, 1-w/100) for w in range(55, 86, 5)]
print(f"\n{'Weight (Ridge:XGB)':<22} {'Pooled R2':>10} {'Median R2':>10} {'RMSE':>10}")
print("-" * 56)

for w_ridge, w_xgb in fine_weights:
    preds = w_ridge * preds_ridge + w_xgb * preds_xgb
    pooled_r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    asset_r2s = []
    for asset in ALL_ASSETS:
        am = (test_df['Asset'] == asset).values
        if am.sum() > 10:
            asset_r2s.append(r2_score(y_te[am], preds[am]))
    med_r2 = np.median(asset_r2s)

    marker = " <-- CURRENT" if abs(w_ridge - 0.7) < 0.01 else ""
    print(f"  {int(w_ridge*100):>3}:{int(w_xgb*100):<3}              {pooled_r2:>10.4f} {med_r2:>10.4f} {rmse:>10.4f}{marker}")

# Save results
out = {
    'experiment': 'WEns Weight Combination (22d)',
    'best_pooled': best_pooled,
    'best_median': best_median,
    'results': results,
}
out_path = '/root/vrp/paper/csv/wens_weight_experiment_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
