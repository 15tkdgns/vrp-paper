"""WEns + ElasticNet-59 median R2 산출 (XGBoost n_estimators 축소)"""
import sys, os
sys.path.insert(0, '/root/vrp')
import numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from src.experiments.creative.v71_model_comparison import build_dataset, ALL_ASSETS, ASSET_GROUPS

print("Loading dataset...", flush=True)
data, feats = build_dataset()
split_idx = int(len(data) * 0.8)
train_df = data.iloc[:split_idx]; test_df = data.iloc[split_idx:]
y_te = test_df['Target'].values
alphas_perclass = {'Equity': 100.0, 'Bond': 10.0, 'Commodity': 10.0}

def compute_metrics(y_true, preds, test_df_sub):
    pooled_r2 = r2_score(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    asset_r2 = {}
    for a in ALL_ASSETS:
        m = (test_df_sub['Asset'] == a).values
        if m.sum() > 10:
            asset_r2[a] = float(r2_score(y_true[m], preds[m]))
    r2_vals = list(asset_r2.values())
    return {'pooled_r2': float(pooled_r2), 'rmse': float(rmse),
            'median_r2': float(np.median(r2_vals)), 'asset_r2': asset_r2}

results = {}

# -- WEns --
print("\n=== WEns ===", flush=True)
sc37 = StandardScaler().fit(train_df[feats])
preds_ridge = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_idx = test_df['Class'] == cls
    if len(tr_c) < 100 or te_idx.sum() == 0: continue
    m = Ridge(alpha=alphas_perclass[cls]).fit(sc37.transform(tr_c[feats]), tr_c['Target'])
    preds_ridge[te_idx.values] = m.predict(sc37.transform(test_df.loc[te_idx, feats]))

from xgboost import XGBRegressor
preds_xgb = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_idx = test_df['Class'] == cls
    if len(tr_c) < 100 or te_idx.sum() == 0: continue
    m = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
        random_state=42, verbosity=0, n_jobs=1, tree_method='hist'
    ).fit(tr_c[feats].values, tr_c['Target'].values)
    preds_xgb[te_idx.values] = m.predict(test_df.loc[te_idx, feats].values)
    print(f"  XGB {cls} done", flush=True)

preds_wens = 0.7 * preds_ridge + 0.3 * preds_xgb
results['WEns'] = compute_metrics(y_te, preds_wens, test_df)
print(f"  WEns: Pooled={results['WEns']['pooled_r2']:.4f}, Median={results['WEns']['median_r2']:.4f}")
print(f"  Asset R2: {results['WEns']['asset_r2']}")

# Ridge only (for reference)
results['Ridge_perclass'] = compute_metrics(y_te, preds_ridge, test_df)
print(f"  Ridge: Pooled={results['Ridge_perclass']['pooled_r2']:.4f}, Median={results['Ridge_perclass']['median_r2']:.4f}")

# -- ElasticNet-59 --
print("\n=== ElasticNet-59 ===", flush=True)
ext_train = train_df.copy(); ext_test = test_df.copy()
cross_lag_feats = []
for f in ['RogersSatchell_22', 'GarmanKlass_22', 'Parkinson_22']:
    for lag in [5, 10]:
        fn = f'{f}_lag{lag}'
        ext_train[fn] = ext_train.groupby('Asset')[f].shift(lag).fillna(0)
        ext_test[fn] = ext_test.groupby('Asset')[f].shift(lag).fillna(0)
        cross_lag_feats.append(fn)
for f in ['LogRV_lag1', 'IV_VIX']:
    for f2 in ['RogersSatchell_22', 'GarmanKlass_22']:
        fn = f'{f}_x_{f2}'
        ext_train[fn] = ext_train[f] * ext_train[f2]
        ext_test[fn] = ext_test[f] * ext_test[f2]
        cross_lag_feats.append(fn)
for f in ['Parkinson_5', 'LogRV_Std5']:
    for f2 in ['Parkinson_22', 'LogRV_Std22']:
        fn = f'{f}_div_{f2}'
        ext_train[fn] = ext_train[f] / (ext_train[f2].abs() + 1e-10)
        ext_test[fn] = ext_test[f] / (ext_test[f2].abs() + 1e-10)
        cross_lag_feats.append(fn)
for f in ['RogersSatchell_22', 'IV_VIX', 'Garch_Weekly']:
    fn = f'{f}_sq'
    ext_train[fn] = ext_train[f] ** 2; ext_test[fn] = ext_test[f] ** 2
    cross_lag_feats.append(fn)
fn = 'LogRV_lag1_ma5'
ext_train[fn] = ext_train.groupby('Asset')['LogRV_lag1'].transform(lambda x: x.rolling(5, min_periods=1).mean())
ext_test[fn] = ext_test.groupby('Asset')['LogRV_lag1'].transform(lambda x: x.rolling(5, min_periods=1).mean())
cross_lag_feats.append(fn)

ext_feats = feats + cross_lag_feats
for df in [ext_train, ext_test]:
    df[ext_feats] = df[ext_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"  Features: {len(ext_feats)}")

sc_ext = StandardScaler().fit(ext_train[ext_feats])
preds_enet = np.full(len(ext_test), np.nan)
for cls in ASSET_GROUPS:
    tr_c = ext_train[ext_train['Class'] == cls]
    te_idx = ext_test['Class'] == cls
    if len(tr_c) < 100 or te_idx.sum() == 0: continue
    m = ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=5000, random_state=42)
    m.fit(sc_ext.transform(tr_c[ext_feats]), tr_c['Target'])
    preds_enet[te_idx.values] = m.predict(sc_ext.transform(ext_test.loc[te_idx, ext_feats]))
    print(f"  ElasticNet {cls} done", flush=True)

results['ElasticNet-59'] = compute_metrics(y_te, preds_enet, test_df)
results['ElasticNet-59']['n_features'] = len(ext_feats)
print(f"  ElasticNet-59: Pooled={results['ElasticNet-59']['pooled_r2']:.4f}, Median={results['ElasticNet-59']['median_r2']:.4f}")
print(f"  Asset R2: {results['ElasticNet-59']['asset_r2']}")

# -- HAR-3 (already computed, re-run for completeness) --
print("\n=== HAR-3 ===", flush=True)
har_feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
sc_h = StandardScaler().fit(train_df[har_feats])
preds_har = np.full(len(test_df), np.nan)
for cls in ASSET_GROUPS:
    tr_c = train_df[train_df['Class'] == cls]
    te_idx = test_df['Class'] == cls
    if len(tr_c) < 100 or te_idx.sum() == 0: continue
    m = Ridge(alpha=alphas_perclass[cls]).fit(sc_h.transform(tr_c[har_feats]), tr_c['Target'])
    preds_har[te_idx.values] = m.predict(sc_h.transform(test_df.loc[te_idx, har_feats]))
results['HAR-3'] = compute_metrics(y_te, preds_har, test_df)
print(f"  HAR-3: Pooled={results['HAR-3']['pooled_r2']:.4f}, Median={results['HAR-3']['median_r2']:.4f}")

# -- Summary --
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
for n in ['HAR-3', 'Ridge_perclass', 'WEns', 'ElasticNet-59']:
    r = results[n]
    print(f"{n:>15}: Pooled={r['pooled_r2']:.4f}, Median={r['median_r2']:.4f}, RMSE={r['rmse']:.4f}")

out_path = '/root/vrp/paper/csv/table_3_1b_missing.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {out_path}")
