"""
LASSO-HAR vs full LASSO comparison (M7).
LASSO-HAR: LASSO with only 3 HAR features (lag1, lag5, lag22).
Full LASSO: existing results from main_benchmark_v6_results.json.
Outputs: paper/csv/lasso_har_comparison.csv
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HORIZONS   = [1, 5, 22, 60, 90, 120, 180, 252]
OUTER_RATIO = 0.8
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS  = [a for g in ASSET_GROUPS.values() for a in g]
ASSET_CLASS = {a: cls for cls, assets in ASSET_GROUPS.items() for a in assets}
HAR_FEATS   = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
ALPHA_GRID  = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(_SCRIPT_DIR, 'results', 'main_benchmark_v6_results.json')
CACHE_DIR    = os.path.join(_SCRIPT_DIR, 'data')
OUT_CSV      = os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv', 'lasso_har_comparison.csv')

with open(RESULTS_JSON) as f:
    saved = json.load(f)


def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)


def load_asset(asset):
    p = os.path.join(CACHE_DIR, f'{asset}.parquet')
    if not os.path.exists(p):
        raise FileNotFoundError(f"Cache not found: {p}. Run run_per_asset_r2.py first.")
    df = pd.read_parquet(p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].ffill()


def compute_har_features(asset):
    df = load_asset(asset)
    c      = df['Adj Close'].squeeze()
    ret    = np.log(c / c.shift(1))
    ret_sq = ret ** 2
    feat   = {}
    for lag, name in [(1, 'LogRV_lag1'), (5, 'LogRV_lag5'), (22, 'LogRV_lag22')]:
        feat[name] = np.log(ret_sq.rolling(lag).mean() * 252 + 1e-12).shift(1)
    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset']  = asset
    d['Class']  = ASSET_CLASS[asset]
    d.index.name = 'Date'
    return d.reset_index()


# Build panel with only HAR features
print("Loading HAR features for all assets...")
frames = {}
for asset in ALL_ASSETS:
    frames[asset] = compute_har_features(asset)
print(f"Loaded {len(frames)} assets")

rows = []

for hz in HORIZONS:
    hz_key = f'{hz}d'
    print(f"\nHorizon: {hz_key}")

    # Build panel
    panel_list = []
    for asset, df in frames.items():
        d = df.copy()
        d['Target'] = forward_rv(d['ret_sq'], hz)
        panel_list.append(d)
    panel = pd.concat(panel_list).sort_values('Date').reset_index(drop=True)
    panel = panel.dropna(subset=HAR_FEATS + ['Target'])

    dates      = panel['Date'].sort_values().unique()
    split_date = dates[int(len(dates) * OUTER_RATIO)]
    train_df   = panel[panel['Date'] < split_date]
    test_df    = panel[panel['Date'] >= split_date]

    y_te  = test_df['Target'].values
    inner_dates = train_df['Date'].sort_values().unique()
    inner_split = inner_dates[int(len(inner_dates) * 0.8)]

    # Inner holdout alpha search (per class)
    best_alpha = {}
    for cls in ASSET_GROUPS:
        tr_inner = train_df[(train_df['Class'] == cls) & (train_df['Date'] < inner_split)]
        va_inner = train_df[(train_df['Class'] == cls) & (train_df['Date'] >= inner_split)]
        if len(tr_inner) < 5 or len(va_inner) < 2:
            best_alpha[cls] = 0.01
            continue
        best_r2, best_a = -np.inf, 0.01
        sc = StandardScaler().fit(tr_inner[HAR_FEATS].values)
        for a in ALPHA_GRID:
            m = Lasso(alpha=a, max_iter=5000).fit(sc.transform(tr_inner[HAR_FEATS]), tr_inner['Target'].values)
            p = m.predict(sc.transform(va_inner[HAR_FEATS]))
            r2 = r2_score(va_inner['Target'].values, p)
            if r2 > best_r2:
                best_r2, best_a = r2, a
        best_alpha[cls] = best_a
    print(f"  best alpha: {best_alpha}")

    # Fit on full train, predict test
    p_lasso_har = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0:
            continue
        sc = StandardScaler().fit(tr_c[HAR_FEATS].values)
        m  = Lasso(alpha=best_alpha[cls], max_iter=5000).fit(
            sc.transform(tr_c[HAR_FEATS]), tr_c['Target'].values)
        p_lasso_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, HAR_FEATS]))

    valid = ~np.isnan(p_lasso_har) & ~np.isnan(y_te)
    pooled_lasso_har = float(r2_score(y_te[valid], p_lasso_har[valid]))
    pooled_lasso_full = saved.get(hz_key, {}).get('LASSO', {}).get('Pooled_R2', np.nan)
    pooled_har3       = saved.get(hz_key, {}).get('HAR-3',  {}).get('Pooled_R2', np.nan)
    pooled_wens       = saved.get(hz_key, {}).get('WEns',   {}).get('Pooled_R2', np.nan)

    print(f"  LASSO-HAR={pooled_lasso_har:.4f}  HAR-3={pooled_har3:.4f}  LASSO(full)={pooled_lasso_full:.4f}  WEns={pooled_wens:.4f}")

    rows.append({
        'Horizon':       hz_key,
        'LASSO_HAR':     round(pooled_lasso_har, 4),
        'HAR_3':         pooled_har3,
        'LASSO_full':    pooled_lasso_full,
        'WEns':          pooled_wens,
        'Delta_vs_HAR3': round(pooled_lasso_har - pooled_har3, 4),
        'Delta_full_vs_HAR': round(pooled_lasso_full - pooled_lasso_har, 4),
    })

df_out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print("\n" + "="*70)
print(f"{'Horizon':<8} {'LASSO-HAR':>10} {'HAR-3':>8} {'LASSO(35)':>10} {'WEns':>8} {'Δ(LH-H3)':>10} {'Δ(L35-LH)':>10}")
print("-"*70)
for _, r in df_out.iterrows():
    print(f"{r['Horizon']:<8} {r['LASSO_HAR']:>10.4f} {r['HAR_3']:>8.4f} {r['LASSO_full']:>10.4f} {r['WEns']:>8.4f} {r['Delta_vs_HAR3']:>10.4f} {r['Delta_full_vs_HAR']:>10.4f}")
