"""
R²_Within (time-series within-asset R²) computation.
Removes asset-level mean from y and ŷ, then computes R².
Contrasts with Pooled R² to show how much prediction comes from
cross-sectional vs. time-series variation.
Output: paper/csv/r2_within.csv
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HORIZONS    = [1, 5, 22, 60, 90, 120, 180, 252]
OUTER_RATIO = 0.8
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS  = [a for g in ASSET_GROUPS.values() for a in g]
ASSET_CLASS = {a: cls for cls, assets in ASSET_GROUPS.items() for a in assets}
HAR_FEATS   = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

RESULTS_JSON = '/root/vrp/results/main_benchmark_v6_results.json'
CACHE_DIR    = '/root/vrp/data'
OUT_CSV      = '/root/vrp/paper/csv/r2_within.csv'

with open(RESULTS_JSON) as f:
    saved = json.load(f)


def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    return np.log((cs.shift(-horizon) - cs) / horizon * 252 + 1e-12)


def load_asset(asset):
    p = os.path.join(CACHE_DIR, f'{asset}.parquet')
    df = pd.read_parquet(p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].ffill()


def compute_features(asset, spy_close):
    df = load_asset(asset)
    c      = df['Adj Close'].squeeze()
    o_r, h_r, l_r = df['Open'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
    v      = df['Volume'].squeeze()
    ret    = np.log(c / c.shift(1))
    ret_sq = ret ** 2
    lrv    = np.log(ret_sq.rolling(22).mean() * 252 + 1e-12)

    feat = {}
    for lag, name in [(1,'LogRV_lag1'),(5,'LogRV_lag5'),(10,'LogRV_lag10'),(22,'LogRV_lag22')]:
        feat[name] = np.log(ret_sq.rolling(lag).mean() * 252 + 1e-12).shift(1)
    feat['LogRV_Std5']  = lrv.rolling(5).std().shift(1)
    feat['LogRV_Std22'] = lrv.rolling(22).std().shift(1)
    feat['RV_Mom5']     = (lrv - lrv.shift(5)).shift(1)
    feat['RV_Mom22']    = (lrv - lrv.shift(22)).shift(1)
    feat['SPY_LogRV']   = np.log(spy_close.pct_change()**2 * 252 + 1e-12).reindex(c.index).ffill().shift(1) \
                          if asset != 'SPY' else lrv.shift(1)
    feat['Ret_lag1']     = ret.shift(1)
    feat['Ret_abs_lag1'] = ret.abs().shift(1)
    spy_ret = spy_close.pct_change()
    feat['Corr_SPY'] = ret.rolling(22).corr(spy_ret.reindex(c.index)).shift(1)

    def park(h, l, w): return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)
    def gk(o, h, l, c2, w):
        hl=np.log(h/l); co=np.log(c2/o)
        return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
    def rs(o, h, l, c2, w):
        r2=np.log(h/c2)*np.log(h/o)+np.log(l/c2)*np.log(l/o)
        return np.sqrt(r2.rolling(w).mean().clip(0)*252)

    feat['Parkinson_5']       = park(h_r, l_r, 5).shift(1)
    feat['Parkinson_22']      = park(h_r, l_r, 22).shift(1)
    feat['GarmanKlass_22']    = gk(o_r, h_r, l_r, c, 22).shift(1)
    feat['RogersSatchell_22'] = rs(o_r, h_r, l_r, c, 22).shift(1)
    p22 = park(h_r, l_r, 22)
    feat['Range_Close_Ratio'] = (np.log(p22+1e-6)-lrv).shift(1)
    on = np.log(o_r / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)

    vix_df = pd.read_parquet(os.path.join(CACHE_DIR, 'VIX.parquet'))
    vix = vix_df['Close'].reindex(c.index).ffill()
    iv  = (vix/100)**2
    feat['IV_VIX']       = iv.shift(1)
    feat['IV_VIX_chg']   = iv.diff().shift(1)
    feat['IV_VIX_ma5']   = iv.rolling(5).mean().shift(1)
    feat['IV_VIX_std5']  = iv.rolling(5).std().shift(1)

    vix3m = vix_df.get('Close_3M', vix).reindex(c.index).ffill()
    iv3m  = (vix3m/100)**2
    feat['IV_VIX3M']          = iv3m.shift(1)
    feat['IV_VIX_TermSlope']  = (iv3m - iv).shift(1)
    vix9d = vix_df.get('Close_9D', vix).reindex(c.index).ffill()
    iv9d  = (vix9d/100)**2
    feat['IV_VIX9D']           = iv9d.shift(1)
    feat['IV_VIX_ShortSlope']  = (iv - iv9d).shift(1)

    spy_rv_val = ret_sq.rolling(22).mean() * 252
    vrp_val    = iv - spy_rv_val
    feat['IV_VRP']      = vrp_val.shift(1)
    feat['IV_VRP_ma22'] = vrp_val.rolling(22).mean().shift(1)

    dv = v * c
    feat['AltVol_Amihud']          = (ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']       = (v.rolling(5).mean()/(v.rolling(22).mean()+1e-10)).shift(1)
    feat['AltVol_PV_Corr']         = ret.rolling(22).corr(np.log(v+1)).shift(1)
    feat['AltVol_Vol_Surprise']    = ((v-v.rolling(22).mean())/(v.rolling(22).std()+1e-10)).shift(1)
    pv = v.where(ret>0, 0).rolling(22).sum()
    nv = v.where(ret<=0, 0).rolling(22).sum()
    feat['AltVol_Order_Imbalance'] = ((pv-nv)/(pv+nv+1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']     = (ret.abs().rolling(22).sum()/(v.rolling(22).sum()+1e-10)*1e6).shift(1)

    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset']  = asset
    d['Class']  = ASSET_CLASS[asset]
    d.index.name = 'Date'
    return d.reset_index()


def r2_within(y_true, y_pred, groups):
    """R²_Within: demean by group (asset) then compute R²."""
    y  = np.array(y_true, dtype=float)
    yh = np.array(y_pred, dtype=float)
    grp = np.array(groups)
    y_dm  = y.copy()
    yh_dm = yh.copy()
    for g in np.unique(grp):
        mask = grp == g
        y_dm[mask]  -= y[mask].mean()
        yh_dm[mask] -= yh[mask].mean()
    ss_res = ((y_dm - yh_dm)**2).sum()
    ss_tot = (y_dm**2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


print("Loading features...")
spy_close = load_asset('SPY')['Adj Close'].squeeze()
asset_frames = {}
for asset in ALL_ASSETS:
    asset_frames[asset] = compute_features(asset, spy_close)
print(f"Loaded {len(asset_frames)} assets")

FEATS = [c for c in next(iter(asset_frames.values())).columns
         if c not in ['Date', 'Asset', 'Class', 'ret_sq']]

# ── Models to evaluate ────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

rows = []

for hz in HORIZONS:
    hz_key = f'{hz}d'
    print(f"\nHorizon: {hz_key}")

    frames = []
    for asset, df in asset_frames.items():
        d = df.copy()
        d['Target'] = forward_rv(d['ret_sq'], hz)
        frames.append(d)
    panel = pd.concat(frames).sort_values('Date').reset_index(drop=True)
    panel = panel.dropna(subset=FEATS + ['Target'])

    dates      = panel['Date'].sort_values().unique()
    split_date = dates[int(len(dates) * OUTER_RATIO)]
    train_df   = panel[panel['Date'] < split_date]
    test_df    = panel[panel['Date'] >= split_date]
    y_te       = test_df['Target'].values
    assets_te  = test_df['Asset'].values

    bp_ridge = saved.get(hz_key, {}).get('Ridge', {}).get('best_params', {})
    bp_wens  = saved.get(hz_key, {}).get('WEns',  {}).get('best_params', {})

    # ── HAR-3 ──
    har_idx = [FEATS.index(f) for f in HAR_FEATS if f in FEATS]
    p_har   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5: continue
        sc = StandardScaler().fit(tr_c[FEATS].values)
        m  = Ridge(alpha=1.0).fit(sc.transform(tr_c[FEATS])[:, har_idx], tr_c['Target'].values)
        p_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS])[:, har_idx])

    # ── Ridge (full 35 feats) ──
    p_ridge  = np.full(len(test_df), np.nan)
    scalers  = {}
    r_models = {}
    for cls in ASSET_GROUPS:
        tr_c  = train_df[train_df['Class'] == cls]
        te_m  = (test_df['Class'] == cls).values
        if len(tr_c) < 5: continue
        alpha = bp_ridge.get(cls, {}).get('alpha', 1000) if isinstance(bp_ridge, dict) and cls in bp_ridge else 1000
        sc = StandardScaler().fit(tr_c[FEATS].values)
        m  = Ridge(alpha=alpha).fit(sc.transform(tr_c[FEATS]), tr_c['Target'].values)
        p_ridge[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS]))
        scalers[cls]  = sc
        r_models[cls] = m

    # ── WEns ──
    if HAS_XGB and bp_wens:
        pw = bp_wens.get('pw', 0.8)
        p_xgb = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = train_df[train_df['Class'] == cls]
            te_m = (test_df['Class'] == cls).values
            if len(tr_c) < 5: continue
            xp = bp_wens.get('XGBoost', {}).get(cls, {'max_depth': 3, 'learning_rate': 0.03})
            sc = scalers.get(cls, StandardScaler().fit(tr_c[FEATS].values))
            m  = XGBRegressor(n_estimators=300, random_state=42,
                              max_depth=xp.get('max_depth', 3),
                              learning_rate=xp.get('learning_rate', 0.03), verbosity=0)
            m.fit(sc.transform(tr_c[FEATS]), tr_c['Target'].values)
            p_xgb[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS]))
        p_wens = pw * p_ridge + (1 - pw) * p_xgb
    else:
        p_wens = p_ridge.copy()

    for mname, preds in [('HAR-3', p_har), ('Ridge', p_ridge), ('WEns', p_wens)]:
        valid = ~np.isnan(preds) & ~np.isnan(y_te)
        pooled = float(r2_score(y_te[valid], preds[valid]))
        within = r2_within(y_te[valid], preds[valid], assets_te[valid])
        print(f"  {mname:<8} Pooled={pooled:.4f}  Within={within:.4f}  diff={pooled-within:+.4f}")
        rows.append({
            'Horizon':      hz_key,
            'Model':        mname,
            'Pooled_R2':    round(pooled, 4),
            'Within_R2':    round(within, 4),
            'Diff':         round(pooled - within, 4),
        })

df_out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

print("\n" + "="*60)
print("WEns — Pooled R² vs Within R²")
print("="*60)
wens = df_out[df_out['Model'] == 'WEns']
print(f"{'Horizon':<8} {'Pooled':>8} {'Within':>8} {'Diff(P-W)':>10}")
print("-"*38)
for _, r in wens.iterrows():
    print(f"{r['Horizon']:<8} {r['Pooled_R2']:>8.4f} {r['Within_R2']:>8.4f} {r['Diff']:>10.4f}")
