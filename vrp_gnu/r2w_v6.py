import os, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HAR_FEATS  = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

def cp(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean() / (4*np.log(2))) * np.sqrt(252)

def cgk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)

def crs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def frv(ret_sq, hz):
    cs = ret_sq.cumsum()
    return np.log((cs.shift(-hz) - cs) / hz * 252 + 1e-12)

def r2w(yt, yp, grp):
    y = np.array(yt, dtype=float); yh = np.array(yp, dtype=float); g = np.array(grp)
    yd = y.copy(); yhd = yh.copy()
    for gg in np.unique(g):
        m = g == gg; yd[m] -= y[m].mean(); yhd[m] -= yh[m].mean()
    ss_r = ((yd - yhd)**2).sum(); ss_t = (yd**2).sum()
    return float(1 - ss_r/ss_t) if ss_t > 0 else np.nan

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH   = 'src/data/v71_ohlcv_cache.pkl'
_DATA_DIR   = os.path.join(_SCRIPT_DIR, 'data')

def _load_from_parquet():
    vix_df = pd.read_parquet(os.path.join(_DATA_DIR, 'VIX.parquet'))
    frames = {}
    for asset in ALL_ASSETS + ['VIX']:
        p = os.path.join(_DATA_DIR, f'{asset}.parquet')
        if not os.path.exists(p): continue
        frames[asset] = pd.read_parquet(p)
    combined = pd.concat(frames.values(), axis=1)
    combined[('Close', 'VIX')]   = vix_df['Close']
    combined[('Close', 'VIX3M')] = vix_df['Close_3M']
    combined[('Close', 'VIX9D')] = vix_df['Close_9D']
    return combined

print('Loading data...', flush=True)
if os.path.exists(_PKL_PATH):
    raw = pd.read_pickle(_PKL_PATH)
else:
    raw = _load_from_parquet()
vix = raw[('Close', 'VIX')]; spy_c = raw[('Close', 'SPY')]
spy_ret = np.log(spy_c / spy_c.shift(1)).dropna()
spy_rv  = (spy_ret**2).rolling(22).mean() * 252 * 10000
spy_lrv = np.log(spy_rv + 1e-6)

ivf = {
    'VIX':           np.log(vix + 1e-6),
    'VIX_chg':       np.log(vix + 1e-6).diff(),
    'VIX_ma5':       np.log(vix + 1e-6).rolling(5).mean(),
    'VIX_std5':      np.log(vix + 1e-6).rolling(5).std(),
    'VIX3M':         np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX_TermSlope': np.log(vix + 1e-6) - np.log(raw[('Close', 'VIX3M')] + 1e-6),
    'VIX9D':         np.log(raw[('Close', 'VIX9D')] + 1e-6),
    'VIX_ShortSlope':np.log(raw[('Close', 'VIX9D')] + 1e-6) - np.log(vix + 1e-6),
}
vrp = (vix**2 / 100) - spy_rv / 10000
ivf['VRP'] = vrp; ivf['VRP_ma22'] = vrp.rolling(22).mean()

af = {}
for asset in ALL_ASSETS:
    c = raw[('Close', asset)]; o = raw[('Open', asset)]
    h = raw[('High', asset)];  l = raw[('Low', asset)]; v = raw[('Volume', asset)]
    ret = np.log(c / c.shift(1)).dropna(); rs2 = ret**2
    rv2 = rs2.rolling(22).mean() * 252 * 10000; lrv = np.log(rv2 + 1e-6)
    ft = {
        'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
        'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
        'LogRV_Std5': lrv.rolling(5).std().shift(1),
        'LogRV_Std22': lrv.rolling(22).std().shift(1),
        'RV_Mom5': (lrv - lrv.shift(5)).shift(1),
        'RV_Mom22': (lrv - lrv.shift(22)).shift(1),
        'SPY_LogRV': spy_lrv.shift(1),
        'Ret_lag1': ret.shift(1), 'Ret_abs_lag1': ret.abs().shift(1),
        'Corr_SPY': (ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                     if asset != 'SPY' else pd.Series(1.0, index=ret.index)),
    }
    p5 = cp(h, l, 5); p22 = cp(h, l, 22)
    gk22 = cgk(o, h, l, c, 22); rs22 = crs(o, h, l, c, 22)
    ft.update({
        'Parkinson_5':       np.log(p5   + 1e-6).shift(1),
        'Parkinson_22':      np.log(p22  + 1e-6).shift(1),
        'GarmanKlass_22':    np.log(gk22 + 1e-6).shift(1),
        'RogersSatchell_22': np.log(rs22 + 1e-6).shift(1),
        'Range_Close_Ratio': (np.log(p22 + 1e-6) - lrv).shift(1),
    })
    on = np.log(o / c.shift(1))
    ft['Overnight_Vol'] = on.rolling(22).std().shift(1)
    ft['Overnight_Ret'] = on.shift(1)
    for k, val in ivf.items(): ft['IV_' + k] = val.shift(1)
    dv = v * c
    ft.update({
        'AltVol_Amihud':         (ret.abs() / (dv + 1e-10)).rolling(22).mean().shift(1),
        'AltVol_Vol_Ratio':      (v.rolling(5).mean() / (v.rolling(22).mean() + 1e-10)).shift(1),
        'AltVol_PV_Corr':        ret.rolling(22).corr(np.log(v + 1)).shift(1),
        'AltVol_Vol_Surprise':   ((v - v.rolling(22).mean()) / (v.rolling(22).std() + 1e-10)).shift(1),
    })
    pv = v.where(ret > 0, 0).rolling(22).sum()
    nv = v.where(ret <= 0, 0).rolling(22).sum()
    ft['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
    ft['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum() + 1e-10) * 1e6).shift(1)
    d = pd.DataFrame(ft); d['ret_sq'] = rs2; d['Asset'] = asset
    d['Class'] = next(cls for cls, assets in ASSET_GROUPS.items() if asset in assets)
    af[asset] = d

print('Data loaded', flush=True)
with open('results/main_benchmark_v6_results.json') as f: saved = json.load(f)

rows = []
for hz in HORIZONS:
    hk = str(hz) + 'd'
    print('Horizon:', hk, flush=True)
    pool = []
    for asset in ALL_ASSETS:
        df = af[asset].copy(); df['Target'] = frv(df['ret_sq'], hz)
        df = df.drop(columns=['ret_sq']).dropna(); pool.append(df)
    data = pd.concat(pool).sort_index().reset_index(drop=True)
    fs = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    sp = int(len(data) * 0.8)
    tr = data.iloc[:sp - hz].copy(); te = data.iloc[sp:].copy()
    yt = te['Target'].values; at = te['Asset'].values
    bpr = saved.get(hk, {}).get('Ridge', {}).get('best_params', {})
    bpw = saved.get(hk, {}).get('WEns',  {}).get('best_params', {})
    hidx = [fs.index(f) for f in HAR_FEATS if f in fs]

    # Global scaler fitted on all train data (same as v6 benchmark)
    sc = StandardScaler().fit(tr[fs].values)
    X_tr = sc.transform(tr[fs]); X_te = sc.transform(te[fs])

    # HAR-3
    ph = np.full(len(te), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        m = Ridge(alpha=1.0).fit(X_tr[tr['Class'].values == cls][:, hidx], trc['Target'].values)
        ph[tem] = m.predict(X_te[te['Class'].values == cls][:, hidx])

    # Ridge
    pr2 = np.full(len(te), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        alpha = bpr.get(cls, {}).get('alpha', 1000) if isinstance(bpr, dict) and cls in bpr else 1000
        m = Ridge(alpha=alpha).fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        pr2[tem] = m.predict(X_te[te['Class'].values == cls])

    # WEns
    pw = bpw.get('pw', 0.8) if isinstance(bpw, dict) else 0.8
    px = np.full(len(te), np.nan)
    for cls in ASSET_GROUPS:
        trc = tr[tr['Class'] == cls]; tem = (te['Class'] == cls).values
        if len(trc) < 5: continue
        xp = bpw.get('XGBoost', {}).get(cls, {'max_depth': 3, 'learning_rate': 0.03}) \
             if isinstance(bpw, dict) else {'max_depth': 3, 'learning_rate': 0.03}
        m = XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                         max_depth=xp.get('max_depth', 3), learning_rate=xp.get('learning_rate', 0.03),
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
                         min_child_weight=5, n_jobs=1, device='cuda', tree_method='hist')
        m.fit(X_tr[tr['Class'].values == cls], trc['Target'].values)
        px[tem] = m.predict(X_te[te['Class'].values == cls])
    pw2 = pw * pr2 + (1 - pw) * px

    for mn, preds in [('HAR-3', ph), ('Ridge', pr2), ('WEns', pw2)]:
        vld = ~np.isnan(preds) & ~np.isnan(yt)
        poolr = float(r2_score(yt[vld], preds[vld]))
        withr = r2w(yt[vld], preds[vld], at[vld])
        print(f'  {mn:<8} Pooled={poolr:.4f}  Within={withr:.4f}')
        rows.append({'Horizon': hk, 'Model': mn,
                     'Pooled_R2': round(poolr, 4), 'Within_R2': round(withr, 4),
                     'Diff': round(poolr - withr, 4)})

df_out = pd.DataFrame(rows)
df_out.to_csv('paper/csv/r2_within.csv', index=False)
print('\nSaved: paper/csv/r2_within.csv')
print(f"\n{'Horizon':<8} {'Pooled':>8} {'Within':>8} {'Diff':>10}")
for _, r in df_out[df_out['Model'] == 'WEns'].iterrows():
    print(f"{r['Horizon']:<8} {r['Pooled_R2']:>8.4f} {r['Within_R2']:>8.4f} {r['Diff']:>+10.4f}")
