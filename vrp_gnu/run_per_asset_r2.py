"""
Per-asset R² breakdown using saved best_params from main_benchmark_v6_results.json.
Re-fits WEns/Ridge/HAR-3 with known best_params (no inner holdout search).
Output: paper/csv/per_asset_r2.csv
"""
import sys, os, json, ast
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]
OUTER_TRAIN_RATIO = 0.8
ASSET_GROUPS = {
    'Equity':    ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond':      ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO'],
}
ALL_ASSETS   = [a for g in ASSET_GROUPS.values() for a in g]
ASSET_CLASS  = {a: cls for cls, assets in ASSET_GROUPS.items() for a in assets}
HAR_FEATS    = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']

RESULTS_JSON = '/root/vrp/results/main_benchmark_v6_results.json'
OUT_CSV      = '/root/vrp/paper/csv/per_asset_r2.csv'

# ── Load best_params ──────────────────────────────────────────────────────────
with open(RESULTS_JSON) as f:
    saved = json.load(f)

def get_best_params(hz_key, model):
    bp = saved.get(hz_key, {}).get(model, {}).get('best_params', {})
    if isinstance(bp, str):
        bp = ast.literal_eval(bp)
    return bp

print("Loading data and computing features...")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ── Try loading from cached parquet if available ──────────────────────────────
CACHE_DIR = '/root/vrp/data'
os.makedirs(CACHE_DIR, exist_ok=True)

def forward_rv(ret_sq, horizon):
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

def load_or_fetch(asset):
    cache = os.path.join(CACHE_DIR, f'{asset}.parquet')
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not HAS_YF:
        return None
    print(f"  Fetching {asset} from yfinance...")
    df = yf.download(asset, start='2009-01-01', end='2025-01-01',
                     auto_adjust=False, progress=False)
    if df.empty:
        return None
    df.to_parquet(cache)
    return df

def compute_features(asset, vix_df):
    raw = load_or_fetch(asset)
    if raw is None:
        return None

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    raw = raw[['Open','High','Low','Close','Adj Close','Volume']].copy()
    raw = raw.ffill()

    c   = raw['Adj Close']
    o_r = raw['Open']
    h_r = raw['High']
    l_r = raw['Low']
    v   = raw['Volume']

    ret     = np.log(c / c.shift(1))
    ret_sq  = ret ** 2
    lrv     = np.log(ret_sq.rolling(22).mean() * 252 + 1e-12)

    feat = {}
    for lag, name in [(1,'LogRV_lag1'),(5,'LogRV_lag5'),(10,'LogRV_lag10'),(22,'LogRV_lag22')]:
        feat[name] = np.log(ret_sq.rolling(lag).mean() * 252 + 1e-12).shift(1)
    feat['LogRV_Std5']  = lrv.rolling(5).std().shift(1)
    feat['LogRV_Std22'] = lrv.rolling(22).std().shift(1)
    feat['RV_Mom5']     = (lrv - lrv.shift(5)).shift(1)
    feat['RV_Mom22']    = (lrv - lrv.shift(22)).shift(1)
    _spy_close = pd.read_parquet(os.path.join(CACHE_DIR, 'SPY.parquet'))['Adj Close'].squeeze()
    feat['SPY_LogRV']   = np.log(
        _spy_close.pct_change() ** 2 * 252 + 1e-12
    ).reindex(c.index).ffill().shift(1) if asset != 'SPY' else lrv.shift(1)
    feat['Ret_lag1']     = ret.shift(1)
    feat['Ret_abs_lag1'] = ret.abs().shift(1)
    spy_ret = _spy_close.pct_change()
    feat['Corr_SPY']    = ret.rolling(22).corr(spy_ret.reindex(c.index)).shift(1)

    def park(h, l, w):
        return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)
    def gk(o, h, l, c2, w):
        hl = np.log(h/l); co = np.log(c2/o)
        return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
    def rs(o, h, l, c2, w):
        r2 = np.log(h/c2)*np.log(h/o)+np.log(l/c2)*np.log(l/o)
        return np.sqrt(r2.rolling(w).mean().clip(0)*252)

    feat['Parkinson_5']    = park(h_r, l_r, 5).shift(1)
    feat['Parkinson_22']   = park(h_r, l_r, 22).shift(1)
    feat['GarmanKlass_22'] = gk(o_r, h_r, l_r, c, 22).shift(1)
    feat['RogersSatchell_22'] = rs(o_r, h_r, l_r, c, 22).shift(1)
    p22 = park(h_r, l_r, 22)
    feat['Range_Close_Ratio'] = (np.log(p22+1e-6)-lrv).shift(1)
    on = np.log(o_r / c.shift(1))
    feat['Overnight_Vol'] = on.rolling(22).std().shift(1)
    feat['Overnight_Ret'] = on.shift(1)

    # IV features
    vix = vix_df['Close'].reindex(c.index).ffill()
    iv  = (vix/100)**2
    feat['IV_VIX']        = iv.shift(1)
    feat['IV_VIX_chg']    = iv.diff().shift(1)
    feat['IV_VIX_ma5']    = iv.rolling(5).mean().shift(1)
    feat['IV_VIX_std5']   = iv.rolling(5).std().shift(1)

    vix3m = vix_df.get('Close_3M', vix).reindex(c.index).ffill()
    iv3m  = (vix3m/100)**2
    feat['IV_VIX3M']       = iv3m.shift(1)
    feat['IV_VIX_TermSlope'] = (iv3m - iv).shift(1)

    vix9d = vix_df.get('Close_9D', vix).reindex(c.index).ffill()
    iv9d  = (vix9d/100)**2
    feat['IV_VIX9D']        = iv9d.shift(1)
    feat['IV_VIX_ShortSlope'] = (iv - iv9d).shift(1)

    spy_rv_val = ret_sq.rolling(22).mean() * 252
    vrp_val    = iv - spy_rv_val
    feat['IV_VRP']      = vrp_val.shift(1)
    feat['IV_VRP_ma22'] = vrp_val.rolling(22).mean().shift(1)

    dv = v * c
    feat['AltVol_Amihud']         = (ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']      = (v.rolling(5).mean()/(v.rolling(22).mean()+1e-10)).shift(1)
    feat['AltVol_PV_Corr']        = ret.rolling(22).corr(np.log(v+1)).shift(1)
    feat['AltVol_Vol_Surprise']   = ((v-v.rolling(22).mean())/(v.rolling(22).std()+1e-10)).shift(1)
    pv = v.where(ret>0,0).rolling(22).sum()
    nv = v.where(ret<=0,0).rolling(22).sum()
    feat['AltVol_Order_Imbalance']= ((pv-nv)/(pv+nv+1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']    = (ret.abs().rolling(22).sum()/(v.rolling(22).sum()+1e-10)*1e6).shift(1)

    d = pd.DataFrame(feat)
    d['ret_sq'] = ret_sq
    d['Asset']  = asset
    d['Class']  = ASSET_CLASS[asset]
    d.index.name = 'Date'
    d = d.reset_index()
    return d

# ── Load VIX ─────────────────────────────────────────────────────────────────
print("Loading VIX data...")
vix_cache = os.path.join(CACHE_DIR, 'VIX.parquet')
if os.path.exists(vix_cache):
    vix_df = pd.read_parquet(vix_cache)
elif HAS_YF:
    vix_raw = yf.download(['^VIX', '^VIX3M', '^VIX9D'],
                          start='2009-01-01', end='2025-01-01',
                          auto_adjust=False, progress=False)
    vix_df = pd.DataFrame({
        'Close':    vix_raw['Close']['^VIX']    if '^VIX'    in vix_raw['Close'].columns else vix_raw['Close'],
        'Close_3M': vix_raw['Close']['^VIX3M']  if '^VIX3M'  in vix_raw['Close'].columns else vix_raw['Close']['^VIX'],
        'Close_9D': vix_raw['Close']['^VIX9D']  if '^VIX9D'  in vix_raw['Close'].columns else vix_raw['Close']['^VIX'],
    })
    vix_df.to_parquet(vix_cache)
else:
    raise RuntimeError("No VIX data available and yfinance not installed.")

# Pre-fetch SPY for cross-asset features
if not os.path.exists(os.path.join(CACHE_DIR, 'SPY.parquet')) and HAS_YF:
    load_or_fetch('SPY')

# ── Build asset frames ────────────────────────────────────────────────────────
print("Computing features for all assets...")
asset_frames = {}
for asset in ALL_ASSETS:
    print(f"  {asset}")
    d = compute_features(asset, vix_df)
    if d is not None:
        asset_frames[asset] = d

print(f"Loaded {len(asset_frames)} assets")

# ── XGBoost import ────────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available — WEns will use Ridge only")

# ── Per-asset R² computation ─────────────────────────────────────────────────
FEATS = [c for c in next(iter(asset_frames.values())).columns
         if c not in ['Date','Asset','Class','ret_sq']]
print(f"Features: {len(FEATS)}")

rows = []

for hz in HORIZONS:
    hz_key = f'{hz}d'
    print(f"\n{'='*50}\nHorizon: {hz_key}")

    # Build panel
    frames = []
    for asset, df in asset_frames.items():
        d = df.copy()
        d['Target'] = forward_rv(d['ret_sq'], hz)
        frames.append(d)
    panel = pd.concat(frames).sort_values('Date').reset_index(drop=True)
    panel = panel.dropna(subset=FEATS + ['Target'])

    # Train/test split
    dates      = panel['Date'].sort_values().unique()
    split_date = dates[int(len(dates) * OUTER_TRAIN_RATIO)]
    train_df   = panel[panel['Date'] < split_date]
    test_df    = panel[panel['Date'] >= split_date]

    y_te = test_df['Target'].values
    X_te = test_df[FEATS].values
    X_tr = train_df[FEATS].values
    y_tr = train_df['Target'].values

    # Load best_params
    bp_wens  = get_best_params(hz_key, 'WEns')
    bp_ridge = get_best_params(hz_key, 'Ridge')

    # Models to evaluate
    models_to_run = ['HAR-3', 'Ridge', 'WEns']

    preds_dict = {}

    # ── HAR-3 ──
    har_idx = [FEATS.index(f) for f in HAR_FEATS if f in FEATS]
    p_har   = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_m = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        sc = StandardScaler().fit(tr_c[FEATS].values)
        m  = Ridge(alpha=1.0).fit(sc.transform(tr_c[FEATS])[:, har_idx], tr_c['Target'].values)
        p_har[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS])[:, har_idx])
    preds_dict['HAR-3'] = p_har

    # ── Ridge (per-class, best alpha) ──
    ridge_params = bp_ridge if bp_ridge else {'Equity':{'alpha':2000},'Bond':{'alpha':10},'Commodity':{'alpha':500}}
    p_ridge = np.full(len(test_df), np.nan)
    ridge_models = {}
    scalers      = {}
    for cls in ASSET_GROUPS:
        tr_c  = train_df[train_df['Class'] == cls]
        te_m  = (test_df['Class'] == cls).values
        if len(tr_c) < 5 or te_m.sum() == 0: continue
        alpha = ridge_params.get('Equity',{}).get('alpha', ridge_params.get(cls,{}).get('alpha',1000)) \
                if cls == 'Equity' else ridge_params.get(cls,{}).get('alpha',1000)
        # Handle nested dict from WEns best_params
        if 'Ridge' in bp_wens:
            alpha = bp_wens['Ridge'].get(cls, {}).get('alpha', 1000)
        elif cls in ridge_params:
            alpha = ridge_params[cls].get('alpha', 1000)
        sc = StandardScaler().fit(tr_c[FEATS].values)
        m  = Ridge(alpha=alpha).fit(sc.transform(tr_c[FEATS]), tr_c['Target'].values)
        p_ridge[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS]))
        ridge_models[cls] = m
        scalers[cls]      = sc
    preds_dict['Ridge'] = p_ridge

    # ── WEns ──
    if HAS_XGB and bp_wens:
        pw = bp_wens.get('pw', 0.8)
        xgb_params_bp = bp_wens.get('XGBoost', {})
        p_xgb = np.full(len(test_df), np.nan)
        for cls in ASSET_GROUPS:
            tr_c = train_df[train_df['Class'] == cls]
            te_m = (test_df['Class'] == cls).values
            if len(tr_c) < 5 or te_m.sum() == 0: continue
            xp   = xgb_params_bp.get(cls, {'max_depth':3,'learning_rate':0.03})
            sc   = scalers.get(cls, StandardScaler().fit(tr_c[FEATS].values))
            m    = XGBRegressor(n_estimators=300, random_state=42,
                                max_depth=xp.get('max_depth',3),
                                learning_rate=xp.get('learning_rate',0.03),
                                verbosity=0)
            m.fit(sc.transform(tr_c[FEATS]), tr_c['Target'].values)
            p_xgb[te_m] = m.predict(sc.transform(test_df.loc[te_m, FEATS]))
        p_wens = pw * p_ridge + (1 - pw) * p_xgb
        preds_dict['WEns'] = p_wens
    else:
        preds_dict['WEns'] = p_ridge.copy()
        print("  WEns = Ridge (XGBoost not available)")

    # ── Per-asset R² ──
    for model_name, preds in preds_dict.items():
        for asset in ALL_ASSETS:
            mask  = (test_df['Asset'] == asset).values
            if mask.sum() < 2: continue
            y_a   = y_te[mask]
            p_a   = preds[mask]
            valid = ~np.isnan(p_a) & ~np.isnan(y_a)
            if valid.sum() < 2: continue
            r2 = float(r2_score(y_a[valid], p_a[valid]))
            rows.append({
                'Model':   model_name,
                'Asset':   asset,
                'Class':   ASSET_CLASS[asset],
                'Horizon': hz_key,
                'R2':      round(r2, 4),
            })
        # Pooled check
        valid_all = ~np.isnan(preds) & ~np.isnan(y_te)
        pooled    = float(r2_score(y_te[valid_all], preds[valid_all]))
        per_asset_r2 = [r for r in [rows[-i]['R2'] for i in range(len(ALL_ASSETS),0,-1)] if True]
        print(f"  {model_name:<8} Pooled={pooled:.4f}  Median={np.median([r['R2'] for r in rows if r['Model']==model_name and r['Horizon']==hz_key]):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
df_long = pd.DataFrame(rows)

# WEns wide table (for paper)
wens = df_long[df_long['Model'] == 'WEns'].copy()
pivot = wens.pivot_table(index=['Asset','Class'], columns='Horizon', values='R2')
hz_cols = [f'{h}d' for h in HORIZONS if f'{h}d' in pivot.columns]
pivot = pivot[hz_cols].reset_index()
class_order = {'Equity':0,'Bond':1,'Commodity':2}
pivot['_ord'] = pivot['Class'].map(class_order)
pivot = pivot.sort_values(['_ord','Asset']).drop(columns='_ord')

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pivot.to_csv(OUT_CSV, index=False)
print(f"\nSaved WEns per-asset R² table: {OUT_CSV}")

# All models long format
df_long.to_csv(OUT_CSV.replace('.csv', '_all_models.csv'), index=False)
print(f"Saved all-models long format: {OUT_CSV.replace('.csv','_all_models.csv')}")

print("\n=== WEns Per-Asset R² (22d) ===")
print(pivot[['Asset','Class','22d']].to_string(index=False))
