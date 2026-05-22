"""
Step 0: Shared Data Builder
============================
V71 데이터셋을 생성하고 캐싱. 모든 후속 스크립트가 이 모듈을 import.
"""

import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from arch import arch_model

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]
DATASET_CACHE = 'src/data/v71_dataset_cache.pkl'
OHLCV_CACHE = 'src/data/v71_ohlcv_cache.pkl'

def fit_garch(returns):
    try:
        ret = returns * 100
        am = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def compute_parkinson_vol(high, low, window=22):
    log_hl = np.log(high / low)
    return np.sqrt((log_hl**2).rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

def compute_garman_klass_vol(open_p, high, low, close, window=22):
    log_hl = np.log(high / low)
    log_co = np.log(close / open_p)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk.rolling(window).mean() * 252)

def compute_rogers_satchell_vol(open_p, high, low, close, window=22):
    log_ho = np.log(high / open_p)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_p)
    log_lc = np.log(low / close)
    rs = log_ho * log_hc + log_lo * log_lc
    return np.sqrt(rs.rolling(window).mean().clip(lower=0) * 252)

def get_asset_class(asset):
    for cls, assets in ASSET_GROUPS.items():
        if asset in assets:
            return cls
    return 'Unknown'

def build_dataset(force_rebuild=False):
    """V71 전체 데이터셋 생성 (캐시 활용)"""
    if os.path.exists(DATASET_CACHE) and not force_rebuild:
        print(f"[DataBuilder] Loading cached dataset: {DATASET_CACHE}", flush=True)
        with open(DATASET_CACHE, 'rb') as f:
            return pickle.load(f)
    
    print("[DataBuilder] Building V71 dataset...", flush=True)
    
    # OHLCV 로드
    if os.path.exists(OHLCV_CACHE):
        raw = pd.read_pickle(OHLCV_CACHE)
    else:
        import yfinance as yf
        tickers = ALL_ASSETS + ['^VIX', '^VIX3M', '^VIX9D']
        raw = yf.download(tickers, start='2010-01-01', end='2025-01-01', progress=True)
        if isinstance(raw.columns, pd.MultiIndex):
            new_cols = [(pt, t.replace('^', '')) for pt, t in raw.columns]
            raw.columns = pd.MultiIndex.from_tuples(new_cols)
        # NOTE on yfinance Close vs Adj Close:
        #   yfinance >= 0.2.18 returns Close as split/dividend-adjusted price.
        #   However, Open/High/Low are NOT adjusted, which may cause minor
        #   inconsistencies in OHLC range-based estimators around ex-dividend dates.
        #   Impact is limited since most ETFs distribute quarterly/semi-annually.
        # Missing values: forward-fill for trading day mismatches (1-2 day gaps).
        #   ffill may artificially create zero-return days, slightly deflating
        #   short-window RV/range estimators. Impact on 22-day windows is minimal.
        raw = raw.ffill()
        raw.to_pickle(OHLCV_CACHE)
    
    available_tickers = raw.columns.get_level_values(1).unique()
    
    # IV features (global)
    iv_features = {}
    if 'VIX' in available_tickers:
        vix = raw[('Close', 'VIX')]
        iv_features['VIX'] = np.log(vix + 1e-6)
        iv_features['VIX_chg'] = iv_features['VIX'].diff()
        iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
        iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
    if 'VIX3M' in available_tickers:
        iv_features['VIX3M'] = np.log(raw[('Close', 'VIX3M')] + 1e-6)
        if 'VIX' in iv_features:
            iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
    if 'VIX9D' in available_tickers:
        iv_features['VIX9D'] = np.log(raw[('Close', 'VIX9D')] + 1e-6)
        if 'VIX' in iv_features:
            iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
    
    spy_close = raw[('Close', 'SPY')]
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)
    
    if 'VIX' in available_tickers:
        vix_raw = raw[('Close', 'VIX')]
        # VRP definition (variance unit, annualized):
        #   VRP = IV^2 - RV
        #   IV^2 = (VIX/100)^2      [VIX in percentage points -> decimal]
        #   RV   = spy_rv / 10000   [spy_rv in bp^2 -> decimal]
        #   Both sides in annual variance (decimal^2) unit.
        # NOTE: VIX reflects 30-calendar-day (~22 trading day) implied volatility.
        #   Using it as IV proxy for all horizons (1d-252d) introduces maturity
        #   mismatch. This VRP is therefore a proxy, not a true risk-neutral VRP.
        vrp = (vix_raw**2 / 100) - spy_rv / 10000
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()
    
    # Per-asset features
    pooled_data = []
    for asset in [a for a in ALL_ASSETS if a in available_tickers]:
        close = raw[('Close', asset)]
        open_p = raw[('Open', asset)]
        high = raw[('High', asset)]
        low = raw[('Low', asset)]
        volume = raw[('Volume', asset)]
        
        ret = np.log(close / close.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)
        
        garch_d = pd.Series(fit_garch(ret), index=ret.index)
        ret_w = ret.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret.index, method='ffill')
        
        corr_spy = ret.rolling(22).corr(spy_ret.reindex(ret.index)) if asset != 'SPY' else pd.Series(1.0, index=ret.index)
        
        park_5 = compute_parkinson_vol(high, low, 5)
        park_22 = compute_parkinson_vol(high, low, 22)
        gk_22 = compute_garman_klass_vol(open_p, high, low, close, 22)
        rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, 22)
        overnight = np.log(open_p / close.shift(1))
        
        dollar_vol = volume * close
        amihud = (ret.abs() / (dollar_vol + 1e-10)).rolling(22).mean()
        vol_ma5 = volume.rolling(5).mean()
        vol_ma22 = volume.rolling(22).mean()
        vol_std = volume.rolling(22).std()
        buy_vol = volume * (ret > 0).astype(float)
        sell_vol = volume * (ret <= 0).astype(float)
        order_imb = (buy_vol - sell_vol) / (volume + 1e-10)
        kyle = (ret.abs() / volume.pct_change().abs().clip(lower=1e-10)).rolling(22).mean()
        
        feat = {
            # Base (14)
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_d.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv.rolling(5).std().shift(1),
            'LogRV_Std22': log_rv.rolling(22).std().shift(1),
            'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
            'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret.shift(1),
            'Ret_abs_lag1': ret.abs().shift(1),
            'Corr_SPY': corr_spy.shift(1),
            # HF Proxy (7)
            'Parkinson_5': np.log(park_5 + 1e-6).shift(1),
            'Parkinson_22': np.log(park_22 + 1e-6).shift(1),
            'GarmanKlass_22': np.log(gk_22 + 1e-6).shift(1),
            'RogersSatchell_22': np.log(rs_22 + 1e-6).shift(1),
            'Range_Close_Ratio': (np.log(park_22 + 1e-6) - log_rv).shift(1),
            'Overnight_Vol': overnight.rolling(22).std().shift(1),
            'Overnight_Ret': overnight.shift(1),
            # Alt Data (6)
            'AltVol_Amihud': np.log(amihud + 1e-10).shift(1),
            'AltVol_Vol_Ratio': np.log(vol_ma5 / (vol_ma22 + 1e-10) + 1e-10).shift(1),
            'AltVol_PV_Corr': ret.rolling(22).corr(volume.pct_change()).shift(1),
            'AltVol_Vol_Surprise': ((volume - vol_ma22) / (vol_std + 1e-10)).shift(1),
            'AltVol_Order_Imbalance': order_imb.rolling(22).mean().shift(1),
            'AltVol_Kyle_Lambda': np.log(kyle.clip(lower=1e-10)).shift(1),
            # Meta
            # Target: log realized volatility 22 trading days ahead (overlapping)
            # This is the "backward-looking" 22d rolling RV shifted forward.
            # For non-overlapping forward RV, see tmp_multi_horizon_nonoverlap.py
            'Target': log_rv.shift(-22),
            'Asset': asset,
            'Class': get_asset_class(asset),
        }
        # IV Surface (10)
        for iv_name, iv_val in iv_features.items():
            feat[f'IV_{iv_name}'] = iv_val.shift(1)
        
        d = pd.DataFrame(feat)
        d = d.reset_index().rename(columns={'index': 'Date'})
        d = d.dropna(subset=['Target'])
        numeric = [c for c in d.columns if c not in ['Asset', 'Class', 'Target', 'Date']]
        d[numeric] = d[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
        pooled_data.append(d)
    
    data = pd.concat(pooled_data).sort_values('Date').reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class', 'Date']]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Feature group definitions
    base_feats = [f for f in feats if not f.startswith(('IV_', 'AltVol_', 'Parkinson', 'Garman', 'Rogers', 'Range_', 'Overnight'))]
    hf_feats = [f for f in feats if any(x in f for x in ['Parkinson', 'Garman', 'Rogers', 'Range_Close', 'Overnight'])]
    iv_feats = [f for f in feats if f.startswith('IV_')]
    alt_feats = [f for f in feats if f.startswith('AltVol_')]
    
    result = {
        'data': data,
        'feats': feats,
        'base_feats': base_feats,
        'hf_feats': hf_feats,
        'iv_feats': iv_feats,
        'alt_feats': alt_feats,
    }
    
    with open(DATASET_CACHE, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"[DataBuilder] Done: {len(data)} samples, {len(feats)} features, "
          f"saved to {DATASET_CACHE}", flush=True)
    return result

def get_train_test(data, ratio=0.8, purge_gap=22):
    """시간 순서 기반 train/test split with purge gap.
    
    Parameters
    ----------
    purge_gap : int
        Number of samples to remove at train/test boundary to prevent
        target leakage from overlapping RV windows. Should match
        the prediction horizon h (default: 22 trading days).
    """
    split = int(len(data) * ratio)
    return data.iloc[:split - purge_gap].copy(), data.iloc[split:].copy()

def train_ridge_perclass(train_df, test_df, feature_list):
    """자산 클래스별 Ridge (alpha 튜닝) 학습 및 예측"""
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    sc = StandardScaler()
    sc.fit(train_df[feature_list])
    
    val_split = int(len(train_df) * 0.8)
    tr_inner, va_inner = train_df.iloc[:val_split], train_df.iloc[val_split:]
    
    best_alphas = {}
    for cls in ASSET_GROUPS:
        best_r2, best_a = -999, 1.0
        tr_c = tr_inner[tr_inner['Class'] == cls]
        va_c = va_inner[va_inner['Class'] == cls]
        if len(tr_c) < 100 or len(va_c) < 30:
            best_alphas[cls] = 1.0; continue
        for a in alphas:
            m = Ridge(alpha=a).fit(sc.transform(tr_c[feature_list]), tr_c['Target'])
            r2 = r2_score(va_c['Target'], m.predict(sc.transform(va_c[feature_list])))
            if r2 > best_r2: best_r2, best_a = r2, a
        best_alphas[cls] = best_a
    
    preds = np.full(len(test_df), np.nan)
    models = {}
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_idx = test_df['Class'] == cls
        if len(tr_c) < 100 or te_idx.sum() == 0: continue
        m = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_c[feature_list]), tr_c['Target'])
        preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feature_list]))
        models[cls] = m
    
    return preds, models, best_alphas, sc


if __name__ == '__main__':
    ds = build_dataset(force_rebuild=True)
    print(f"Features ({len(ds['feats'])}): {ds['feats']}")
    print(f"Base: {len(ds['base_feats'])}, HF: {len(ds['hf_feats'])}, "
          f"IV: {len(ds['iv_feats'])}, Alt: {len(ds['alt_feats'])}")
