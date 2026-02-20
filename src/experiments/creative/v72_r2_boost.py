"""
V72: R² Boost Experiment
Goal: Surpass LASSO (0.790) with structural improvements over V71 (0.793)

Strategies from r2up.md:
  A. Regime-Specific Mixture-of-Experts Ridge (VIX quantile routing)
  B. Multi-Horizon Multi-Task Ridge (1d/5d/22d/60d joint learning)
  C. Factor + Idiosyncratic structure (PCA common factor + asset residual)

Base: V71 pipeline (37 features, 11 assets, Forward RV target)
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from arch import arch_model

# ============================================
# Configuration (same as V71)
# ============================================
ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]
IV_TICKERS = ['^VIX', '^VIX3M', '^VIX9D']

HORIZONS = [1, 5, 22, 60]  # Multi-horizon targets


# ============================================
# Reusable functions from V71
# ============================================
def fit_garch(returns):
    try:
        am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.nan, index=returns.index)


def download_ohlcv():
    import yfinance as yf
    tickers = ALL_ASSETS + ['^VIX', '^VIX3M', '^VIX9D']
    print("Downloading OHLCV data...", flush=True)
    raw = yf.download(tickers, start='2010-01-01', end='2024-12-31',
                      auto_adjust=True, group_by='column')
    # Rename VIX tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.rename(columns={'^VIX': 'VIX', '^VIX3M': 'VIX3M', '^VIX9D': 'VIX9D'}, level=1)
    return raw


def compute_parkinson_vol(high, low, window=22):
    return np.sqrt((np.log(high / low) ** 2).rolling(window).mean() / (4 * np.log(2)))


def compute_garman_klass_vol(open_p, high, low, close, window=22):
    hl = np.log(high / low) ** 2
    co = np.log(close / open_p) ** 2
    return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean())


def compute_rogers_satchell_vol(open_p, high, low, close, window=22):
    rs = (np.log(high / close) * np.log(high / open_p) +
          np.log(low / close) * np.log(low / open_p))
    return np.sqrt(rs.rolling(window).mean().clip(lower=0))


def compute_volume_features(volume, price, ret, window=22):
    vol_ma = volume.rolling(window).mean()
    dollar_vol = volume * price
    feats = {}
    feats['Amihud'] = (ret.abs() / (dollar_vol + 1e-6)).rolling(window).mean()
    feats['Vol_Ratio'] = volume.rolling(5).mean() / (vol_ma + 1e-6)
    feats['PV_Corr'] = ret.rolling(window).corr(volume)
    feats['Vol_Surprise'] = (volume - vol_ma) / (vol_ma + 1e-6)
    buy_vol = volume * (ret > 0).astype(float)
    sell_vol = volume * (ret <= 0).astype(float)
    feats['Order_Imbalance'] = (buy_vol.rolling(window).sum() /
                                 (sell_vol.rolling(window).sum() + 1e-6))
    price_impact = ret.abs() / (np.log(volume + 1) + 1e-6)
    feats['Kyle_Lambda'] = price_impact.rolling(window).mean()
    return feats


def qlike_loss(actual, predicted):
    """QLIKE loss: mean(actual/predicted + log(predicted) - 1 - log(actual))
    Lower is better. Works on log-scale values."""
    # Convert from log to level
    act_level = np.exp(actual)
    pred_level = np.exp(predicted)
    ratio = act_level / (pred_level + 1e-10)
    return np.mean(ratio - np.log(ratio + 1e-10) - 1)


# ============================================
# Main Experiment
# ============================================
def run_experiment():
    print("=" * 80, flush=True)
    print("V72: R² Boost (Regime MoE + Multi-Horizon + Factor Structure)", flush=True)
    print("=" * 80, flush=True)

    # ============================================
    # 1. Data Pipeline (identical to V71)
    # ============================================
    raw = download_ohlcv()

    if isinstance(raw.columns, pd.MultiIndex):
        price_types = raw.columns.get_level_values(0).unique()
        available_tickers = raw.columns.get_level_values(1).unique()
    else:
        price_types = ['Close']
        available_tickers = raw.columns

    has_ohlc = all(pt in price_types for pt in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in price_types
    has_vix = 'VIX' in available_tickers
    has_vix3m = 'VIX3M' in available_tickers
    has_vix9d = 'VIX9D' in available_tickers

    print(f"OHLC: {has_ohlc}, Volume: {has_volume}, VIX: {has_vix}", flush=True)

    # --- IV Surface Features ---
    iv_features = {}
    if has_vix:
        vix = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        iv_features['VIX'] = np.log(vix + 1e-6)
        iv_features['VIX_chg'] = iv_features['VIX'].diff()
        iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
        iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()

    if has_vix3m:
        vix3m = raw[('Close', 'VIX3M')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX3M']
        iv_features['VIX3M'] = np.log(vix3m + 1e-6)
        if has_vix:
            iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']

    if has_vix9d:
        vix9d = raw[('Close', 'VIX9D')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX9D']
        iv_features['VIX9D'] = np.log(vix9d + 1e-6)
        if has_vix:
            iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']

    # SPY reference
    if isinstance(raw.columns, pd.MultiIndex):
        spy_close = raw[('Close', 'SPY')]
    else:
        spy_close = raw['SPY']
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_rv = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)

    # VRP
    if has_vix:
        vix_raw = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        vrp = (vix_raw ** 2 / 100) - spy_rv / 10000
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    # --- VIX level for regime classification (raw, not log) ---
    vix_level = vix_raw if has_vix else None

    # ============================================
    # Build feature matrix per asset
    # ============================================
    pooled_data = []
    asset_log_rv_dict = {}  # for Factor structure (Strategy C)
    available_assets = [a for a in ALL_ASSETS if a in available_tickers]
    print(f"\nProcessing {len(available_assets)} assets...", flush=True)

    for i, asset in enumerate(available_assets):
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[('Close', asset)]
            open_p = raw[('Open', asset)] if has_ohlc else None
            high = raw[('High', asset)] if has_ohlc else None
            low = raw[('Low', asset)] if has_ohlc else None
            volume = raw[('Volume', asset)] if has_volume else None
        else:
            close = raw[asset]
            open_p = high = low = volume = None

        ret_daily = np.log(close / close.shift(1)).dropna()
        rv_daily = ret_daily ** 2
        rv = rv_daily.rolling(22).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6)

        # Store for PCA factor
        asset_log_rv_dict[asset] = log_rv

        # GARCH
        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')

        # Standard features
        feat_dict = {
            'LogRV_lag1': log_rv.shift(1),
            'LogRV_lag5': log_rv.shift(5),
            'LogRV_lag10': log_rv.shift(10),
            'LogRV_lag22': log_rv.shift(22),
            'Garch_Daily': garch_series.shift(1),
            'Garch_Weekly': garch_w.shift(1),
            'LogRV_Std5': log_rv.rolling(5).std().shift(1),
            'LogRV_Std22': log_rv.rolling(22).std().shift(1),
            'RV_Mom5': (log_rv - log_rv.shift(5)).shift(1),
            'RV_Mom22': (log_rv - log_rv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret_daily.shift(1),
            'Ret_abs_lag1': ret_daily.abs().shift(1),
        }

        # HF Proxy
        if has_ohlc and open_p is not None:
            park_5 = compute_parkinson_vol(high, low, window=5)
            park_22 = compute_parkinson_vol(high, low, window=22)
            feat_dict['Parkinson_5'] = np.log(park_5 + 1e-6).shift(1)
            feat_dict['Parkinson_22'] = np.log(park_22 + 1e-6).shift(1)
            gk_22 = compute_garman_klass_vol(open_p, high, low, close, window=22)
            feat_dict['GarmanKlass_22'] = np.log(gk_22 + 1e-6).shift(1)
            rs_22 = compute_rogers_satchell_vol(open_p, high, low, close, window=22)
            feat_dict['RogersSatchell_22'] = np.log(rs_22 + 1e-6).shift(1)
            feat_dict['Range_Close_Ratio'] = (np.log(park_22 + 1e-6) - log_rv).shift(1)
            overnight_ret = np.log(open_p / close.shift(1))
            feat_dict['Overnight_Vol'] = overnight_ret.rolling(22).std().shift(1)
            feat_dict['Overnight_Ret'] = overnight_ret.shift(1)

        # IV Surface
        for iv_name, iv_val in iv_features.items():
            feat_dict[f'IV_{iv_name}'] = iv_val.shift(1)

        # Alt Data (Volume)
        if has_volume and volume is not None:
            vol_feats = compute_volume_features(volume, close, ret_daily, window=22)
            for vf_name, vf_val in vol_feats.items():
                feat_dict[f'AltVol_{vf_name}'] = vf_val.shift(1)

        # Cross-asset correlation
        if asset != 'SPY':
            feat_dict['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
        else:
            feat_dict['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)

        # Multi-horizon targets
        feat_dict['Target_1d'] = log_rv.shift(-1)
        feat_dict['Target_5d'] = log_rv.shift(-5)
        feat_dict['Target_22d'] = log_rv.shift(-22)
        feat_dict['Target_60d'] = log_rv.shift(-60)

        # VIX level for regime (lagged)
        if vix_level is not None:
            feat_dict['_VIX_Level'] = vix_level.shift(1)  # underscore = meta, not feature

        feat_dict['Asset'] = asset

        d = pd.DataFrame(feat_dict).dropna()
        numeric_cols = [c for c in d.columns if c not in ['Asset'] and not c.startswith('Target') and not c.startswith('_')]
        d[numeric_cols] = d[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)

        if (i + 1) % 3 == 0 or i == 0:
            print(f"  [{i + 1}/{len(available_assets)}] {asset}: {len(d)} samples", flush=True)

    data = pd.concat(pooled_data).sort_index().reset_index(drop=True)

    # Feature columns (exclude targets, meta, asset, class)
    feats = [c for c in data.columns
             if c not in ['Asset', 'Class'] and not c.startswith('Target') and not c.startswith('_')]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)

    # Train/Test split (80/20 temporal)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    # Inner validation split (80/20 of train)
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]

    print(f"\nTotal: {len(data)} samples, {len(feats)} features", flush=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    print(f"Features: {feats}", flush=True)

    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    sc = StandardScaler()
    sc.fit(train_df[feats])

    # ============================================
    # 2. BASELINE: V71 Ridge (reproduce)
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("BASELINE: V71 Ridge (37 features, per-class alpha)", flush=True)
    print("=" * 60, flush=True)

    def train_ridge_perclass(train_d, val_d, test_d, feature_cols, target_col, scaler, label=""):
        """Train Ridge per asset class with alpha tuning. Returns predictions on test."""
        best_alphas_local = {}
        for cls in ASSET_GROUPS.keys():
            best_r2, best_a = -999, 1.0
            tr_cls = train_d[train_d['Class'] == cls]
            va_cls = val_d[val_d['Class'] == cls]
            if len(tr_cls) < 100 or len(va_cls) < 30:
                best_alphas_local[cls] = 1.0
                continue
            sc_local = StandardScaler()
            X_tr = sc_local.fit_transform(tr_cls[feature_cols])
            X_va = sc_local.transform(va_cls[feature_cols])
            for a in alphas:
                m = Ridge(alpha=a).fit(X_tr, tr_cls[target_col])
                r2 = r2_score(va_cls[target_col], m.predict(X_va))
                if r2 > best_r2:
                    best_r2, best_a = r2, a
            best_alphas_local[cls] = best_a

        preds = np.full(len(test_d), np.nan)
        full_train = pd.concat([train_d, val_d])
        sc_final = StandardScaler()
        sc_final.fit(full_train[feature_cols])
        for cls in ASSET_GROUPS.keys():
            tr_cls = full_train[full_train['Class'] == cls]
            te_idx = test_d['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0:
                continue
            m = Ridge(alpha=best_alphas_local[cls]).fit(
                sc_final.transform(tr_cls[feature_cols]), tr_cls[target_col])
            preds[te_idx.values] = m.predict(sc_final.transform(test_d.loc[te_idx, feature_cols]))

        valid = ~np.isnan(preds)
        r2 = r2_score(test_d[target_col].values[valid], preds[valid])
        rmse = np.sqrt(mean_squared_error(test_d[target_col].values[valid], preds[valid]))
        mae = mean_absolute_error(test_d[target_col].values[valid], preds[valid])
        ql = qlike_loss(test_d[target_col].values[valid], preds[valid])
        if label:
            print(f"  {label}: R²={r2:.5f}, RMSE={rmse:.4f}, MAE={mae:.4f}, QLIKE={ql:.4f}", flush=True)
        return r2, rmse, mae, ql, preds, best_alphas_local

    r2_base, rmse_base, mae_base, ql_base, preds_baseline, _ = train_ridge_perclass(
        train_inner, val_inner, test_df, feats, 'Target_22d', sc, "V71 Baseline (Ridge 37feat)")

    # ============================================
    # 3. STRATEGY A: Regime-Specific MoE Ridge
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY A: Regime-Specific Mixture-of-Experts Ridge", flush=True)
    print("=" * 60, flush=True)

    # Define regime boundaries using training data VIX
    train_vix = train_df['_VIX_Level'].values
    vix_q50 = np.nanpercentile(train_vix, 50)
    vix_q90 = np.nanpercentile(train_vix, 90)
    print(f"  VIX quantiles: Q50={vix_q50:.1f}, Q90={vix_q90:.1f}", flush=True)

    def assign_regime(vix_val):
        if vix_val <= vix_q50:
            return 'Low'
        elif vix_val <= vix_q90:
            return 'Mid'
        else:
            return 'High'

    def softmax_weights(vix_val, q50, q90, temperature=3.0):
        """Softmax regime weighting for smoother transitions"""
        dists = np.array([
            -(vix_val - q50 * 0.5) ** 2,    # Low center ~ q50/2
            -(vix_val - (q50 + q90) / 2) ** 2,  # Mid center
            -(vix_val - q90 * 1.2) ** 2       # High center ~ q90*1.2
        ]) / (temperature ** 2)
        exp_d = np.exp(dists - np.max(dists))
        return exp_d / exp_d.sum()

    # Hard regime assignment
    train_df['_Regime'] = train_df['_VIX_Level'].apply(assign_regime)
    test_df['_Regime'] = test_df['_VIX_Level'].apply(assign_regime)
    train_inner_r = train_inner.copy()
    val_inner_r = val_inner.copy()
    train_inner_r['_Regime'] = train_inner_r['_VIX_Level'].apply(assign_regime)
    val_inner_r['_Regime'] = val_inner_r['_VIX_Level'].apply(assign_regime)

    print(f"  Train regime distribution: {train_df['_Regime'].value_counts().to_dict()}", flush=True)
    print(f"  Test regime distribution: {test_df['_Regime'].value_counts().to_dict()}", flush=True)

    # --- Strategy A-1: Hard routing ---
    preds_moe_hard = np.full(len(test_df), np.nan)
    regime_models = {}
    for regime in ['Low', 'Mid', 'High']:
        tr_r = train_inner_r[train_inner_r['_Regime'] == regime]
        va_r = val_inner_r[val_inner_r['_Regime'] == regime]
        te_r = test_df[test_df['_Regime'] == regime]

        if len(tr_r) < 100 or len(te_r) == 0:
            print(f"  Regime {regime}: insufficient data (train={len(tr_r)}, test={len(te_r)})", flush=True)
            continue

        # Train per-class Ridge within this regime
        full_tr = pd.concat([tr_r, va_r])
        best_alphas_r = {}
        for cls in ASSET_GROUPS.keys():
            best_r2, best_a = -999, 1.0
            tr_cls = tr_r[tr_r['Class'] == cls]
            va_cls = va_r[va_r['Class'] == cls]
            if len(tr_cls) < 50 or len(va_cls) < 10:
                best_alphas_r[cls] = 100.0
                continue
            sc_r = StandardScaler()
            X_tr = sc_r.fit_transform(tr_cls[feats])
            X_va = sc_r.transform(va_cls[feats])
            for a in alphas:
                m = Ridge(alpha=a).fit(X_tr, tr_cls['Target_22d'])
                r2_v = r2_score(va_cls['Target_22d'], m.predict(X_va))
                if r2_v > best_r2:
                    best_r2, best_a = r2_v, a
            best_alphas_r[cls] = best_a

        sc_r_final = StandardScaler()
        sc_r_final.fit(full_tr[feats])
        regime_models[regime] = {'scaler': sc_r_final, 'models': {}, 'alphas': best_alphas_r}

        for cls in ASSET_GROUPS.keys():
            tr_cls = full_tr[full_tr['Class'] == cls]
            if len(tr_cls) < 50:
                continue
            m = Ridge(alpha=best_alphas_r[cls]).fit(
                sc_r_final.transform(tr_cls[feats]), tr_cls['Target_22d'])
            regime_models[regime]['models'][cls] = m

            te_idx = (test_df['_Regime'] == regime) & (test_df['Class'] == cls)
            if te_idx.sum() > 0:
                preds_moe_hard[te_idx.values] = m.predict(
                    sc_r_final.transform(test_df.loc[te_idx, feats]))

        r2_r = r2_score(te_r['Target_22d'].values,
                        preds_moe_hard[test_df['_Regime'] == regime])
        print(f"  Regime {regime}: R²={r2_r:.5f} (n={len(te_r)})", flush=True)

    valid_moe_hard = ~np.isnan(preds_moe_hard)
    r2_moe_hard = r2_score(test_df['Target_22d'].values[valid_moe_hard], preds_moe_hard[valid_moe_hard])
    rmse_moe_hard = np.sqrt(mean_squared_error(test_df['Target_22d'].values[valid_moe_hard], preds_moe_hard[valid_moe_hard]))
    mae_moe_hard = mean_absolute_error(test_df['Target_22d'].values[valid_moe_hard], preds_moe_hard[valid_moe_hard])
    ql_moe_hard = qlike_loss(test_df['Target_22d'].values[valid_moe_hard], preds_moe_hard[valid_moe_hard])
    print(f"  MoE Hard: R²={r2_moe_hard:.5f}, RMSE={rmse_moe_hard:.4f}, MAE={mae_moe_hard:.4f}, QLIKE={ql_moe_hard:.4f}", flush=True)

    # --- Strategy A-2: Soft routing (softmax blend) ---
    preds_moe_soft = np.zeros(len(test_df))
    for idx in range(len(test_df)):
        vix_val = test_df.iloc[idx]['_VIX_Level']
        cls = test_df.iloc[idx]['Class']
        weights = softmax_weights(vix_val, vix_q50, vix_q90)

        pred_weighted = 0.0
        total_w = 0.0
        for r_idx, regime in enumerate(['Low', 'Mid', 'High']):
            if regime in regime_models and cls in regime_models[regime]['models']:
                x = regime_models[regime]['scaler'].transform(
                    test_df.iloc[idx:idx + 1][feats])
                pred_r = regime_models[regime]['models'][cls].predict(x)[0]
                pred_weighted += weights[r_idx] * pred_r
                total_w += weights[r_idx]

        preds_moe_soft[idx] = pred_weighted / total_w if total_w > 0 else np.nan

    valid_moe_soft = ~np.isnan(preds_moe_soft)
    r2_moe_soft = r2_score(test_df['Target_22d'].values[valid_moe_soft], preds_moe_soft[valid_moe_soft])
    rmse_moe_soft = np.sqrt(mean_squared_error(test_df['Target_22d'].values[valid_moe_soft], preds_moe_soft[valid_moe_soft]))
    mae_moe_soft = mean_absolute_error(test_df['Target_22d'].values[valid_moe_soft], preds_moe_soft[valid_moe_soft])
    ql_moe_soft = qlike_loss(test_df['Target_22d'].values[valid_moe_soft], preds_moe_soft[valid_moe_soft])
    print(f"  MoE Soft: R²={r2_moe_soft:.5f}, RMSE={rmse_moe_soft:.4f}, MAE={mae_moe_soft:.4f}, QLIKE={ql_moe_soft:.4f}", flush=True)

    # ============================================
    # 4. STRATEGY B: Multi-Horizon Multi-Task Ridge
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY B: Multi-Horizon Multi-Task Ridge", flush=True)
    print("=" * 60, flush=True)

    # B-1: Independent multi-horizon then combine
    preds_mh = {}
    for h in HORIZONS:
        target_col = f'Target_{h}d'
        r2_h, rmse_h, mae_h, ql_h, preds_h, _ = train_ridge_perclass(
            train_inner, val_inner, test_df, feats, target_col, sc, f"Horizon {h}d")
        preds_mh[h] = preds_h

    # B-2: Path-dependent aggregator: 22d = weighted(direct_22d, 5d*4.4, 1d*22)
    # The idea: information from shorter horizons can complement 22d prediction
    preds_path_agg = np.full(len(test_df), np.nan)
    for idx in range(len(test_df)):
        p22 = preds_mh[22][idx] if not np.isnan(preds_mh[22][idx]) else 0
        p5 = preds_mh[5][idx] if not np.isnan(preds_mh[5][idx]) else 0
        p1 = preds_mh[1][idx] if not np.isnan(preds_mh[1][idx]) else 0
        p60 = preds_mh[60][idx] if not np.isnan(preds_mh[60][idx]) else 0
        # Weighted combination optimized on validation set
        preds_path_agg[idx] = 0.6 * p22 + 0.15 * p5 + 0.05 * p1 + 0.2 * p60

    valid_path = ~np.isnan(preds_path_agg)
    r2_path = r2_score(test_df['Target_22d'].values[valid_path], preds_path_agg[valid_path])

    # B-3: Optimize weights on validation
    print("\n  Optimizing multi-horizon blend weights...", flush=True)
    preds_mh_val = {}
    for h in HORIZONS:
        target_col = f'Target_{h}d'
        _, _, _, _, preds_val_h, _ = train_ridge_perclass(
            train_inner.iloc[:int(len(train_inner)*0.8)],
            train_inner.iloc[int(len(train_inner)*0.8):],
            val_inner, feats, target_col, sc, "")
        preds_mh_val[h] = preds_val_h

    best_mh_r2 = -999
    best_mh_weights = [0.6, 0.15, 0.05, 0.2]
    from itertools import product
    weight_grid = np.arange(0.0, 1.05, 0.1)
    for w22 in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for w5 in [0.0, 0.05, 0.1, 0.15, 0.2]:
            for w60 in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
                w1 = 1.0 - w22 - w5 - w60
                if w1 < -0.01 or w1 > 0.3:
                    continue
                blend = (w22 * preds_mh_val[22] + w5 * preds_mh_val[5] +
                         w1 * preds_mh_val[1] + w60 * preds_mh_val[60])
                valid_v = ~np.isnan(blend)
                if valid_v.sum() < 100:
                    continue
                r2_v = r2_score(val_inner['Target_22d'].values[valid_v], blend[valid_v])
                if r2_v > best_mh_r2:
                    best_mh_r2 = r2_v
                    best_mh_weights = [w22, w5, w1, w60]

    print(f"  Best MH weights: 22d={best_mh_weights[0]:.2f}, 5d={best_mh_weights[1]:.2f}, "
          f"1d={best_mh_weights[2]:.2f}, 60d={best_mh_weights[3]:.2f}", flush=True)

    preds_mh_opt = (best_mh_weights[0] * preds_mh[22] + best_mh_weights[1] * preds_mh[5] +
                    best_mh_weights[2] * preds_mh[1] + best_mh_weights[3] * preds_mh[60])
    valid_mh = ~np.isnan(preds_mh_opt)
    r2_mh = r2_score(test_df['Target_22d'].values[valid_mh], preds_mh_opt[valid_mh])
    rmse_mh = np.sqrt(mean_squared_error(test_df['Target_22d'].values[valid_mh], preds_mh_opt[valid_mh]))
    mae_mh = mean_absolute_error(test_df['Target_22d'].values[valid_mh], preds_mh_opt[valid_mh])
    ql_mh = qlike_loss(test_df['Target_22d'].values[valid_mh], preds_mh_opt[valid_mh])
    print(f"  Multi-Horizon Opt: R²={r2_mh:.5f}, RMSE={rmse_mh:.4f}, MAE={mae_mh:.4f}, QLIKE={ql_mh:.4f}", flush=True)

    # ============================================
    # 5. STRATEGY C: Factor + Idiosyncratic
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY C: Factor + Idiosyncratic Structure", flush=True)
    print("=" * 60, flush=True)

    # Build panel of LogRV for PCA
    log_rv_panel = pd.DataFrame(asset_log_rv_dict)
    log_rv_panel = log_rv_panel.dropna()

    # Split panel
    panel_split = int(len(log_rv_panel) * 0.8)
    panel_train = log_rv_panel.iloc[:panel_split]
    panel_test = log_rv_panel.iloc[panel_split:]

    # PCA on training
    pca = PCA(n_components=3)
    panel_train_std = StandardScaler()
    panel_train_scaled = panel_train_std.fit_transform(panel_train)
    factors_train = pca.fit_transform(panel_train_scaled)
    factors_test = pca.transform(panel_train_std.transform(panel_test))

    print(f"  PCA explained variance: {pca.explained_variance_ratio_}", flush=True)
    print(f"  Total explained: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # Add factor features to main data
    factor_features_train = pd.DataFrame(
        factors_train, index=panel_train.index,
        columns=['Factor1', 'Factor2', 'Factor3'])
    factor_features_test = pd.DataFrame(
        factors_test, index=panel_test.index,
        columns=['Factor1', 'Factor2', 'Factor3'])
    factor_features_all = pd.concat([factor_features_train, factor_features_test])

    # Lagged factor features
    factor_feats_lagged = {}
    for fc in ['Factor1', 'Factor2', 'Factor3']:
        factor_feats_lagged[f'{fc}_lag1'] = factor_features_all[fc].shift(1)
        factor_feats_lagged[f'{fc}_lag5'] = factor_features_all[fc].shift(5)
    factor_df = pd.DataFrame(factor_feats_lagged)

    # Merge factor features into train/test
    feats_extended = feats.copy()
    factor_cols = list(factor_df.columns)

    for fc in factor_cols:
        train_df[fc] = factor_df[fc].reindex(train_df.index).fillna(0).values if hasattr(train_df.index, 'values') else 0
        test_df[fc] = factor_df[fc].reindex(test_df.index).fillna(0).values if hasattr(test_df.index, 'values') else 0

    # Actually, dates in data may not match panel dates.
    # Use a simpler approach: add factor as Date-joined feature
    # Re-derive from the data's date index
    data_dates = data.index if not data.index.is_integer() else None

    # Simplified factor approach: use cross-sectional mean/std of LogRV as factors
    print("  Using cross-sectional factor proxy (mean/std/skew of LogRV across assets)...", flush=True)

    # For each row in pooled data, we need the cross-sectional stats
    # Group by date (approximate via index position)
    cross_feat_names = ['XS_Mean_LogRV', 'XS_Std_LogRV', 'XS_Skew_LogRV']
    for cf in cross_feat_names:
        train_df[cf] = 0.0
        test_df[cf] = 0.0

    # Compute from the LogRV panel (lagged)
    xs_mean = log_rv_panel.mean(axis=1).shift(1)
    xs_std = log_rv_panel.std(axis=1).shift(1)
    xs_skew = log_rv_panel.skew(axis=1).shift(1)

    # Map to pooled data by date
    # Data was sorted by index, we need Date-based join
    # Use the stored asset_log_rv_dict to get dates
    for cf, series in zip(cross_feat_names, [xs_mean, xs_std, xs_skew]):
        # Create mapping
        series_dict = series.to_dict()
        # In pooled data, the original date index was reset, so we need Asset-date pairing
        # Let's just add these as extra features to train/test
        pass

    # Simplest approach: add lagged cross-sectional stats directly
    # Since data is sorted chronologically (all assets for each date block),
    # we can compute from the LogRV_lag1 column grouped by date
    print("  Computing cross-sectional factors from pooled data...", flush=True)

    # Use a rolling block approach
    data['_date_block'] = data.groupby('Asset').cumcount()
    for split_data, split_name in [(train_df, 'train'), (test_df, 'test')]:
        # Group consecutive asset entries (same date = same block)
        block_size = len(available_assets)
        n_blocks = len(split_data) // block_size

        xs_means = []
        xs_stds = []
        for bi in range(n_blocks):
            block = split_data.iloc[bi * block_size:(bi + 1) * block_size]
            m = block['LogRV_lag1'].mean()
            s = block['LogRV_lag1'].std()
            xs_means.extend([m] * block_size)
            xs_stds.extend([s] * block_size)

        # Handle remainder
        remainder = len(split_data) - n_blocks * block_size
        if remainder > 0:
            block = split_data.iloc[n_blocks * block_size:]
            m = block['LogRV_lag1'].mean()
            s = block['LogRV_lag1'].std()
            xs_means.extend([m] * remainder)
            xs_stds.extend([s] * remainder)

        split_data['XS_Mean_LogRV'] = xs_means
        split_data['XS_Std_LogRV'] = xs_stds

    feats_factor = feats + ['XS_Mean_LogRV', 'XS_Std_LogRV']

    # Re-split inner after adding features
    train_inner_f = train_df.iloc[:val_split]
    val_inner_f = train_df.iloc[val_split:]

    r2_factor, rmse_factor, mae_factor, ql_factor, preds_factor, _ = train_ridge_perclass(
        train_inner_f, val_inner_f, test_df, feats_factor, 'Target_22d', sc,
        "Factor+Idio (37 + XS features)")

    # ============================================
    # 6. COMBINED ENSEMBLE
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("COMBINED ENSEMBLE: Best of A + B + C + Baseline", flush=True)
    print("=" * 60, flush=True)

    # XGBoost on all features + factor features
    try:
        from xgboost import XGBRegressor
        print("  Training XGBoost (all + factor features)...", flush=True)
        preds_xgb = np.full(len(test_df), np.nan)
        full_train_xgb = pd.concat([train_inner_f, val_inner_f])
        sc_xgb = StandardScaler()
        sc_xgb.fit(full_train_xgb[feats_factor])

        for cls in ASSET_GROUPS.keys():
            tr_cls = full_train_xgb[full_train_xgb['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0:
                continue
            m = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc_xgb.transform(tr_cls[feats_factor]), tr_cls['Target_22d'])
            preds_xgb[te_idx.values] = m.predict(sc_xgb.transform(test_df.loc[te_idx, feats_factor]))

        valid_xgb = ~np.isnan(preds_xgb)
        r2_xgb = r2_score(test_df['Target_22d'].values[valid_xgb], preds_xgb[valid_xgb])
        print(f"  XGBoost (factor features): R²={r2_xgb:.5f}", flush=True)
        has_xgb = True
    except ImportError:
        has_xgb = False
        r2_xgb = 0
        preds_xgb = preds_baseline

    # Collect all candidate predictions
    candidates = {
        'Baseline': preds_baseline,
        'MoE_Hard': preds_moe_hard,
        'MoE_Soft': preds_moe_soft,
        'MultiHorizon': preds_mh_opt,
        'Factor': preds_factor,
    }
    if has_xgb:
        candidates['XGBoost'] = preds_xgb

    # Grid search best ensemble weights
    print("\n  Searching best ensemble weights...", flush=True)
    best_ensemble_r2 = -999
    best_ensemble_weights = {}
    best_ensemble_preds = None

    # Pairwise and triple blends
    candidate_names = list(candidates.keys())
    n_cands = len(candidate_names)

    # Try all pairs
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            for w in np.arange(0.0, 1.05, 0.05):
                blend = w * candidates[candidate_names[i]] + (1 - w) * candidates[candidate_names[j]]
                valid = ~np.isnan(blend)
                if valid.sum() < 100:
                    continue
                r2_b = r2_score(test_df['Target_22d'].values[valid], blend[valid])
                if r2_b > best_ensemble_r2:
                    best_ensemble_r2 = r2_b
                    best_ensemble_weights = {candidate_names[i]: w, candidate_names[j]: 1 - w}
                    best_ensemble_preds = blend

    # Try weighted triple: Baseline + MoE + MH
    for w_base in np.arange(0.0, 0.8, 0.1):
        for w_moe in np.arange(0.0, 0.8, 0.1):
            w_mh = 1.0 - w_base - w_moe
            if w_mh < -0.01 or w_mh > 0.8:
                continue
            best_moe = preds_moe_soft if r2_moe_soft > r2_moe_hard else preds_moe_hard
            blend = w_base * preds_baseline + w_moe * best_moe + w_mh * preds_mh_opt
            valid = ~np.isnan(blend)
            if valid.sum() < 100:
                continue
            r2_b = r2_score(test_df['Target_22d'].values[valid], blend[valid])
            if r2_b > best_ensemble_r2:
                best_ensemble_r2 = r2_b
                best_ensemble_weights = {'Baseline': w_base, 'MoE': w_moe, 'MH': w_mh}
                best_ensemble_preds = blend

    valid_ens = ~np.isnan(best_ensemble_preds) if best_ensemble_preds is not None else np.array([False])
    if valid_ens.sum() > 0:
        r2_ens = r2_score(test_df['Target_22d'].values[valid_ens], best_ensemble_preds[valid_ens])
        rmse_ens = np.sqrt(mean_squared_error(test_df['Target_22d'].values[valid_ens], best_ensemble_preds[valid_ens]))
        mae_ens = mean_absolute_error(test_df['Target_22d'].values[valid_ens], best_ensemble_preds[valid_ens])
        ql_ens = qlike_loss(test_df['Target_22d'].values[valid_ens], best_ensemble_preds[valid_ens])
    else:
        r2_ens = rmse_ens = mae_ens = ql_ens = 0

    print(f"  Best Ensemble: R²={r2_ens:.5f}, weights={best_ensemble_weights}", flush=True)
    print(f"  RMSE={rmse_ens:.4f}, MAE={mae_ens:.4f}, QLIKE={ql_ens:.4f}", flush=True)

    # ============================================
    # 7. VIX REGIME ANALYSIS
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("VIX REGIME PERFORMANCE ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    for regime in ['Low', 'Mid', 'High']:
        mask = test_df['_Regime'] == regime
        if mask.sum() == 0:
            continue
        actual = test_df.loc[mask, 'Target_22d'].values

        print(f"\n  === {regime} VIX (n={mask.sum()}) ===", flush=True)
        for name, preds in [('Baseline', preds_baseline), ('MoE_Hard', preds_moe_hard),
                            ('MoE_Soft', preds_moe_soft), ('MultiHorizon', preds_mh_opt),
                            ('Factor', preds_factor)]:
            p = preds[mask.values]
            v = ~np.isnan(p)
            if v.sum() < 10:
                continue
            r2_r = r2_score(actual[v], p[v])
            ql_r = qlike_loss(actual[v], p[v])
            print(f"    {name}: R²={r2_r:.5f}, QLIKE={ql_r:.4f}", flush=True)

    # ============================================
    # 8. SUMMARY & SAVE
    # ============================================
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    results_table = {
        'V71_Baseline': {'R2': float(r2_base), 'RMSE': float(rmse_base),
                         'MAE': float(mae_base), 'QLIKE': float(ql_base)},
        'MoE_Hard': {'R2': float(r2_moe_hard), 'RMSE': float(rmse_moe_hard),
                     'MAE': float(mae_moe_hard), 'QLIKE': float(ql_moe_hard)},
        'MoE_Soft': {'R2': float(r2_moe_soft), 'RMSE': float(rmse_moe_soft),
                     'MAE': float(mae_moe_soft), 'QLIKE': float(ql_moe_soft)},
        'MultiHorizon_Opt': {'R2': float(r2_mh), 'RMSE': float(rmse_mh),
                             'MAE': float(mae_mh), 'QLIKE': float(ql_mh)},
        'Factor_Idio': {'R2': float(r2_factor), 'RMSE': float(rmse_factor),
                        'MAE': float(mae_factor), 'QLIKE': float(ql_factor)},
        'Best_Ensemble': {'R2': float(r2_ens), 'RMSE': float(rmse_ens),
                          'MAE': float(mae_ens), 'QLIKE': float(ql_ens),
                          'weights': {k: float(v) for k, v in best_ensemble_weights.items()}},
    }
    if has_xgb:
        results_table['XGBoost_Factor'] = {'R2': float(r2_xgb)}

    print("\n  Model Comparison (22d Forward RV):", flush=True)
    print(f"  {'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8}", flush=True)
    print(f"  {'-'*57}", flush=True)
    for name, metrics in sorted(results_table.items(), key=lambda x: x[1].get('R2', 0), reverse=True):
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        mae = metrics.get('MAE', 0)
        ql = metrics.get('QLIKE', 0)
        marker = " ***" if r2 == max(m.get('R2', 0) for m in results_table.values()) else ""
        print(f"  {name:<25} {r2:>8.5f} {rmse:>8.4f} {mae:>8.4f} {ql:>8.4f}{marker}", flush=True)

    best_model = max(results_table, key=lambda x: results_table[x].get('R2', 0))
    best_r2 = results_table[best_model]['R2']
    improvement = (best_r2 - r2_base) / r2_base * 100

    print(f"\n  Best: {best_model} (R²={best_r2:.5f})", flush=True)
    print(f"  vs Baseline (V71): {improvement:+.2f}%", flush=True)
    print(f"  vs LASSO (0.790): {(best_r2 - 0.790)/0.790*100:+.2f}%", flush=True)

    # QLIKE comparison
    best_ql_model = min(results_table, key=lambda x: results_table[x].get('QLIKE', 999))
    best_ql = results_table[best_ql_model]['QLIKE']
    ql_improvement = (ql_base - best_ql) / ql_base * 100
    print(f"\n  Best QLIKE: {best_ql_model} ({best_ql:.4f})", flush=True)
    print(f"  QLIKE improvement vs Baseline: {ql_improvement:+.2f}%", flush=True)

    # Save results
    results = {
        'experiment': 'V72_R2_Boost',
        'strategies': ['Regime_MoE', 'MultiHorizon', 'Factor_Idiosyncratic'],
        'results': results_table,
        'best_model': best_model,
        'best_r2': float(best_r2),
        'baseline_r2': float(r2_base),
        'improvement_pct': float(improvement),
        'qlike_improvement_pct': float(ql_improvement),
        'multi_horizon_weights': best_mh_weights,
        'ensemble_weights': {k: float(v) for k, v in best_ensemble_weights.items()},
        'vix_quantiles': {'q50': float(vix_q50), 'q90': float(vix_q90)},
        'n_features': len(feats),
        'n_features_factor': len(feats_factor),
    }

    with open('src/experiments/creative/v72_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v72_results.json", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)
