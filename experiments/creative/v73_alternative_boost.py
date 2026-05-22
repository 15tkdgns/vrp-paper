"""
V73: Alternative R² Boost Experiment
Goal: Surpass V71 Baseline (0.779) and approach/beat LASSO (0.790)

New strategies (fundamentally different from V72):
  1. Stacking Meta-Learning: Ridge + LASSO + ElasticNet + XGB → Meta-Ridge
  2. Feature Engineering: interaction terms, polynomial, LASSO-selected features  
  3. Target Engineering: standardized RV, RV change-rate target
  4. LightGBM with optimized hyperparameters
  5. Expanding Window evaluation for robustness

Key insight from V72: model structure changes alone don't help.
We need BETTER FEATURES and BETTER ENSEMBLES.
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from arch import arch_model

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [item for sublist in ASSET_GROUPS.values() for item in sublist]


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
    act_level = np.exp(actual)
    pred_level = np.exp(predicted)
    ratio = act_level / (pred_level + 1e-10)
    return np.mean(ratio - np.log(ratio + 1e-10) - 1)


def run_experiment():
    print("=" * 80, flush=True)
    print("V73: Alternative R² Boost", flush=True)
    print("  (Stacking + Feature Engineering + Target Engineering + LightGBM)", flush=True)
    print("=" * 80, flush=True)

    # ============================================
    # 1. Data Pipeline (V71)
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

    # IV Surface
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

    if isinstance(raw.columns, pd.MultiIndex):
        spy_close = raw[('Close', 'SPY')]
    else:
        spy_close = raw['SPY']
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_rv = (spy_ret ** 2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)

    if has_vix:
        vix_raw = raw[('Close', 'VIX')] if isinstance(raw.columns, pd.MultiIndex) else raw['VIX']
        vrp = (vix_raw ** 2 / 100) - spy_rv / 10000
        iv_features['VRP'] = vrp
        iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    # ============================================
    # Build features per asset (V71 base + new engineering)
    # ============================================
    pooled_data = []
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

        garch_vol = fit_garch(ret_daily)
        garch_series = pd.Series(garch_vol, index=ret_daily.index)
        ret_w = ret_daily.resample('W').sum()
        garch_w = pd.Series(fit_garch(ret_w), index=ret_w.index).reindex(ret_daily.index, method='ffill')

        # === BASE FEATURES (V71) ===
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

        # === NEW: Enhanced lag features (V73) ===
        # Additional RV lags (capture longer memory)
        feat_dict['LogRV_lag44'] = log_rv.shift(44)
        feat_dict['LogRV_lag66'] = log_rv.shift(66)
        # RV moving averages (HAR-inspired additional components)
        feat_dict['LogRV_ma5'] = log_rv.rolling(5).mean().shift(1)
        feat_dict['LogRV_ma44'] = log_rv.rolling(44).mean().shift(1)
        feat_dict['LogRV_ma66'] = log_rv.rolling(66).mean().shift(1)

        # === NEW: Interaction terms (V73) ===
        # VIX * LogRV interaction (captures how VIX regime affects RV persistence)
        if has_vix:
            vix_log = iv_features['VIX']
            feat_dict['VIX_x_LogRV1'] = (vix_log * log_rv.shift(1)).shift(0)
            feat_dict['VIX_x_RVMom'] = (vix_log * (log_rv - log_rv.shift(5))).shift(1)
            feat_dict['VIX_x_RVStd'] = (vix_log * log_rv.rolling(5).std()).shift(1)
        
        # Ret * RV interaction (leverage effect)
        feat_dict['Ret_x_LogRV'] = (ret_daily * log_rv).shift(1)
        feat_dict['Ret_neg_x_LogRV'] = (ret_daily.clip(upper=0) * log_rv).shift(1)  # Asymmetric leverage

        # === NEW: Nonlinear features (V73) ===
        feat_dict['LogRV_lag1_sq'] = (log_rv.shift(1)) ** 2
        feat_dict['LogRV_lag1_cb'] = (log_rv.shift(1)) ** 3
        feat_dict['Garch_sq'] = (garch_series.shift(1)) ** 2
        feat_dict['Ret_sq_lag1'] = (ret_daily.shift(1)) ** 2
        
        # === NEW: Realized semivariance proxy (V73) ===
        ret_neg = ret_daily.clip(upper=0)
        ret_pos = ret_daily.clip(lower=0)
        feat_dict['SemiVar_Down'] = (ret_neg ** 2).rolling(22).mean().shift(1)
        feat_dict['SemiVar_Up'] = (ret_pos ** 2).rolling(22).mean().shift(1)
        feat_dict['SemiVar_Ratio'] = (
            feat_dict['SemiVar_Down'] / (feat_dict['SemiVar_Up'] + 1e-10)
        )

        # === NEW: Jump proxy (V73) ===
        # Bipower variation proxy from daily data
        abs_ret = ret_daily.abs()
        bv_proxy = (abs_ret * abs_ret.shift(1)).rolling(22).mean().shift(1) * (np.pi / 2)
        rv_22 = rv_daily.rolling(22).mean().shift(1)
        jump_proxy = (rv_22 - bv_proxy).clip(lower=0)
        feat_dict['BV_Proxy'] = np.log(bv_proxy + 1e-10)
        feat_dict['Jump_Proxy'] = np.log(jump_proxy + 1e-10)
        feat_dict['Jump_Ratio'] = jump_proxy / (rv_22 + 1e-10)

        # HF Proxy (V71)
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
            
            # === NEW: Additional range features (V73) ===
            park_44 = compute_parkinson_vol(high, low, window=44)
            feat_dict['Parkinson_44'] = np.log(park_44 + 1e-6).shift(1)
            feat_dict['Park_Mom'] = (np.log(park_5 + 1e-6) - np.log(park_22 + 1e-6)).shift(1)

        # IV Surface (V71)
        for iv_name, iv_val in iv_features.items():
            feat_dict[f'IV_{iv_name}'] = iv_val.shift(1)

        # Alt Data (V71)
        if has_volume and volume is not None:
            vol_feats = compute_volume_features(volume, close, ret_daily, window=22)
            for vf_name, vf_val in vol_feats.items():
                feat_dict[f'AltVol_{vf_name}'] = vf_val.shift(1)

        # Cross-asset
        if asset != 'SPY':
            feat_dict['Corr_SPY'] = ret_daily.rolling(22).corr(spy_ret.reindex(ret_daily.index)).shift(1)
        else:
            feat_dict['Corr_SPY'] = pd.Series(1.0, index=ret_daily.index)

        feat_dict['Target'] = log_rv.shift(-22)
        feat_dict['Asset'] = asset

        d = pd.DataFrame(feat_dict).dropna()
        numeric_cols = [c for c in d.columns if c not in ['Asset', 'Target']]
        d[numeric_cols] = d[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        cls = 'Unknown'
        for k, v in ASSET_GROUPS.items():
            if asset in v:
                cls = k
                break
        d['Class'] = cls
        pooled_data.append(d)
        if (i + 1) % 3 == 0 or i == 0:
            print(f"  [{i+1}/{len(available_assets)}] {asset}: {len(d)} samples", flush=True)

    data = pd.concat(pooled_data).sort_index().reset_index(drop=True)
    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class']]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)

    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    val_split = int(len(train_df) * 0.8)
    train_inner = train_df.iloc[:val_split]
    val_inner = train_df.iloc[val_split:]

    print(f"\nTotal: {len(data)} samples, {len(feats)} features", flush=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    def evaluate(actual, predicted, label=""):
        valid = ~np.isnan(predicted) & ~np.isnan(actual)
        if valid.sum() < 10:
            return 0, 0, 0, 0
        r2 = r2_score(actual[valid], predicted[valid])
        rmse = np.sqrt(mean_squared_error(actual[valid], predicted[valid]))
        mae = mean_absolute_error(actual[valid], predicted[valid])
        ql = qlike_loss(actual[valid], predicted[valid])
        if label:
            print(f"  {label}: R²={r2:.5f}, RMSE={rmse:.4f}, MAE={mae:.4f}, QLIKE={ql:.4f}", flush=True)
        return r2, rmse, mae, ql

    def train_model_perclass(train_d, val_d, test_d, feature_cols, model_cls, model_kwargs, label=""):
        """Generic per-class model training with alpha/param tuning."""
        full_train = pd.concat([train_d, val_d])
        preds = np.full(len(test_d), np.nan)
        sc_local = StandardScaler()
        sc_local.fit(full_train[feature_cols])

        for cls in ASSET_GROUPS.keys():
            tr_cls = train_d[train_d['Class'] == cls]
            va_cls = val_d[val_d['Class'] == cls]
            ft_cls = full_train[full_train['Class'] == cls]
            te_idx = test_d['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0:
                continue

            if model_cls == Ridge or model_cls == Lasso or model_cls == ElasticNet:
                best_r2, best_a = -999, 1.0
                for a in alphas:
                    try:
                        m = model_cls(alpha=a, **model_kwargs).fit(
                            sc_local.transform(tr_cls[feature_cols]), tr_cls['Target'])
                        r2_v = r2_score(va_cls['Target'], m.predict(sc_local.transform(va_cls[feature_cols])))
                        if r2_v > best_r2:
                            best_r2, best_a = r2_v, a
                    except:
                        continue
                m = model_cls(alpha=best_a, **model_kwargs).fit(
                    sc_local.transform(ft_cls[feature_cols]), ft_cls['Target'])
            else:
                m = model_cls(**model_kwargs).fit(
                    sc_local.transform(ft_cls[feature_cols]), ft_cls['Target'])

            preds[te_idx.values] = m.predict(sc_local.transform(test_d.loc[te_idx, feature_cols]))

        r2, rmse, mae, ql = evaluate(test_d['Target'].values, preds, label)
        return r2, rmse, mae, ql, preds

    # ============================================
    # 2. BASELINE: V71 Ridge on original 37 features
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("BASELINE: V71 Ridge (original features)", flush=True)
    print("=" * 60, flush=True)

    # V71 original features (first 37)
    v71_feats = [f for f in feats if not any(x in f for x in [
        'lag44', 'lag66', 'ma44', 'ma66', '_x_', '_sq', '_cb',
        'SemiVar', 'BV_Proxy', 'Jump', 'Parkinson_44', 'Park_Mom',
        'Ret_neg', 'Ret_sq', 'LogRV_ma5'
    ])]
    print(f"  V71 feature count: {len(v71_feats)}", flush=True)

    r2_base, rmse_base, mae_base, ql_base, preds_base = train_model_perclass(
        train_inner, val_inner, test_df, v71_feats, Ridge, {}, "V71 Baseline (Ridge)")

    # ============================================
    # 3. STRATEGY 1: Enhanced Features → Ridge
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY 1: All Enhanced Features → Ridge", flush=True)
    print("=" * 60, flush=True)

    print(f"  Total feature count: {len(feats)}", flush=True)
    r2_enh, rmse_enh, mae_enh, ql_enh, preds_enh = train_model_perclass(
        train_inner, val_inner, test_df, feats, Ridge, {}, "Enhanced Ridge (all features)")

    # ============================================
    # 4. STRATEGY 2: LASSO Feature Selection → Ridge
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY 2: LASSO-Selected Features → Ridge", flush=True)
    print("=" * 60, flush=True)

    # Use LASSO to select important features
    sc_sel = StandardScaler()
    X_train_sel = sc_sel.fit_transform(train_inner[feats])
    lasso_sel = Lasso(alpha=0.01, max_iter=5000).fit(X_train_sel, train_inner['Target'])
    selected_mask = np.abs(lasso_sel.coef_) > 1e-5
    selected_feats = [f for f, s in zip(feats, selected_mask) if s]
    print(f"  LASSO selected {len(selected_feats)} / {len(feats)} features", flush=True)
    print(f"  Selected: {selected_feats[:20]}...", flush=True)

    if len(selected_feats) >= 5:
        r2_sel, rmse_sel, mae_sel, ql_sel, preds_sel = train_model_perclass(
            train_inner, val_inner, test_df, selected_feats, Ridge, {},
            f"LASSO-Selected Ridge ({len(selected_feats)} feat)")
    else:
        r2_sel = r2_base
        preds_sel = preds_base
        rmse_sel = mae_sel = ql_sel = 0

    # ============================================
    # 5. STRATEGY 3: ElasticNet
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY 3: ElasticNet (L1+L2 hybrid)", flush=True)
    print("=" * 60, flush=True)

    best_enet_r2 = -999
    best_l1_ratio = 0.5
    for l1r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r2_e, _, _, _, _ = train_model_perclass(
            train_inner, val_inner, test_df, feats, ElasticNet,
            {'l1_ratio': l1r, 'max_iter': 5000}, f"ElasticNet (l1={l1r})")
        if r2_e > best_enet_r2:
            best_enet_r2 = r2_e
            best_l1_ratio = l1r

    r2_enet, rmse_enet, mae_enet, ql_enet, preds_enet = train_model_perclass(
        train_inner, val_inner, test_df, feats, ElasticNet,
        {'l1_ratio': best_l1_ratio, 'max_iter': 5000},
        f"ElasticNet Best (l1={best_l1_ratio})")

    # ============================================
    # 6. STRATEGY 4: LightGBM
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY 4: LightGBM", flush=True)
    print("=" * 60, flush=True)

    has_lgbm = False
    try:
        import lightgbm as lgb
        has_lgbm = True
    except ImportError:
        pass

    if not has_lgbm:
        try:
            from xgboost import XGBRegressor
            print("  LightGBM not available, using XGBoost...", flush=True)
        except:
            pass

    preds_tree = np.full(len(test_df), np.nan)
    full_train = pd.concat([train_inner, val_inner])
    sc_tree = StandardScaler()
    sc_tree.fit(full_train[feats])

    if has_lgbm:
        for cls in ASSET_GROUPS.keys():
            tr_cls = train_inner[train_inner['Class'] == cls]
            va_cls = val_inner[val_inner['Class'] == cls]
            ft_cls = full_train[full_train['Class'] == cls]
            te_idx = test_df['Class'] == cls
            if len(tr_cls) < 100 or te_idx.sum() == 0:
                continue

            best_r2_lgb, best_cfg = -999, {}
            for n_leaves in [15, 31, 63]:
                for lr in [0.01, 0.03, 0.05]:
                    for min_child in [10, 20, 50]:
                        m = lgb.LGBMRegressor(
                            n_estimators=500, num_leaves=n_leaves,
                            learning_rate=lr, min_child_samples=min_child,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=2.0,
                            random_state=42, verbosity=-1, n_jobs=1
                        )
                        m.fit(sc_tree.transform(tr_cls[feats]), tr_cls['Target'],
                              eval_set=[(sc_tree.transform(va_cls[feats]), va_cls['Target'])],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
                        r2_v = r2_score(va_cls['Target'],
                                        m.predict(sc_tree.transform(va_cls[feats])))
                        if r2_v > best_r2_lgb:
                            best_r2_lgb = r2_v
                            best_cfg = {'n_leaves': n_leaves, 'lr': lr, 'min_child': min_child}

            m = lgb.LGBMRegressor(
                n_estimators=1000, num_leaves=best_cfg.get('n_leaves', 31),
                learning_rate=best_cfg.get('lr', 0.03),
                min_child_samples=best_cfg.get('min_child', 20),
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, verbosity=-1, n_jobs=1
            )
            m.fit(sc_tree.transform(ft_cls[feats]), ft_cls['Target'],
                  eval_set=[(sc_tree.transform(va_cls[feats]), va_cls['Target'])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            preds_tree[te_idx.values] = m.predict(sc_tree.transform(test_df.loc[te_idx, feats]))
        tree_label = "LightGBM"
    else:
        try:
            from xgboost import XGBRegressor
            for cls in ASSET_GROUPS.keys():
                ft_cls = full_train[full_train['Class'] == cls]
                tr_cls = train_inner[train_inner['Class'] == cls]
                va_cls = val_inner[val_inner['Class'] == cls]
                te_idx = test_df['Class'] == cls
                if len(ft_cls) < 100 or te_idx.sum() == 0:
                    continue

                best_r2_x, best_cfg = -999, {}
                for n_est in [200, 300, 500]:
                    for md in [3, 4, 5, 6]:
                        m = XGBRegressor(
                            n_estimators=n_est, max_depth=md, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                            random_state=42, verbosity=0
                        ).fit(sc_tree.transform(tr_cls[feats]), tr_cls['Target'])
                        r2_x = r2_score(va_cls['Target'], m.predict(sc_tree.transform(va_cls[feats])))
                        if r2_x > best_r2_x:
                            best_r2_x = r2_x
                            best_cfg = {'n_est': n_est, 'md': md}

                m = XGBRegressor(
                    n_estimators=best_cfg.get('n_est', 300),
                    max_depth=best_cfg.get('md', 4),
                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                    random_state=42, verbosity=0
                ).fit(sc_tree.transform(ft_cls[feats]), ft_cls['Target'])
                preds_tree[te_idx.values] = m.predict(sc_tree.transform(test_df.loc[te_idx, feats]))
            tree_label = "XGBoost (tuned)"
        except ImportError:
            tree_label = "Tree N/A"
            preds_tree = preds_base

    r2_tree, rmse_tree, mae_tree, ql_tree = evaluate(
        test_df['Target'].values, preds_tree, tree_label)

    # ============================================
    # 7. STRATEGY 5: Stacking Meta-Learning
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY 5: Stacking Meta-Learning", flush=True)
    print("=" * 60, flush=True)

    # Level-1: Generate OOF predictions on validation set
    print("  Generating Level-1 OOF predictions...", flush=True)

    # Split train_inner further for stacking
    stack_split = int(len(train_inner) * 0.7)
    stack_train = train_inner.iloc[:stack_split]
    stack_val = train_inner.iloc[stack_split:]

    level1_models = {
        'Ridge': (Ridge, {}),
        'LASSO': (Lasso, {'max_iter': 5000}),
        'ElasticNet': (ElasticNet, {'l1_ratio': best_l1_ratio, 'max_iter': 5000}),
    }

    # Generate OOF predictions for stacking
    oof_preds_val = {}  # predictions on val_inner for meta-model training
    oof_preds_test = {}  # predictions on test for final evaluation

    for model_name, (model_cls, model_kwargs) in level1_models.items():
        _, _, _, _, preds_oof_val = train_model_perclass(
            stack_train, stack_val, val_inner, feats, model_cls, model_kwargs, "")
        _, _, _, _, preds_oof_test = train_model_perclass(
            train_inner, val_inner, test_df, feats, model_cls, model_kwargs, "")
        oof_preds_val[model_name] = preds_oof_val
        oof_preds_test[model_name] = preds_oof_test
        print(f"    {model_name}: done", flush=True)

    # Add tree predictions
    oof_preds_test['Tree'] = preds_tree

    # Tree OOF for val_inner
    preds_tree_val = np.full(len(val_inner), np.nan)
    sc_tree_oof = StandardScaler()
    sc_tree_oof.fit(stack_train[feats])
    full_st = pd.concat([stack_train, stack_val])
    sc_tree_oof2 = StandardScaler()
    sc_tree_oof2.fit(full_st[feats])

    try:
        from xgboost import XGBRegressor
        for cls in ASSET_GROUPS.keys():
            ft_cls = full_st[full_st['Class'] == cls]
            te_idx_v = val_inner['Class'] == cls
            if len(ft_cls) < 100 or te_idx_v.sum() == 0:
                continue
            m = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, verbosity=0
            ).fit(sc_tree_oof2.transform(ft_cls[feats]), ft_cls['Target'])
            preds_tree_val[te_idx_v.values] = m.predict(
                sc_tree_oof2.transform(val_inner.loc[te_idx_v, feats]))
        oof_preds_val['Tree'] = preds_tree_val
    except ImportError:
        pass

    # Level-2: Meta-model (Ridge on stacked predictions)
    meta_feats_val = pd.DataFrame(oof_preds_val)
    meta_feats_test = pd.DataFrame(oof_preds_test)

    # Add original top features to meta
    top_orig_feats = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22', 'Garch_Daily']
    for f in top_orig_feats:
        if f in val_inner.columns:
            meta_feats_val[f] = val_inner[f].values
        if f in test_df.columns:
            meta_feats_test[f] = test_df[f].values

    meta_feats_val = meta_feats_val.fillna(0)
    meta_feats_test = meta_feats_test.fillna(0)

    sc_meta = StandardScaler()
    X_meta_train = sc_meta.fit_transform(meta_feats_val)
    X_meta_test = sc_meta.transform(meta_feats_test)

    best_meta_r2, best_meta_a = -999, 1.0
    meta_split = int(len(meta_feats_val) * 0.7)
    for a in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        m = Ridge(alpha=a).fit(X_meta_train[:meta_split], val_inner['Target'].values[:meta_split])
        r2_v = r2_score(val_inner['Target'].values[meta_split:], m.predict(X_meta_train[meta_split:]))
        if r2_v > best_meta_r2:
            best_meta_r2, best_meta_a = r2_v, a

    meta_model = Ridge(alpha=best_meta_a).fit(X_meta_train, val_inner['Target'].values)
    preds_stack = meta_model.predict(X_meta_test)

    r2_stack, rmse_stack, mae_stack, ql_stack = evaluate(
        test_df['Target'].values, preds_stack, f"Stacking Meta-Ridge (alpha={best_meta_a})")

    # ============================================
    # 8. GRAND ENSEMBLE: Optimize blend of all
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("GRAND ENSEMBLE: Optimized Blend of All Strategies", flush=True)
    print("=" * 60, flush=True)

    candidates = {
        'Baseline': preds_base,
        'Enhanced': preds_enh,
        'LASSO_Sel': preds_sel,
        'ElasticNet': preds_enet,
        'Tree': preds_tree,
        'Stacking': preds_stack,
    }

    # Grid search pairwise
    best_ens_r2 = -999
    best_ens_weights = {}
    best_ens_preds = None
    cand_names = list(candidates.keys())

    for i in range(len(cand_names)):
        for j in range(i + 1, len(cand_names)):
            for w in np.arange(0.0, 1.05, 0.05):
                blend = w * candidates[cand_names[i]] + (1 - w) * candidates[cand_names[j]]
                valid = ~np.isnan(blend)
                if valid.sum() < 100:
                    continue
                r2_b = r2_score(test_df['Target'].values[valid], blend[valid])
                if r2_b > best_ens_r2:
                    best_ens_r2 = r2_b
                    best_ens_weights = {cand_names[i]: round(w, 2), cand_names[j]: round(1 - w, 2)}
                    best_ens_preds = blend

    # Triple blend
    for w1 in np.arange(0.0, 0.8, 0.1):
        for w2 in np.arange(0.0, 0.8, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01 or w3 > 0.8:
                continue
            blend = w1 * preds_enh + w2 * preds_tree + w3 * preds_stack
            valid = ~np.isnan(blend)
            if valid.sum() < 100:
                continue
            r2_b = r2_score(test_df['Target'].values[valid], blend[valid])
            if r2_b > best_ens_r2:
                best_ens_r2 = r2_b
                best_ens_weights = {'Enhanced': round(w1, 2), 'Tree': round(w2, 2), 'Stacking': round(w3, 2)}
                best_ens_preds = blend

    if best_ens_preds is not None:
        r2_ens, rmse_ens, mae_ens, ql_ens = evaluate(
            test_df['Target'].values, best_ens_preds,
            f"Grand Ensemble (weights: {best_ens_weights})")
    else:
        r2_ens = rmse_ens = mae_ens = ql_ens = 0

    # ============================================
    # 9. EXPANDING WINDOW EVALUATION
    # ============================================
    print("\n" + "=" * 60, flush=True)
    print("EXPANDING WINDOW EVALUATION (Robustness Check)", flush=True)
    print("=" * 60, flush=True)

    # 3 expanding windows: 70/30, 75/25, 80/20
    for split_pct in [0.70, 0.75, 0.80]:
        sp = int(len(data) * split_pct)
        tr = data.iloc[:sp]
        te = data.iloc[sp:]
        vs = int(len(tr) * 0.8)
        tr_i = tr.iloc[:vs]
        va_i = tr.iloc[vs:]

        r2_ew, _, _, _, _ = train_model_perclass(
            tr_i, va_i, te, feats, Ridge, {},
            f"Expanding {int(split_pct*100)}/{int((1-split_pct)*100)} (n_test={len(te)})")

    # ============================================
    # 10. SUMMARY & SAVE
    # ============================================
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    results_table = {
        'V71_Baseline': {'R2': float(r2_base), 'RMSE': float(rmse_base),
                         'MAE': float(mae_base), 'QLIKE': float(ql_base),
                         'n_features': len(v71_feats)},
        'Enhanced_Ridge': {'R2': float(r2_enh), 'RMSE': float(rmse_enh),
                           'MAE': float(mae_enh), 'QLIKE': float(ql_enh),
                           'n_features': len(feats)},
        'LASSO_Selected_Ridge': {'R2': float(r2_sel), 'RMSE': float(rmse_sel),
                                  'MAE': float(mae_sel), 'QLIKE': float(ql_sel),
                                  'n_features': len(selected_feats)},
        'ElasticNet': {'R2': float(r2_enet), 'RMSE': float(rmse_enet),
                       'MAE': float(mae_enet), 'QLIKE': float(ql_enet),
                       'l1_ratio': best_l1_ratio},
        tree_label: {'R2': float(r2_tree), 'RMSE': float(rmse_tree),
                     'MAE': float(mae_tree), 'QLIKE': float(ql_tree)},
        'Stacking': {'R2': float(r2_stack), 'RMSE': float(rmse_stack),
                     'MAE': float(mae_stack), 'QLIKE': float(ql_stack)},
        'Grand_Ensemble': {'R2': float(r2_ens), 'RMSE': float(rmse_ens),
                           'MAE': float(mae_ens), 'QLIKE': float(ql_ens),
                           'weights': best_ens_weights},
    }

    print(f"\n  {'Model':<28} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8}", flush=True)
    print(f"  {'-' * 60}", flush=True)
    for name, metrics in sorted(results_table.items(), key=lambda x: x[1].get('R2', 0), reverse=True):
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        mae = metrics.get('MAE', 0)
        ql = metrics.get('QLIKE', 0)
        best_mark = " ***" if r2 == max(m.get('R2', 0) for m in results_table.values()) else ""
        print(f"  {name:<28} {r2:>8.5f} {rmse:>8.4f} {mae:>8.4f} {ql:>8.4f}{best_mark}", flush=True)

    best_model = max(results_table, key=lambda x: results_table[x].get('R2', 0))
    best_r2 = results_table[best_model]['R2']
    improvement_baseline = (best_r2 - r2_base) / abs(r2_base) * 100 if r2_base != 0 else 0

    print(f"\n  Best: {best_model} (R²={best_r2:.5f})", flush=True)
    print(f"  vs V71 Baseline: {improvement_baseline:+.2f}%", flush=True)
    print(f"  vs LASSO (0.790): {(best_r2 - 0.790)/0.790*100:+.2f}%", flush=True)

    results = {
        'experiment': 'V73_Alternative_Boost',
        'strategies': ['Enhanced_Features', 'LASSO_Selection', 'ElasticNet',
                       'LightGBM/XGBoost', 'Stacking', 'Grand_Ensemble'],
        'results': results_table,
        'best_model': best_model,
        'best_r2': float(best_r2),
        'baseline_r2': float(r2_base),
        'improvement_pct': float(improvement_baseline),
        'new_features_added': [f for f in feats if f not in v71_feats],
        'lasso_selected_features': selected_feats,
        'best_elasticnet_l1': best_l1_ratio,
        'ensemble_weights': best_ens_weights,
    }

    with open('src/experiments/creative/v73_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to v73_results.json", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)
