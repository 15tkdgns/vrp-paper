"""
V71 Academic Robustness Verification Suite
===========================================
Reviewer-requested robustness checks:

1) GVZ/OVX Commodity IV robustness
   - Replace VIX with asset-specific IV indices for commodity assets
2) Block Bootstrap DM comparison
   - Non-parametric significance test as alternative to HAC-based DM
3) Pooled vs Median R² reporting with asset×horizon matrix
   - Breakdown of negative R² assets/horizons
4) ffill removal comparison
   - Compare performance with vs without forward-fill
5) Negative R² diagnostics
   - Identify and analyze failure cases

Uses cached V71 dataset infrastructure from data_builder.
"""

import numpy as np
import pandas as pd
import json, time, os, pickle, warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

ASSET_GROUPS = {
    'Equity': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
    'Bond': ['TLT', 'IEF', 'AGG'],
    'Commodity': ['GLD', 'SLV', 'USO']
}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
OHLCV_CACHE = 'src/data/v71_ohlcv_cache.pkl'

# =====================================================================
# Utility functions
# =====================================================================
def fit_garch(returns):
    try:
        from arch import arch_model
        ret = returns * 100
        am = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        return res.conditional_volatility / 100
    except:
        return pd.Series(np.zeros(len(returns)), index=returns.index)

def compute_parkinson(h, l, w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)

def compute_gk(o, h, l, c, w=22):
    hl = np.log(h/l); co = np.log(c/o)
    return np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)

def compute_rs(o, h, l, c, w=22):
    rs = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def get_asset_class(asset):
    for cls, assets in ASSET_GROUPS.items():
        if asset in assets:
            return cls
    return 'Unknown'

def train_ridge_perclass(train_df, test_df, feats, purge_gap=22):
    """Ridge per asset class with alpha tuning and purge gap."""
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    sc = StandardScaler()
    sc.fit(train_df[feats])

    val_split = int(len(train_df) * 0.8)
    tr_inner = train_df.iloc[:val_split - purge_gap]
    va_inner = train_df.iloc[val_split:]

    best_alphas = {}
    for cls in ASSET_GROUPS:
        best_r2, best_a = -999, 1.0
        tr_c = tr_inner[tr_inner['Class'] == cls]
        va_c = va_inner[va_inner['Class'] == cls]
        if len(tr_c) < 100 or len(va_c) < 30:
            best_alphas[cls] = 1.0; continue
        for a in alphas:
            m = Ridge(alpha=a).fit(sc.transform(tr_c[feats]), tr_c['Target'])
            r2 = r2_score(va_c['Target'], m.predict(sc.transform(va_c[feats])))
            if r2 > best_r2: best_r2, best_a = r2, a
        best_alphas[cls] = best_a

    preds = np.full(len(test_df), np.nan)
    for cls in ASSET_GROUPS:
        tr_c = train_df[train_df['Class'] == cls]
        te_idx = test_df['Class'] == cls
        if len(tr_c) < 100 or te_idx.sum() == 0: continue
        m = Ridge(alpha=best_alphas[cls]).fit(sc.transform(tr_c[feats]), tr_c['Target'])
        preds[te_idx.values] = m.predict(sc.transform(test_df.loc[te_idx, feats]))

    return preds

def diebold_mariano_test(e1, e2, horizon=22):
    """DM test with Newey-West HAC (bandwidth = horizon - 1)."""
    d = e1**2 - e2**2
    n = len(d)
    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    hac_var = gamma_0
    for k in range(1, horizon):
        weight = 1 - k / horizon
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / (n - 1)
        hac_var += 2 * weight * gamma_k
    hac_var = max(hac_var, 1e-15)
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# =====================================================================
# Build dataset with optional IV replacement & ffill control
# =====================================================================
def build_dataset_robust(use_commodity_iv=False, use_ffill=True):
    """Build V71 dataset with robustness options.

    Parameters
    ----------
    use_commodity_iv : bool
        If True, use GVZ for GLD, OVX for USO instead of VIX for IV features.
    use_ffill : bool
        If False, drop NaN rows instead of forward-filling.
    """
    raw = pd.read_pickle(OHLCV_CACHE)
    if use_ffill:
        raw = raw.ffill()
    # else: keep NaN, rows will be dropped later

    available_tickers = raw.columns.get_level_values(1).unique()

    # Download commodity IV indices if needed
    commodity_iv = {}
    if use_commodity_iv:
        try:
            import yfinance as yf
            for ticker, name in [('^GVZ', 'GVZ'), ('^OVX', 'OVX')]:
                try:
                    iv_data = yf.download(ticker, start='2010-01-01', end='2025-01-01',
                                          progress=False)
                    if len(iv_data) > 100:
                        col = iv_data['Close'].squeeze()  # ensure 1D Series
                        if isinstance(col, pd.DataFrame):
                            col = col.iloc[:, 0]
                        commodity_iv[name] = col
                        print(f"  Loaded {name}: {len(iv_data)} rows", flush=True)
                except Exception as ex:
                    print(f"  Failed to load {name}: {ex}", flush=True)
        except ImportError:
            print("  yfinance not available for commodity IV", flush=True)

    # SPY reference
    spy_close = raw[('Close', 'SPY')]
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    spy_rv = (spy_ret**2).rolling(22).mean() * 252 * 10000
    spy_log_rv = np.log(spy_rv + 1e-6)

    # IV features (global VIX-based)
    vix = raw[('Close', 'VIX')]
    iv_features = {}
    iv_features['VIX'] = np.log(vix + 1e-6)
    iv_features['VIX_chg'] = iv_features['VIX'].diff()
    iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
    iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
    iv_features['VIX3M'] = np.log(raw[('Close', 'VIX3M')] + 1e-6)
    iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
    iv_features['VIX9D'] = np.log(raw[('Close', 'VIX9D')] + 1e-6)
    iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
    vrp = (vix**2 / 100) - spy_rv / 10000
    iv_features['VRP'] = vrp
    iv_features['VRP_ma22'] = vrp.rolling(22).mean()

    # Per-asset features
    pooled_data = []
    for asset in [a for a in ALL_ASSETS if a in available_tickers]:
        c = raw[('Close', asset)]; o = raw[('Open', asset)]
        h = raw[('High', asset)]; l = raw[('Low', asset)]
        v = raw[('Volume', asset)]

        ret = np.log(c / c.shift(1)).dropna()
        rv = (ret**2).rolling(22).mean() * 252 * 10000
        lrv = np.log(rv + 1e-6)
        gd = pd.Series(fit_garch(ret), index=ret.index)
        rw = ret.resample('W').sum()
        gw = pd.Series(fit_garch(rw), index=rw.index).reindex(ret.index, method='ffill')

        feat = {
            'LogRV_lag1': lrv.shift(1), 'LogRV_lag5': lrv.shift(5),
            'LogRV_lag10': lrv.shift(10), 'LogRV_lag22': lrv.shift(22),
            'Garch_Daily': gd.shift(1), 'Garch_Weekly': gw.shift(1),
            'LogRV_Std5': lrv.rolling(5).std().shift(1),
            'LogRV_Std22': lrv.rolling(22).std().shift(1),
            'RV_Mom5': (lrv - lrv.shift(5)).shift(1),
            'RV_Mom22': (lrv - lrv.shift(22)).shift(1),
            'SPY_LogRV': spy_log_rv.shift(1),
            'Ret_lag1': ret.shift(1), 'Ret_abs_lag1': ret.abs().shift(1),
            'Corr_SPY': ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                        if asset != 'SPY' else pd.Series(1.0, index=ret.index),
        }
        # HF proxy
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

        # IV features with commodity IV substitution
        if use_commodity_iv and asset == 'GLD' and 'GVZ' in commodity_iv:
            gvz = commodity_iv['GVZ'].reindex(ret.index)
            if use_ffill:
                gvz = gvz.ffill()
            feat['IV_VIX'] = np.log(gvz + 1e-6).shift(1)
            # Replace VRP with GVZ-based
            gvz_vrp = (gvz**2 / 100) - rv / 10000
            feat['IV_VRP'] = gvz_vrp.shift(1)
            feat['IV_VRP_ma22'] = gvz_vrp.rolling(22).mean().shift(1)
            # Keep other IV features as VIX-based
            for n2, v2 in iv_features.items():
                if n2 not in ['VIX', 'VRP', 'VRP_ma22']:
                    feat[f'IV_{n2}'] = v2.shift(1)
        elif use_commodity_iv and asset == 'USO' and 'OVX' in commodity_iv:
            ovx = commodity_iv['OVX'].reindex(ret.index)
            if use_ffill:
                ovx = ovx.ffill()
            feat['IV_VIX'] = np.log(ovx + 1e-6).shift(1)
            ovx_vrp = (ovx**2 / 100) - rv / 10000
            feat['IV_VRP'] = ovx_vrp.shift(1)
            feat['IV_VRP_ma22'] = ovx_vrp.rolling(22).mean().shift(1)
            for n2, v2 in iv_features.items():
                if n2 not in ['VIX', 'VRP', 'VRP_ma22']:
                    feat[f'IV_{n2}'] = v2.shift(1)
        else:
            for n2, v2 in iv_features.items():
                feat[f'IV_{n2}'] = v2.shift(1)

        # Alt data
        dv = v * c
        feat['AltVol_Amihud'] = (ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1)
        feat['AltVol_Vol_Ratio'] = (v.rolling(5).mean() / (v.rolling(22).mean()+1e-10)).shift(1)
        feat['AltVol_PV_Corr'] = ret.rolling(22).corr(np.log(v+1)).shift(1)
        feat['AltVol_Vol_Surprise'] = ((v - v.rolling(22).mean()) / (v.rolling(22).std()+1e-10)).shift(1)
        pv = v.where(ret > 0, 0).rolling(22).sum()
        nv = v.where(ret <= 0, 0).rolling(22).sum()
        feat['AltVol_Order_Imbalance'] = ((pv - nv) / (pv + nv + 1e-10)).shift(1)
        feat['AltVol_Kyle_Lambda'] = (ret.abs().rolling(22).sum() / (v.rolling(22).sum()+1e-10)*1e6).shift(1)

        feat['Target'] = lrv.shift(-22)
        feat['Asset'] = asset
        feat['Class'] = get_asset_class(asset)

        d = pd.DataFrame(feat)
        if use_ffill:
            d = d.dropna(subset=['Target'])
            nc = [x for x in d.columns if x not in ['Asset', 'Class', 'Target']]
            d[nc] = d[nc].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            # Drop ALL NaN rows (no fill)
            d = d.replace([np.inf, -np.inf], np.nan).dropna()

        pooled_data.append(d)

    data = pd.concat(pooled_data).sort_values(
        'Date' if 'Date' in pooled_data[0].columns else pooled_data[0].index.name or pooled_data[0].index.names[0],
        kind='mergesort'
    ).reset_index(drop=True) if 'Date' in pd.concat(pooled_data).columns else pd.concat(pooled_data).sort_index().reset_index(drop=True)

    feats = [c for c in data.columns if c not in ['Target', 'Asset', 'Class', 'Date']]
    data[feats] = data[feats].fillna(0).replace([np.inf, -np.inf], 0)

    return data, feats


# =====================================================================
# Test 1: GVZ/OVX Commodity IV Robustness
# =====================================================================
def test_commodity_iv():
    print("\n" + "="*70)
    print("TEST 1: GVZ/OVX Commodity IV Robustness")
    print("="*70, flush=True)

    # Baseline (VIX only)
    data_base, feats = build_dataset_robust(use_commodity_iv=False)
    purge = 22
    split = int(len(data_base) * 0.8)
    train_b = data_base.iloc[:split - purge]
    test_b = data_base.iloc[split:]
    preds_base = train_ridge_perclass(train_b, test_b, feats)

    # With commodity IV
    data_cv, feats_cv = build_dataset_robust(use_commodity_iv=True)
    split_cv = int(len(data_cv) * 0.8)
    train_cv = data_cv.iloc[:split_cv - purge]
    test_cv = data_cv.iloc[split_cv:]
    preds_cv = train_ridge_perclass(train_cv, test_cv, feats_cv)

    actual_b = test_b['Target'].values
    actual_cv = test_cv['Target'].values

    print(f"\n{'Config':<25} {'Pooled R²':>10} {'Median R²':>10} {'RMSE':>10}")
    print("-"*60)

    # Per-asset comparison for commodity
    for label, test, preds, actual in [
        ('VIX only (baseline)', test_b, preds_base, actual_b),
        ('GVZ/OVX (commodity)', test_cv, preds_cv, actual_cv),
    ]:
        r2_pooled = r2_score(actual, preds)
        asset_r2s = []
        for a in ALL_ASSETS:
            m = (test['Asset'] == a).values
            if m.sum() > 10:
                asset_r2s.append(r2_score(actual[m], preds[m]))
        med_r2 = float(np.median(asset_r2s))
        rmse = np.sqrt(mean_squared_error(actual, preds))
        print(f"{label:<25} {r2_pooled:>10.4f} {med_r2:>10.4f} {rmse:>10.4f}")

    # Commodity-specific comparison
    print("\n  Commodity assets detail:")
    print(f"  {'Asset':<8} {'VIX R²':>10} {'GVZ/OVX R²':>10} {'Diff':>10}")
    print("  " + "-"*42)
    for a in ['GLD', 'SLV', 'USO']:
        m_b = (test_b['Asset'] == a).values
        m_cv = (test_cv['Asset'] == a).values
        r2_b = r2_score(actual_b[m_b], preds_base[m_b]) if m_b.sum() > 10 else float('nan')
        r2_cv = r2_score(actual_cv[m_cv], preds_cv[m_cv]) if m_cv.sum() > 10 else float('nan')
        diff = r2_cv - r2_b
        print(f"  {a:<8} {r2_b:>10.4f} {r2_cv:>10.4f} {diff:>+10.4f}")

    return {'baseline_r2': r2_score(actual_b, preds_base),
            'commodity_iv_r2': r2_score(actual_cv, preds_cv)}


# =====================================================================
# Test 2: Block Bootstrap DM Comparison
# =====================================================================
def test_block_bootstrap_dm():
    print("\n" + "="*70)
    print("TEST 2: Block Bootstrap DM Comparison")
    print("="*70, flush=True)

    data, feats = build_dataset_robust()
    purge = 22
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split - purge]
    test_df = data.iloc[split:]

    # HAR baseline
    har3 = ['LogRV_lag1', 'LogRV_lag5', 'LogRV_lag22']
    preds_har = train_ridge_perclass(train_df, test_df, har3)
    # V71 full
    preds_v71 = train_ridge_perclass(train_df, test_df, feats)

    actual = test_df['Target'].values
    e_v71 = actual - preds_v71
    e_har = actual - preds_har

    # Standard DM test
    dm_stat, dm_p = diebold_mariano_test(e_v71, e_har, horizon=22)
    print(f"\n  Standard DM test: stat={dm_stat:.4f}, p={dm_p:.6f}")

    # Block Bootstrap DM
    block_size = 22  # match horizon
    n = len(e_v71)
    n_bootstrap = 5000
    d_original = np.mean(e_v71**2) - np.mean(e_har**2)

    np.random.seed(42)
    bootstrap_diffs = []
    n_blocks = n // block_size + 1

    for _ in range(n_bootstrap):
        # Circular block bootstrap
        starts = np.random.randint(0, n, size=n_blocks)
        indices = []
        for s in starts:
            indices.extend(range(s, min(s + block_size, n)))
        indices = np.array(indices[:n])

        d_boot = np.mean(e_v71[indices]**2) - np.mean(e_har[indices]**2)
        bootstrap_diffs.append(d_boot)

    bootstrap_diffs = np.array(bootstrap_diffs)
    # Two-sided p-value
    centered = bootstrap_diffs - np.mean(bootstrap_diffs)
    boot_p = np.mean(np.abs(centered) >= np.abs(d_original))

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    print(f"  Block Bootstrap (B={n_bootstrap}, block={block_size}):")
    print(f"    MSE diff (V71-HAR): {d_original:.6f}")
    print(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"    p-value: {boot_p:.6f}")
    print(f"    {'***' if boot_p < 0.01 else '**' if boot_p < 0.05 else '*' if boot_p < 0.1 else 'n.s.'}")

    # HAC DM vs Bootstrap comparison
    print(f"\n  {'Method':<30} {'p-value':>10} {'Sig':>6}")
    print("  " + "-"*50)
    print(f"  {'HAC DM (Newey-West)':<30} {dm_p:>10.6f} {'***' if dm_p < 0.01 else '**' if dm_p < 0.05 else 'n.s.':>6}")
    print(f"  {'Block Bootstrap (B=5000)':<30} {boot_p:>10.6f} {'***' if boot_p < 0.01 else '**' if boot_p < 0.05 else 'n.s.':>6}")

    return {'dm_p': dm_p, 'bootstrap_p': boot_p, 'mse_diff': d_original,
            'ci_lower': ci_lower, 'ci_upper': ci_upper}


# =====================================================================
# Test 3: Pooled vs Median R² + Asset×Class Matrix + Negative R² diagnostics
# =====================================================================
def test_reporting_metrics():
    print("\n" + "="*70)
    print("TEST 3: Pooled vs Median R² Reporting + Negative R² Diagnostics")
    print("="*70, flush=True)

    data, feats = build_dataset_robust()
    purge = 22
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split - purge]
    test_df = data.iloc[split:]

    preds = train_ridge_perclass(train_df, test_df, feats)
    actual = test_df['Target'].values

    # Overall
    r2_pooled = r2_score(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)

    # Per-asset
    asset_results = {}
    for a in ALL_ASSETS:
        m = (test_df['Asset'] == a).values
        if m.sum() > 10:
            r2_a = r2_score(actual[m], preds[m])
            rmse_a = np.sqrt(mean_squared_error(actual[m], preds[m]))
            asset_results[a] = {'R2': r2_a, 'RMSE': rmse_a, 'n': int(m.sum()),
                                'Class': get_asset_class(a)}

    # Per-class
    class_results = {}
    for cls in ASSET_GROUPS:
        m = (test_df['Class'] == cls).values
        if m.sum() > 10:
            class_results[cls] = {
                'R2': r2_score(actual[m], preds[m]),
                'RMSE': np.sqrt(mean_squared_error(actual[m], preds[m])),
                'n': int(m.sum())
            }

    # Summary statistics
    r2_values = [v['R2'] for v in asset_results.values()]
    r2_median = np.median(r2_values)
    r2_mean = np.mean(r2_values)
    r2_q25 = np.percentile(r2_values, 25)
    r2_q75 = np.percentile(r2_values, 75)
    num_negative = sum(1 for r in r2_values if r < 0)

    print(f"\n  === Overall Metrics ===")
    print(f"  Pooled R²:   {r2_pooled:.4f} (cross-sectional variation included)")
    print(f"  Median R²:   {r2_median:.4f} (per-asset, recommended primary)")
    print(f"  Mean R²:     {r2_mean:.4f}")
    print(f"  IQR:         [{r2_q25:.4f}, {r2_q75:.4f}]")
    print(f"  Negative R²: {num_negative}/{len(r2_values)} assets")

    print(f"\n  === Per-Asset R² (sorted) ===")
    print(f"  {'Asset':<8} {'Class':<12} {'R²':>8} {'RMSE':>8} {'n':>6}")
    print("  " + "-"*46)
    for a, v in sorted(asset_results.items(), key=lambda x: x[1]['R2']):
        marker = " <-- NEGATIVE" if v['R2'] < 0 else ""
        print(f"  {a:<8} {v['Class']:<12} {v['R2']:>8.4f} {v['RMSE']:>8.4f} {v['n']:>6}{marker}")

    print(f"\n  === Per-Class R² ===")
    print(f"  {'Class':<12} {'R²':>8} {'RMSE':>8} {'n':>6}")
    print("  " + "-"*36)
    for cls, v in class_results.items():
        print(f"  {cls:<12} {v['R2']:>8.4f} {v['RMSE']:>8.4f} {v['n']:>6}")

    # Negative R² diagnostics
    if num_negative > 0:
        print(f"\n  === Negative R² Diagnostics ===")
        for a, v in asset_results.items():
            if v['R2'] < 0:
                m = (test_df['Asset'] == a).values
                y = actual[m]; p = preds[m]
                naive_mse = np.mean((y - np.mean(y))**2)
                model_mse = np.mean((y - p)**2)
                print(f"  {a}: Naive MSE={naive_mse:.4f}, Model MSE={model_mse:.4f}")
                print(f"    -> Model is {model_mse/naive_mse:.2f}x worse than mean prediction")
                print(f"    -> Target std={np.std(y):.4f}, pred std={np.std(p):.4f}")

    return {
        'pooled_r2': r2_pooled, 'median_r2': r2_median, 'mean_r2': r2_mean,
        'iqr': [r2_q25, r2_q75], 'num_negative': num_negative,
        'asset_results': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                              for kk, vv in v.items()} for k, v in asset_results.items()},
        'class_results': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                              for kk, vv in v.items()} for k, v in class_results.items()},
    }


# =====================================================================
# Test 4: ffill Removal Comparison
# =====================================================================
def test_ffill_comparison():
    print("\n" + "="*70)
    print("TEST 4: Forward-Fill vs No-Fill Comparison")
    print("="*70, flush=True)

    results = {}
    for label, use_ff in [('With ffill', True), ('Without ffill', False)]:
        print(f"\n  Building dataset ({label})...", flush=True)
        data, feats = build_dataset_robust(use_ffill=use_ff)
        print(f"  Samples: {len(data)}, Features: {len(feats)}", flush=True)

        purge = 22
        split = int(len(data) * 0.8)
        train_df = data.iloc[:split - purge]
        test_df = data.iloc[split:]

        preds = train_ridge_perclass(train_df, test_df, feats)
        actual = test_df['Target'].values

        r2_pooled = r2_score(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))

        # Per-asset
        asset_r2 = []
        for a in ALL_ASSETS:
            m = (test_df['Asset'] == a).values
            if m.sum() > 10:
                asset_r2.append(r2_score(actual[m], preds[m]))
        med_r2 = float(np.median(asset_r2))

        results[label] = {
            'n_samples': len(data), 'pooled_r2': r2_pooled,
            'median_r2': med_r2, 'rmse': rmse
        }
        print(f"  {label}: Pooled R²={r2_pooled:.4f}, Median R²={med_r2:.4f}, RMSE={rmse:.4f}")

    print(f"\n  === Comparison ===")
    print(f"  {'Config':<20} {'Samples':>8} {'Pooled R²':>10} {'Med R²':>10} {'RMSE':>8}")
    print("  " + "-"*60)
    for label, v in results.items():
        print(f"  {label:<20} {v['n_samples']:>8} {v['pooled_r2']:>10.4f} {v['median_r2']:>10.4f} {v['rmse']:>8.4f}")

    diff_r2 = results['With ffill']['pooled_r2'] - results['Without ffill']['pooled_r2']
    print(f"\n  R² difference (ffill - no_fill): {diff_r2:+.4f}")
    if abs(diff_r2) < 0.005:
        print("  -> Negligible difference, results are robust to ffill choice")
    else:
        print(f"  -> Non-trivial difference ({abs(diff_r2):.4f}), warrants attention")

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    print("="*70)
    print("V71 Academic Robustness Verification Suite")
    print("="*70)
    t0 = time.time()

    all_results = {}

    # Test 1: Commodity IV
    try:
        all_results['commodity_iv'] = test_commodity_iv()
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        all_results['commodity_iv'] = {'error': str(e)}

    # Test 2: Block Bootstrap DM
    try:
        all_results['block_bootstrap'] = test_block_bootstrap_dm()
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        all_results['block_bootstrap'] = {'error': str(e)}

    # Test 3: Reporting Metrics
    try:
        all_results['reporting'] = test_reporting_metrics()
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        all_results['reporting'] = {'error': str(e)}

    # Test 4: ffill comparison
    try:
        all_results['ffill'] = test_ffill_comparison()
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        all_results['ffill'] = {'error': str(e)}

    # Save results
    out_path = 'src/experiments/creative/v71_robustness_academic_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    return all_results


if __name__ == "__main__":
    main()
