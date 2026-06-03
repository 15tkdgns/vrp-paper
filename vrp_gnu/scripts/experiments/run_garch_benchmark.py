"""GARCH Benchmark (Exact Date Alignment)"""
import sys; sys.path.insert(0, '/root/vrp')
import numpy as np, pandas as pd, json, warnings, time
warnings.filterwarnings('ignore')
from arch import arch_model
from sklearn.metrics import r2_score, mean_squared_error
from src.experiments.creative.v71_model_comparison import build_dataset, ALL_ASSETS

print("Loading...", flush=True)

# MOCK reset_index to prevent dropping Date
orig_reset = pd.DataFrame.reset_index
def fake_reset(*args, **kwargs):
    if 'drop' in kwargs: kwargs['drop']=False
    return orig_reset(*args, **kwargs)
pd.DataFrame.reset_index = fake_reset
data, feats = build_dataset()
pd.DataFrame.reset_index = orig_reset

split_idx = int(len(data) * 0.8)
test_df = data.iloc[split_idx:].copy()
if 'index' in test_df.columns:
    test_df['Date'] = test_df['index']
y_te = test_df['Target'].values
raw = pd.read_pickle('src/data/v71_ohlcv_cache.pkl').ffill()

def metrics(y_true, preds, tdf):
    v = ~np.isnan(preds)
    if v.sum() < 10: return {}
    ar = {}
    for a in ALL_ASSETS:
        m = (tdf['Asset']==a).values & v
        if m.sum()>10: ar[a] = float(r2_score(y_true[m], preds[m]))
    rv = list(ar.values())
    return {'pooled_r2': float(r2_score(y_true[v], preds[v])),
            'rmse': float(np.sqrt(mean_squared_error(y_true[v], preds[v]))),
            'median_r2': float(np.median(rv)) if rv else float('nan'),
            'asset_r2': ar}

results = {}
for gname, kwargs in [('GARCH(1,1)', {'p':1,'q':1}), ('GJR-GARCH', {'p':1,'o':1,'q':1})]:
    print(f"\n=== {gname} ===", flush=True)
    preds = np.full(len(test_df), np.nan)
    for asset in ALL_ASSETS:
        te_mask = (test_df['Asset']==asset).values
        if te_mask.sum()<10: continue
        
        c = raw[('Close',asset)].dropna()
        ret = np.log(c/c.shift(1)).dropna()
        dates = ret.index
        
        n_tr = int(len(ret)*0.8)
        train_r = ret.iloc[:n_tr] * 100
        
        try:
            am = arch_model(train_r, vol='Garch', mean='Zero', dist='normal', rescale=False, **kwargs)
            res = am.fit(disp='off', show_warning=False)
            p = res.params
            om = p.get('omega',0.01); a1 = p.get('alpha[1]',0.05)
            b1 = p.get('beta[1]',0.90); g1 = p.get('gamma[1]',0.0) if 'gamma[1]' in p else 0.0
            pers = a1 + g1/2 + b1
            lr = om/(1-pers) if pers<1 else np.var(train_r)
            
            full_r = ret*100
            cv = np.zeros(len(ret)); cv[0] = np.var(train_r)
            for i in range(1,len(ret)):
                s = full_r.iloc[i-1]
                cv[i] = om + a1*s**2 + g1*(s<0)*s**2 + b1*cv[i-1]
            
            fcs = pd.Series(index=dates, dtype=float)
            for i in range(len(ret)):
                h1 = cv[i] # conditional var for day i (using i-1 shock)
                tv = sum(lr + pers**k*(h1-lr) for k in range(1,23))
                ann_rv = (tv/22)*252
                fcs.iloc[i] = np.log(max(ann_rv, 1e-10))
                
            # Explicit string matching via Date column
            str_dates_fcs = pd.to_datetime(fcs.index).strftime('%Y-%m-%d').values
            str_dates_test = pd.to_datetime(test_df['Date'][te_mask]).dt.strftime('%Y-%m-%d').values
            fcs_dict = dict(zip(str_dates_fcs, fcs.values))
            
            asset_test_idx = np.where(te_mask)[0]
            for j, d_str in enumerate(str_dates_test):
                if d_str in fcs_dict:
                    preds[asset_test_idx[j]] = fcs_dict[d_str]
                    
            vm = te_mask & ~np.isnan(preds)
            if vm.sum()>10:
                print(f"  {asset}: R2={r2_score(y_te[vm],preds[vm]):.4f}", flush=True)
                
        except Exception as e:
            print(f"  {asset}: {e}", flush=True)
            
    results[gname] = metrics(y_te, preds, test_df)
    r = results[gname]
    print(f"  >> Pooled={r.get('pooled_r2',0):.4f}, Median={r.get('median_r2',0):.4f}")

print("\n"+"="*60)
for n,r in results.items():
    print(f"{n:>15}: P={r.get('pooled_r2',0):.4f} M={r.get('median_r2',0):.4f} RMSE={r.get('rmse',0):.4f}")

print(f"{'HAR-3':>15}: P=0.7610 M=0.1709 RMSE=0.5415")
print(f"{'WEns':>15}: P=0.8030 M=0.2150 RMSE=0.4680")

with open('/root/vrp/paper/csv/garch_benchmark_results.json','w') as f:
    json.dump(results, f, indent=2, default=float)
print("Saved.")
