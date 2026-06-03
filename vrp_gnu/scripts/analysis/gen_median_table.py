import json, os
import numpy as np

path = '/root/vrp/src/experiments/universal/v5_multi_horizon.json'
out = []
if os.path.exists(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if 'model_comparison' in data:
        comps = data['model_comparison']
        for h in ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']:
            if h not in comps: continue
            v = comps[h]
            row = {'horizon': h}
            for model, mdata in v.items():
                if isinstance(mdata, dict) and 'median_r2' in mdata:
                    row[model] = mdata['median_r2']
            out.append(row)
    print("Multi horizon found.")
else:
    print("No multi horizon json.")

if out:
    print("| 지평 | Ridge per-class(37) | BiLSTM-A(3) | HAR-3(3) | LASSO(37) | RF(37) |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|")
    for r in out:
        h = r['horizon']
        ridge = r.get('Ridge_perclass', r.get('Ridge(37)', r.get('Ridge', float('nan'))))
        lstm = r.get('BiLSTM-A(3)', r.get('BiLSTM-A', float('nan')))
        har = r.get('HAR-3', r.get('HAR_only', float('nan')))
        lasso = r.get('Lasso', r.get('LASSO(37)', float('nan')))
        rf = r.get('RandomForest', r.get('RF', r.get('RF(37)', float('nan'))))
        
        def fmt(val):
            return f"{val:.3f}" if not np.isnan(val) else "--"
            
        print(f"| {h} | {fmt(ridge)} | {fmt(lstm)} | {fmt(har)} | {fmt(lasso)} | {fmt(rf)} |")
