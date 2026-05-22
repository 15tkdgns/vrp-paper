"""Multi-Horizon with Literature Benchmarks (Non-Overlapping Forward RV)
Adds 3 literature benchmarks to existing 4 models:
  - LASSO-HAR (Audrino & Knaus, 2016, J. Financial Econometrics)
  - HAR-CJ (Andersen, Bollerslev, Diebold, 2007, Econometrica)
  - Random Forest (Christensen, Siggaard, Veliyev, 2023, J. Financial Economics)
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, json, time, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model
warnings.filterwarnings('ignore')

ASSET_GROUPS = {'Equity':['SPY','QQQ','IWM','EFA','EEM'],
                'Bond':['TLT','IEF','AGG'],'Commodity':['GLD','SLV','USO']}
ALL_ASSETS = [a for g in ASSET_GROUPS.values() for a in g]
HORIZONS = [1, 5, 22, 60, 90, 120, 180, 365]

class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h*2,1)
    def forward(self, x):
        w = torch.softmax(torch.tanh(self.fc(x)).squeeze(-1), dim=1)
        return torch.sum(x*w.unsqueeze(-1), dim=1)

class LSTMModel(nn.Module):
    def __init__(self, idim, hdim=32, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(idim, hdim, batch_first=True, bidirectional=True)
        self.attn = Attention(hdim)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Sequential(nn.Linear(hdim*2,hdim),nn.ReLU(),nn.Dropout(drop),nn.Linear(hdim,1))
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(self.drop(self.attn(o)))

def fit_garch(r):
    try:
        am = arch_model(r*100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        return pd.Series(res.conditional_volatility.values.flatten()/100, index=r.index)
    except: return r.rolling(22).std()

def compute_parkinson(h,l,w=22):
    return np.sqrt((np.log(h/l)**2).rolling(w).mean()/(4*np.log(2)))*np.sqrt(252)
def compute_gk(o,h,l,c,w=22):
    hl=np.log(h/l);co=np.log(c/o)
    return np.sqrt((0.5*hl**2-(2*np.log(2)-1)*co**2).rolling(w).mean().clip(0)*252)
def compute_rs(o,h,l,c,w=22):
    rs=np.log(h/c)*np.log(h/o)+np.log(l/c)*np.log(l/o)
    return np.sqrt(rs.rolling(w).mean().clip(0)*252)

def qlike(a,p):
    r=np.exp(a-p); return np.mean(r-(a-p)-1)

def eval_metrics(y,p,assets):
    r2=r2_score(y,p); rmse=np.sqrt(mean_squared_error(y,p))
    mae_v=mean_absolute_error(y,p); ql=qlike(y,p)
    ta=np.array(assets); ar2s=[]
    for a in ALL_ASSETS:
        m=ta==a
        if m.sum()>10: ar2s.append(r2_score(y[m],p[m]))
    med = float(np.median(ar2s)) if ar2s else 0.0
    return {'R2':round(r2,4),'RMSE':round(rmse,4),'MAE':round(mae_v,4),
            'QLIKE':round(ql,4),'Med_R2':round(med,4),'n_assets':len(ar2s)}

def train_lstm(X_tr,y_tr,X_te,y_te,ta,hdim=32):
    torch.manual_seed(42); np.random.seed(42)
    m = LSTMModel(X_tr.shape[2],hdim,0.2)
    opt = optim.Adam(m.parameters(),lr=0.001,weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
    dl = torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True)
    m.train()
    for e in range(10):
        for bx,by in dl:
            opt.zero_grad()
            loss=nn.MSELoss()(m(bx).squeeze(),by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step()
    m.eval()
    with torch.no_grad():
        p = m(torch.from_numpy(X_te)).squeeze().numpy()
    return eval_metrics(y_te,p,ta)

def forward_rv(ret_sq, horizon):
    """Non-overlapping forward RV: target[t] = log(mean(ret²[t+1:t+h]) * 252)"""
    cs = ret_sq.cumsum()
    fwd_mean = (cs.shift(-horizon) - cs) / horizon
    return np.log(fwd_mean * 252 + 1e-12)

# ============ Load Data ============
print("="*70)
print("Multi-Horizon + Literature Benchmarks (Non-Overlapping Forward RV)")
print("="*70)
print("\nModels: LSTM_Top3, LSTM_Orig, Ridge(37), HAR-3, LASSO-HAR, HAR-CJ, RF")
print("References:")
print("  HAR-RV:  Corsi (2009), J. Financial Econometrics")
print("  HAR-CJ:  Andersen, Bollerslev, Diebold (2007), Econometrica")
print("  LASSO:   Audrino & Knaus (2016), J. Financial Econometrics")
print("  RF:      Christensen, Siggaard, Veliyev (2023), J. Financial Economics")
print()
t0 = time.time()
raw = pd.read_pickle('src/data/v71_ohlcv_cache.pkl')

vix = raw[('Close','VIX')]
spy_c = raw[('Close','SPY')]
spy_ret = np.log(spy_c/spy_c.shift(1)).dropna()
spy_rv = (spy_ret**2).rolling(22).mean()*252*10000
spy_lrv = np.log(spy_rv+1e-6)

iv_features = {}
iv_features['VIX'] = np.log(vix+1e-6)
iv_features['VIX_chg'] = iv_features['VIX'].diff()
iv_features['VIX_ma5'] = iv_features['VIX'].rolling(5).mean()
iv_features['VIX_std5'] = iv_features['VIX'].rolling(5).std()
iv_features['VIX3M'] = np.log(raw[('Close','VIX3M')]+1e-6)
iv_features['VIX_TermSlope'] = iv_features['VIX'] - iv_features['VIX3M']
iv_features['VIX9D'] = np.log(raw[('Close','VIX9D')]+1e-6)
iv_features['VIX_ShortSlope'] = iv_features['VIX9D'] - iv_features['VIX']
vrp = (vix**2/100) - spy_rv/10000
iv_features['VRP'] = vrp; iv_features['VRP_ma22'] = vrp.rolling(22).mean()

print("Building features...", flush=True)
base_frames = {}
for asset in ALL_ASSETS:
    c=raw[('Close',asset)];o=raw[('Open',asset)];h=raw[('High',asset)]
    l=raw[('Low',asset)];v=raw[('Volume',asset)]
    ret=np.log(c/c.shift(1)).dropna()
    ret_sq = ret**2
    rv=(ret_sq).rolling(22).mean()*252*10000
    lrv=np.log(rv+1e-6)
    gd=fit_garch(ret)
    rw=ret.resample('W').sum()
    gw=fit_garch(rw).reindex(ret.index,method='ffill')
    
    feat = {
        'LogRV_lag1':lrv.shift(1),'LogRV_lag5':lrv.shift(5),
        'LogRV_lag10':lrv.shift(10),'LogRV_lag22':lrv.shift(22),
        'Garch_Daily':gd.shift(1),'Garch_Weekly':gw.shift(1),
        'LogRV_Std5':lrv.rolling(5).std().shift(1),
        'LogRV_Std22':lrv.rolling(22).std().shift(1),
        'RV_Mom5':(lrv-lrv.shift(5)).shift(1),
        'RV_Mom22':(lrv-lrv.shift(22)).shift(1),
        'SPY_LogRV':spy_lrv.shift(1),
        'Ret_lag1':ret.shift(1),'Ret_abs_lag1':ret.abs().shift(1),
        'Corr_SPY': ret.rolling(22).corr(spy_ret.reindex(ret.index)).shift(1)
                    if asset!='SPY' else pd.Series(1.0,index=ret.index),
    }
    p5=compute_parkinson(h,l,5);p22=compute_parkinson(h,l,22)
    gk22=compute_gk(o,h,l,c,22);rs22=compute_rs(o,h,l,c,22)
    feat['Parkinson_5']=np.log(p5+1e-6).shift(1)
    feat['Parkinson_22']=np.log(p22+1e-6).shift(1)
    feat['GarmanKlass_22']=np.log(gk22+1e-6).shift(1)
    feat['RogersSatchell_22']=np.log(rs22+1e-6).shift(1)
    feat['Range_Close_Ratio']=(np.log(p22+1e-6)-lrv).shift(1)
    on=np.log(o/c.shift(1))
    feat['Overnight_Vol']=on.rolling(22).std().shift(1)
    feat['Overnight_Ret']=on.shift(1)
    for n2,v2 in iv_features.items(): feat[f'IV_{n2}']=v2.shift(1)
    dv=v*c
    feat['AltVol_Amihud']=(ret.abs()/(dv+1e-10)).rolling(22).mean().shift(1)
    feat['AltVol_Vol_Ratio']=(v.rolling(5).mean()/(v.rolling(22).mean()+1e-10)).shift(1)
    feat['AltVol_PV_Corr']=ret.rolling(22).corr(np.log(v+1)).shift(1)
    feat['AltVol_Vol_Surprise']=((v-v.rolling(22).mean())/(v.rolling(22).std()+1e-10)).shift(1)
    pv=v.where(ret>0,0).rolling(22).sum();nv=v.where(ret<=0,0).rolling(22).sum()
    feat['AltVol_Order_Imbalance']=((pv-nv)/(pv+nv+1e-10)).shift(1)
    feat['AltVol_Kyle_Lambda']=(ret.abs().rolling(22).sum()/(v.rolling(22).sum()+1e-10)*1e6).shift(1)
    
    # HAR-CJ features (Andersen, Bollerslev, Diebold 2007)
    # Jump proxy: J = max(RV - BV, 0), where BV = bipower variation
    # BV approximation using |ret_t| * |ret_{t-1}| * (pi/2)
    abs_ret = ret.abs()
    bv = (abs_ret * abs_ret.shift(1)).rolling(22).mean() * (np.pi/2) * 252 * 10000
    log_bv = np.log(bv.clip(lower=1e-6))
    jump = (rv - bv).clip(lower=0)
    log_jump = np.log(jump + 1e-6)
    feat['HAR_CJ_C_lag1'] = log_bv.shift(1)   # Continuous component
    feat['HAR_CJ_J_lag1'] = log_jump.shift(1)  # Jump component
    feat['HAR_CJ_C_lag5'] = log_bv.rolling(5).mean().shift(1)
    feat['HAR_CJ_J_lag5'] = log_jump.rolling(5).mean().shift(1)
    feat['HAR_CJ_C_lag22'] = log_bv.rolling(22).mean().shift(1)
    feat['HAR_CJ_J_lag22'] = log_jump.rolling(22).mean().shift(1)
    
    feat['ret_sq'] = ret_sq
    feat['Asset'] = asset
    
    d = pd.DataFrame(feat)
    nc=[x for x in d.columns if x not in ['Asset','ret_sq']]
    d[nc]=d[nc].replace([np.inf,-np.inf],np.nan)
    base_frames[asset] = d

print(f"Features built for {len(base_frames)} assets", flush=True)

top3 = ['RogersSatchell_22','Range_Close_Ratio','GarmanKlass_22']
original2 = ['LogRV_lag1','Ret_lag1']
har3 = ['LogRV_lag1','LogRV_lag5','LogRV_lag22']
# HAR-CJ: Andersen, Bollerslev, Diebold (2007) - continuous + jump decomposition
har_cj = ['HAR_CJ_C_lag1','HAR_CJ_C_lag5','HAR_CJ_C_lag22',
           'HAR_CJ_J_lag1','HAR_CJ_J_lag5','HAR_CJ_J_lag22']

# ============ Run All Horizons ============
results_all = {}

MODEL_NAMES = ['LSTM_Top3','LSTM_Orig','Ridge_37','HAR3','LASSO_HAR','HAR_CJ','RF_37']

for horizon in HORIZONS:
    print(f"\n{'='*70}")
    print(f"Horizon: {horizon}-day FORWARD RV (non-overlapping)")
    print(f"{'='*70}", flush=True)
    
    pooled = []
    for asset in ALL_ASSETS:
        d = base_frames[asset].copy()
        d['Target'] = forward_rv(d['ret_sq'], horizon)
        d = d.drop(columns=['ret_sq']).dropna()
        nc=[x for x in d.columns if x not in ['Asset','Target']]
        d[nc]=d[nc].fillna(0)
        pooled.append(d)
    
    data = pd.concat(pooled).sort_index().reset_index(drop=True)
    all_feats = [c for c in data.columns if c not in ['Target','Asset']]
    data[all_feats] = data[all_feats].fillna(0).replace([np.inf,-np.inf],0)
    
    split = int(len(data)*0.8)
    train_df = data.iloc[:split]; test_df = data.iloc[split:]
    
    print(f"Data: {len(data)} (Train:{len(train_df)}, Test:{len(test_df)})", flush=True)
    
    horizon_results = {}
    
    def make_seqs(df, fc, sl=22):
        sc=StandardScaler(); sc.fit(train_df[fc])
        Xs,ys,assets=[],[],[]
        for a in ALL_ASSETS:
            adf=df[df['Asset']==a].sort_index()
            if len(adf)<sl+1: continue
            Xsc=sc.transform(adf[fc]); yv=adf['Target'].values
            for i in range(sl,len(Xsc)):
                Xs.append(Xsc[i-sl:i]); ys.append(yv[i]); assets.append(a)
        return np.array(Xs,dtype=np.float32), np.array(ys,dtype=np.float32), assets
    
    # 1) LSTM Top-3 (V50 Tuned)
    X_tr,y_tr,_ = make_seqs(train_df, top3)
    X_te,y_te,ta = make_seqs(test_df, top3)
    if len(X_tr)>0 and len(X_te)>0:
        r = train_lstm(X_tr,y_tr,X_te,y_te,ta)
        horizon_results['LSTM_Top3'] = r
        print(f"  LSTM Top-3:    R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}", flush=True)
    
    # 2) LSTM Original (V50)
    X_tr,y_tr,_ = make_seqs(train_df, original2)
    X_te,y_te,ta = make_seqs(test_df, original2)
    if len(X_tr)>0 and len(X_te)>0:
        r = train_lstm(X_tr,y_tr,X_te,y_te,ta)
        horizon_results['LSTM_Orig'] = r
        print(f"  LSTM Orig(2):  R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}", flush=True)
    
    # 3) Ridge 37 (V71)
    sc37=StandardScaler(); sc37.fit(train_df[all_feats])
    X_tr_r=sc37.transform(train_df[all_feats]); y_tr_r=train_df['Target'].values
    X_te_r=sc37.transform(test_df[all_feats]); y_te_r=test_df['Target'].values
    ridge=Ridge(alpha=100); ridge.fit(X_tr_r, y_tr_r)
    p_r=ridge.predict(X_te_r)
    r = eval_metrics(y_te_r, p_r, test_df['Asset'].values)
    horizon_results['Ridge_37'] = r
    print(f"  Ridge(37):     R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}", flush=True)
    
    # 4) HAR-3: Corsi (2009)
    sc3=StandardScaler(); sc3.fit(train_df[har3])
    X_tr_h=sc3.transform(train_df[har3]); X_te_h=sc3.transform(test_df[har3])
    har=Ridge(alpha=10); har.fit(X_tr_h, y_tr_r)
    p_h=har.predict(X_te_h)
    r = eval_metrics(y_te_r, p_h, test_df['Asset'].values)
    horizon_results['HAR3'] = r
    print(f"  HAR-3:         R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}  [Corsi 2009]", flush=True)
    
    # 5) LASSO-HAR: Audrino & Knaus (2016)
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X_tr_r, y_tr_r)  # Same 37 features as Ridge
    p_l = lasso.predict(X_te_r)
    r = eval_metrics(y_te_r, p_l, test_df['Asset'].values)
    horizon_results['LASSO_HAR'] = r
    print(f"  LASSO(37):     R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}  [Audrino & Knaus 2016]", flush=True)
    
    # 6) HAR-CJ: Andersen, Bollerslev, Diebold (2007)
    sc_cj = StandardScaler(); sc_cj.fit(train_df[har_cj])
    X_tr_cj = sc_cj.transform(train_df[har_cj])
    X_te_cj = sc_cj.transform(test_df[har_cj])
    har_cj_model = Ridge(alpha=10)
    har_cj_model.fit(X_tr_cj, y_tr_r)
    p_cj = har_cj_model.predict(X_te_cj)
    r = eval_metrics(y_te_r, p_cj, test_df['Asset'].values)
    horizon_results['HAR_CJ'] = r
    print(f"  HAR-CJ:        R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}  [Andersen+ 2007]", flush=True)
    
    # 7) Random Forest: Christensen, Siggaard, Veliyev (2023)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr_r, y_tr_r)  # Same 37 features
    p_rf = rf.predict(X_te_r)
    r = eval_metrics(y_te_r, p_rf, test_df['Asset'].values)
    horizon_results['RF_37'] = r
    print(f"  RF(37):        R²={r['R2']:.4f}  Med={r['Med_R2']:.4f}  RMSE={r['RMSE']:.4f}  [Christensen+ 2023]", flush=True)
    
    results_all[f'{horizon}d'] = horizon_results

# ============ Final Tables ============
print(f"\n{'='*70}")
print("FINAL: Pooled R² (Non-Overlapping Forward RV)")
print(f"{'='*70}")
header = f"{'Horizon':<8}"
for m in MODEL_NAMES: header += f" {m:>11}"
print(f"\n{header}")
print("-"*93)
for h in HORIZONS:
    key=f'{h}d'
    row = f"{key:<8}"
    for m in MODEL_NAMES:
        v=results_all[key].get(m,{}).get('R2','-')
        row += f" {v:>11.4f}" if isinstance(v,float) else f" {v:>11}"
    print(row)

print(f"\n{'='*70}")
print("FINAL: Median R² (Non-Overlapping Forward RV)")
print(f"{'='*70}")
print(f"\n{header}")
print("-"*93)
for h in HORIZONS:
    key=f'{h}d'
    row = f"{key:<8}"
    for m in MODEL_NAMES:
        v=results_all[key].get(m,{}).get('Med_R2','-')
        row += f" {v:>11.4f}" if isinstance(v,float) else f" {v:>11}"
    print(row)

# Model references
print(f"\n{'='*70}")
print("Model References:")
print(f"{'='*70}")
print("  LSTM_Top3:  V50 Tuned (Bi-LSTM+Attention, 3 features)")
print("  LSTM_Orig:  V50 Original (Bi-LSTM+Attention, 2 features)")
print("  Ridge_37:   V71 (Ridge alpha=100, 37 features)")
print("  HAR3:       Corsi (2009), J. Financial Econometrics, 11(2)")
print("  LASSO_HAR:  Audrino & Knaus (2016), J. Financial Econometrics, 14(2)")
print("  HAR_CJ:     Andersen, Bollerslev, Diebold (2007), Econometrica, 75(4)")
print("  RF_37:      Christensen, Siggaard, Veliyev (2023), J. Financial Economics, 150(2)")

print(f"\nTotal time: {time.time()-t0:.1f}s")

with open('src/experiments/creative/multi_horizon_benchmark_results.json','w') as f:
    json.dump({
        'results': results_all,
        'references': {
            'LSTM_Top3': 'V50 Tuned (Bi-LSTM+Attention, 3 features: RS, GK, RC)',
            'LSTM_Orig': 'V50 Original (Bi-LSTM+Attention, 2 features: LogRV, Ret)',
            'Ridge_37': 'V71 (Ridge alpha=100, 37 features)',
            'HAR3': 'Corsi (2009), Journal of Financial Econometrics, 11(2), 174-196',
            'LASSO_HAR': 'Audrino & Knaus (2016), Journal of Financial Econometrics, 14(2), 392-427',
            'HAR_CJ': 'Andersen, Bollerslev, Diebold (2007), Econometrica, 75(4), 979-1002',
            'RF_37': 'Christensen, Siggaard, Veliyev (2023), Journal of Financial Economics, 150(2), 103737'
        }
    }, f, indent=2, default=lambda o:float(o) if hasattr(o,'item') else o)
print("Saved: multi_horizon_benchmark_results.json")
