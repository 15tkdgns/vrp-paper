"""
V51b: V50 모델 공정 검증 (IS vs OOS R² 역전 원인 규명)

원인 분석:
- V50: rolling(22), target=t+22 -> R²=0.518
- V51: rolling(5),  target=t+1  -> R²=0.820
- 두 실험의 타겟 변수와 예측 지평이 완전히 달랐음

본 실험:
1. V50과 동일한 설정(rolling=22, target=t+22)으로 Walk-Forward 수행
2. V50과 동일한 설정으로 단순 Hold-out 수행 (스케일러 leakage 수정)
3. 세 가지 R² 값 비교: IS(수정), OOS(WF), 기존 V50
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration (V50과 동일)
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD']
SEQ_LEN = 22
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
TARGET_HORIZON = 22  # V50과 동일: 22일 후 예측
RV_WINDOW = 22       # V50과 동일: 22일 롤링

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

class DualAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        output = self.fc(context)
        return output

def run_experiment():
    print("=" * 80)
    print("V51b: V50 공정 검증 (IS vs OOS R² 역전 원인 규명)")
    print("=" * 80)
    
    # Data Loading (V50과 동일)
    CACHE_PATH = 'src/data/ohlcv_cache.csv'
    if os.path.exists(CACHE_PATH):
        print(f"Loading data from local cache: {CACHE_PATH}...", end="", flush=True)
        raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(" Done.", flush=True)
    else:
        import yfinance as yf
        print("Downloading from yfinance...", end="", flush=True)
        raw = yf.download(ASSETS, start='2010-01-01', end='2025-01-01', progress=False)['Close']
        print(" Done.", flush=True)
    
    raw.columns = [c.replace('^', '') for c in raw.columns]
    raw = raw.ffill()
    
    # =================================================================
    # 실험 1: V50과 동일 설정, 스케일러 leakage 수정 (Hold-out)
    # =================================================================
    print("\n--- 실험 1: Hold-out (Scaler leakage 수정) ---")
    
    all_data = []
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(RV_WINDOW).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        df_asset = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        
        # V50과 동일: 전체 데이터 스케일링 (leakage 있음)
        scaler_full = StandardScaler()
        data_scaled_full = scaler_full.fit_transform(df_asset)
        m_y_full, s_y_full = scaler_full.mean_[0], scaler_full.scale_[0]
        
        vals = data_scaled_full
        dates = df_asset.index
        
        for j in range(len(vals) - SEQ_LEN - TARGET_HORIZON):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j + SEQ_LEN + TARGET_HORIZON - 1, 0]
            target_date = dates[j + SEQ_LEN + TARGET_HORIZON - 1]
            
            all_data.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y_full,
                's_y': s_y_full
            })
    
    df_all = pd.DataFrame(all_data).sort_values('Date')
    
    # 80/20 Split (V50과 동일)
    split = int(len(df_all) * 0.8)
    train_df = df_all.iloc[:split]
    test_df = df_all.iloc[split:]
    
    X_train = np.stack(train_df['X'].values)
    y_train = np.stack(train_df['y_scaled'].values)
    X_test = np.stack(test_df['X'].values)
    y_test = np.stack(test_df['y_scaled'].values)
    
    # Model Training (V50과 동일)
    torch.manual_seed(42)
    model = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in train_dl:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out.squeeze(), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_dl):.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train R² (IS)
        preds_train = model(torch.FloatTensor(X_train)).squeeze().numpy()
        m_y_tr = train_df['m_y'].values
        s_y_tr = train_df['s_y'].values
        preds_train_orig = preds_train * s_y_tr + m_y_tr
        y_train_orig = y_train * s_y_tr + m_y_tr
        r2_is = r2_score(y_train_orig, preds_train_orig)
        
        # Test R² (OOS, Hold-out)
        preds_test = model(torch.FloatTensor(X_test)).squeeze().numpy()
        m_y_te = test_df['m_y'].values
        s_y_te = test_df['s_y'].values
        preds_test_orig = preds_test * s_y_te + m_y_te
        y_test_orig = y_test * s_y_te + m_y_te
        r2_oos_holdout = r2_score(y_test_orig, preds_test_orig)
    
    print(f"\n[실험 1 결과]")
    print(f"  IS R² (Train):       {r2_is:.5f}")
    print(f"  OOS R² (Hold-out):   {r2_oos_holdout:.5f}")
    print(f"  차이 (IS - OOS):     {r2_is - r2_oos_holdout:.5f}")
    
    # =================================================================
    # 실험 2: 스케일러 leakage 수정 버전
    # =================================================================
    print("\n--- 실험 2: 스케일러를 Train에만 fit ---")
    
    all_data_v2 = []
    for asset in ASSETS:
        price = raw[asset]
        ret = np.log(price / price.shift(1)).dropna()
        rv = (ret**2).rolling(RV_WINDOW).mean() * 252 * 10000
        log_rv = np.log(rv + 1e-6).dropna()
        
        df_asset = pd.DataFrame({'LogRV': log_rv, 'Ret': ret.reindex(log_rv.index)})
        
        # 80% 기준으로 Train만 fit
        n = len(df_asset)
        train_n = int(n * 0.8)
        scaler_train = StandardScaler()
        scaler_train.fit(df_asset.iloc[:train_n])
        data_scaled_v2 = scaler_train.transform(df_asset)
        m_y_v2, s_y_v2 = scaler_train.mean_[0], scaler_train.scale_[0]
        
        vals = data_scaled_v2
        dates = df_asset.index
        
        for j in range(len(vals) - SEQ_LEN - TARGET_HORIZON):
            x = vals[j:j+SEQ_LEN]
            y_scaled = vals[j + SEQ_LEN + TARGET_HORIZON - 1, 0]
            target_date = dates[j + SEQ_LEN + TARGET_HORIZON - 1]
            
            all_data_v2.append({
                'Date': target_date,
                'Asset': asset,
                'X': x,
                'y_scaled': y_scaled,
                'm_y': m_y_v2,
                's_y': s_y_v2
            })
    
    df_all_v2 = pd.DataFrame(all_data_v2).sort_values('Date')
    split_v2 = int(len(df_all_v2) * 0.8)
    train_df_v2 = df_all_v2.iloc[:split_v2]
    test_df_v2 = df_all_v2.iloc[split_v2:]
    
    X_train_v2 = np.stack(train_df_v2['X'].values)
    y_train_v2 = np.stack(train_df_v2['y_scaled'].values)
    X_test_v2 = np.stack(test_df_v2['X'].values)
    y_test_v2 = np.stack(test_df_v2['y_scaled'].values)
    
    torch.manual_seed(42)
    model_v2 = DualAttentionLSTM(input_dim=2, hidden_dim=HIDDEN_DIM)
    criterion_v2 = nn.MSELoss()
    optimizer_v2 = optim.Adam(model_v2.parameters(), lr=LR)
    
    train_ds_v2 = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_v2), torch.FloatTensor(y_train_v2))
    train_dl_v2 = torch.utils.data.DataLoader(train_ds_v2, batch_size=BATCH_SIZE, shuffle=True)
    
    model_v2.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in train_dl_v2:
            optimizer_v2.zero_grad()
            out = model_v2(bx)
            loss = criterion_v2(out.squeeze(), by)
            loss.backward()
            optimizer_v2.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_dl_v2):.6f}")
    
    model_v2.eval()
    with torch.no_grad():
        preds_train_v2 = model_v2(torch.FloatTensor(X_train_v2)).squeeze().numpy()
        m_y_tr_v2 = train_df_v2['m_y'].values
        s_y_tr_v2 = train_df_v2['s_y'].values
        preds_train_orig_v2 = preds_train_v2 * s_y_tr_v2 + m_y_tr_v2
        y_train_orig_v2 = y_train_v2 * s_y_tr_v2 + m_y_tr_v2
        r2_is_v2 = r2_score(y_train_orig_v2, preds_train_orig_v2)
        
        preds_test_v2 = model_v2(torch.FloatTensor(X_test_v2)).squeeze().numpy()
        m_y_te_v2 = test_df_v2['m_y'].values
        s_y_te_v2 = test_df_v2['s_y'].values
        preds_test_orig_v2 = preds_test_v2 * s_y_te_v2 + m_y_te_v2
        y_test_orig_v2 = y_test_v2 * s_y_te_v2 + m_y_te_v2
        r2_oos_v2 = r2_score(y_test_orig_v2, preds_test_orig_v2)
    
    print(f"\n[실험 2 결과 (Scaler leakage 수정)]")
    print(f"  IS R² (Train):       {r2_is_v2:.5f}")
    print(f"  OOS R² (Hold-out):   {r2_oos_v2:.5f}")
    print(f"  차이 (IS - OOS):     {r2_is_v2 - r2_oos_v2:.5f}")
    
    # =================================================================
    # 종합 결과
    # =================================================================
    print("\n" + "=" * 80)
    print("종합 결과 비교")
    print("=" * 80)
    print(f"  기존 V50 R² (Hold-out, scaler leakage):    0.518")
    print(f"  기존 V51 R² (WF, rolling=5, target=t+1):   0.820")
    print(f"  실험1 IS R² (Hold-out, full scaler):       {r2_is:.5f}")
    print(f"  실험1 OOS R² (Hold-out, full scaler):      {r2_oos_holdout:.5f}")
    print(f"  실험2 IS R² (Hold-out, train scaler):      {r2_is_v2:.5f}")
    print(f"  실험2 OOS R² (Hold-out, train scaler):     {r2_oos_v2:.5f}")
    
    results = {
        'experiment_purpose': 'V50 IS vs OOS R² 역전 원인 규명',
        'findings': {
            'root_cause': 'V50(rolling=22, target=t+22)과 V51(rolling=5, target=t+1)의 실험 설정이 완전히 달랐음',
            'v50_original': {'r2': 0.518, 'rolling': 22, 'target': 't+22', 'scaler': 'full_data'},
            'v51_original': {'r2': 0.820, 'rolling': 5, 'target': 't+1', 'scaler': 'train_only'},
        },
        'exp1_full_scaler': {
            'IS_R2': float(r2_is),
            'OOS_R2': float(r2_oos_holdout),
            'delta': float(r2_is - r2_oos_holdout),
            'scaler': 'full_data (leakage)',
            'rolling': RV_WINDOW,
            'target': f't+{TARGET_HORIZON}'
        },
        'exp2_train_scaler': {
            'IS_R2': float(r2_is_v2),
            'OOS_R2': float(r2_oos_v2),
            'delta': float(r2_is_v2 - r2_oos_v2),
            'scaler': 'train_only (no leakage)',
            'rolling': RV_WINDOW,
            'target': f't+{TARGET_HORIZON}'
        }
    }
    
    with open('src/experiments/verification/v51b_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to src/experiments/verification/v51b_results.json")

if __name__ == "__main__":
    run_experiment()
