#!/usr/bin/env python
# coding: utf-8

# In[1]:


import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime

# 1. 读取 CSV 文件构建股票池
csv_file_path = os.path.join('ui', 'stock_pool.csv')  # 请确保你的文件名是这个

if not os.path.exists(csv_file_path):
    print(f"❌ 错误：在当前目录下未找到 {csv_file_path} 文件。")
    exit()

print(f"📖 正在读取 {csv_file_path} ...")

# 注意：
# dtype={'代码': str} 必须加，否则 002281 会变成 2281
# encoding='utf-8-sig' 用于去除文件开头的 BOM 字符
try:
    df_pool = pd.read_csv(csv_file_path, dtype={'代码': str}, encoding='utf-8-sig')
    
    # 去除列名的空格（防止CSV表头有空格）
    df_pool.columns = df_pool.columns.str.strip()
    
    # 检查列名是否正确
    required_columns = ['代码', '名称', '板块']
    if not all(col in df_pool.columns for col in required_columns):
        print(f"❌ CSV格式错误，必须包含列: {required_columns}")
        print(f"当前列名: {df_pool.columns.tolist()}")
        exit()
        
except Exception as e:
    print(f"❌ 读取CSV失败: {e}")
    exit()

print(f"✅ 成功加载 {len(df_pool)} 只股票信息。")

# 2. 设置数据保存目录和时间范围
save_dir = os.path.join("ui", "stock_data_csv")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"📁 已创建数据存放文件夹: {save_dir}")

# 设置获取数据的时间段
start_date = "20230101"
end_date = "20260223" # 自动获取今天日期 (例如 20260221)
print(f"📅 数据时间范围: {start_date} 至 {end_date}")

# 3. 遍历 DataFrame，抓取并保存数据
print(f"🚀 开始批量获取量价数据 (前复权) ...")

for index, row in df_pool.iterrows():
    symbol = row['代码']
    name = row['名称']
    sector = row['板块'] # 读取板块信息

    # 为了文件名合法，去除板块名称中的特殊字符（如下划线等如果是路径分隔符）
    safe_sector = sector.replace('/', '_').replace('\\', '_')

    try:
        print(f"[{index+1}/{len(df_pool)}] 正在获取: {safe_sector} - {name}({symbol}) ...", end=" ")
        
        # 调用 AkShare 接口获取 A 股历史行情数据
        # adjust="qfq" 代表前复权
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df.empty:
            print("⚠️ 数据为空 (可能是停牌或新股)，跳过。")
            continue
            
        # 重命名列名（转为英文，方便后续 LSTM 处理）
        df.rename(columns={
            '日期': 'date',
            '股票代码': 'symbol',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change_amount',
            '换手率': 'turnover'
        }, inplace=True)
        
        # 将日期设置为索引并排序
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        # 保存到本地 CSV 文件
        # 文件名格式建议：板块_代码_名称.csv，这样文件夹里会自动按板块排序
        file_name = f"{safe_sector}_{symbol}_{name}.csv"
        file_path = os.path.join(save_dir, file_name)
        
        df.to_csv(file_path)
        print(f"✅ 已保存")
        
        # 礼貌性休眠
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ 抓取失败: {e}")

print("🎉 所有数据获取完毕！")


# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import glob
import time
import random

# ==========================================
# 0. 宇宙法则：锁定随机种子，保证绝对可复现
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

seed_everything(42) # 启动定海神针

# 自动检测是否可以使用 GPU 加速
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 核心特征工程 (11维黄金技术指标)
# ==========================================
def add_technical_indicators(df):
    data = df.copy()
    
    # RSI (14天)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 布林带 Bollinger Bands (20天)
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['STD20'] = data['close'].rolling(window=20).std()
    data['BB_Upper'] = data['MA20'] + (data['STD20'] * 2)
    data['BB_Lower'] = data['MA20'] - (data['STD20'] * 2)
    
    return data

# ==========================================
# 2. 究极版 LSTM 网络架构 (抗噪音拟合)
# ==========================================
class UltimateStockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(UltimateStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.layer_norm(out) 
        out = self.fc(out)
        return out

# ==========================================
# 3. 稳健的数据流水线
# ==========================================
def prepare_robust_data(file_path, seq_length=20):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    
    df['target_return'] = df['close'].pct_change()
    df = add_technical_indicators(df)
    
    # 极值与缺失值清洗
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if len(df) < seq_length + 20:
        return None
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower']
    
    data_X = df[feature_cols].values
    data_y = df[['target_return']].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaled_X = scaler_X.fit_transform(data_X)
    scaled_y = scaler_y.fit_transform(data_y)
    
    X, y = [], []
    for i in range(len(scaled_X) - seq_length):
        X.append(scaled_X[i : i + seq_length])
        y.append(scaled_y[i + seq_length])
        
    dates = df.index[seq_length:]
        
    return (torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE), 
            torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE), 
            scaler_y, scaled_X, df.iloc[-1]['close'], dates)

# ==========================================
# 4. 单股高精度训练与预测核心逻辑
# ==========================================
def train_and_predict_single(file_path, seq_length=20, epochs=100):
    base_name = os.path.basename(file_path).replace('.csv', '')
    
    # 完美兼容你的神级命名法: 上游_AI芯片_300223_北京君正
    parts = base_name.split('_')
    if len(parts) >= 2:
        symbol = parts[-2]
        name = parts[-1]
    else:
        symbol = base_name
        name = "未知股票"

    prep_result = prepare_robust_data(file_path, seq_length)
    if prep_result is None:
        return symbol, name, None, None, None, "有效数据不足", []
        
    X_tensor, y_tensor, scaler_y, scaled_X_full, last_close, dates = prep_result
    
    train_size = int(len(X_tensor) * 0.8) # 回测取后 20% 作为测试集
    X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
    X_test, y_test = X_tensor[train_size:], y_tensor[train_size:]
    test_dates = dates[train_size:]
    
    model = UltimateStockLSTM(input_size=11, hidden_size=64, num_layers=2, output_size=1).to(DEVICE)
    
    # Huber Loss 抵抗异动噪音，L2正则化抵抗过拟合
    criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
            
    model.eval()
    historical_preds = []
    with torch.no_grad():
        # 获取测试集上的预测得分
        if len(X_test) > 0:
            preds_scaled = model(X_test)
            preds_return = scaler_y.inverse_transform(preds_scaled.cpu().numpy())
            actuals_return = scaler_y.inverse_transform(y_test.cpu().numpy())
            
            for i in range(len(preds_return)):
                historical_preds.append({
                    "Date": test_dates[i].strftime('%Y-%m-%d'),
                    "Symbol": symbol,
                    "LSTM_Pred_Return": float(preds_return[i][0]),
                    "True_Next_Return": float(actuals_return[i][0])
                })
        
        latest_window = scaled_X_full[-seq_length:]
        latest_window_tensor = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        pred_scaled = model(latest_window_tensor)
        pred_return = scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0][0]
        pred_price = last_close * (1 + pred_return)
        
    return symbol, name, last_close, pred_price, pred_return, "成功", historical_preds

# ==========================================
# 5. 全量引擎
# ==========================================
if __name__ == "__main__":
    # 指向你的数据文件夹，现在文件在 agent/ 下，而数据在 ui/stock_data_csv 下
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(base_dir, "ui", "stock_data_csv")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        print(f"❌ 未在 [{data_folder}] 找到数据，请检查路径。")
        exit()
        
    print(f"🚀 发现 {len(csv_files)} 只股票，采用最优参数启动量化级 LSTM 集群...")
    print(f"🖥️ 当前计算设备: {DEVICE.type.upper()}")
    print("-" * 75)
    print(f"{'代码':<8} | {'名称':<10} | {'最新收盘价':<10} | {'预测明日价':<10} | {'LSTM 基准涨跌幅':<15} | {'状态'}")
    print("-" * 75)
    
    results = []
    all_historical_preds = []
    start_time = time.time()
    
    for i, file_path in enumerate(csv_files, 1):
        try:
            sym, name, actual, pred_p, pred_r, status, h_preds = train_and_predict_single(file_path)
            
            if status == "成功":
                print(f"{sym:<10} | {name:<10} | ¥ {actual:<10.2f} | ¥ {pred_p:<10.2f} | {pred_r*100:>+8.2f}%       | ✅")
                results.append({
                    "Symbol": sym, "Name": name, 
                    "Last_Close": round(actual, 2), 
                    "LSTM_Base_Return(%)": round(pred_r * 100, 2)
                })
                all_historical_preds.extend(h_preds)
            else:
                print(f"{sym:<10} | {name:<10} | {'-':<12} | {'-':<12} | {'-':<17} | ⚠️ {status}")
                
        except Exception as e:
            file_name = os.path.basename(file_path)
            print(f"{file_name[:10]}... | {'Error':<10} | {'-':<12} | {'-':<12} | {'-':<17} | ❌ 异常报错 {e}")

    print("-" * 75)
    print(f"🎉 物理引擎基准测试完毕！耗时: {time.time() - start_time:.1f} 秒。")
    
    output_csv = "lstm_ultimate_baseline.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"💾 高精度技术面基准池已保存至: {output_csv}")
    
    hist_csv = "lstm_historical_predictions.csv"
    if all_historical_preds:
        pd.DataFrame(all_historical_preds).to_csv(hist_csv, index=False)
        print(f"💾 历史回测打分池已保存至: {hist_csv}")

