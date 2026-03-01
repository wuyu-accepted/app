# -*- coding: utf-8 -*-
"""
train_thresholds.py

量化信号分离器的启发式网格寻优 (Grid Search for Classification Thresholds)
脱离拍脑袋的超参数设定，通过历史预测得分与真实收益标签池，遍历寻找最佳交易动作触发点。
以此证明交易阈值的“数据驱动”合理性。
"""

import math
import argparse
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import pandas as pd

def generate_real_backtest_data():
    """
    使用真实的 LSTM 历史预测数据和真实标签来寻找阈值。
    对于 Agent 的动态融合，我们依然使用稀疏随机注入来模拟具有一定准确率的新闻信号。
    """
    print("   [系统] 正在加载真实 LSTM测试集预测结果...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hist_file = os.path.join(base_dir, "lstm_historical_predictions.csv")
    
    if not os.path.exists(hist_file):
        print(f"   [错误] 未找到历史预测文件：{hist_file}")
        return []
        
    data = []
    
    try:
        df = pd.read_csv(hist_file)
        # 将 LSTM 的收益率预测转化为 -1.0 到 1.0 的打分
        # 假设 10% 的预测涨跌幅作为上下限，乘以 10 映射到 [-1.0, 1.0]
        # 这是为了适配 dynamic_gating 里的模型假设分数范围
        df['lstm_score'] = (df['LSTM_Pred_Return'] * 10).clip(lower=-1.0, upper=1.0)
        
        for _, row in df.iterrows():
            lstm_mock_score = row['lstm_score']
            
            # Agent 新闻是稀疏的，偶尔发生大偏差 (假定我们的大模型准确率为 65%)
            agent_mock_score = 0.0
            next_ret = row['True_Next_Return']
            
            if random.random() < 0.1: # 10%的概率有突发新闻
                accuracy_chance = random.random()
                if next_ret > 0:
                    if accuracy_chance < 0.65:
                        agent_mock_score = random.uniform(0.1, 1.0)  # 预测准确，看多
                    else:
                        agent_mock_score = random.uniform(-1.0, -0.1) # 预测错误，看空
                else:
                    if accuracy_chance < 0.65:
                        agent_mock_score = random.uniform(-1.0, -0.1) # 预测准确，看空
                    else:
                        agent_mock_score = random.uniform(0.1, 1.0)   # 预测错误，看多
                
            # 我们在回测池中仅仅提取出未融合的代理信号
            # 真正的 score 会在循环寻找 k 时动态计算
            # 过滤掉涨跌停板以上的无效跳空数据（如果是极端数据）
            if abs(next_ret) < 0.21: 
                data.append((lstm_mock_score, agent_mock_score, next_ret))
                
    except Exception as e:
        print(f"   [错误] 处理预测数据异常: {e}")
            
    print(f"   [系统] 成功提取了 {len(data)} 条真实深度学习历史日线验证集交易样本！")
    return data

def simulate_sharpe_ratio(data, k_param, thresholds):
    """
    给定融合参数 k 和 交易阈值，跑一遍回测，计算策略的简易夏普率或总收益
    thresholds 格式：(strong_buy_th, buy_th, sell_th, strong_sell_th)
    要求: strong_buy > buy > sell > strong_sell
    """
    s_buy, buy, sell, s_sell = thresholds
    if not (s_buy > buy and buy > sell and sell > s_sell):
        return -999.0 # 无效的阈值排序
        
    portfolio_returns = []
    
    for lstm_mock_score, agent_mock_score, true_ret in data:
        # 1. 动态生成 final_score
        w_agent = min(math.pow(abs(agent_mock_score), k_param), 1.0)
        if abs(agent_mock_score) > 0.7:
            w_agent = max(w_agent, 0.8)
            
        w_lstm = 1.0 - w_agent
        final_score = w_lstm * lstm_mock_score + w_agent * agent_mock_score
        
        # 2. 执行动作决策
        position = 0.0
        if final_score >= s_buy:
            position = 1.0     # 强力看多，满仓
        elif final_score >= buy:
            position = 0.5     # 轻仓试盘
        elif final_score > sell:
            position = 0.0     # 空仓观望 (Hold)
        elif final_score > s_sell:
            position = -0.5    # 轻仓融券做空
        else:
            position = -1.0    # 强力看空，满仓做空
            
        # 扣除滑点和手续费 (假设万分之二)
        cost = abs(position) * 0.0002
        trade_profit = (position * true_ret) - cost
        portfolio_returns.append(trade_profit)
        
    # 计算年化夏普比率 (简易化：平均收益 / 收益标准差)
    avg_ret = sum(portfolio_returns) / len(portfolio_returns)
    variance = sum((r - avg_ret) ** 2 for r in portfolio_returns) / len(portfolio_returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001
    
    # 假设每日交易，年化倍数 approx sqrt(252)
    sharpe = (avg_ret / std_dev) * math.sqrt(252)
    return sharpe

def grid_search_thresholds(data):
    print("🚀 启动端到端超参数回测网格搜索...")
    
    # 构建超参数遍历空间 (Hyperparameter Space)
    # 取值范围：
    # k: 控制 Agent 的放大比例，我们将搜索的颗粒度切细一点
    # s_buy: 0.4 到 0.8
    # buy:   0.1 到 0.4
    # sell: -0.4 到 -0.1
    # s_sell: -0.8 到 -0.4
    
    # 增加 k 的网格，从非常不信任(0.5)到非常信任(3.5)，步长 0.5
    k_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    s_buy_range = [0.4, 0.5, 0.6, 0.7, 0.8]
    buy_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    sell_range = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.4]
    s_sell_range = [-0.4, -0.5, -0.6, -0.7, -0.8]
    
    best_sharpe = -999.0
    best_thresh = None
    best_k = 2.0
    
    total_combinations = len(k_range) * len(s_buy_range) * len(buy_range) * len(sell_range) * len(s_sell_range)
    print(f"📊 即将验证的参数组合总数 (5维空间): {total_combinations} 次跑批")
    
    count = 0
    for k in k_range:
        for sb in s_buy_range:
            for b in buy_range:
                for s in sell_range:
                    for ss in s_sell_range:
                        count += 1
                        thresh = (sb, b, s, ss)
                        sharpe = simulate_sharpe_ratio(data, k, thresh)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = thresh
                            best_k = k
                            
                        if count % 1000 == 0:
                            print(f"   执行进度: {count} / {total_combinations} ...")
                            
    return best_k, best_thresh, best_sharpe

if __name__ == "__main__":
    import time
    import numpy as np
    start_time = time.time()
    
    # 设定蒙特卡洛迭代次数
    N_ITERATIONS = 10
    print(f"🌀 准备执行 {N_ITERATIONS} 次蒙特卡洛随机迭代寻优...")
    
    best_thresholds_history = []
    
    for i in range(N_ITERATIONS):
        print(f"\n=======================================================")
        print(f"▶️ 开始迭代 {i+1} / {N_ITERATIONS}")
        print("=======================================================")
        
        print("📈 步骤1/2: 构建/加载真实 A 股历史打分基准数据集...")
        real_data = generate_real_backtest_data()
        
        # 防止空数据运行
        if not real_data:
            print("未找到真实数据，退回生成模拟数据...")
            real_data = [] # Fallback
            break
        
        print(f"\n🔍 步骤2/2: 执行超空间网格扫描以寻找夏普最优截断点 ({i+1}/{N_ITERATIONS})...")
        best_k, best_t, best_s = grid_search_thresholds(real_data)
        
        if best_t is not None:
            best_thresholds_history.append((best_k, *best_t))
            print(f"   [迭代 {i+1} 最佳结果] K={best_k:.2f}, Thresholds: {best_t}, Max Sharpe: {best_s:.4f}")
        else:
            print(f"   [迭代 {i+1}] 未找到有效参数。")

    
    print("\n=======================================================")
    print("✅ 【全局参数寻优完成】Optimal Thresholds Found!")
    
    if best_thresholds_history:
        # 包含 k 的 5 个参数找稳定点
        final_k = np.median([x[0] for x in best_thresholds_history])
        final_s_buy = np.median([x[1] for x in best_thresholds_history])
        final_buy = np.median([x[2] for x in best_thresholds_history])
        final_sell = np.median([x[3] for x in best_thresholds_history])
        final_s_sell = np.median([x[4] for x in best_thresholds_history])
        
        import json
        with open("best_k_t.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": best_thresholds_history,
                "final": [final_k, final_s_buy, final_buy, final_sell, final_s_sell]
            }, f, indent=4)
        print("\n   [建议固化进入 dynamic_gating.py 的硬核交易参数 (基于10次迭代中位数)]")
        print(f"   Math Mode 放大系数 (k): {final_k:.2f}    (原设定为 2.00)")
        print(f"   STRONG BUY       >= {final_s_buy:.2f}    (原设定为 0.50)")
        print(f"   BUY              >= {final_buy:.2f}    (原设定为 0.15)")
        print(f"   SELL              < {final_sell:.2f}    (原设定为 -0.40)")
        print(f"   STRONG SELL       < {final_s_sell:.2f}    (原设定为 -0.80)")
    else:
        print("未收集到足够的阈值数据。")
        
    print("=======================================================")
    print(f"⏱️ 寻优总耗时: {time.time() - start_time:.2f} 秒")
