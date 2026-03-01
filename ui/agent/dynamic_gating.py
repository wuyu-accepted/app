# -*- coding: utf-8 -*-
"""
dynamic_gating.py

动态门控融合模块：负责融合 LSTM 历史模型分数和 Agent 即时多跳分数的最终中枢决策。
包含两种可配置的融合模式：数学门控驱动 (Mode A) 与 双向注意力适配机制 (Mode B)。
"""

import math
from typing import List

class FusionEngine:
    """
    负责动态权重分配与最终买卖量化信号映射的引挚。
    """
    def __init__(self, mode: str = 'math', k: float = 0.5):
        """
        初始化动态门控引擎。
        
        参数:
            mode (str): 'math' 为数学公式门控，'attention' 为双轨注意力适配。
            k (float): 数学模式下的超参数，用于控制 Agent 分数的非线性放大力度。
        """
        if mode not in ['math', 'attention']:
            raise ValueError("mode 必须是 'math' 或 'attention'")
        self.mode = mode
        self.k = k

    def calculate_final_score(self, lstm_features: List[float], agent_features: List[float]) -> dict:
        """
        根据指定的融合机制，计算个股最终得分。
        
        参数:
            lstm_features (List[float]): LSTM 模型的 64 维隐状态输出向量。
                                         我们约定第 0 维代表趋势主预测分 [-1.0, 1.0]。
            agent_features (List[float]): Agent 给出的 1 维特征向量 [total_score]。
            
        返回:
            dict: 包含 final_score (最终得分) 与 action (操作指令)、状态特征等。
        """
        lstm_score_scalar = lstm_features[0]
        agent_score_scalar = agent_features[0]
        
        if self.mode == 'math':
            result = self._fusion_math_mode(lstm_score_scalar, agent_score_scalar)
        else:
            result = self._fusion_attention_mode(lstm_features, agent_features)
            
        # 将连续的 -1 到 +1 分数映射到实际的离散交易动作
        final_score = result['final_score']
        action = self._map_score_to_action(final_score)
        result['action'] = action
        result['mode'] = self.mode
        
        return result
        
    def _fusion_math_mode(self, lstm_score: float, agent_score: float) -> dict:
        """
        模式 A (数学门控)：
        W_event = |S_agent|^k
        Final_score = (1 - W_event) * LSTM_score + W_event * Agent_score
        """
        # 基础动态赋权公式
        w_agent = math.pow(abs(agent_score), self.k)
        # 确保权重不过界
        w_agent = min(w_agent, 1.0)
        
        # 利用自身连续打分（而非离散标志位）决定接管阈值
        if abs(agent_score) > 0.7:
            w_agent = max(w_agent, 0.8)
            status = "重大事件/断档接管 (Math)"
        else:
            status = "常态化基础融合 (Math)"
            
        w_lstm = 1.0 - w_agent
        
        final_score = w_lstm * lstm_score + w_agent * agent_score
        
        return {
            "final_score": round(final_score, 4),
            "status": status,
            "weights": {"w_lstm": w_lstm, "w_agent": w_agent}
        }
        
    def _fusion_attention_mode(self, lstm_features: List[float], agent_features: List[float]) -> dict:
        """
        模式 B (注意力适配)：
        在此逻辑中模拟论文中的交叉注意力权重对齐。
        计算 Query(Agent) 与 Keys(LSTM) 的点积注意力分数，来隐式推断最终结合状态。
        """
        lstm_score = lstm_features[0]
        agent_score = agent_features[0]
        
        # 伪全连接与点积计算模拟 (Attention Engine Mock)
        # 假设我们通过计算 lstm 高维特征中的方差/激活程度来表示模型的不确定性
        lstm_variance = sum(abs(x) for x in lstm_features[1:]) / (len(lstm_features) - 1)
        
        # Attention score 伪算法: 当 Agent 强度大，或者 LSTM 内部特征极度分散(不确定)时，Agent 注意力上升
        attention_agent_raw = abs(agent_score) * 1.5 + lstm_variance * 0.5
        attention_lstm_raw = abs(lstm_score) + 0.1  # 基础平滑
        
        # Softmax 归一化模拟
        exp_agent = math.exp(attention_agent_raw)
        exp_lstm = math.exp(attention_lstm_raw)
        sum_exp = exp_agent + exp_lstm
        
        w_agent = exp_agent / sum_exp
        w_lstm = exp_lstm / sum_exp
        
        final_score = w_lstm * lstm_score + w_agent * agent_score
        
        status = "交叉注意力对齐 (Attention)"
        
        return {
            "final_score": round(final_score, 4),
            "status": status,
            "weights": {"w_lstm": w_lstm, "w_agent": w_agent}
        }

    def _map_score_to_action(self, score: float) -> str:
        """
        将连续的分数映射为具体的离散交易信号。
        (阈值由历史网格搜索回测寻优产生)
        """
        if score >= 0.40:
            return "STRONG BUY"
        elif score >= 0.15:
            return "BUY"
        elif score > -0.40:
            return "HOLD"
        elif score > -0.70:
            return "SELL"
        else:
            return "STRONG SELL"
