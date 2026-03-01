# -*- coding: utf-8 -*-
"""
reasoning.py

核心推理引擎：负责基于知识图谱与突发事件进行多跳衰减传导计算。
完全符合面向对象及控制反转（依赖注入），不强绑定任何图谱具体实现。
"""

from collections import deque
from typing import Dict, List, Tuple

# 引入基类以做类型提示
from abc import ABC, abstractmethod

class BaseGraphProvider(ABC):
    """
    抽象图谱提供者基类。
    强制解耦 Agent 逻辑与底层图谱的具体实现。
    """
    @abstractmethod
    def get_neighbors(self, node_name: str) -> List[Dict]:
        """
        获取节点的直接相邻节点信息。
        """
        pass

class FinancialAgent:
    """
    负责维护状态并执行基于金融图谱逻辑的推理实体。
    不再直接持有静态网络，而是通过依赖传入提供者。
    """
    def __init__(self, decay_lambda: float = 0.8):
        """
        初始化 Agent 及其资产记忆状态。
        
        参数:
            decay_lambda (float): 多跳传递过程中的级联衰减系数（λ），默认 0.8。
        """
        self.decay_lambda = decay_lambda
        # 维护内存：目前跟踪 50 级节点状态池
        self.scores: Dict[str, float] = {}

    def _get_score(self, stock: str) -> float:
        """获取当前个股得分，处理默认0的值"""
        return self.scores.get(stock, 0.0)

    def _set_score(self, stock: str, increment: float):
        """更新个股得分，处理默认0的值，并将结果限制在 [-1.0, 1.0]"""
        current = self.scores.get(stock, 0.0)
        new_score = current + increment
        # 限制分数极值防溢出
        self.scores[stock] = max(min(new_score, 1.0), -1.0)

    def daily_decay(self):
        """
        每日记忆衰减（平滑化版本）：
        使用 Tanh 双曲正切函数进行连续映射：
        当事件绝对值极小（如 0.0）时，衰减率下限为 0.8
        当事件绝对值极大（趋于 1.0）时，衰减率上限逼近 0.95
        """
        import math
        for stock, score in list(self.scores.items()):
            # 基于事件强度的动态阻尼
            dynamic_gamma = 0.80 + 0.15 * math.tanh(abs(score) * 2.0)
            self.scores[stock] = score * dynamic_gamma

    def propagate_impact(self, target_stock: str, initial_power: float, graph_provider: BaseGraphProvider):
        """
        处理突发新闻并基于 BFS 在产业链中扩散信号。
        
        传导公式：$Power_{next} = Power_{curr} \\times base\\_weight \\times \\lambda$
        其中 $base\\_weight$ 自带正负属性，直接反映了上下游协同或是竞对利空。
        
        参数:
            target_stock (str): 受到直接冲击（利好/利空）的首发目标个股。
            initial_power (float): $[-1.0, 1.0]$ 新闻带来的初始冲击强度聚合分数。
            graph_provider (BaseGraphProvider): 提供图谱网络拓扑查询的抽象接口。
        """
        # 记录已访问节点防止图循环死锁
        visited = set()
        
        # 广度优先队栈: 存储 (当前节点名, 到达此节点的能量强度)
        queue = deque([(target_stock, initial_power)])
        
        while queue:
            current_node, current_power = queue.popleft()
            
            # 截断机制：如果波及到此节点的强度不足 0.05，则该波动平息，停止向下级传导
            if abs(current_power) < 0.05:
                continue
                
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 将冲击量注入自身状态记忆中
            self._set_score(current_node, current_power)
            
            # 使用依赖注入提供者，获取所有一级邻居
            neighbors = graph_provider.get_neighbors(current_node)
            
            for nb in neighbors:
                nb_name = nb.get("name")
                base_weight = nb.get("base_weight", 0.0)
                
                if nb_name not in visited:
                    # 计算传导波及
                    # 公式: Power_next = Power_curr * base_weight * λ
                    next_power = current_power * base_weight * self.decay_lambda
                    queue.append((nb_name, next_power))

    def get_feature_vectors(self, stock: str) -> List[float]:
        """
        输出目标股票目前的 Agent 侧提取的主观特征向量。
        供后续网络双轨融合使用。
        
        参数:
            stock (str): 目标股票名称
            
        返回:
            List[float]: [total_score, is_major_event_flag]
                         其中 is_major_event_flag 当 |score| > 0.7 为 1.0，否则为 0.0。
        """
        total_score = self._get_score(stock)
        return [total_score]
