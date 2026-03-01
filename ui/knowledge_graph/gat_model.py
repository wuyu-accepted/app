"""
GAT 图神经网络模型 — 聚合知识图谱邻居信息，输出传导修正量

架构:
  - 2层 GATConv，heads=4，dropout=0.2
  - 层间 ELU 激活 + 残差连接
  - 按关系类型选择性传播（创新点①）
  - 最终线性层 → [num_nodes, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from knowledge_graph.graph_builder import RELATION_TO_IDX


class StockGAT(nn.Module):
    """
    输入:
        node_features: [num_nodes, in_dim]   来自 LSTM 隐状态
        edge_index:    [2, num_edges]
        edge_type:     [num_edges]
        edge_attr:     [num_edges, 1]        sentiment 分数（由 LLM 给出）
    输出:
        y_graph:       [num_nodes, 1]        每个节点的传导修正量
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
        num_relation_types: int = 5,
        edge_attr_dim: int = 1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.num_relation_types = num_relation_types
        self.edge_attr_dim = edge_attr_dim

        head_dim = hidden_dim // heads  # 16

        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_attr_dim,
        )

        if in_dim != hidden_dim:
            self.res_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.res_proj = None

        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_attr_dim,
        )

        self.relation_bias = nn.Embedding(num_relation_types, heads)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _filter_edges(self, edge_index, edge_type, edge_attr, active_relations):
        """按关系类型过滤边（创新点①：事件感知选择性传播）"""
        if active_relations is None:
            return edge_index, edge_type, edge_attr

        active_indices = [RELATION_TO_IDX[r] for r in active_relations if r in RELATION_TO_IDX]
        if not active_indices:
            device = edge_index.device
            empty_attr = torch.zeros(0, self.edge_attr_dim, device=device) if edge_attr is not None else None
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                empty_attr,
            )

        mask = torch.zeros(edge_type.shape[0], dtype=torch.bool, device=edge_type.device)
        for idx in active_indices:
            mask |= (edge_type == idx)

        filtered_attr = edge_attr[mask] if edge_attr is not None else None
        return edge_index[:, mask], edge_type[mask], filtered_attr

    def forward(self, node_features, edge_index, edge_type, edge_attr=None, active_relations=None):
        edge_index, edge_type, edge_attr = self._filter_edges(
            edge_index, edge_type, edge_attr, active_relations
        )

        x = node_features

        residual = x if self.res_proj is None else self.res_proj(x)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x + residual)
        x = F.elu(x)

        y_graph = self.out_linear(x)
        return y_graph
