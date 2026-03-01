"""
图构建工具 — 将 edges.py 的边表转为 PyTorch Geometric 图结构

功能:
  - 读取 EDGES 列表，构建 code→idx 映射
  - 生成 edge_index [2, num_edges]、edge_type 和 edge_attr(sentiment) 张量
  - 无向边（compete/peer/cooperate）双向添加
  - 有向边（supply/invest）保持方向
  - 支持 edge_mask 参数按关系类型过滤边（创新点①：事件感知选择性传播）
"""

import torch
from torch_geometric.data import Data

from knowledge_graph.edges import EDGES
from knowledge_graph.stock_pool import get_all_codes

# 关系类型 → 整数编码
RELATION_TO_IDX = {
    "supply": 0,
    "compete": 1,
    "peer": 2,
    "invest": 3,
    "cooperate": 4,
}

# 无向关系类型（双向添加）
UNDIRECTED_RELATIONS = {"compete", "peer", "cooperate"}


def get_code_to_idx() -> dict[str, int]:
    """节点索引映射：股票代码 → 图节点编号"""
    codes = get_all_codes()
    return {code: idx for idx, code in enumerate(codes)}


def get_relation_types() -> list[str]:
    """返回所有关系类型列表"""
    return list(RELATION_TO_IDX.keys())


def build_pyg_graph(edge_mask: list[str] | None = None) -> Data:
    """
    将 edges.py 的边表转为 PyG Data 对象。

    Args:
        edge_mask: 只保留指定关系类型的边，如 ["supply", "compete"]。
                   None 表示保留所有边。

    Returns:
        torch_geometric.data.Data，包含:
          - edge_index: [2, num_edges]
          - edge_type:  [num_edges]
          - edge_attr:  [num_edges, 1]  sentiment 分数（基础边默认 0.0）
          - num_nodes:  50
    """
    code_to_idx = get_code_to_idx()
    num_nodes = len(code_to_idx)

    src_list = []
    tgt_list = []
    type_list = []
    sentiment_list = []

    for source, target, relation_type, _desc in EDGES:
        # 按 edge_mask 过滤
        if edge_mask is not None and relation_type not in edge_mask:
            continue

        # 跳过不在股票池中的代码
        if source not in code_to_idx or target not in code_to_idx:
            continue

        s_idx = code_to_idx[source]
        t_idx = code_to_idx[target]
        r_idx = RELATION_TO_IDX[relation_type]

        # 添加正向边
        src_list.append(s_idx)
        tgt_list.append(t_idx)
        type_list.append(r_idx)
        sentiment_list.append(0.0)  # 基础边默认 sentiment=0

        # 无向关系：添加反向边
        if relation_type in UNDIRECTED_RELATIONS:
            src_list.append(t_idx)
            tgt_list.append(s_idx)
            type_list.append(r_idx)
            sentiment_list.append(0.0)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    edge_attr = torch.tensor(sentiment_list, dtype=torch.float32).unsqueeze(-1)  # [E, 1]

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    data.edge_type = edge_type
    data.edge_attr = edge_attr

    return data


def get_neighbor_indices(node_idx: int, data: Data | None = None) -> list[int]:
    """获取某节点的1-hop邻居索引列表。"""
    if data is None:
        data = build_pyg_graph()

    mask = data.edge_index[0] == node_idx
    neighbors = data.edge_index[1, mask].tolist()

    mask_rev = data.edge_index[1] == node_idx
    neighbors_rev = data.edge_index[0, mask_rev].tolist()

    return sorted(set(neighbors + neighbors_rev))
