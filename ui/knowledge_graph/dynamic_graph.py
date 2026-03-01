"""
动态图谱管理 — 支持运行时新增/删除边（创新点②：动态图谱演化）

功能:
  - 基于 edges.py 的 EDGES 初始化基础图
  - LLM 分析新闻时调用 add_edge() / remove_edge() 实时更新
  - 每条边携带 sentiment ∈ [-1, 1]（由 LLM 给出，基础边默认 0.0）
  - get_current_graph() 返回当前状态的 PyG Data（含 edge_attr，可带边类型过滤）
  - reset_to_base() 重置到初始状态
"""

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from knowledge_graph.edges import EDGES
from knowledge_graph.graph_builder import (
    RELATION_TO_IDX,
    UNDIRECTED_RELATIONS,
    get_code_to_idx,
)

# 默认关系文件路径
_DEFAULT_RELATIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "all_extracted_relations.json"


class DynamicKnowledgeGraph:
    """维护一个可动态更新的图谱状态"""

    def __init__(self, auto_load: bool = True):
        self._code_to_idx = get_code_to_idx()
        self._num_nodes = len(self._code_to_idx)

        # 基础边列表：[(source_code, target_code, relation_type, description, sentiment), ...]
        self._base_edges = [
            (s, t, r, d, 0.0) for s, t, r, d in EDGES
        ]

        # 动态新增的边（同样是 5-tuple）
        self._dynamic_edges: list[tuple[str, str, str, str, float]] = []

        # 动态删除的边（记录 (source, target) 对，用于从基础边中排除）
        self._removed_edges: set[tuple[str, str]] = set()

        if auto_load:
            self.load_extracted_relations()

    def load_extracted_relations(self, path: str | Path | None = None) -> int:
        """从 all_extracted_relations.json 加载 LLM 提取的关系到动态边。

        对同一 (source, target, relation) 的重复边，取 sentiment 均值聚合。
        返回实际新增的边数。
        """
        fpath = Path(path) if path else _DEFAULT_RELATIONS_FILE
        if not fpath.exists():
            return 0

        with open(fpath, encoding="utf-8") as f:
            raw_relations = json.load(f)

        # 聚合：(source, target, relation) → [sentiments]
        agg: dict[tuple[str, str, str], list[float]] = {}
        for r in raw_relations:
            key = (r.get("source", ""), r.get("target", ""), r.get("relation", ""))
            if not all(key):
                continue
            agg.setdefault(key, []).append(float(r.get("sentiment", 0.0)))

        added = 0
        for (src, tgt, rel), sents in agg.items():
            avg_sent = sum(sents) / len(sents)
            desc = f"LLM提取(n={len(sents)},avg_sent={avg_sent:.2f})"
            if self.add_edge(src, tgt, rel, description=desc, sentiment=avg_sent):
                added += 1

        return added

    def add_edge(
        self,
        source_code: str,
        target_code: str,
        relation_type: str,
        description: str = "",
        sentiment: float = 0.0,
    ) -> bool:
        """动态新增一条边。"""
        if source_code not in self._code_to_idx:
            return False
        if target_code not in self._code_to_idx:
            return False
        if relation_type not in RELATION_TO_IDX:
            return False

        sentiment = max(-1.0, min(1.0, sentiment))

        for edges in [self._base_edges, self._dynamic_edges]:
            for s, t, r, _d, _sent in edges:
                if s == source_code and t == target_code and r == relation_type:
                    return False

        self._dynamic_edges.append((source_code, target_code, relation_type, description, sentiment))
        self._removed_edges.discard((source_code, target_code))
        return True

    def remove_edge(self, source_code: str, target_code: str) -> bool:
        """删除一条边（标记删除）。"""
        found = False

        for i, (s, t, r, _d, _sent) in enumerate(self._dynamic_edges):
            if s == source_code and t == target_code:
                self._dynamic_edges.pop(i)
                found = True
                break

        for s, t, r, _d, _sent in self._base_edges:
            if s == source_code and t == target_code:
                self._removed_edges.add((source_code, target_code))
                if r in UNDIRECTED_RELATIONS:
                    self._removed_edges.add((target_code, source_code))
                found = True
                break

        return found

    def get_current_graph(self, active_relations: list[str] | None = None) -> Data:
        """返回当前状态的 PyG Data 对象。"""
        src_list = []
        tgt_list = []
        type_list = []
        sentiment_list = []

        all_edges = self._base_edges + self._dynamic_edges

        for source, target, relation_type, _desc, sentiment in all_edges:
            if (source, target) in self._removed_edges:
                continue
            if active_relations is not None and relation_type not in active_relations:
                continue
            if source not in self._code_to_idx or target not in self._code_to_idx:
                continue

            s_idx = self._code_to_idx[source]
            t_idx = self._code_to_idx[target]
            r_idx = RELATION_TO_IDX[relation_type]

            src_list.append(s_idx)
            tgt_list.append(t_idx)
            type_list.append(r_idx)
            sentiment_list.append(sentiment)

            if relation_type in UNDIRECTED_RELATIONS:
                src_list.append(t_idx)
                tgt_list.append(s_idx)
                type_list.append(r_idx)
                sentiment_list.append(sentiment)

        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
        edge_attr = torch.tensor(sentiment_list, dtype=torch.float32).unsqueeze(-1)

        data = Data(edge_index=edge_index, num_nodes=self._num_nodes)
        data.edge_type = edge_type
        data.edge_attr = edge_attr
        return data

    def reset_to_base(self):
        """重置到初始状态"""
        self._dynamic_edges.clear()
        self._removed_edges.clear()

    def update_edge_sentiment(self, source_code: str, target_code: str, sentiment: float) -> bool:
        """更新已有边的 sentiment 值。"""
        sentiment = max(-1.0, min(1.0, sentiment))

        for i, (s, t, r, d, _old) in enumerate(self._dynamic_edges):
            if s == source_code and t == target_code:
                self._dynamic_edges[i] = (s, t, r, d, sentiment)
                return True

        for i, (s, t, r, d, _old) in enumerate(self._base_edges):
            if s == source_code and t == target_code:
                self._base_edges[i] = (s, t, r, d, sentiment)
                return True

        return False

    def get_neighbors_raw(self, code: str) -> list[tuple[str, str, str, float]]:
        """
        获取某节点的所有邻居（用于适配 BaseGraphProvider 接口）。

        Returns:
            [(neighbor_code, relation_type, description, sentiment), ...]
        """
        neighbors = []
        all_edges = self._base_edges + self._dynamic_edges

        for source, target, relation_type, desc, sentiment in all_edges:
            if (source, target) in self._removed_edges:
                continue
            if source == code:
                neighbors.append((target, relation_type, desc, sentiment))
            elif target == code and relation_type in UNDIRECTED_RELATIONS:
                neighbors.append((source, relation_type, desc, sentiment))

        return neighbors

    def get_all_current_edges(self) -> list[tuple[str, str, str, str, float]]:
        """返回所有当前生效的边"""
        edges = []
        all_edges = self._base_edges + self._dynamic_edges
        for s, t, r, d, sent in all_edges:
            if (s, t) not in self._removed_edges:
                edges.append((s, t, r, d, sent))
        return edges

    @property
    def num_dynamic_edges(self) -> int:
        return len(self._dynamic_edges)

    @property
    def num_removed_edges(self) -> int:
        return len(self._removed_edges)

    def get_dynamic_edges_info(self) -> list[dict]:
        """返回所有动态新增边的信息"""
        return [
            {"source": s, "target": t, "relation_type": r, "description": d, "sentiment": sent}
            for s, t, r, d, sent in self._dynamic_edges
        ]
