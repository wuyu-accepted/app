"""
图谱适配器 — 将 DynamicKnowledgeGraph 适配为队友 Agent 模块的 BaseGraphProvider 接口

队友接口要求:
    get_neighbors(node_name) -> [{"name": str, "relation": str, "base_weight": float}, ...]
    get_graph_snapshot()     -> {node_name: [neighbor_dicts]}
    get_latest_news()        -> {"target_stock": str, "impact_score": float, "news_text": str}

使用方式:
    # 在队友的 main.py 中，替换 MockTeammateA:
    #   mock_a = MockTeammateA()
    # 改为:
    from knowledge_graph.graph_adapter import RealGraphProvider
    mock_a = RealGraphProvider()
"""

import json
from pathlib import Path

from knowledge_graph.dynamic_graph import DynamicKnowledgeGraph
from knowledge_graph.stock_pool import get_all_stocks_flat, get_all_codes

# 关系类型 → 默认传导权重（用于基础边，无 LLM sentiment 时）
DEFAULT_RELATION_WEIGHT = {
    "supply":    0.7,    # 供应链正相关
    "compete":  -0.5,    # 竞争负相关
    "peer":      0.4,    # 板块联动
    "invest":    0.85,   # 控股强正相关
    "cooperate": 0.6,    # 合作正相关
}

_DEFAULT_RELATIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "all_extracted_relations.json"


class RealGraphProvider:
    """
    适配 DynamicKnowledgeGraph → 队友 BaseGraphProvider 接口。

    支持 name（股票名称）和 code（股票代码）两种方式查询。
    输出格式完全兼容队友的 reasoning.py 和 dynamic_gating.py。
    """

    def __init__(self, dkg: DynamicKnowledgeGraph | None = None):
        self.dkg = dkg or DynamicKnowledgeGraph()

        self._name_to_code: dict[str, str] = {}
        self._code_to_name: dict[str, str] = {}
        for code, name, _sector in get_all_stocks_flat():
            self._name_to_code[name] = code
            self._code_to_name[code] = name

        self._news_cache: list[dict] = []
        self._load_news_from_relations()

    # ---- 核心接口：兼容 BaseGraphProvider ----

    def get_neighbors(self, node_name: str) -> list[dict]:
        """
        获取节点的直接邻居（兼容 BaseGraphProvider 接口）。

        Args:
            node_name: 股票名称（如 "中际旭创"）或代码（如 "300308"）

        Returns:
            [{"name": str, "relation": str, "base_weight": float}, ...]
        """
        code = self._resolve_code(node_name)
        if code is None:
            return []

        raw_neighbors = self.dkg.get_neighbors_raw(code)
        result = []
        for nb_code, relation_type, _desc, sentiment in raw_neighbors:
            nb_name = self._code_to_name.get(nb_code, nb_code)
            result.append({
                "name": nb_name,
                "relation": relation_type.upper(),
                "base_weight": self._compute_weight(relation_type, sentiment),
            })
        return result

    def get_graph_snapshot(self) -> dict:
        """返回完整图谱快照。"""
        snapshot = {}
        for code in get_all_codes():
            name = self._code_to_name.get(code, code)
            snapshot[name] = self.get_neighbors(code)
        return snapshot

    def get_latest_news(self, stock_name: str = "") -> dict:
        """返回最新新闻事件。如果指定 stock_name，优先返回该股票的新闻。"""
        if stock_name and self._news_cache:
            # 优先找该股票相关的新闻
            code = self._resolve_code(stock_name)
            name = self._code_to_name.get(code, stock_name) if code else stock_name
            for item in self._news_cache:
                if item["target_stock"] == name or item["target_stock"] == stock_name:
                    return item

        if self._news_cache:
            top = max(self._news_cache, key=lambda x: abs(x.get("impact_score", 0)))
            return top

        return {
            "target_stock": stock_name or "寒武纪",
            "impact_score": 0.0,
            "news_text": "暂无新闻数据，请先运行 news_pipeline 获取实时新闻。",
        }

    # ---- 新闻集成 ----

    def load_news_from_pipeline(self, results: list[dict]):
        """从 news_pipeline 的提取结果中加载新闻事件。"""
        self._news_cache.clear()
        for r in results:
            source_name = self._code_to_name.get(r.get("source", ""), r.get("source", ""))
            self._news_cache.append({
                "target_stock": source_name,
                "impact_score": r.get("sentiment", 0.0),
                "news_text": r.get("description", r.get("news_title", "")),
            })

    def add_news_event(self, stock_name: str, impact_score: float, news_text: str):
        """手动添加一条新闻事件"""
        self._news_cache.append({
            "target_stock": stock_name,
            "impact_score": max(-1.0, min(1.0, impact_score)),
            "news_text": news_text,
        })

    # ---- 内部工具 ----

    def _resolve_code(self, name_or_code: str) -> str | None:
        """将名称或代码统一解析为股票代码"""
        if name_or_code in self._code_to_name:
            return name_or_code
        if name_or_code in self._name_to_code:
            return self._name_to_code[name_or_code]
        return None

    def _compute_weight(self, relation_type: str, sentiment: float) -> float:
        """计算 base_weight：有 LLM sentiment 用 sentiment，否则用默认权重。"""
        if abs(sentiment) > 0.01:
            return round(sentiment, 4)
        return DEFAULT_RELATION_WEIGHT.get(relation_type, 0.3)

    def _load_news_from_relations(self):
        """启动时从 all_extracted_relations.json 加载新闻事件到 _news_cache"""
        if not _DEFAULT_RELATIONS_FILE.exists():
            return
        try:
            with open(_DEFAULT_RELATIONS_FILE, encoding="utf-8") as f:
                relations = json.load(f)
        except Exception:
            return

        # 按 from_stock 分组，计算每只股票的平均 sentiment 和最新新闻
        stock_sents: dict[str, list[float]] = {}
        stock_news: dict[str, str] = {}
        for r in relations:
            stock = r.get("from_stock", "")
            if not stock:
                continue
            stock_sents.setdefault(stock, []).append(float(r.get("sentiment", 0)))
            # 保留最后一条描述
            desc = r.get("description", r.get("news_title", ""))
            if desc:
                stock_news[stock] = desc

        for code, sents in stock_sents.items():
            name = self._code_to_name.get(code, code)
            avg_sent = sum(sents) / len(sents)
            self._news_cache.append({
                "target_stock": name,
                "impact_score": round(avg_sent, 3),
                "news_text": stock_news.get(code, ""),
            })
