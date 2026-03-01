"""
知识图谱模块 (Module B) — 数据获取 + 知识图谱 + GAT 图神经网络

对接方式:
    # 在 main.py 中替换 MockTeammateA:
    from knowledge_graph.graph_adapter import RealGraphProvider
    provider = RealGraphProvider()

    # provider 兼容 BaseGraphProvider 接口:
    #   provider.get_neighbors("中际旭创")
    #   provider.get_graph_snapshot()
    #   provider.get_latest_news()
"""
