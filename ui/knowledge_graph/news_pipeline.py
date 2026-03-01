"""
新闻分析流水线 — 爬取新闻 → LLM提取关系 → 更新动态图谱

使用方式:
    python -m knowledge_graph.news_pipeline              # 全部50只
    python -m knowledge_graph.news_pipeline 002230       # 指定股票
    python -m knowledge_graph.news_pipeline --top 5      # 前5只（测试）
"""

import json
import time
from pathlib import Path

from knowledge_graph.stock_pool import get_all_codes, get_all_stocks_flat
from knowledge_graph.news_fetcher import fetch_stock_news_merged
from knowledge_graph.dynamic_graph import DynamicKnowledgeGraph
from knowledge_graph.news_analyzer import NewsAnalyzer, analyze_and_update, get_api_key, get_base_url


def run_pipeline(
    codes: list[str] | None = None,
    max_news_per_stock: int = 10,
    only_event: bool = True,
    delay: float = 1.0,
) -> tuple[DynamicKnowledgeGraph, list[dict]]:
    """完整流水线：爬新闻 → LLM分析 → 更新图谱"""
    import sys

    api_key = get_api_key()
    if not api_key:
        print("错误: 未配置 API key，请编辑 knowledge_graph/.env")
        sys.exit(1)

    base_url = get_base_url()
    analyzer = NewsAnalyzer(api_key=api_key, base_url=base_url)
    dkg = DynamicKnowledgeGraph()

    if codes is None:
        codes = get_all_codes()

    code_to_name = {code: name for code, name, _ in get_all_stocks_flat()}

    print(f"待分析股票: {len(codes)} 只")
    print(f"初始图谱边数: {dkg.get_current_graph().edge_index.shape[1]}")
    print()

    all_results = []
    total_news = 0
    total_relations = 0

    for i, code in enumerate(codes):
        name = code_to_name.get(code, code)
        print(f"[{i+1}/{len(codes)}] {code} {name}", end=" ")

        try:
            news_list = fetch_stock_news_merged(code, page_size=max_news_per_stock,
                                                 scrape_content=True)
        except Exception as e:
            print(f"  爬取失败: {e}")
            continue

        if only_event:
            news_list = [n for n in news_list if n.get("news_type") == "event"]

        news_list = news_list[:max_news_per_stock]
        print(f"→ {len(news_list)} 条新闻")

        for news in news_list:
            text = news.get("title", "")
            content = news.get("content", "")
            if content:
                text = f"{text}。{content}"
            text = text[:300]

            if len(text) < 10:
                continue

            try:
                results = analyze_and_update(analyzer, dkg, text)
            except Exception as e:
                print(f"    LLM分析失败: {e}")
                continue

            total_news += 1
            if results:
                total_relations += len(results)
                for r in results:
                    r["from_stock"] = code
                    r["news_title"] = news.get("title", "")
                all_results.extend(results)
                print(f"    [{news['title'][:30]}...] → {len(results)} 条关系")

        if i < len(codes) - 1:
            time.sleep(delay)

    graph = dkg.get_current_graph()
    print(f"\n{'='*60}")
    print(f"分析完成!")
    print(f"  处理新闻: {total_news} 条")
    print(f"  提取关系: {total_relations} 条")
    print(f"  新增动态边: {dkg.num_dynamic_edges} 条")
    print(f"  更新后图谱边数: {graph.edge_index.shape[1]}")
    print(f"{'='*60}")

    return dkg, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="新闻→知识图谱分析流水线")
    parser.add_argument("codes", nargs="*", help="指定股票代码")
    parser.add_argument("--top", type=int, default=0, help="只分析前N只")
    parser.add_argument("--max-news", type=int, default=5, help="每只股票最多分析几条")
    parser.add_argument("--delay", type=float, default=1.0, help="爬取间隔")
    args = parser.parse_args()

    codes = args.codes if args.codes else None
    if codes is None and args.top > 0:
        codes = get_all_codes()[:args.top]

    dkg, results = run_pipeline(
        codes=codes,
        max_news_per_stock=args.max_news,
        delay=args.delay,
    )

    if results:
        out_dir = Path(__file__).resolve().parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "extracted_relations.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n提取结果已保存到: {out_path}")

        dynamic_path = out_dir / "dynamic_edges.json"
        with open(dynamic_path, "w", encoding="utf-8") as f:
            json.dump(dkg.get_dynamic_edges_info(), f, ensure_ascii=False, indent=2)
        print(f"动态边已保存到: {dynamic_path}")
