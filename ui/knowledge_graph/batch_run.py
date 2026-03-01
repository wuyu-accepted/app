"""
批量新闻→知识图谱处理脚本 — 并发加速 + 断点续传

用法:
    cd teammate-repo2/
    python -m knowledge_graph.batch_run                    # 全部50只股票
    python -m knowledge_graph.batch_run --top 5            # 前5只测试
    python -m knowledge_graph.batch_run --workers 5        # 5线程并发调LLM
    python -m knowledge_graph.batch_run --resume           # 从上次中断处继续

数据量估算:
    50只股票 × ~200条新闻/只 ≈ 10,000条新闻
    其中 event 类 ~40% ≈ 4,000条
    LLM 分析 4,000条 × ~700 tokens ≈ 2.8M tokens
    API2D 成本约 $2-5

时间估算:
    单线程: ~2小时 (4000条 × 2秒/条)
    5线程:  ~25分钟
"""

import json
import time
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from knowledge_graph.stock_pool import get_all_codes, get_all_stocks_flat
from knowledge_graph.news_fetcher import (
    fetch_news_listapi, fetch_news_akshare, fetch_news_search,
    fetch_announcements, _classify_news, backfill_content,
)
from knowledge_graph.news_analyzer import NewsAnalyzer, get_api_key, get_base_url

# =====================================================================
#  配置
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
NEWS_CACHE_FILE = DATA_DIR / "all_news_cache.json"          # 抓取的全部新闻
RELATIONS_FILE = DATA_DIR / "all_extracted_relations.json"   # LLM提取的关系
PROGRESS_FILE = DATA_DIR / "batch_progress.json"             # 断点续传进度

_print_lock = Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# =====================================================================
#  Step 1: 批量抓取新闻（尽可能多页）
# =====================================================================

def fetch_all_news_for_stock(code: str, name: str = "",
                             start_date: str = "2025-01-01") -> list[dict]:
    """抓取一只股票的所有可用新闻（四个数据源合并去重）

    数据源:
      1. 东方财富搜索API (curl_cffi) — 按股票名搜索，可翻页，覆盖~半年 (~1200条)
      2. 东方财富 listapi — 近1个月新闻 (~200条)
      3. AKShare stock_news_em — 近2周新闻 (~10条，带正文)
      4. 东方财富公告 — 可翻页，覆盖数年历史 (~180条/年)
    """
    all_news = []
    seen_titles = set()

    def _add(items):
        for item in items:
            title = item.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                if "news_type" not in item:
                    item["news_type"] = _classify_news(title)
                all_news.append(item)

    # 1. 东方财富搜索API（主力数据源，~1200条，半年历史）
    if name:
        try:
            _add(fetch_news_search(name, max_pages=30, page_size=50,
                                   stock_code=code))
        except Exception:
            pass

    # 2. AKShare（事件类为主，带正文内容）
    try:
        _add(fetch_news_akshare(code))
    except Exception:
        pass

    # 3. 东方财富新闻 listapi（近1个月，~200条）
    try:
        prefix = "1" if code.startswith("6") else "0"
        import requests
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        params = {
            "client": "wap", "type": 1,
            "mTypeAndCode": f"{prefix}.{code}",
            "pageSize": 200, "pageNo": 1,
        }
        r = session.get(url, params=params, verify=False, timeout=15)
        data = r.json()
        items = data.get("data", {}).get("list", [])
        news_items = []
        for it in items:
            title = it.get("Art_Title", "").strip()
            if title:
                news_items.append({
                    "stock_code": code,
                    "title": title,
                    "time": it.get("Art_ShowTime", ""),
                    "source": it.get("Art_MediaName", ""),
                    "url": it.get("Art_Url", ""),
                    "news_type": _classify_news(title),
                })
        _add(news_items)
    except Exception:
        pass

    # 4. 东方财富公告（可翻页，覆盖长历史）
    try:
        announcements = fetch_announcements(code, page_size=100, max_pages=10,
                                            start_date=start_date)
        _add(announcements)
    except Exception:
        pass

    return all_news


def batch_fetch_news(codes: list[str], delay: float = 0.5) -> list[dict]:
    """批量抓取所有股票新闻"""
    code_to_name = {c: n for c, n, _ in get_all_stocks_flat()}
    all_news = []

    for i, code in enumerate(codes):
        name = code_to_name.get(code, code)
        news = fetch_all_news_for_stock(code, name=name)
        event_news = [n for n in news if n.get("news_type") == "event"]
        safe_print(f"[{i+1}/{len(codes)}] {code} {name}: {len(news)} 条 (event: {len(event_news)})")
        all_news.extend(news)
        if i < len(codes) - 1:
            time.sleep(delay)

    # 保存新闻缓存
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)
    safe_print(f"\n新闻缓存已保存: {NEWS_CACHE_FILE} ({len(all_news)} 条)")

    return all_news


# =====================================================================
#  Step 2: 并发 LLM 分析
# =====================================================================

def analyze_single_news(analyzer: NewsAnalyzer, news: dict) -> list[dict]:
    """分析单条新闻，返回提取的关系"""
    text = news.get("title", "")
    content = news.get("content", "")
    if content:
        text = f"{text}。{content}"
    text = text[:300]

    if len(text) < 10:
        return []

    try:
        return analyzer.analyze(text)
    except Exception as e:
        return []


BATCH_SIZE = 5  # 每次API调用分析的新闻数


def _prepare_text(news: dict) -> str:
    """准备单条新闻文本"""
    text = news.get("title", "")
    content = news.get("content", "")
    if content:
        text = f"{text}。{content}"
    return text[:200]


def batch_analyze(news_list: list[dict], workers: int = 3, done_titles: set = None) -> list[dict]:
    """并发 LLM 分析（批量模式：每次API调用分析5条新闻），支持断点续传"""
    api_key = get_api_key()
    if not api_key:
        print("错误: 未配置 API key")
        sys.exit(1)

    base_url = get_base_url()

    # 过滤已处理的
    if done_titles:
        news_list = [n for n in news_list if n.get("title", "") not in done_titles]

    if not news_list:
        print("所有新闻已分析完毕，无需重跑。")
        return []

    # 分批：每 BATCH_SIZE 条为一组
    batches = []
    for i in range(0, len(news_list), BATCH_SIZE):
        batches.append(news_list[i:i + BATCH_SIZE])

    total_news = len(news_list)
    print(f"\n待分析: {total_news} 条新闻, 分为 {len(batches)} 批 (每批{BATCH_SIZE}条), 并发线程: {workers}")

    all_results = []
    processed_news = 0
    processed_batches = 0
    failed = 0
    start_time = time.time()

    def worker_fn(batch):
        """处理一批新闻"""
        analyzer = NewsAnalyzer(api_key=api_key, base_url=base_url)
        texts = [_prepare_text(n) for n in batch]
        texts = [t for t in texts if len(t) >= 10]
        if not texts:
            return [], batch

        try:
            results = analyzer.analyze_batch_multi(texts)
            # 标注来源
            for r in results:
                r["from_stock"] = batch[0].get("stock_code", "")
            return results, batch
        except Exception:
            return [], batch

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, batch): batch for batch in batches}

        for future in as_completed(futures):
            processed_batches += 1
            try:
                results, batch = future.result()
                processed_news += len(batch)
                if results:
                    # 给每个结果标注新闻标题（用batch中第一条）
                    for r in results:
                        if "news_title" not in r:
                            r["news_title"] = batch[0].get("title", "")
                    all_results.extend(results)
            except Exception:
                failed += 1

            # 进度打印
            if processed_batches % 20 == 0 or processed_batches == len(batches):
                elapsed = time.time() - start_time
                rate = processed_news / elapsed if elapsed > 0 else 0
                eta = (total_news - processed_news) / rate if rate > 0 else 0
                safe_print(f"  进度: {processed_news}/{total_news} ({processed_batches}/{len(batches)}批) | "
                          f"提取关系: {len(all_results)} | "
                          f"失败: {failed} | "
                          f"速度: {rate:.1f}条/s | "
                          f"ETA: {eta:.0f}s")

            # 每50批保存一次（断点续传）
            if processed_batches % 50 == 0:
                _save_results(all_results)
                _save_progress(news_list[:processed_news])

    _save_results(all_results)
    _save_progress(news_list)
    elapsed = time.time() - start_time
    print(f"\n分析完成! 耗时: {elapsed:.1f}s, 提取关系: {len(all_results)} 条")

    return all_results


def _save_results(results):
    """保存提取结果（增量追加）"""
    existing = []
    if RELATIONS_FILE.exists():
        try:
            existing = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 合并去重
    seen = set()
    merged = []
    for r in existing + results:
        key = (r.get("source", ""), r.get("target", ""), r.get("relation", ""), r.get("news_title", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    with open(RELATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _save_progress(processed_news):
    """保存已处理新闻标题（用于断点续传）"""
    titles = [n.get("title", "") for n in processed_news]
    existing = set()
    if PROGRESS_FILE.exists():
        try:
            existing = set(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    existing.update(titles)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(existing), f, ensure_ascii=False)


def _load_progress() -> set:
    """加载已处理新闻标题"""
    if PROGRESS_FILE.exists():
        try:
            return set(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


# =====================================================================
#  主入口
# =====================================================================

if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="批量新闻→知识图谱处理")
    parser.add_argument("--top", type=int, default=0, help="只处理前N只股票（测试）")
    parser.add_argument("--workers", type=int, default=3, help="LLM并发线程数")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--fetch-only", action="store_true", help="只抓新闻不分析")
    parser.add_argument("--analyze-only", action="store_true", help="只分析（用缓存新闻）")
    args = parser.parse_args()

    codes = get_all_codes()
    if args.top > 0:
        codes = codes[:args.top]

    print(f"=" * 60)
    print(f"批量新闻→知识图谱处理")
    print(f"股票数: {len(codes)}, 并发: {args.workers}, 续传: {args.resume}")
    print(f"=" * 60)

    # Step 1: 抓取新闻
    if not args.analyze_only:
        print(f"\n[Step 1/2] 批量抓取新闻...")
        all_news = batch_fetch_news(codes, delay=0.3)
    else:
        if NEWS_CACHE_FILE.exists():
            all_news = json.loads(NEWS_CACHE_FILE.read_text(encoding="utf-8"))
            print(f"[Step 1/2] 从缓存加载 {len(all_news)} 条新闻")
        else:
            print("错误: 无新闻缓存，请先不带 --analyze-only 运行")
            sys.exit(1)

    if args.fetch_only:
        print("仅抓取模式，跳过LLM分析。")
        sys.exit(0)

    # 只分析 event 类新闻
    event_news = [n for n in all_news if n.get("news_type") == "event"]
    print(f"\n总新闻: {len(all_news)}, event类: {len(event_news)}")

    # Step 2: LLM 分析
    done_titles = _load_progress() if args.resume else set()
    if done_titles:
        print(f"[续传] 已处理 {len(done_titles)} 条，跳过")

    print(f"\n[Step 2/2] LLM 并发分析...")
    results = batch_analyze(event_news, workers=args.workers, done_titles=done_titles)

    print(f"\n{'=' * 60}")
    print(f"全部完成!")
    print(f"  关系文件: {RELATIONS_FILE}")
    if RELATIONS_FILE.exists():
        all_rels = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
        print(f"  总提取关系: {len(all_rels)} 条")
    print(f"{'=' * 60}")
