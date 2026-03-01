"""
LLM 新闻分析模块 — 用 Claude API 从新闻中提取公司关系和 sentiment
"""

import json
import re
from pathlib import Path

import anthropic

from knowledge_graph.stock_pool import get_all_stocks_flat

# =====================================================================
#  从 .env 文件读取配置
# =====================================================================

_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_env() -> dict[str, str]:
    """读取 knowledge_graph/.env 文件中的配置"""
    config = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


def get_api_key() -> str:
    """获取 API key"""
    import os
    config = _load_env()
    key = (config.get("ANTHROPIC_API_KEY")
           or os.environ.get("ANTHROPIC_API_KEY")
           or "")
    if not key or not key.isascii() or not key.startswith("sk-"):
        return ""
    return key


def get_base_url() -> str | None:
    """获取 API base_url"""
    config = _load_env()
    return config.get("ANTHROPIC_BASE_URL") or None


# =====================================================================
#  股票池信息（嵌入 prompt）
# =====================================================================

def _build_stock_context() -> str:
    lines = []
    for code, name, sector in get_all_stocks_flat():
        lines.append(f"{code}={name}")
    return "、".join(lines)


STOCK_CONTEXT = _build_stock_context()

# =====================================================================
#  Prompt 模板
# =====================================================================

SYSTEM_PROMPT = f"""你是JSON提取器。从新闻中提取股票池内公司之间的关系。

规则：
1. 只输出JSON数组，禁止任何其他文字
2. 只关注以下50只股票之间的关系：{STOCK_CONTEXT}
3. 关系类型（5选1）：supply=供应链(有向)、compete=竞争、peer=板块联动、invest=控股(有向)、cooperate=合作
4. sentiment∈[-1,1]：正=利好，负=利空
5. 无关系输出[]

JSON格式：[{{"source":"股票代码","target":"股票代码","relation":"类型","sentiment":数值,"description":"一句话"}}]"""

USER_PROMPT_TEMPLATE = """新闻：{news_text}

输出JSON："""


# =====================================================================
#  分析器
# =====================================================================

class NewsAnalyzer:
    """用 Claude API 分析新闻，提取公司关系和 sentiment"""

    def __init__(self, api_key: str, base_url: str | None = None,
                 model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com",
            auth_token=None,
        )
        self.model = model

    def analyze(self, news_text: str, max_retries: int = 2) -> list[dict]:
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(news_text=news_text)},
                        {"role": "assistant", "content": "["},
                    ],
                )
                raw = "[" + message.content[0].text.strip()
                result = self._parse_response(raw)
                if result or "[]" in raw or raw.strip() == "[]":
                    return result
            except Exception:
                pass
        return []

    def analyze_batch_multi(self, news_items: list[str]) -> list[dict]:
        """批量分析多条新闻（一次API调用），返回合并的关系列表"""
        numbered = "\n".join(f"{i+1}. {t[:150]}" for i, t in enumerate(news_items))
        prompt = f"以下{len(news_items)}条新闻，提取所有公司关系，合并输出一个JSON数组：\n{numbered}\n\n输出JSON："

        for attempt in range(2):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "["},
                    ],
                )
                raw = "[" + message.content[0].text.strip()
                result = self._parse_response(raw)
                if result is not None:
                    return result
            except Exception:
                pass
        return []

    def _parse_response(self, raw: str) -> list[dict]:
        """解析 LLM 返回的 JSON，容忍混入的解释文字"""
        # 策略1：直接解析
        try:
            results = json.loads(raw.strip())
            if isinstance(results, list):
                return self._validate(results)
        except json.JSONDecodeError:
            pass

        # 策略2：提取代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group(1).strip())
                if isinstance(results, list):
                    return self._validate(results)
            except json.JSONDecodeError:
                pass

        # 策略3：找 [ 到 ] 之间（贪心匹配最外层）
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                results = json.loads(raw[start:end + 1])
                if isinstance(results, list):
                    return self._validate(results)
            except json.JSONDecodeError:
                pass

        # 策略4：找 { 到 } 单个对象
        m = re.search(r'\{[^{}]*"source"[^{}]*"target"[^{}]*\}', raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                return self._validate([obj])
            except json.JSONDecodeError:
                pass

        return []

    def _validate(self, results: list) -> list[dict]:
        valid = []
        required_keys = {"source", "target", "relation", "sentiment", "description"}
        valid_relations = {"supply", "compete", "peer", "invest", "cooperate"}

        for item in results:
            if not isinstance(item, dict):
                continue
            if not required_keys.issubset(item.keys()):
                continue
            if item["relation"] not in valid_relations:
                continue
            item["sentiment"] = max(-1.0, min(1.0, float(item["sentiment"])))
            valid.append(item)

        return valid


def analyze_and_update(analyzer: NewsAnalyzer, dkg, news_text: str) -> list[dict]:
    """分析一条新闻并更新动态图谱。"""
    results = analyzer.analyze(news_text)

    for r in results:
        added = dkg.add_edge(
            source_code=r["source"],
            target_code=r["target"],
            relation_type=r["relation"],
            description=r["description"],
            sentiment=r["sentiment"],
        )
        if not added:
            dkg.update_edge_sentiment(r["source"], r["target"], r["sentiment"])

    return results
