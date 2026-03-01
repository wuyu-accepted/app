# -*- coding: utf-8 -*-
"""
llm_reporter.py

大模型自然语言生成模块：负责将 Agent（及其 LSTM 融合后）的量化输出转化为自然语言报告。
通过调用 LLM API，解释最终的交易决策（Action）、分值（Score）及其背后的状态流转机制。
"""

import os
from pathlib import Path
import anthropic
from typing import Dict, Any

# 读取 knowledge_graph/.env 统一配置
_ENV_FILE = Path(__file__).resolve().parent.parent / "knowledge_graph" / ".env"

def _load_env() -> dict[str, str]:
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

class LLMReporter:
    """
    接收最终门控计算出来的信号字典，将其拼接成 Prompt 交由 LLM 转换为人类可读的投资建议报告。
    """
    def __init__(self, api_key: str = None, model: str = "claude-haiku-4-5-20251001", base_url: str = None):
        """
        初始化 LLM 报告生成器。

        参数:
            api_key (str): 大模型 API 密钥。如果为空则从 .env 或环境变量读取。
            model (str): 使用的 LLM 模型名称。
            base_url (str): API 的基准 URL。如果为空则从 .env 或环境变量读取。
        """
        config = _load_env()
        self.api_key = api_key or config.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or config.get("ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"

        if not self.api_key:
            raise ValueError("未找到 API_KEY，请在 knowledge_graph/.env 中配置 ANTHROPIC_API_KEY。")

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            auth_token=None,
        )
        self.model = model

    def generate_prompt(self, stock: str, fusion_result: Dict[str, Any]) -> str:
        """
        构造用于请求 LLM 的提示词 (Prompt)。
        """
        prompt = f"""你是一个资深的量化金融分析师，你的主要任务是将我们AI投资系统的机器输出参数转化成通俗易懂、逻辑严密的自然语言投资报告。

                目前针对股票【{stock}】的AI融合模型评估结果如下：
                - 模型评估量化总分 (Final Score)：{fusion_result.get('final_score')} (范围在 -1.0 到 1.0 之间，1.0为最强买入信号)
                - 系统给出的交易动作 (Action)：{fusion_result.get('action')} (选项包含 STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
                - 融合引擎当前触发的状态 (Status)：{fusion_result.get('status')}
                - 使用的融合策略模式 (Mode)：{fusion_result.get('mode')}
                - 权重分配 (Weights)：
                * 纯量化/LSTM 侧权重 (w_lstm)：{fusion_result.get('weights', {}).get('w_lstm')}
                * 多跳图谱/事件 Agent 侧权重 (w_agent)：{fusion_result.get('weights', {}).get('w_agent')}

                要求：
                1. 第一部分直接给出交易结论（Action）与个股评级。
                2. 第二部分根据"评估量化总分"以及"触发状态（Status）"和"权重分配"，解释做出该决策的核心原因。
                如果是"核弹级事件/断档接管"，请着重强调近期强突发影响超越了技术结构。
                如果是常规模式/交叉注意力等模式，请说明技术面评估（LSTM）和事件信息传导（Agent）是如何有机结合且谁占主导的。
                3. 语言需专业、客观，作为研究分享使用，并在末尾加入适当且简短的风险提示。
                4. 整体输出格式要求清晰，不要过于冗长。
                """
        return prompt

    def get_natural_language_report(self, stock: str, fusion_result: Dict[str, Any]) -> str:
        """
        调用 LLM API 生成自然语言投资报告。
        """
        prompt = self.generate_prompt(stock, fusion_result)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=[
                    {"role": "user", "content": "你是一位专注于人工智能选股与量化分析的金融分析师。\n\n" + prompt}
                ],
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"LLM API 调用失败: {str(e)}"

if __name__ == "__main__":
    mock_fusion_res = {
        "final_score": 0.8521,
        "status": "核弹级事件/断档接管 (Math)",
        "weights": {"w_lstm": 0.2, "w_agent": 0.8},
        "action": "STRONG BUY",
        "mode": "math"
    }

    reporter = LLMReporter()
    print("-------------------- 构造的提示词如下 --------------------\n")
    print(reporter.generate_prompt("英伟达 (NVDA)", mock_fusion_res))
    print("\n----------------------------------------------------------")

    print("\n\n正在调用 LLM API 进行真实的自然语言生成测试，请稍候...")
    report = reporter.get_natural_language_report("英伟达 (NVDA)", mock_fusion_res)
    print("\n-------------------- 生成的自然语言报告 --------------------\n")
    print(report)
    print("\n----------------------------------------------------------")
