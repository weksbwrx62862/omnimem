"""LLMFactExtractor — L0 感知层的 LLM 事实精炼（hybrid 抽取模式）。

extraction_mode 配置项（默认 hybrid）:
  - rule:   仅规则抽取（原有行为，零 LLM 开销）
  - hybrid: 规则信号触发 should_memorize 后，用 LLM 精炼事实内容与类型；
            LLM 不可用或失败时静默回退规则结果
  - llm:    语义同 hybrid（保留区分位，便于未来实现纯 LLM 信号检测）

设计约束:
  - 仅在 should_memorize 为真的少数轮次调用 LLM，不增加常规轮次延迟
  - 复用 AsyncLLMClient 的并发限制与 TTL 缓存
  - 任何异常都不得中断写入主流程（返回 None 由调用方回退）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 允许的记忆类型（与 mcp_server 工具 schema 保持一致）
_ALLOWED_TYPES = {
    "fact", "preference", "correction", "skill",
    "procedural", "event", "action", "reasoning",
}

_SYSTEM_PROMPT = (
    "You are a memory extraction engine for an AI agent. "
    "Extract atomic, self-contained facts worth remembering long-term. "
    "Respond with a JSON array only, no prose."
)

_PROMPT_TEMPLATE = """从下面的用户消息中抽取值得长期记忆的原子事实。

要求：
- 每条事实独立完整、简洁（≤50字），第三人称陈述（如"用户偏好深色主题"）
- type 从以下选择：fact / preference / correction / skill / procedural / event
- 跳过寒暄、疑问句、临时性内容；没有值得记忆的内容时返回 []
- 最多抽取 3 条

用户消息：
{content}

规则引擎初步抽取（供参考，可修正）：
{rule_hint}

返回 JSON 数组：[{{"content": "...", "type": "..."}}]
只返回 JSON。"""


class LLMFactExtractor:
    """LLM 事实精炼器。持有 AsyncLLMClient 兼容对象（需提供 call_sync）。"""

    def __init__(self, llm_client: Any, max_tokens: int = 400, temperature: float = 0.2):
        self._llm_client = llm_client
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def available(self) -> bool:
        return self._llm_client is not None

    def refine(self, user_content: str, rule_fact: str = "") -> list[dict[str, str]] | None:
        """用 LLM 精炼事实。

        Args:
            user_content: 清洗后的用户消息
            rule_fact: 规则引擎的初步抽取结果（作为提示传给 LLM）

        Returns:
            [{"content": ..., "type": ...}]；失败或无产出返回 None（调用方回退规则）
        """
        if self._llm_client is None or not user_content.strip():
            return None
        try:
            prompt = _PROMPT_TEMPLATE.format(
                content=user_content[:800],
                rule_hint=rule_fact[:200] or "（无）",
            )
            result = self._llm_client.call_sync(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            raw = result.content if result and getattr(result, "content", "") else ""
            if not raw:
                return None
            return self._parse_response(raw)
        except Exception as e:
            logger.warning("LLMFactExtractor.refine failed (fallback to rule): %s", e)
            return None

    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, str]] | None:
        """解析 LLM 返回的 JSON 数组，校验类型与内容长度。"""
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return None
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list):
            return None

        facts: list[dict[str, str]] = []
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            mtype = str(item.get("type", "fact")).strip().lower()
            # 内容长度门槛：过短无信息量，过长非原子事实
            if not content or len(content) < 4 or len(content) > 200:
                continue
            if mtype not in _ALLOWED_TYPES:
                mtype = "fact"
            facts.append({"content": content, "type": mtype})
        return facts or None
