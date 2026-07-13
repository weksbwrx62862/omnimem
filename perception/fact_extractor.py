"""AtomicFactExtractor — LLM 驱动的原子事实提取。

对标 Mem0 的分层提取策略：从一条对话中提取多条独立原子事实，
每条事实只包含一个信息点，提升检索命中率。

示例：
  输入: "我喜欢 Python，尤其是 3.11 版本"
  输出: ["用户喜欢 Python", "用户偏好 Python 3.11 版本"]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omnimem.utils.llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)

# 原子事实提取 Prompt
_FACT_EXTRACTION_SYSTEM = """你是一个事实提取专家。从给定文本中提取独立的原子事实。

规则：
1. 每条事实只包含一个信息点（主语+谓语+宾语）
2. 保持原始信息，不推断不编造
3. 用第三人称表述（如"用户喜欢..."而非"我喜欢..."）
4. 输出 JSON 数组格式
5. 如果文本中没有可提取的事实，返回空数组 []

示例：
输入: "我喜欢 Python，尤其是 3.11 版本"
输出: ["用户喜欢 Python", "用户偏好 Python 3.11 版本"]

输入: "我搬到柏林了"
输出: ["用户搬到柏林"]

输入: "好的"
输出: []"""

_FACT_EXTRACTION_USER = """从以下文本中提取独立的原子事实：

文本: {content}

原子事实（JSON 数组）:"""


class AtomicFactExtractor:
    """LLM 驱动的原子事实提取器。"""

    def __init__(
        self,
        llm_client: AsyncLLMClient | None = None,
        fallback_extractor: Any | None = None,
    ) -> None:
        """初始化原子事实提取器。

        Args:
            llm_client: LLM 客户端，None 时使用回退方法
            fallback_extractor: 回退提取器（如 PerceptionEngine._extract_core_fact）
        """
        self._llm_client = llm_client
        self._fallback_extractor = fallback_extractor

    def extract_facts(self, content: str) -> list[str]:
        """从内容中提取原子事实。

        Args:
            content: 输入文本

        Returns:
            原子事实列表，每条事实只含一个信息点
        """
        if not content or len(content.strip()) < 5:
            return []

        # 短文本（<=20字）不需要 LLM 提取，直接作为单条事实
        if len(content.strip()) <= 20:
            return [content.strip()]

        # 尝试 LLM 提取
        facts = self._extract_with_llm(content)
        if facts is not None:
            return facts

        # LLM 失败，回退到正则方法
        return self._extract_with_fallback(content)

    def _extract_with_llm(self, content: str) -> list[str] | None:
        """使用 LLM 提取原子事实。

        Returns:
            原子事实列表，LLM 失败返回 None
        """
        if self._llm_client is None:
            return None

        try:
            user_prompt = _FACT_EXTRACTION_USER.format(content=content)
            result = self._llm_client.call_sync(
                prompt=user_prompt,
                system=_FACT_EXTRACTION_SYSTEM,
                max_tokens=300,
                temperature=0.0,
                use_cache=False,
            )

            if not result or not result.content:
                return None

            # 解析 JSON 数组
            raw = result.content.strip()
            facts = self._parse_json_facts(raw)
            if facts is not None and len(facts) > 0:
                logger.debug("LLM 提取 %d 条原子事实: %s", len(facts), facts[:3])
                return facts

            return None

        except Exception as e:
            logger.warning("LLM 原子事实提取失败: %s", e)
            return None

    @staticmethod
    def _parse_json_facts(raw: str) -> list[str] | None:
        """解析 LLM 返回的 JSON 数组。"""
        # 尝试直接解析
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(f).strip() for f in parsed if str(f).strip()]
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 数组部分
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(f).strip() for f in parsed if str(f).strip()]
            except json.JSONDecodeError:
                pass

        # 尝试按行分割（兼容非 JSON 格式）
        lines = [line.strip().lstrip('- •*0123456789.').strip() for line in raw.split('\n')]
        facts = [line for line in lines if line and len(line) >= 5]
        if facts:
            return facts

        return None

    def _extract_with_fallback(self, content: str) -> list[str]:
        """使用回退方法提取事实。"""
        if self._fallback_extractor is not None:
            try:
                fact = self._fallback_extractor(content)
                if fact:
                    return [fact]
            except Exception as e:
                logger.warning("回退事实提取失败: %s", e)

        # 最终回退：按句子分割
        sentences = re.split(r'[。！？.!?;\n]', content)
        facts = [s.strip() for s in sentences if len(s.strip()) >= 10]
        return facts if facts else [content[:100]]
