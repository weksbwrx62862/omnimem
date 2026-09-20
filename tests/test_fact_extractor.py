"""perception.fact_extractor.AtomicFactExtractor 离线单元测试。

覆盖：
  - extract_facts 短文本快速通道（<=20 字）
  - LLM 主路径 + JSON/行了列表多种解析回退
  - LLM 失败/异常 → 回退到 fallback_extractor
  - fallback 缺失 → 按句子/标点分割
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omnimem.perception.fact_extractor import AtomicFactExtractor


class _FakeLLM:
    """最小 AsyncLLMClient 替身：返回固定文本或抛异常。"""

    def __init__(self, response_text: str | None = None, raises: Exception | None = None) -> None:
        self._response_text = response_text
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def call_sync(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if self._response_text is None:
            return None
        return SimpleNamespace(content=self._response_text)


# ── extract_facts 前置过滤 ──


def test_extract_facts_empty_returns_empty() -> None:
    assert AtomicFactExtractor().extract_facts("") == []


def test_extract_facts_whitespace_only_returns_empty() -> None:
    assert AtomicFactExtractor().extract_facts("   \n  ") == []


def test_extract_facts_short_input_returns_empty() -> None:
    # strip 后 <5 字符
    assert AtomicFactExtractor().extract_facts("你好") == []


def test_extract_facts_medium_input_returns_single_fact() -> None:
    # 5-20 字符：直接原样返回（不触发 LLM）
    llm = _FakeLLM(response_text='["x"]')
    ex = AtomicFactExtractor(llm_client=llm)
    out = ex.extract_facts("用户喜欢 Python")
    assert out == ["用户喜欢 Python"]
    assert llm.calls == []


# ── _extract_with_llm 主路径 ──


def test_extract_with_llm_json_array() -> None:
    llm = _FakeLLM(response_text='["用户喜欢 Python", "用户偏好 3.11 版本"]')
    ex = AtomicFactExtractor(llm_client=llm)
    out = ex.extract_facts("我喜欢 Python，特别是 3.11 版本，因为性能提升了很多")
    assert out == ["用户喜欢 Python", "用户偏好 3.11 版本"]
    # 使用缓存关闭 + temperature 0
    call = llm.calls[0]
    assert call["use_cache"] is False
    assert call["temperature"] == 0.0


def test_extract_with_llm_embedded_json_array() -> None:
    # 前后带噪声文本，正则提取 [...] 部分
    llm = _FakeLLM(response_text='这是我的答案：\n["用户喜欢 Python", "用户偏好 3.11"]\n完毕')
    ex = AtomicFactExtractor(llm_client=llm)
    out = ex.extract_facts("我喜欢 Python，特别是 3.11 版本，因为性能有了提升")
    assert out == ["用户喜欢 Python", "用户偏好 3.11"]


def test_extract_with_llm_line_split_fallback_when_not_json() -> None:
    # 非 JSON、非数组 → 按行解析（去除列表前缀符号）
    llm = _FakeLLM(response_text="- 用户喜欢 Python 语言\n- 用户偏好 3.11 版本")
    ex = AtomicFactExtractor(llm_client=llm)
    out = ex.extract_facts("我喜欢 Python，特别是 3.11 版本，性能提升非常明显")
    assert any("用户喜欢 Python 语言" in f for f in out)
    assert any("用户偏好 3.11 版本" in f for f in out)


def test_extract_with_llm_returns_none_on_empty_response() -> None:
    # LLM 返回空字符串 → 视为 None → 走 fallback（未提供 → 句子分割）
    llm = _FakeLLM(response_text="   ")
    ex = AtomicFactExtractor(llm_client=llm)
    out = ex.extract_facts("我喜欢 Python；也写 JavaScript；两者各有所长")
    # 无 fallback → 分割句子（每句 >=10 字符）
    assert isinstance(out, list)


def test_extract_with_llm_returns_none_on_error_and_falls_back() -> None:
    llm = _FakeLLM(raises=RuntimeError("boom"))
    called = {"n": 0}

    def fallback(content: str) -> str:
        called["n"] += 1
        return f"回退事实：{content[:8]}"

    ex = AtomicFactExtractor(llm_client=llm, fallback_extractor=fallback)
    out = ex.extract_facts("这是一段比较长的中文文本，用来测试回退路径是否被调用")
    assert out == ["回退事实：这是一段比较长的"]
    assert called["n"] == 1


# ── _parse_json_facts ──


def test_parse_json_facts_plain_array() -> None:
    assert AtomicFactExtractor._parse_json_facts('["a", "b"]') == ["a", "b"]


def test_parse_json_facts_filters_empty_strings() -> None:
    assert AtomicFactExtractor._parse_json_facts('["a", "", "   "]') == ["a"]


def test_parse_json_facts_coerces_non_strings() -> None:
    assert AtomicFactExtractor._parse_json_facts("[1, 2.0, true]") == ["1", "2.0", "True"]


def test_parse_json_facts_embedded_array() -> None:
    out = AtomicFactExtractor._parse_json_facts('前缀 ["x", "y"] 后缀')
    assert out == ["x", "y"]


def test_parse_json_facts_line_mode_when_not_json() -> None:
    out = AtomicFactExtractor._parse_json_facts("- 用户喜欢 Python\n- 用户偏好 3.11 版本")
    assert len(out) == 2


def test_parse_json_facts_returns_line_list_for_non_list_json() -> None:
    # 非数组的合法 JSON → json.loads 返回 dict（不是 list）→ 无 [...] 匹配 →
    # 走行模式，长度 ≥5 的整行原样保留
    assert AtomicFactExtractor._parse_json_facts('{"a": 1}') == ['{"a": 1}']


def test_parse_json_facts_returns_none_when_no_long_enough_line() -> None:
    assert AtomicFactExtractor._parse_json_facts("ab\ncd") is None


# ── _extract_with_fallback ──


def test_fallback_uses_injector_when_available() -> None:
    ex = AtomicFactExtractor(fallback_extractor=lambda _c: "注入的事实")
    out = ex._extract_with_fallback("这是一段较长的中文文本，应该走注入器路径")
    assert out == ["注入的事实"]


def test_fallback_swallows_injector_exception_and_uses_sentences() -> None:
    def boom(_c: str) -> str:
        raise ValueError("nope")

    ex = AtomicFactExtractor(fallback_extractor=boom)
    content = "第一句是足够长的内容用于测试。第二句也是足够长的内容用于测试。"
    out = ex._extract_with_fallback(content)
    assert isinstance(out, list)
    assert out  # 至少一条


def test_fallback_returns_original_when_no_sentences_long_enough() -> None:
    # 全部句子都 <10 字符 → 兜底返回原文前 100 字符
    ex = AtomicFactExtractor()
    out = ex._extract_with_fallback("短句。也更短。")
    assert out == ["短句。也更短。"]


def test_fallback_splits_on_multiple_punctuation() -> None:
    ex = AtomicFactExtractor()
    content = "这是第一句足够长度的内容。这是第二句足够长度的内容！这是第三句足够长度的内容？"
    out = ex._extract_with_fallback(content)
    assert len(out) >= 2


def test_fallback_splits_on_newline() -> None:
    ex = AtomicFactExtractor()
    content = "第一行有足够长度的内容\n第二行也有足够长度的内容\n第三行同样足够长度内容啊"
    out = ex._extract_with_fallback(content)
    assert len(out) >= 2


# ── 默认无客户端 ──


def test_no_llm_client_skips_directly_to_fallback() -> None:
    ex = AtomicFactExtractor(fallback_extractor=lambda _c: "回退")
    out = ex.extract_facts("这是一段比较长的中文文本，用于测试无 LLM 客户端场景")
    assert out == ["回退"]


def test_repr_or_sanity_of_instance() -> None:
    ex = AtomicFactExtractor()
    assert isinstance(ex._llm_client, type(None))
    assert ex._fallback_extractor is None


@pytest.mark.parametrize(
    ("raw", "expected_first"),
    [
        ('["用户喜欢 Python"]', "用户喜欢 Python"),
        ('结果：["用户喜欢 Python"]', "用户喜欢 Python"),
    ],
)
def test_parse_json_facts_parametrized(raw: str, expected_first: str) -> None:
    out = AtomicFactExtractor._parse_json_facts(raw)
    assert out is not None
    assert out[0] == expected_first
