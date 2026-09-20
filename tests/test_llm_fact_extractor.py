"""perception.llm_extraction.LLMFactExtractor 离线单元测试。

覆盖：
  - available 属性
  - refine：无客户端 / 空内容 / LLM 抛异常 / 空返回 → None
  - refine：正常返回 + 参数透传（max_tokens/temperature）
  - _parse_response：JSON 数组、嵌入数组、非数组、字段过滤、类型回退
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omnimem.perception.llm_extraction import LLMFactExtractor


class _FakeLLM:
    def __init__(self, response: Any = None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.kwargs: list[dict[str, Any]] = []

    def call_sync(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if self._response is None:
            return None
        if isinstance(self._response, str):
            return SimpleNamespace(content=self._response)
        return self._response


# ── available ──


def test_available_true_when_client_present() -> None:
    assert LLMFactExtractor(_FakeLLM()).available is True


def test_available_false_when_client_none() -> None:
    assert LLMFactExtractor(None).available is False


# ── refine 前置 ──


def test_refine_returns_none_when_no_client() -> None:
    assert LLMFactExtractor(None).refine("hello world this is content") is None


def test_refine_returns_none_for_blank_content() -> None:
    ex = LLMFactExtractor(_FakeLLM())
    assert ex.refine("   ") is None


# ── refine 主路径 ──


def test_refine_passes_prompts_and_params() -> None:
    llm = _FakeLLM('[]')
    ex = LLMFactExtractor(llm, max_tokens=256, temperature=0.1)
    ex.refine("用户偏好深色主题", "规则提示")
    call = llm.kwargs[0]
    assert call["max_tokens"] == 256
    assert call["temperature"] == 0.1
    assert "用户偏好深色主题" in call["prompt"]
    assert "规则提示" in call["prompt"]


def test_refine_truncates_long_content() -> None:
    llm = _FakeLLM('[]')
    ex = LLMFactExtractor(llm)
    ex.refine("a" * 2000, "b" * 500)
    prompt = llm.kwargs[0]["prompt"]
    # content 800 上限；rule_hint 200 上限
    assert prompt.count("a" * 200) < 5  # 不可能出现 5 段 200 连续字符
    assert "b" * 201 not in prompt


def test_refine_rule_hint_empty_uses_placeholder() -> None:
    llm = _FakeLLM('[]')
    ex = LLMFactExtractor(llm)
    ex.refine("用户内容 abcdefg", "")
    assert "（无）" in llm.kwargs[0]["prompt"]


def test_refine_returns_none_when_llm_raises() -> None:
    ex = LLMFactExtractor(_FakeLLM(raises=RuntimeError("boom")))
    assert ex.refine("用户消息 abc") is None


def test_refine_returns_none_when_result_none() -> None:
    ex = LLMFactExtractor(_FakeLLM(None))
    assert ex.refine("用户消息 abc") is None


def test_refine_returns_none_when_content_empty() -> None:
    ex = LLMFactExtractor(_FakeLLM(SimpleNamespace(content="")))
    assert ex.refine("用户消息 abc") is None


def test_refine_returns_parsed_on_valid_json() -> None:
    payload = '[{"content": "用户偏好深色主题", "type": "preference"}]'
    ex = LLMFactExtractor(_FakeLLM(payload))
    out = ex.refine("用户消息 abcdefgh")
    assert out == [{"content": "用户偏好深色主题", "type": "preference"}]


# ── _parse_response ──


def test_parse_returns_none_without_array() -> None:
    assert LLMFactExtractor._parse_response("no json here") is None


def test_parse_returns_none_on_malformed_json() -> None:
    assert LLMFactExtractor._parse_response('[{"content": "abc" broken') is None


def test_parse_returns_none_for_non_list_json() -> None:
    # 正则 \[[\s\S]*\] 贪婪要求以 ] 结尾；输入不含 ] → 无匹配
    assert LLMFactExtractor._parse_response('{"content": "abc"}') is None


def test_parse_embedded_array_extraction() -> None:
    raw = '这是我的答案：[{"content": "用户喜欢 Python", "type": "fact"}] 完毕'
    out = LLMFactExtractor._parse_response(raw)
    assert out == [{"content": "用户喜欢 Python", "type": "fact"}]


def test_parse_skips_non_dict_items() -> None:
    out = LLMFactExtractor._parse_response('["str", 42, {"content": "有效事实内容", "type": "fact"}]')
    assert out == [{"content": "有效事实内容", "type": "fact"}]


def test_parse_rejects_content_too_short() -> None:
    # content 长度 <4 → 丢弃
    assert LLMFactExtractor._parse_response('[{"content": "abc", "type": "fact"}]') is None


def test_parse_rejects_content_too_long() -> None:
    assert LLMFactExtractor._parse_response(
        '[{"content": "' + "x" * 210 + '", "type": "fact"}]'
    ) is None


def test_parse_defaults_invalid_type_to_fact() -> None:
    out = LLMFactExtractor._parse_response('[{"content": "有效事实", "type": "nonsense"}]')
    assert out == [{"content": "有效事实", "type": "fact"}]


def test_parse_lowercases_type() -> None:
    out = LLMFactExtractor._parse_response('[{"content": "有效事实", "type": "PREFERENCE"}]')
    assert out == [{"content": "有效事实", "type": "preference"}]


def test_parse_caps_at_three_items() -> None:
    payload = (
        '[{"content": "aaaa bbbb", "type": "fact"},'
        ' {"content": "cccc dddd", "type": "fact"},'
        ' {"content": "eeee ffff", "type": "fact"},'
        ' {"content": "gggg hhhh", "type": "fact"}]'
    )
    out = LLMFactExtractor._parse_response(payload)
    assert out is not None
    assert len(out) == 3


def test_parse_returns_none_when_all_items_rejected() -> None:
    # 三条都太短 → facts 空 → None
    assert LLMFactExtractor._parse_response(
        '[{"content": "ab"}, {"content": "cd"}, {"content": "ef"}]'
    ) is None


def test_parse_defaults_type_when_missing() -> None:
    out = LLMFactExtractor._parse_response('[{"content": "有效事实内容"}]')
    assert out == [{"content": "有效事实内容", "type": "fact"}]


@pytest.mark.parametrize(
    "raw,expect_len",
    [
        ('[{"content": "有效事实", "type": "fact"}]', 1),
        ('[{"content": "有效事实内容", "type": "preference"}, {"content": "另一条事实", "type": "correction"}]', 2),
    ],
)
def test_parse_parametrized_shapes(raw: str, expect_len: int) -> None:
    out = LLMFactExtractor._parse_response(raw)
    assert out is not None
    assert len(out) == expect_len
