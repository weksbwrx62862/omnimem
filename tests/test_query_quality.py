"""垃圾查询检测与 Token 预算裁剪单元测试。"""

from __future__ import annotations

import pytest

from omnimem.retrieval.query_quality import is_garbage_query, trim_to_budget


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", True),
        ("a", True),
        ("12345", True),
        ("111", True),
        ("!!!@#$%", True),
        ("zzzzzzxyz", True),
        ("qwertyuiop", True),
        ("啊啊啊啊", True),
        ("测测测", True),
        ("测试测试测试", True),
        ("how are you", False),
        ("用户偏好是什么", False),
        ("Python 和 JavaScript 区别", False),
    ],
)
def test_is_garbage_query_cases(query: str, expected: bool) -> None:
    """常见垃圾/正常查询判定。"""
    assert is_garbage_query(query) is expected


def test_garbage_query_common_word_threshold() -> None:
    """仅包含常见词且长度<=3 应判定为垃圾。"""
    assert is_garbage_query("how") is True


def test_garbage_query_noise_ratio() -> None:
    """主体为噪声字符时判定为垃圾。"""
    assert is_garbage_query("what!!!@#$") is True


def test_garbage_query_chinese_repeat_group() -> None:
    """中文重复词组应判定为垃圾。"""
    assert is_garbage_query("测试测试测试内容") is True


def test_garbage_query_english_vowelless() -> None:
    """无元音的长英文字母串应判定为垃圾。"""
    assert is_garbage_query("bcdfg") is True


def test_garbage_query_only_digits_spaces() -> None:
    """纯数字/空格串应判定为垃圾。"""
    assert is_garbage_query("123 456") is True


def test_trim_to_budget_respects_limit() -> None:
    """trim_to_budget 应裁剪结果到 Token 预算内。"""
    results = [
        {"content": "a" * 40},
        {"content": "b" * 40},
        {"content": "c" * 40},
    ]
    trimmed = trim_to_budget(results, max_tokens=15)
    assert len(trimmed) == 1
    assert trimmed[0]["content"].startswith("a")


def test_trim_to_budget_empty_input() -> None:
    """空结果列表应返回空列表。"""
    assert trim_to_budget([], max_tokens=100) == []


def test_trim_to_budget_exact_budget() -> None:
    """结果刚好占满预算时应全部保留。"""
    results = [{"content": "a" * 40}]
    trimmed = trim_to_budget(results, max_tokens=10)
    assert len(trimmed) == 1
