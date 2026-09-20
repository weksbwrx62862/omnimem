"""retrieval.organizer.EvidenceOrganizer 离线单元测试。

覆盖：
  - should_organize：Gated 信号词、空串
  - organize：disabled 直通、空入参、去重合并、时间排序、来源标注
  - _sort_key：多字段兜底、无时间戳排最后
  - _source_label：wing/hall/room/session 组合、metadata 兜底
  - _dedup：相似度阈值触发合并、_merged_count 计数
  - _simhash / _hamming_similarity
"""

from __future__ import annotations

from typing import Any

import pytest
from omnimem.retrieval.organizer import EvidenceOrganizer

# ── should_organize ──


def test_should_organize_empty_query() -> None:
    assert EvidenceOrganizer().should_organize("") is False


def test_should_organize_plain_query() -> None:
    assert EvidenceOrganizer().should_organize("你好啊") is False


@pytest.mark.parametrize(
    "q",
    [
        "有几个问题",
        "how many sessions",
        "按时间排序",
        "对比 A 和 B",
        "最近发生了什么",
        "第一次会议在何时",
        "统计总数",
    ],
)
def test_should_organize_gate_signals(q: str) -> None:
    assert EvidenceOrganizer().should_organize(q) is True


# ── organize 顶层 ──


def test_organize_disabled_passthrough() -> None:
    o = EvidenceOrganizer(enabled=False)
    src = [{"content": "a", "score": 1.0}]
    out = o.organize(src)
    assert out == src
    assert out[0] is src[0]  # 直通：同一 dict 对象


def test_organize_empty_results() -> None:
    assert EvidenceOrganizer().organize([]) == []


def test_organize_annotates_source_and_flag() -> None:
    results = [{"content": "x", "score": 1.0, "stored_at": "2024-01-01T00:00:00Z"}]
    out = EvidenceOrganizer().organize(results)
    assert out[0]["_organized"] is True
    assert out[0]["_source_label"] == "未知来源"


# ── _sort_key ──


def test_sort_key_reads_stored_at() -> None:
    key = EvidenceOrganizer._sort_key({"stored_at": "2024-01-01"})
    assert key == (0, "2024-01-01")


def test_sort_key_falls_back_to_metadata() -> None:
    key = EvidenceOrganizer._sort_key({"metadata": {"stored_at": "2023-06-06"}})
    assert key == (0, "2023-06-06")


def test_sort_key_missing_time_last() -> None:
    key = EvidenceOrganizer._sort_key({"content": "x"})
    assert key == (1, "")


def test_organize_sorts_ascending_by_time() -> None:
    results = [
        {"content": "late", "score": 1.0, "stored_at": "2024-06-01"},
        {"content": "no_time", "score": 1.0},
        {"content": "early", "score": 1.0, "stored_at": "2024-01-01"},
    ]
    out = EvidenceOrganizer().organize(results)
    assert [r["content"] for r in out] == ["early", "late", "no_time"]


# ── _source_label ──


def test_source_label_all_fields() -> None:
    label = EvidenceOrganizer._source_label(
        {"wing": "W", "hall": "H", "room": "R", "session_id": "abcdefghij"}
    )
    assert label == "W/H/R/会话:abcdefgh"


def test_source_label_metadata_fallback() -> None:
    label = EvidenceOrganizer._source_label({"metadata": {"wing": "MW", "room": "MR"}})
    assert label == "MW/MR"


def test_source_label_missing_all() -> None:
    assert EvidenceOrganizer._source_label({}) == "未知来源"


def test_source_label_short_session_id_safe() -> None:
    label = EvidenceOrganizer._source_label({"session_id": "abc"})
    assert label == "会话:abc"


# ── _simhash + _hamming_similarity ──


def test_simhash_empty_returns_short_zero_string() -> None:
    # 源实现空串走 "0"*16 分支（长度与主路径 64 不一致，测试锁定现状）
    assert EvidenceOrganizer._simhash("") == "0" * 16


def test_simhash_deterministic() -> None:
    a = EvidenceOrganizer._simhash("hello world this is a test")
    b = EvidenceOrganizer._simhash("hello world this is a test")
    assert a == b
    assert len(a) == 64


def test_simhash_short_text_single_gram() -> None:
    out = EvidenceOrganizer._simhash("ab")
    assert len(out) == 64


def test_hamming_similarity_identical() -> None:
    assert EvidenceOrganizer._hamming_similarity("10101", "10101") == 1.0


def test_hamming_similarity_all_different() -> None:
    assert EvidenceOrganizer._hamming_similarity("0000", "1111") == 0.0


def test_hamming_similarity_length_mismatch() -> None:
    assert EvidenceOrganizer._hamming_similarity("000", "11") == 0.0


def test_hamming_similarity_partial_diff() -> None:
    # 4 bits, 1 diff → 0.75
    assert EvidenceOrganizer._hamming_similarity("0101", "0100") == 0.75


# ── _dedup ──


def test_dedup_single_passthrough() -> None:
    o = EvidenceOrganizer()
    r = [{"content": "x", "score": 1.0}]
    assert o._dedup(r) == r


def test_dedup_merges_identical_content() -> None:
    o = EvidenceOrganizer(dedup_threshold=0.85)
    results = [
        {"content": "用户喜欢 Python 语言开发项目", "score": 0.5},
        {"content": "用户喜欢 Python 语言开发项目", "score": 0.9},
    ]
    out = o._dedup(results)
    assert len(out) == 1
    assert out[0]["score"] == 0.9
    assert out[0]["_merged_count"] == 2


def test_dedup_keeps_distinct_content() -> None:
    o = EvidenceOrganizer(dedup_threshold=0.99)
    results = [
        {"content": "苹果香蕉橙子葡萄西瓜", "score": 0.5},
        {"content": "飞机火车汽车轮船自行", "score": 0.6},
    ]
    out = o._dedup(results)
    assert len(out) == 2
    assert "_merged_count" not in out[0]


def test_dedup_uses_summary_when_no_content() -> None:
    o = EvidenceOrganizer()
    results = [
        {"summary": "完全相同的摘要文本 abcdefg", "score": 0.1},
        {"summary": "完全相同的摘要文本 abcdefg", "score": 0.8},
    ]
    out = o._dedup(results)
    assert len(out) == 1
    assert out[0]["score"] == 0.8


def test_dedup_uses_text_field_as_fallback() -> None:
    o = EvidenceOrganizer()
    results = [
        {"text": "另一种字段承载 abcdefgh", "score": 0.1},
        {"text": "另一种字段承载 abcdefgh", "score": 0.2},
    ]
    out = o._dedup(results)
    assert len(out) == 1


# ── 端到端 ──


def test_organize_end_to_end_dedup_and_sort() -> None:
    results: list[dict[str, Any]] = [
        {
            "content": "用户在 2024 年喜欢 Python 编程",
            "score": 0.4,
            "stored_at": "2024-03-01",
            "wing": "personal",
        },
        {
            "content": "用户在 2024 年喜欢 Python 编程",
            "score": 0.9,
            "stored_at": "2024-03-01",
            "wing": "personal",
        },
        {
            "content": "完全不同的话题讨论关于架构演进",
            "score": 0.5,
            "stored_at": "2024-01-01",
            "room": "work",
        },
    ]
    out = EvidenceOrganizer().organize(results)
    assert len(out) == 2
    # 时间线：2024-01-01 在前
    assert out[0]["content"].startswith("完全不同")
    assert out[0]["_source_label"] == "work"
    assert out[1]["_merged_count"] == 2
    assert out[1]["_source_label"] == "personal"


def test_repr_includes_threshold_and_enabled() -> None:
    o = EvidenceOrganizer(enabled=True, dedup_threshold=0.9)
    assert repr(o) == "EvidenceOrganizer(dedup_threshold=0.9, enabled=True)"
