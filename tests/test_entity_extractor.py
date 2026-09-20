"""retrieval.entity_extractor.EntityExtractor 离线单元测试。

覆盖：
  - extract：引号 / 英文专名 / 中文短语 / jieba 词性 / 归一化 / 去重 / max_entities
  - _normalize / _is_generic
  - extract_from_metadata
  - compute_entity_overlap：边界与阈值
"""

from __future__ import annotations

import pytest
from omnimem.retrieval.entity_extractor import (
    ENTITY_BOOST_WEIGHT,
    EntityExtractor,
)


@pytest.fixture
def extractor() -> EntityExtractor:
    return EntityExtractor()


# ── extract 前置 ──


def test_extract_empty_returns_empty(extractor: EntityExtractor) -> None:
    assert extractor.extract("") == []


def test_extract_whitespace_returns_empty(extractor: EntityExtractor) -> None:
    assert extractor.extract("   \n\t ") == []


# ── 引号 ──


def test_extract_quoted_chinese_corner_brackets(extractor: EntityExtractor) -> None:
    out = extractor.extract("关于「向量检索」的讨论")
    assert "向量检索" in out


def test_extract_quoted_double_curly(extractor: EntityExtractor) -> None:
    out = extractor.extract("所谓“混合检索”其实是多路融合")
    assert "混合检索" in out


def test_extract_quoted_single_quote(extractor: EntityExtractor) -> None:
    out = extractor.extract("we call it 'RRF' internally")
    assert "RRF" in out


# ── 英文专名 ──


def test_extract_english_capitalized(extractor: EntityExtractor) -> None:
    out = extractor.extract("Alice Bob and Charlie discussed it")
    for name in ("Alice", "Bob", "Charlie"):
        assert name in out


def test_extract_skips_generic_english(extractor: EntityExtractor) -> None:
    # 通用名词在 proper 阶段被过滤（system/data 均在 _GENERIC_NOUNS）
    out = extractor.extract("The System Processes Data Everywhere")
    assert "System" not in out
    assert "Data" not in out


# ── 中文 2-4 字 ──


def test_extract_chinese_before_particle(extractor: EntityExtractor) -> None:
    # "茅台" 后紧跟 "是"（particles）→ 优先命中
    out = extractor.extract("茅台是贵")
    assert "茅台" in out


def test_extract_generic_chinese_filtered(extractor: EntityExtractor) -> None:
    # "问题"/"方法" 等在 generic 表中的中文词被过滤
    out = extractor.extract("问题是方法")
    for w in ("问题", "方法"):
        assert w not in out


# ── jieba 通道 ──


def test_extract_includes_jieba_pos_entities(extractor: EntityExtractor) -> None:
    # 中文长句 → jieba 会切出名词/动词
    text = "华为公司发布了鸿蒙操作系统并在中国市场引起轰动"
    out = extractor.extract(text)
    assert len(out) >= 1


# ── 去重 & 截断 ──


def test_extract_deduplicates(extractor: EntityExtractor) -> None:
    text = "Alice 与 Alice 和 Alice 都提到了「Alice」这个项目"
    out = extractor.extract(text)
    assert out.count("Alice") == 1


def test_extract_respects_max_entities(extractor: EntityExtractor) -> None:
    text = "「A1」 「A2」 「A3」 「A4」 「A5」 「A6」 「A7」 Alice Bob Charlie David"
    out = extractor.extract(text, max_entities=3)
    assert len(out) == 3


# ── _normalize ──


def test_normalize_strips_surrounding_punctuation(extractor: EntityExtractor) -> None:
    assert extractor._normalize("  !!abc?? ") == "abc"


def test_normalize_preserves_inner_punctuation(extractor: EntityExtractor) -> None:
    # 内部标点在 re 只锚定两端的情况下保留
    assert extractor._normalize("a.b") == "a.b"


# ── _is_generic ──


def test_is_generic_case_insensitive(extractor: EntityExtractor) -> None:
    assert extractor._is_generic("System") is True
    assert extractor._is_generic("system") is True
    assert extractor._is_generic("UNIQUE_TOKEN") is False


def test_is_generic_chinese(extractor: EntityExtractor) -> None:
    assert extractor._is_generic("问题") is True
    assert extractor._is_generic("茅台") is False


# ── extract_from_metadata ──


def test_extract_from_metadata_returns_list(extractor: EntityExtractor) -> None:
    assert extractor.extract_from_metadata({"entities": ["A", "B"]}) == ["A", "B"]


def test_extract_from_metadata_missing_key_returns_empty(extractor: EntityExtractor) -> None:
    assert extractor.extract_from_metadata({}) == []


# ── compute_entity_overlap ──


def test_overlap_empty_query_returns_zero(extractor: EntityExtractor) -> None:
    assert extractor.compute_entity_overlap([], ["a"]) == 0.0


def test_overlap_empty_doc_returns_zero(extractor: EntityExtractor) -> None:
    assert extractor.compute_entity_overlap(["a"], []) == 0.0


def test_overlap_full_match(extractor: EntityExtractor) -> None:
    score = extractor.compute_entity_overlap(["Alice", "Bob"], ["Alice", "Bob"])
    assert score == pytest.approx(ENTITY_BOOST_WEIGHT, rel=1e-6)


def test_overlap_partial_match(extractor: EntityExtractor) -> None:
    score = extractor.compute_entity_overlap(["Alice", "Bob"], ["Alice"])
    assert score == pytest.approx(ENTITY_BOOST_WEIGHT / 2, rel=1e-6)


def test_overlap_no_common_returns_zero(extractor: EntityExtractor) -> None:
    assert extractor.compute_entity_overlap(["Alice"], ["Bob"]) == 0.0


def test_overlap_case_insensitive(extractor: EntityExtractor) -> None:
    score = extractor.compute_entity_overlap(["alice"], ["ALICE"])
    assert score == pytest.approx(ENTITY_BOOST_WEIGHT, rel=1e-6)


def test_overlap_deduplicates_query_entities(extractor: EntityExtractor) -> None:
    # q_set 去重：{a} → 交集大小 1 / q_set 1 → 满权重
    score = extractor.compute_entity_overlap(["a", "a", "a"], ["a"])
    assert score == pytest.approx(ENTITY_BOOST_WEIGHT, rel=1e-6)


def test_overlap_returns_bounded_by_boost_weight(extractor: EntityExtractor) -> None:
    score = extractor.compute_entity_overlap(["a", "b", "c"], ["a"])
    # 交集 1 / q 3 → 0.333 * 0.3 ≈ 0.1
    assert 0.0 < score < ENTITY_BOOST_WEIGHT
