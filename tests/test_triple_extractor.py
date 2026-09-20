"""Tests for governance.triple_extractor — regex-first triple extraction + LLM fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from governance import triple_extractor as te
from governance.triple_extractor import (
    TripleExtractor,
    _is_valid_entity,
    get_triple_extractor,
)

# ── _is_valid_entity ────────────────────────────────────────────────────


def test_valid_entity_chinese() -> None:
    assert _is_valid_entity("记忆系统") is True


def test_valid_entity_english() -> None:
    assert _is_valid_entity("Python") is True


def test_valid_entity_two_char_chinese() -> None:
    assert _is_valid_entity("记忆") is True


def test_invalid_entity_empty() -> None:
    assert _is_valid_entity("") is False


def test_invalid_entity_single_char() -> None:
    assert _is_valid_entity("a") is False


def test_invalid_entity_two_char_english() -> None:
    # 2-char English without Chinese is rejected
    assert _is_valid_entity("py") is False


def test_invalid_entity_noise_word() -> None:
    assert _is_valid_entity("什么") is False
    assert _is_valid_entity("这个") is False


def test_invalid_entity_noise_prefix() -> None:
    assert _is_valid_entity("这个问题") is False


def test_invalid_entity_with_space() -> None:
    assert _is_valid_entity("hello world") is False


def test_invalid_entity_with_newline() -> None:
    assert _is_valid_entity("hello\nworld") is False


def test_invalid_entity_punctuation_only() -> None:
    assert _is_valid_entity("//") is False
    assert _is_valid_entity("--") is False


def test_valid_entity_with_hyphen() -> None:
    assert _is_valid_entity("foo-bar") is True


def test_valid_entity_with_underscore() -> None:
    assert _is_valid_entity("foo_bar") is True


def test_valid_entity_with_dot() -> None:
    assert _is_valid_entity("foo.bar") is True


# ── TripleExtractor constructor ─────────────────────────────────────────


def test_constructor_defaults() -> None:
    t = TripleExtractor()
    assert t._llm_client is None
    assert t._llm_model == ""
    assert t._entity_extractor is None
    assert len(t._batch_queue) == 0


def test_inject_llm_client() -> None:
    t = TripleExtractor()
    client = MagicMock()
    t.inject_llm_client(client, model="test-model")
    assert t._llm_client is client
    assert t._llm_model == "test-model"


# ── _normalize_entity ───────────────────────────────────────────────────


def test_normalize_strips_whitespace() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("a   b") == "a b"


def test_normalize_removes_chinese_punct() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("你好，世界！") == "你好世界"


def test_normalize_lowercases_english() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("Python") == "python"


def test_normalize_preserves_chinese_case() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("记忆系统") == "记忆系统"


def test_normalize_mixed_starts_english_lowercases() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("Python记忆") == "python记忆"


def test_normalize_mixed_starts_chinese_preserves() -> None:
    t = TripleExtractor()
    assert t._normalize_entity("记忆Python") == "记忆Python"


# ── extract (regex path) ────────────────────────────────────────────────


def test_extract_empty_returns_empty() -> None:
    t = TripleExtractor()
    assert t.extract("") == []


def test_extract_short_returns_empty() -> None:
    t = TripleExtractor()
    assert t.extract("ab") == []


def test_extract_uses_pattern_chinese() -> None:
    t = TripleExtractor()
    triples = t.extract("项目使用Python开发", use_llm=False)
    assert any(p == "uses" for _, p, _ in triples)


def test_extract_belongs_to_pattern() -> None:
    t = TripleExtractor()
    triples = t.extract("内存属于硬件资源", use_llm=False)
    assert any(p == "belongs_to" for _, p, _ in triples)


def test_extract_causes_pattern() -> None:
    t = TripleExtractor()
    triples = t.extract("内存泄漏导致系统崩溃", use_llm=False)
    assert any(p == "causes" for _, p, _ in triples)


def test_extract_replaces_pattern() -> None:
    t = TripleExtractor()
    triples = t.extract("新版本替代了旧版本", use_llm=False)
    assert any(p == "replaces" for _, p, _ in triples)


def test_extract_contains_pattern() -> None:
    t = TripleExtractor()
    triples = t.extract("系统包含多个模块", use_llm=False)
    assert any(p == "contains" for _, p, _ in triples)


def test_extract_english_integrates_with() -> None:
    t = TripleExtractor()
    triples = t.extract("FastAPI integrates with SQLAlchemy", use_llm=False)
    assert any(p == "integrates_with" for _, p, _ in triples)


def test_extract_english_in_part_of() -> None:
    t = TripleExtractor()
    triples = t.extract("PPO in RLHF training", use_llm=False)
    assert any(p == "part_of" for _, p, _ in triples)


def test_extract_mixed_pattern() -> None:
    t = TripleExtractor()
    triples = t.extract("系统的Python接口", use_llm=False)
    assert any(p == "has_property" for _, p, _ in triples)


def test_extract_dedup() -> None:
    t = TripleExtractor()
    # Same relation twice
    triples = t.extract("A使用B，A使用B", use_llm=False)
    keys = [(s, p, o) for s, p, o in triples]
    assert len(keys) == len(set(keys))


def test_extract_filters_self_relation() -> None:
    t = TripleExtractor()
    # "X使用X" → subj == obj → filtered
    triples = t.extract("记忆使用记忆", use_llm=False)
    assert all(s != o for s, _, o in triples)


def test_extract_filters_invalid_entities() -> None:
    t = TripleExtractor()
    triples = t.extract("这个使用那个", use_llm=False)
    # Both are noise words → filtered
    assert triples == []


# ── extract (LLM fallback) ──────────────────────────────────────────────


def test_extract_llm_fallback_called_when_regex_insufficient() -> None:
    t = TripleExtractor()
    client = MagicMock()
    payload = json.dumps([{"s": "Alice", "p": "knows", "o": "Bob"}])
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    t.inject_llm_client(client, model="m")
    # Text >30 chars with no regex matches
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    triples = t.extract(text, use_llm=True)
    assert ("alice", "knows", "bob") in triples


def test_extract_llm_not_called_when_regex_sufficient() -> None:
    t = TripleExtractor()
    client = MagicMock()
    t.inject_llm_client(client, model="m")
    # Text with 2+ regex matches → LLM not called
    text = "项目使用Python开发，Python依赖标准库"
    t.extract(text, use_llm=True)
    client.chat.completions.create.assert_not_called()


def test_extract_llm_disabled() -> None:
    t = TripleExtractor()
    client = MagicMock()
    t.inject_llm_client(client, model="m")
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    t.extract(text, use_llm=False)
    client.chat.completions.create.assert_not_called()


def test_extract_llm_exception_returns_regex_only() -> None:
    t = TripleExtractor()
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("llm down")
    t.inject_llm_client(client, model="m")
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    triples = t.extract(text, use_llm=True)
    assert triples == []


def test_extract_llm_invalid_json_returns_empty() -> None:
    t = TripleExtractor()
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
    )
    t.inject_llm_client(client, model="m")
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    triples = t.extract(text, use_llm=True)
    assert triples == []


def test_extract_llm_filters_invalid_entities() -> None:
    t = TripleExtractor()
    client = MagicMock()
    payload = json.dumps([{"s": "a", "p": "knows", "o": "b"}])  # too short
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    t.inject_llm_client(client, model="m")
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    triples = t.extract(text, use_llm=True)
    assert triples == []


def test_extract_llm_filters_missing_fields() -> None:
    t = TripleExtractor()
    client = MagicMock()
    payload = json.dumps([{"s": "Alice", "p": "knows"}])  # missing o
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    t.inject_llm_client(client, model="m")
    text = "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"
    triples = t.extract(text, use_llm=True)
    assert triples == []


# ── extract_with_entities ───────────────────────────────────────────────


def test_extract_with_entities_returns_tuple() -> None:
    t = TripleExtractor()
    entities, triples = t.extract_with_entities("项目使用Python开发")
    assert isinstance(entities, list)
    assert isinstance(triples, list)


def test_extract_with_entities_collects_from_triples() -> None:
    t = TripleExtractor()
    entities, triples = t.extract_with_entities("项目使用Python开发")
    for s, _, o in triples:
        assert s in entities or o in entities


# ── batch_extract ───────────────────────────────────────────────────────


def test_batch_extract_empty() -> None:
    t = TripleExtractor()
    assert t.batch_extract([]) == []


def test_batch_extract_regex_only() -> None:
    t = TripleExtractor()
    texts = ["项目使用Python开发", "内存属于硬件资源"]
    results = t.batch_extract(texts)
    assert len(results) == 2
    assert any(p == "uses" for _, p, _ in results[0])
    assert any(p == "belongs_to" for _, p, _ in results[1])


def test_batch_extract_llm_for_insufficient() -> None:
    t = TripleExtractor()
    client = MagicMock()
    payload = json.dumps([[{"s": "Alice", "p": "knows", "o": "Bob"}]])
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    t.inject_llm_client(client, model="m")
    texts = [
        "这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已",
        "项目使用Python开发",
    ]
    results = t.batch_extract(texts)
    assert len(results) == 2
    # First text gets LLM fallback
    assert any(s == "alice" for s, _, _ in results[0])


def test_batch_extract_no_llm_client() -> None:
    t = TripleExtractor()
    texts = ["这是一段没有任何关系模式的文本，只是描述了一些背景信息和上下文内容而已"]
    results = t.batch_extract(texts)
    assert results == [[]]


# ── _extract_batch_via_llm ──────────────────────────────────────────────


def test_extract_batch_via_llm_no_client() -> None:
    t = TripleExtractor()
    assert t._extract_batch_via_llm(["a", "b"]) == [[], []]


def test_extract_batch_via_llm_pads_short_results() -> None:
    t = TripleExtractor()
    client = MagicMock()
    payload = json.dumps([[{"s": "A", "p": "r", "o": "B"}]])  # only 1 group
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    t.inject_llm_client(client, model="m")
    results = t._extract_batch_via_llm(["text1", "text2", "text3"])
    assert len(results) == 3
    assert results[0] == [("a", "r", "b")]
    assert results[1] == []
    assert results[2] == []


def test_extract_batch_via_llm_exception() -> None:
    t = TripleExtractor()
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    t.inject_llm_client(client, model="m")
    results = t._extract_batch_via_llm(["a", "b"])
    assert results == [[], []]


# ── _get_entity_extractor ───────────────────────────────────────────────


def test_get_entity_extractor_caches() -> None:
    t = TripleExtractor()
    e1 = t._get_entity_extractor()
    e2 = t._get_entity_extractor()
    assert e1 is e2


# ── get_triple_extractor singleton ──────────────────────────────────────


def test_get_triple_extractor_singleton(monkeypatch) -> None:
    monkeypatch.setattr(te, "_triple_extractor", None)
    t1 = get_triple_extractor()
    t2 = get_triple_extractor()
    assert t1 is t2
    assert isinstance(t1, TripleExtractor)
