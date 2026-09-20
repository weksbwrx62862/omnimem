"""retrieval.fts5.FTS5Retriever 离线单元测试。

覆盖：
  - name / search_sync / asearch 顶层
  - search 前置：连接缺失、空查询、get_read_conn 异常
  - _build_fts_query：jieba 分词路径 + 转义 + 无 jieba 分支
  - _escape_fts5：特殊字符替换
  - add / flush / warmup / document_count no-op 兼容接口
  - search 真实走一次 sqlite3 in-memory FTS5 表（若可用）
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest
from omnimem.retrieval import fts5 as fts5_mod
from omnimem.retrieval.fts5 import FTS5Retriever


@pytest.fixture
def retriever() -> FTS5Retriever:
    return FTS5Retriever()


# ── 基本属性 ──


def test_default_name_is_bm25(retriever: FTS5Retriever) -> None:
    assert retriever.name == "bm25"


def test_add_is_noop(retriever: FTS5Retriever) -> None:
    retriever.add("c", "m", {})


def test_flush_is_noop(retriever: FTS5Retriever) -> None:
    retriever.flush()


def test_warmup_is_noop(retriever: FTS5Retriever) -> None:
    retriever.warmup()


def test_document_count_always_zero(retriever: FTS5Retriever) -> None:
    assert retriever.document_count == 0


# ── _escape_fts5 ──


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('abc"def', "abc def"),
        ("a*b", "a b"),
        ("(x)", " x "),
        ("a-b", "a b"),
        ("clean", "clean"),
        ("", ""),
    ],
)
def test_escape_fts5_replaces_specials(raw: str, expected: str) -> None:
    result = FTS5Retriever._escape_fts5(raw)
    assert result == expected.strip()


def test_escape_fts5_strips_outer_whitespace() -> None:
    assert FTS5Retriever._escape_fts5(' "hello" ') == "hello"


# ── _build_fts_query ──


def test_build_fts_query_uses_jieba_when_available(
    retriever: FTS5Retriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jieba = MagicMock()
    fake_jieba.lcut.return_value = ["用户", "喜欢", "Python"]
    monkeypatch.setattr(fts5_mod, "_HAS_JIEBA", True)
    monkeypatch.setattr(fts5_mod, "jieba", fake_jieba, raising=False)
    q = retriever._build_fts_query("用户喜欢Python")
    # 中文 token 加引号；英文 token 不加引号
    assert '"用户"' in q
    assert '"喜欢"' in q
    assert "Python" in q
    assert q.count(" OR ") == 2


def test_build_fts_query_without_jieba_escapes_original(
    retriever: FTS5Retriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fts5_mod, "_HAS_JIEBA", False)
    out = retriever._build_fts_query('hello"world')
    assert '"' not in out
    assert "hello" in out and "world" in out


def test_build_fts_query_empty_tokens_falls_back_to_escape(
    retriever: FTS5Retriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jieba = MagicMock()
    fake_jieba.lcut.return_value = ["", " "]
    monkeypatch.setattr(fts5_mod, "_HAS_JIEBA", True)
    monkeypatch.setattr(fts5_mod, "jieba", fake_jieba, raising=False)
    out = retriever._build_fts_query('a"b')
    # 空 tokens → 走 _escape_fts5(query)
    assert '"' not in out


# ── search 前置 ──


def test_search_returns_empty_without_get_conn(retriever: FTS5Retriever) -> None:
    assert retriever.search("q") == []


def test_search_returns_empty_for_blank_query() -> None:
    r = FTS5Retriever(get_read_conn=lambda: MagicMock())
    assert r.search("   ") == []


def test_search_returns_empty_when_conn_none() -> None:
    r = FTS5Retriever(get_read_conn=lambda: None)
    assert r.search("q") == []


def test_search_swallows_get_conn_exception() -> None:
    def boom() -> Any:
        raise RuntimeError("no conn")

    r = FTS5Retriever(get_read_conn=boom)
    assert r.search("q") == []


def test_search_swallows_operational_error() -> None:
    conn = MagicMock()
    conn.execute.side_effect = sqlite3.OperationalError("fts: syntax error")
    r = FTS5Retriever(get_read_conn=lambda: conn)
    assert r.search("q") == []


def test_search_returns_empty_when_fts_query_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    monkeypatch.setattr(r, "_build_fts_query", lambda _q: "")
    assert r.search("anything") == []
    conn.execute.assert_not_called()


# ── search 结构：使用真实 sqlite3 FTS5 ──


def _build_fts5_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memory_index (rowid INTEGER PRIMARY KEY, content, content_preview, memory_id, type, stored_at, confidence)")
    try:
        conn.execute("CREATE VIRTUAL TABLE memory_index_fts USING fts5(content, content_preview, tokenize='unicode61')")
    except sqlite3.OperationalError:
        pytest.skip("sqlite3 build lacks FTS5")
    conn.execute(
        "INSERT INTO memory_index (rowid, content, content_preview, memory_id, type, stored_at, confidence) VALUES (1, 'Python 语言用户指南', 'preview', 'm1', 'fact', '2024-01-01', 0.9)"
    )
    conn.execute("INSERT INTO memory_index_fts (rowid, content, content_preview) VALUES (1, 'Python 语言用户指南', 'preview')")
    conn.execute(
        "INSERT INTO memory_index (rowid, content, content_preview, memory_id, type, stored_at, confidence) VALUES (2, 'Java 虚拟机', 'prev2', 'm2', 'fact', '2024-02-02', 0.8)"
    )
    conn.execute("INSERT INTO memory_index_fts (rowid, content, content_preview) VALUES (2, 'Java 虚拟机', 'prev2')")
    conn.commit()
    return conn


def test_search_hits_indexed_row() -> None:
    conn = _build_fts5_conn()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    hits = r.search("Python", top_k=5)
    assert any(h["memory_id"] == "m1" for h in hits)
    assert hits[0]["_source"] == "fts5"
    assert hits[0]["score"] == 0.8  # 固定分


def test_search_truncates_to_top_k() -> None:
    conn = _build_fts5_conn()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    hits = r.search("Python", top_k=1)
    assert len(hits) <= 1


def test_search_defaults_for_missing_fields() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memory_index (rowid INTEGER PRIMARY KEY, content, content_preview, memory_id, type, stored_at, confidence)")
    try:
        conn.execute("CREATE VIRTUAL TABLE memory_index_fts USING fts5(content, content_preview, tokenize='unicode61')")
    except sqlite3.OperationalError:
        pytest.skip("sqlite3 build lacks FTS5")
    conn.execute(
        "INSERT INTO memory_index (rowid, content, memory_id) VALUES (1, 'unique_marker', 'mid-1')"
    )
    conn.execute("INSERT INTO memory_index_fts (rowid, content) VALUES (1, 'unique_marker')")
    conn.commit()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    hits = r.search("unique_marker")
    assert hits[0]["summary"] == ""
    assert hits[0]["type"] == "fact"
    assert hits[0]["stored_at"] == ""
    assert hits[0]["confidence"] == 0.5


# ── search_sync / asearch ──


def test_search_sync_wraps_as_retrieval_result() -> None:
    conn = _build_fts5_conn()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    out = r.search_sync("Python", top_k=5)
    assert out.channel == "bm25"
    assert len(out.results) >= 1
    assert all(s == 0.8 for s in out.scores)


def test_asearch_returns_same_as_sync() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE memory_index (rowid INTEGER PRIMARY KEY, content, content_preview, memory_id, type, stored_at, confidence)")
    try:
        conn.execute("CREATE VIRTUAL TABLE memory_index_fts USING fts5(content, content_preview, tokenize='unicode61')")
    except sqlite3.OperationalError:
        pytest.skip("sqlite3 build lacks FTS5")
    conn.execute(
        "INSERT INTO memory_index (rowid, content, content_preview, memory_id, type, stored_at, confidence) VALUES (1, 'Python 语言', 'p', 'm1', 'fact', '2024-01-01', 0.9)"
    )
    conn.execute("INSERT INTO memory_index_fts (rowid, content, content_preview) VALUES (1, 'Python 语言', 'p')")
    conn.commit()
    r = FTS5Retriever(get_read_conn=lambda: conn)
    out = asyncio.run(r.asearch("Python", top_k=5))
    assert out.channel == "bm25"
    assert any(h["memory_id"] == "m1" for h in out.results)
