"""改进项 #6 可插拔图后端单测（离线，无需真实 Neo4j）。

- SQLite 后端：对真实 TemporalKnowledgeGraph 做端到端双时序验证（矛盾消解 + 时点查询）。
- Neo4j 后端：注入 FakeDriver 断言 Cypher 文本与参数、谓词注入防护、结果映射。
- 工厂：默认 sqlite、协议一致性、未知后端报错。
"""
from __future__ import annotations

from typing import Any

import pytest
from omnimem.deep.kg.backends import (
    SUPPORTED_GRAPH_BACKENDS,
    GraphStoreProtocol,
    Neo4jGraphStore,
    create_graph_store,
)


# ── Fake Neo4j driver ─────────────────────────────────
class _FakeSession:
    def __init__(self, log: list, records_fn) -> None:
        self._log = log
        self._records_fn = records_fn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query: str, **params):
        self._log.append((query, params))
        return list(self._records_fn(query))


class FakeDriver:
    def __init__(self, records_fn=None) -> None:
        self.log: list[tuple[str, dict]] = []
        self.closed = False
        self._records_fn = records_fn or (lambda q: [])

    def session(self, database=None):
        return _FakeSession(self.log, self._records_fn)

    def close(self):
        self.closed = True


def _row(**over) -> dict[str, Any]:
    base = {
        "subject": "Alice", "predicate": "WORKS_AT", "object": "Acme", "id": "t_x",
        "valid_at": "2020-01-01T00:00:00+00:00", "invalid_at": None, "superseded_by": None,
        "source_memory_id": "m1", "confidence": 3, "created_at": "2020-01-01T00:00:00+00:00",
    }
    base.update(over)
    return base


# ── factory / protocol ────────────────────────────────
def test_factory_defaults_to_sqlite(tmp_path):
    store = create_graph_store("sqlite", data_dir=tmp_path)
    assert isinstance(store, GraphStoreProtocol)  # runtime_checkable 结构校验


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        create_graph_store("nope")


def test_supported_backends_listed():
    assert {"sqlite", "neo4j", "falkordb"} <= set(SUPPORTED_GRAPH_BACKENDS)


# ── SQLite 端到端（真实双时序）──────────────────────
def test_sqlite_bi_temporal_flow(tmp_path):
    store = create_graph_store("sqlite", data_dir=tmp_path)
    id1 = store.add_triple("Alice", "WORKS_AT", "Acme", "2020-01-01T00:00:00+00:00", source_memory_id="m1")
    assert id1
    cur = store.query_current("Alice", "WORKS_AT")
    assert any(t.object == "Acme" for t in cur)

    # 新关系应把旧关系标记过时（矛盾消解）
    store.add_triple("Alice", "WORKS_AT", "Beta", "2023-01-01T00:00:00+00:00", source_memory_id="m2")
    cur2 = store.query_current("Alice", "WORKS_AT")
    assert any(t.object == "Beta" for t in cur2)
    assert all(t.object != "Acme" for t in cur2)

    # 历史时点仍可查回旧关系
    past = store.query_at_time("Alice", "WORKS_AT", "2021-06-01T00:00:00+00:00")
    assert any(t.object == "Acme" for t in past)

    stats = store.get_stats()
    assert isinstance(stats, dict)
    store.flush()
    store.close()


# ── Neo4j Cypher 断言（注入 FakeDriver）───────────────
def test_neo4j_add_triple_emits_cypher_and_guards_predicate():
    drv = FakeDriver()
    store = Neo4jGraphStore(driver=drv, database="neo4j")
    tid = store.add_triple("Alice", "WORKS_AT", "Acme", "2020-01-01T00:00:00+00:00", source_memory_id="m1")
    assert tid.startswith("t_")
    joined = "\n".join(q for q, _ in drv.log)
    assert "SET old.invalid_at" in joined          # 矛盾消解
    assert "CREATE (s)-[r:WORKS_AT" in joined       # 谓词写入关系类型
    # 参数化：值走 params，不拼进文本
    assert any(p.get("subject") == "Alice" for _, p in drv.log)


def test_neo4j_rejects_illegal_predicate():
    store = Neo4jGraphStore(driver=FakeDriver())
    with pytest.raises(ValueError):
        store.add_triple("A", "WORKS_AT); DROP", "B", "2020-01-01T00:00:00+00:00")


def test_neo4j_query_at_time_maps_records():
    drv = FakeDriver(records_fn=lambda q: [_row(object="Acme")] if "valid_at <= $at" in q else [])
    store = Neo4jGraphStore(driver=drv)
    out = store.query_at_time("Alice", "WORKS_AT", "2021-01-01T00:00:00+00:00")
    assert out and out[0].object == "Acme"
    q, p = drv.log[-1]
    assert "r.invalid_at > $at" in q
    assert p.get("at") == "2021-01-01T00:00:00+00:00"


def test_neo4j_temporal_search_empty_noop():
    drv = FakeDriver()
    store = Neo4jGraphStore(driver=drv)
    assert store.temporal_search([]) == []
    assert drv.log == []


def test_neo4j_delete_and_stats():
    drv = FakeDriver(records_fn=lambda q: [{"c": 2}] if ("size(rs)" in q or "count(r)" in q) else [])
    store = Neo4jGraphStore(driver=drv)
    assert store.delete_by_memory_id("m1") == 2
    stats = store.get_stats()
    assert stats["backend"] == "neo4j"
    assert stats["triples"] == 2


def test_neo4j_close_reaches_driver():
    drv = FakeDriver()
    store = Neo4jGraphStore(driver=drv)
    store.close()
    assert drv.closed is True
