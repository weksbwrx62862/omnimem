"""改进项 #6 SQLite 图后端补充单测（离线，委托真实 TemporalKnowledgeGraph）。

覆盖 test_graph_backends.py 未触及的转发方法：协议一致性/别名、
invalidate_triple、delete_by_memory_id、get_timeline、temporal_search。
"""
from __future__ import annotations

from omnimem.deep.kg.backends.protocol import GraphStoreProtocol, GraphTriple, TemporalTriple
from omnimem.deep.kg.backends.sqlite_backend import SQLiteGraphStore


def _store(tmp_path) -> SQLiteGraphStore:
    return SQLiteGraphStore(tmp_path)


def test_sqlite_store_satisfies_protocol_and_alias(tmp_path):
    assert isinstance(_store(tmp_path), GraphStoreProtocol)  # runtime_checkable 结构校验
    # GraphTriple 是 TemporalTriple 的语义别名
    assert GraphTriple is TemporalTriple


def test_invalidate_excludes_from_current_keeps_history(tmp_path):
    s = _store(tmp_path)
    tid = s.add_triple("Alice", "WORKS_AT", "Acme", "2020-01-01T00:00:00+00:00", source_memory_id="m1")
    assert tid
    assert any(t.object == "Acme" for t in s.query_current("Alice", "WORKS_AT"))
    s.invalidate_triple(tid, "2021-01-01T00:00:00+00:00")
    assert s.query_current("Alice", "WORKS_AT") == []
    past = s.query_at_time("Alice", "WORKS_AT", "2020-06-01T00:00:00+00:00")
    assert any(t.object == "Acme" for t in past)


def test_delete_by_memory_id(tmp_path):
    s = _store(tmp_path)
    s.add_triple("Bob", "LIVES_IN", "Paris", "2019-01-01T00:00:00+00:00", source_memory_id="mx")
    s.add_triple("Bob", "WORKS_AT", "Foo", "2019-02-01T00:00:00+00:00", source_memory_id="mx")
    assert s.delete_by_memory_id("mx") == 2
    assert s.query_current("Bob", "") == []
    # 空 memory_id 直接返回 0，不触碰存储
    assert s.delete_by_memory_id("") == 0


def test_get_timeline_ascending_and_subject_or_object(tmp_path):
    s = _store(tmp_path)
    s.add_triple("Carol", "KNOWS", "Dave", "2022-03-01T00:00:00+00:00", source_memory_id="m1")
    s.add_triple("Carol", "KNOWS", "Eve", "2021-01-01T00:00:00+00:00", source_memory_id="m2")
    s.add_triple("Zed", "KNOWS", "Carol", "2020-01-01T00:00:00+00:00", source_memory_id="m3")
    timeline = s.get_timeline("Carol")
    # 作为主语或宾语的三元组都应出现
    assert len(timeline) == 3
    valids = [t.valid_at for t in timeline]
    assert valids == sorted(valids)  # 升序
    assert len(s.get_timeline("Carol", limit=1)) == 1


def test_temporal_search_branches(tmp_path):
    s = _store(tmp_path)
    s.add_triple("Erin", "WORKS_AT", "Acme", "2020-01-01T00:00:00+00:00", source_memory_id="m1")
    assert s.temporal_search([]) == []
    hits = s.temporal_search(["Erin"])
    assert any(t.subject == "Erin" for t in hits)
    assert len(s.temporal_search(["Erin"], limit=0)) == 0
