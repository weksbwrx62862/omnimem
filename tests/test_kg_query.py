"""deep/kg/query.py 单元测试：查询/邻居/路径/Graph RAG/社区发现。"""

from __future__ import annotations

import pytest
from omnimem.deep.kg import entity as entity_mod
from omnimem.deep.kg.builder import KnowledgeGraph


@pytest.fixture
def kg(tmp_path):
    g = KnowledgeGraph(tmp_path / "kg")
    try:
        yield g
    finally:
        g.close()


def _seed_chain(g: KnowledgeGraph) -> None:
    """a → b → c 两跳链。"""
    g.add_triple("a", "rel", "b", confidence=0.9)
    g.add_triple("b", "rel", "c", confidence=0.8)


# ─── query_by_* ────────────────────────────────────────────


def test_query_by_subject_case_insensitive(kg):
    kg.add_triple("Alice", "uses", "Python")
    assert kg.query_by_subject("alice")
    assert kg.query_by_subject("ALICE")[0]["object"] == "Python"


def test_query_by_subject_excludes_expired_by_default(kg):
    kg.add_triple("alice", "uses", "python")
    kg.add_triple("alice", "uses", "java", valid_to="2020-01-01T00:00:00+00:00")
    current = kg.query_by_subject("alice")
    assert len(current) == 1
    assert current[0]["object"] == "python"

    all_rows = kg.query_by_subject("alice", include_expired=True)
    assert len(all_rows) == 2


def test_query_by_object_case_insensitive(kg):
    kg.add_triple("alice", "uses", "Python")
    assert kg.query_by_object("python")[0]["subject"] == "alice"
    assert kg.query_by_object("PYTHON")[0]["subject"] == "alice"


def test_query_by_predicate_filters_expired_and_respects_limit(kg):
    for i in range(5):
        kg.add_triple(f"s{i}", "knows", f"o{i}")
    kg.add_triple("sx", "knows", "ox", valid_to="2020-01-01T00:00:00+00:00")
    rows = kg.query_by_predicate("knows", limit=3)
    assert len(rows) == 3
    assert all(r["predicate"] == "knows" for r in rows)
    assert all(r["subject"] != "sx" for r in rows)


# ─── get_neighbors ─────────────────────────────────────────


def test_get_neighbors_depth_1_returns_direct_edges(kg):
    _seed_chain(kg)
    n = kg.get_neighbors("a", depth=1)
    # a 只出现在 a→b 中
    assert len(n) == 1
    assert n[0]["subject"] == "a"


def test_get_neighbors_depth_2_expands_transitively(kg):
    _seed_chain(kg)
    n = kg.get_neighbors("a", depth=2)
    # 覆盖 a→b 和 b→c
    edges = {(t["subject"], t["object"]) for t in n}
    assert ("a", "b") in edges
    assert ("b", "c") in edges


def test_get_neighbors_dedups_edges_by_id(kg):
    # a→b 和 b→a 从 a 出发会同时命中；不应重复
    kg.add_triple("a", "knows", "b")
    kg.add_triple("b", "knows", "a")
    n = kg.get_neighbors("a", depth=1)
    ids = [r["id"] for r in n]
    assert len(ids) == len(set(ids))


# ─── find_path / find_path_context ─────────────────────────


def test_find_path_start_equals_end_returns_empty(kg):
    _seed_chain(kg)
    assert kg.find_path("a", "a") == []


def test_find_path_direct_edge(kg):
    _seed_chain(kg)
    p = kg.find_path("a", "b", max_depth=3)
    assert len(p) == 1
    assert p[0]["subject"] == "a" and p[0]["object"] == "b"


def test_find_path_multi_hop(kg):
    _seed_chain(kg)
    p = kg.find_path("a", "c", max_depth=5)
    assert len(p) == 2
    assert p[0]["object"] == "b"
    assert p[1]["object"] == "c"


def test_find_path_unreachable(kg):
    kg.add_triple("a", "rel", "b")
    kg.add_triple("x", "rel", "y")
    assert kg.find_path("a", "y", max_depth=5) == []


def test_find_path_context_no_path_message(kg):
    kg.add_triple("a", "rel", "b")
    text = kg.find_path_context("a", "zzz")
    assert "未找到" in text


def test_find_path_context_formats_labels_and_hops(kg):
    kg.add_triple("Alice", "uses", "Python", confidence=0.8)
    kg.add_triple("Python", "is_a", "Language", confidence=0.9)
    text = kg.find_path_context("Alice", "Language", max_depth=3)
    assert "知识链路（2 跳）" in text
    assert "使用" in text  # uses 映射为 "使用"
    assert "是一种" in text  # is_a 映射为 "是一种"


# ─── graph_search / graph_rag ──────────────────────────────


def test_graph_search_extracts_entities_and_expands(kg, monkeypatch):
    kg.add_triple("Alice", "uses", "Python")
    kg.add_triple("Bob", "uses", "Python")
    monkeypatch.setattr(entity_mod, "extract_entities", lambda _q: ["Alice"])
    results = kg.graph_search("Alice", max_depth=1, limit=10)
    assert any(r["subject"] == "Alice" for r in results)


def test_graph_search_falls_back_to_like_when_no_entities(kg, monkeypatch):
    kg.add_triple("project-x", "belongs_to", "portfolio")
    monkeypatch.setattr(entity_mod, "extract_entities", lambda _q: [])
    rows = kg.graph_search("project", limit=5)
    assert rows and rows[0]["subject"] == "project-x"


def test_graph_search_like_escapes_wildcards(kg, monkeypatch):
    kg.add_triple("alpha", "rel", "beta")
    monkeypatch.setattr(entity_mod, "extract_entities", lambda _q: [])
    # "%" 若不转义会匹配所有行；转义后应无匹配。
    assert kg.graph_search("%", limit=5) == []


def test_graph_rag_context_empty_when_no_match(kg):
    kg.add_triple("a", "rel", "b")
    assert kg.graph_rag_context("no_such_entity_xyz") == ""


def test_graph_rag_context_formats_relation_labels(kg):
    kg.add_triple("Alice", "uses", "Python")
    kg.add_triple("Alice", "located_in", "Beijing")
    text = kg.graph_rag_context("Alice", depth=1)
    assert "[Knowledge Graph: Alice]" in text
    assert "Alice 使用 Python" in text
    assert "Alice 位于 Beijing" in text


def test_graph_rag_search_returns_empty_without_entities(kg, monkeypatch):
    monkeypatch.setattr(entity_mod, "extract_entities", lambda _q: [])
    assert kg.graph_rag_search("随机无实体查询") == ""


def test_graph_rag_search_joins_multiple_entities(kg, monkeypatch):
    kg.add_triple("Alice", "uses", "Python")
    kg.add_triple("Bob", "uses", "Go")
    monkeypatch.setattr(entity_mod, "extract_entities", lambda _q: ["Alice", "Bob"])
    ctx = kg.graph_rag_search("Alice Bob", max_depth=1)
    assert "Alice" in ctx and "Bob" in ctx


# ─── 实体接口 ─────────────────────────────────────────────


def test_get_entity_returns_none_for_missing(kg):
    assert kg.get_entity("ghost") is None


def test_get_entity_shape_after_add(kg):
    kg.add_triple("Alice", "uses", "Python")
    ent = kg.get_entity("Alice")
    assert ent is not None
    assert {"name", "entity_type", "mention_count", "first_seen", "last_seen"}.issubset(ent.keys())
    assert ent["mention_count"] >= 1


def test_get_all_entities_sorted_by_mention_count(kg):
    kg.add_triple("alice", "knows", "bob")
    kg.add_triple("alice", "knows", "carol")
    kg.add_triple("alice", "knows", "dave")
    ents = kg.get_all_entities(limit=10)
    assert ents[0]["name"] == "alice"
    assert ents[0]["mention_count"] >= 3


def test_get_entity_graph_buckets_unknown_to_object(kg, monkeypatch):
    # 强制推断为未知类型 → 应归入 Object 桶
    monkeypatch.setattr(KnowledgeGraph, "_infer_entity_type", lambda _self, _name: "Unknown")
    kg.add_triple("weird", "rel", "thing")
    graph = kg.get_entity_graph(limit=10)
    assert "Object" in graph
    names_in_object = [e["name"] for e in graph["Object"]]
    assert "weird" in names_in_object


# ─── 图算法 ───────────────────────────────────────────────


def test_shortest_path_returns_multi_hop_chain(kg):
    _seed_chain(kg)
    p = kg.shortest_path("a", "c", max_depth=5)
    # 与 find_path 语义等价：a→b, b→c
    assert [(t["subject"], t["object"]) for t in p] == [("a", "b"), ("b", "c")]


def test_shortest_path_unreachable_returns_empty(kg):
    kg.add_triple("a", "rel", "b")
    kg.add_triple("x", "rel", "y")
    assert kg.shortest_path("a", "y", max_depth=5) == []


def test_shortest_path_no_connection_returns_empty(kg):
    # _conn 已关闭 → 直接返回 []
    kg.close()
    assert kg.shortest_path("a", "b", max_depth=2) == []


def test_connected_components_filters_by_min_size(kg):
    # 组件 1：a,b,c,d（4 个）；组件 2：x,y（2 个）
    kg.add_triple("a", "rel", "b")
    kg.add_triple("b", "rel", "c")
    kg.add_triple("c", "rel", "d")
    kg.add_triple("x", "rel", "y")
    comps = kg.connected_components(min_size=3, limit=100)
    assert len(comps) == 1
    assert set(comps[0]) == {"a", "b", "c", "d"}


def test_connected_components_no_connection_returns_empty(kg):
    kg.close()
    assert kg.connected_components(min_size=1, limit=10) == []
