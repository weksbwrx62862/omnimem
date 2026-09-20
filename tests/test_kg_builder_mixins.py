"""deep/kg 时序/关系混入单测（离线，真实 KnowledgeGraph + 临时 SQLite）。

覆盖 builder.py 动态绑定的 mixin：relationships.get_stats / sync_relationships_from_triples
与 temporal.get_timeline / get_entity_timeline_text。
"""
from __future__ import annotations

import pytest
from omnimem.deep.kg.builder import KnowledgeGraph


@pytest.fixture
def kg(tmp_path):
    g = KnowledgeGraph(tmp_path / "kg")
    try:
        yield g
    finally:
        g.close()


def test_get_stats_counts(kg):
    assert kg.get_stats() == {"triples": 0, "entities": 0, "relationships": 0}
    kg.add_triple("alice", "uses", "python", source_memory_id="m1")
    kg.add_triple("bob", "likes", "tea", source_memory_id="m2")
    stats = kg.get_stats()
    assert stats["triples"] == 2
    assert stats["relationships"] >= 1
    assert stats["entities"] >= 2


def test_timeline_subject_or_object_and_ordering(kg):
    kg.add_triple("alice", "knows", "bob", source_memory_id="m1")
    kg.add_triple("carol", "knows", "alice", source_memory_id="m2")  # alice 作宾语
    timeline = kg.get_timeline("alice")
    # 作为主语或宾语的当前三元组都应出现
    assert len(timeline) == 2
    created = [t["created_at"] for t in timeline]
    assert created == sorted(created)  # 按 created_at 升序


def test_timeline_excludes_invalidated(kg):
    kg.add_triple("alice", "uses", "python", source_memory_id="m1")
    kg.add_triple("alice", "uses", "java", source_memory_id="m2", valid_to="2020-01-01T00:00:00+00:00")
    timeline = kg.get_timeline("alice")
    objs = {t["object"] for t in timeline}
    assert "python" in objs
    assert "java" not in objs  # valid_to 已置 -> 排除


def test_sync_relationships_idempotent(kg):
    kg.add_triple("alice", "uses", "python", source_memory_id="m1")
    first = kg.sync_relationships_from_triples()
    assert isinstance(first, int) and first >= 0
    # 已回填过 -> 再次运行不应新增
    assert kg.sync_relationships_from_triples() == 0


def test_timeline_text_header_and_relation_label(kg):
    kg.add_triple("alice", "uses", "python", source_memory_id="m1")
    text = kg.get_entity_timeline_text("Alice")
    assert text.startswith("[Alice 时间线]")
    assert "开始使用" in text  # uses -> 中文关系标签
    assert "python" in text


def test_empty_timeline_text_blank(kg):
    assert kg.get_entity_timeline_text("nobody") == ""
