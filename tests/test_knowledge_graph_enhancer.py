"""Tests for governance.knowledge_graph_enhancer — semantic + co-access + save/query."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import pytest

from governance import knowledge_graph_enhancer as kge
from governance.knowledge_graph_enhancer import (
    GraphStats,
    KnowledgeGraphEnhancer,
    Relationship,
    discover_and_save_relationships,
    get_enhancer,
)


def _inject_embeddings(e: KnowledgeGraphEnhancer, data: dict[str, list[float]]) -> KnowledgeGraphEnhancer:
    e._embeddings = dict(data)
    e._loaded = True
    return e


# ── dataclasses ─────────────────────────────────────────────────────────


def test_relationship_defaults() -> None:
    r = Relationship(source_id="a", target_id="b", relation_type="semantic", strength=0.9)
    assert r.created_at is not None
    assert isinstance(r.created_at, datetime)


def test_graph_stats_fields() -> None:
    g = GraphStats(total_nodes=10, total_edges=20, avg_degree=4.0, connected_components=2)
    assert g.densest_cluster is None
    assert g.avg_degree == 4.0


# ── constructor ─────────────────────────────────────────────────────────


def test_default_paths(tmp_path) -> None:
    db = tmp_path / "kg.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "emb.json"))
    assert os.path.exists(db)
    assert e._loaded is False


def test_creates_db_schema(tmp_path) -> None:
    db = tmp_path / "kg.db"
    KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "emb.json"))
    conn = sqlite3.connect(str(db))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(relationships)").fetchall()]
    conn.close()
    for c in ["id", "source_id", "target_id", "relation_type", "strength", "created_at"]:
        assert c in cols


# ── _load_data ──────────────────────────────────────────────────────────


def test_load_data_idempotent(tmp_path, monkeypatch) -> None:
    import omnimem.retrieval.vector_store as vs

    calls: list[str] = []

    def _fake(path):
        calls.append(str(path))
        return {"a": [1.0]}

    monkeypatch.setattr(vs, "load_embedding_cache_dict", _fake)
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    e._load_data()
    e._load_data()
    assert len(calls) == 1


def test_load_data_embeddings_exception(tmp_path, monkeypatch) -> None:
    import omnimem.retrieval.vector_store as vs

    def _boom(path):
        raise RuntimeError("nope")

    monkeypatch.setattr(vs, "load_embedding_cache_dict", _boom)
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    e._load_data()
    assert e._embeddings == {}


def test_load_data_relationships_from_db(tmp_path, monkeypatch) -> None:
    db = tmp_path / "k.db"
    # Pre-seed
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE relationships (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source_id TEXT NOT NULL, target_id TEXT NOT NULL, relation_type TEXT NOT NULL, "
        "strength REAL NOT NULL, created_at TEXT NOT NULL, "
        "UNIQUE(source_id, target_id, relation_type))"
    )
    conn.execute(
        "INSERT INTO relationships (source_id, target_id, relation_type, strength, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("a", "b", "semantic", 0.9, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    import omnimem.retrieval.vector_store as vs

    monkeypatch.setattr(vs, "load_embedding_cache_dict", lambda p: {})

    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    # Re-init won't happen because we created the table already
    # Force load
    e._load_data()
    assert len(e._relationships) == 1
    assert e._relationships[0].source_id == "a"


# ── _cosine_similarity ──────────────────────────────────────────────────


def test_cosine_identical_vectors(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_mismatched_dims(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cosine_empty_vectors(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([], []) == 0.0


def test_cosine_zero_norm(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ── discover_semantic_relationships ─────────────────────────────────────


def test_discover_semantic_few_embeddings(tmp_path) -> None:
    e = _inject_embeddings(
        KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json")),
        {"a": [1.0, 0.0]},
    )
    assert e.discover_semantic_relationships(threshold=0.5) == []


def test_discover_semantic_similar_pair(tmp_path) -> None:
    e = _inject_embeddings(
        KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json")),
        {"a": [1.0, 0.0], "b": [0.99, 0.01]},
    )
    rels = e.discover_semantic_relationships(threshold=0.7)
    assert len(rels) == 1
    assert rels[0].relation_type == "semantic"


def test_discover_semantic_below_threshold(tmp_path) -> None:
    e = _inject_embeddings(
        KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json")),
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},  # orthogonal
    )
    rels = e.discover_semantic_relationships(threshold=0.7)
    assert rels == []


def test_discover_semantic_respects_max_relationships(tmp_path) -> None:
    # 4 embeddings all similar → 6 pairs; cap at 2
    e = _inject_embeddings(
        KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json")),
        {
            "a": [1.0, 0.0],
            "b": [1.0, 0.001],
            "c": [1.0, 0.002],
            "d": [1.0, 0.003],
        },
    )
    rels = e.discover_semantic_relationships(threshold=0.7, max_relationships=2)
    assert len(rels) == 2


def test_discover_semantic_sorted_by_similarity(tmp_path) -> None:
    e = _inject_embeddings(
        KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json")),
        {
            "a": [1.0, 0.0],
            "b": [1.0, 0.001],  # very close to a
            "c": [1.0, 0.5],  # less close
            "d": [1.0, 0.6],
        },
    )
    rels = e.discover_semantic_relationships(threshold=0.5, max_relationships=100)
    strengths = [r.strength for r in rels]
    assert strengths == sorted(strengths, reverse=True)


# ── discover_co_access_relationships ────────────────────────────────────


def test_co_access_missing_db(tmp_path, monkeypatch) -> None:
    # Point HOME to tmp_path so forgetting.db path is missing
    monkeypatch.setenv("HOME", str(tmp_path))
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e.discover_co_access_relationships() == []


def test_co_access_above_threshold(tmp_path, monkeypatch) -> None:
    # Create forgetting.db with access_log
    home = tmp_path / "home"
    gov_dir = home / ".hermes/omnimem/governance"
    gov_dir.mkdir(parents=True)
    fdb = gov_dir / "forgetting.db"
    conn = sqlite3.connect(str(fdb))
    conn.execute("CREATE TABLE access_log (memory_id TEXT, accessed_at TEXT)")
    # 4 distinct hours, each containing both a and b → pair (a,b) count = 4
    for i in range(4):
        ts = f"2025-01-0{i + 1}T10:00:00"
        conn.execute("INSERT INTO access_log VALUES (?, ?)", ("a", ts))
        conn.execute("INSERT INTO access_log VALUES (?, ?)", ("b", ts))
    conn.commit()
    conn.close()
    monkeypatch.setenv("HOME", str(home))

    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    rels = e.discover_co_access_relationships(threshold=3)
    assert len(rels) >= 1
    ab = [r for r in rels if {r.source_id, r.target_id} == {"a", "b"}]
    assert len(ab) == 1
    assert ab[0].relation_type == "co_access"
    assert ab[0].strength == pytest.approx(0.4)  # 4/10


def test_co_access_below_threshold(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    gov_dir = home / ".hermes/omnimem/governance"
    gov_dir.mkdir(parents=True)
    fdb = gov_dir / "forgetting.db"
    conn = sqlite3.connect(str(fdb))
    conn.execute("CREATE TABLE access_log (memory_id TEXT, accessed_at TEXT)")
    conn.execute("INSERT INTO access_log VALUES (?, ?)", ("a", "2025-01-01T10:00"))
    conn.execute("INSERT INTO access_log VALUES (?, ?)", ("b", "2025-01-01T10:00"))
    conn.commit()
    conn.close()
    monkeypatch.setenv("HOME", str(home))

    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    rels = e.discover_co_access_relationships(threshold=5)
    assert rels == []


def test_co_access_strength_capped_at_1(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    gov_dir = home / ".hermes/omnimem/governance"
    gov_dir.mkdir(parents=True)
    fdb = gov_dir / "forgetting.db"
    conn = sqlite3.connect(str(fdb))
    conn.execute("CREATE TABLE access_log (memory_id TEXT, accessed_at TEXT)")
    # 20 co-visits across 20 different hours
    for i in range(20):
        ts = f"2025-01-{i + 1:02d}T10:00"
        conn.execute("INSERT INTO access_log VALUES (?, ?)", ("a", ts))
        conn.execute("INSERT INTO access_log VALUES (?, ?)", ("b", ts))
    conn.commit()
    conn.close()
    monkeypatch.setenv("HOME", str(home))

    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    rels = e.discover_co_access_relationships(threshold=3)
    ab = [r for r in rels if {r.source_id, r.target_id} == {"a", "b"}]
    assert ab[0].strength == 1.0


def test_co_access_db_exception(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    gov_dir = home / ".hermes/omnimem/governance"
    gov_dir.mkdir(parents=True)
    # forgetting.db exists but is not a valid sqlite file
    (gov_dir / "forgetting.db").write_bytes(b"not a database")
    monkeypatch.setenv("HOME", str(home))

    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    # Should return [] via exception path
    assert e.discover_co_access_relationships() == []


# ── save_relationships ──────────────────────────────────────────────────


def test_save_relationships_basic(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    rels = [Relationship("a", "b", "semantic", 0.9)]
    saved = e.save_relationships(rels)
    assert saved == 1

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT source_id, target_id FROM relationships").fetchall()
    conn.close()
    assert rows == [("a", "b")]


def test_save_relationships_empty(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e.save_relationships([]) == 0


def test_save_relationships_replace_on_conflict(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    e.save_relationships([Relationship("a", "b", "semantic", 0.9)])
    e.save_relationships([Relationship("a", "b", "semantic", 0.5)])

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT strength FROM relationships").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.5)


def test_save_relationships_multiple(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    rels = [
        Relationship("a", "b", "semantic", 0.9),
        Relationship("c", "d", "co_access", 0.7),
    ]
    assert e.save_relationships(rels) == 2


# ── get_relationships ───────────────────────────────────────────────────


def test_get_relationships_as_source(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    e.save_relationships([Relationship("a", "b", "semantic", 0.9)])
    rels = e.get_relationships("a")
    assert len(rels) == 1
    assert rels[0]["related_id"] == "b"
    assert rels[0]["type"] == "semantic"


def test_get_relationships_as_target(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    e.save_relationships([Relationship("a", "b", "semantic", 0.9)])
    rels = e.get_relationships("b")
    assert rels[0]["related_id"] == "a"


def test_get_relationships_missing(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    assert e.get_relationships("nope") == []


def test_get_relationships_bidirectional(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    e.save_relationships([
        Relationship("a", "b", "semantic", 0.9),  # a → b
        Relationship("c", "a", "co_access", 0.5),  # c → a
    ])
    rels = e.get_relationships("a")
    assert len(rels) == 2


# ── get_graph_stats ─────────────────────────────────────────────────────


def test_get_graph_stats_empty(tmp_path) -> None:
    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    stats = e.get_graph_stats()
    assert stats.total_nodes == 0
    assert stats.total_edges == 0
    assert stats.avg_degree == 0
    assert stats.connected_components == 1


def test_get_graph_stats_with_edges(tmp_path) -> None:
    db = tmp_path / "k.db"
    e = KnowledgeGraphEnhancer(db_path=str(db), embedding_path=str(tmp_path / "e.json"))
    e.save_relationships([
        Relationship("a", "b", "semantic", 0.9),
        Relationship("b", "c", "co_access", 0.5),
    ])
    stats = e.get_graph_stats()
    assert stats.total_edges == 2
    # distinct nodes: a, b, c → 3
    assert stats.total_nodes == 3
    # avg_degree = 2*2/3
    assert stats.avg_degree == pytest.approx(4 / 3)


# ── get_enhancer singleton ──────────────────────────────────────────────


def test_get_enhancer_singleton(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kge, "_enhancer", None)
    db = str(tmp_path / "k.db")
    emb = str(tmp_path / "e.json")
    e1 = get_enhancer(db_path=db, embedding_path=emb)
    e2 = get_enhancer(db_path=db, embedding_path=emb)
    assert e1 is e2


# ── discover_and_save_relationships ─────────────────────────────────────


def test_discover_and_save_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kge, "_enhancer", None)
    import omnimem.retrieval.vector_store as vs

    monkeypatch.setattr(vs, "load_embedding_cache_dict", lambda p: {})
    # Point HOME somewhere without forgetting.db
    monkeypatch.setenv("HOME", str(tmp_path))

    e = KnowledgeGraphEnhancer(db_path=str(tmp_path / "k.db"), embedding_path=str(tmp_path / "e.json"))
    monkeypatch.setattr(kge, "_enhancer", e)
    result = discover_and_save_relationships()
    assert result == {"semantic": 0, "co_access": 0, "total_saved": 0}


def test_discover_and_save_with_semantic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(kge, "_enhancer", None)
    monkeypatch.setenv("HOME", str(tmp_path))
    db = str(tmp_path / "k.db")
    e = KnowledgeGraphEnhancer(db_path=db, embedding_path=str(tmp_path / "e.json"))
    _inject_embeddings(e, {"a": [1.0, 0.0], "b": [1.0, 0.001]})
    monkeypatch.setattr(kge, "_enhancer", e)
    result = discover_and_save_relationships(threshold=0.7)
    assert result["semantic"] == 1
    assert result["total_saved"] == 1
