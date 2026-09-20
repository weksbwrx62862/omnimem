"""governance.provenance.ProvenanceTracker 离线单元测试。

覆盖：
  - track：字段生成、哈希、时间戳
  - record + lookup：内存索引、miss → not_found
  - _persist + _restore：SQLite 往返、parent_id/metadata JSON
  - get_chain：多跳 parent、无 parent、循环防护
  - close / 无 data_dir 纯内存模式
  - 非法 metadata JSON 恢复不抛
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnimem.governance.provenance import ProvenanceTracker


@pytest.fixture
def tracker(tmp_path: Path):
    t = ProvenanceTracker(data_dir=tmp_path / "prov")
    yield t
    t.close()


# ── track ──


def test_track_returns_expected_shape(tracker: ProvenanceTracker) -> None:
    p = tracker.track("hello", source="s", method="manual")
    assert p["source"] == "s"
    assert p["method"] == "manual"
    assert "timestamp" in p
    assert len(p["content_hash"]) == 16


def test_track_defaults_method_to_auto_detect(tracker: ProvenanceTracker) -> None:
    assert tracker.track("x")["method"] == "auto_detect"


def test_hash_stable_and_different(tracker: ProvenanceTracker) -> None:
    a = tracker._hash("content")
    b = tracker._hash("content")
    c = tracker._hash("other")
    assert a == b
    assert a != c


# ── record + lookup ──


def test_lookup_missing_returns_not_found(tracker: ProvenanceTracker) -> None:
    r = tracker.lookup("unknown")
    assert r["status"] == "not_found"
    assert r["memory_id"] == "unknown"


def test_record_then_lookup_roundtrip(tracker: ProvenanceTracker) -> None:
    prov = tracker.track("hello", "s1", "manual")
    tracker.record("m1", prov)
    out = tracker.lookup("m1")
    assert out["source"] == "s1"
    assert out["content_hash"] == prov["content_hash"]


def test_record_persists_to_sqlite(tmp_path: Path) -> None:
    t = ProvenanceTracker(data_dir=tmp_path / "prov2")
    t.record("m1", t.track("hello", "src", "manual"))
    t.close()

    # 重开：从磁盘恢复
    t2 = ProvenanceTracker(data_dir=tmp_path / "prov2")
    out = t2.lookup("m1")
    assert out["source"] == "src"
    assert out["method"] == "manual"
    t2.close()


def test_record_replaces_existing(tracker: ProvenanceTracker) -> None:
    tracker.record("m1", {"source": "a", "method": "manual"})
    tracker.record("m1", {"source": "b", "method": "auto"})
    assert tracker.lookup("m1")["source"] == "b"


def test_record_with_parent_and_metadata(tracker: ProvenanceTracker) -> None:
    tracker.record(
        "m1",
        {"source": "s", "method": "manual", "parent_id": "p0", "metadata": {"k": "v"}},
    )
    out = tracker.lookup("m1")
    assert out["parent_id"] == "p0"
    assert out["metadata"] == {"k": "v"}


# ── restore 边界 ──


def test_restore_ignores_bad_metadata_json(tmp_path: Path) -> None:
    # 先建 DB
    t = ProvenanceTracker(data_dir=tmp_path / "prov3")
    t.record("m1", {"source": "s", "method": "manual"})
    t.close()

    # 手工把 metadata 列写入非 JSON
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "prov3" / "provenance.db"))
    conn.execute("UPDATE provenance SET metadata = 'not-json' WHERE memory_id = 'm1'")
    conn.commit()
    conn.close()

    t2 = ProvenanceTracker(data_dir=tmp_path / "prov3")
    out = t2.lookup("m1")
    # 非法 JSON 被 suppress，不写入 metadata
    assert "metadata" not in out
    assert out["source"] == "s"
    t2.close()


# ── get_chain ──


def test_get_chain_walks_parent_ids(tracker: ProvenanceTracker) -> None:
    tracker.record("c", {"source": "s", "method": "manual", "parent_id": "b"})
    tracker.record("b", {"source": "s", "method": "manual", "parent_id": "a"})
    tracker.record("a", {"source": "s", "method": "manual"})
    chain = tracker.get_chain("c")
    assert len(chain) == 3
    assert chain[0]["parent_id"] == "b"
    assert "parent_id" not in chain[-1] or not chain[-1].get("parent_id")


def test_get_chain_missing_root_returns_empty(tracker: ProvenanceTracker) -> None:
    assert tracker.get_chain("nope") == []


def test_get_chain_handles_cycle_safely(tracker: ProvenanceTracker) -> None:
    # a → b → a 循环
    tracker.record("a", {"parent_id": "b"})
    tracker.record("b", {"parent_id": "a"})
    chain = tracker.get_chain("a")
    # visited 集合防止死循环；两次访问后 break
    assert len(chain) == 2


def test_get_chain_single_node(tracker: ProvenanceTracker) -> None:
    tracker.record("solo", {"source": "s"})
    chain = tracker.get_chain("solo")
    assert len(chain) == 1


# ── 无 data_dir ──


def test_no_data_dir_pure_memory() -> None:
    t = ProvenanceTracker()
    t.record("m1", {"source": "s", "method": "manual"})
    assert t.lookup("m1")["source"] == "s"
    # _persist 无 conn → 静默跳过
    t.close()


def test_persist_without_conn_is_noop() -> None:
    t = ProvenanceTracker()
    t.close()  # 无 conn，静默
    # 二次 close 也不抛
    t.close()


def test_record_after_close_still_writes_memory(tracker: ProvenanceTracker) -> None:
    tracker.close()
    tracker.record("m1", {"source": "post-close"})
    assert tracker.lookup("m1")["source"] == "post-close"
