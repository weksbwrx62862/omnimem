"""Tests for governance.performance_optimizer — LRU / batch retention / parallel evaluate."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from governance import performance_optimizer as po
from governance.performance_optimizer import (
    CacheEntry,
    LRUCache,
    PerformanceOptimizer,
    PerformanceStats,
    batch_calculate_retention,
    get_optimizer,
)

# ── CacheEntry / PerformanceStats dataclasses ───────────────────────────


def test_cache_entry_defaults() -> None:
    ts = datetime(2025, 1, 1)
    e = CacheEntry(key="k", value=42, created_at=ts)
    assert e.access_count == 0
    assert e.last_accessed is not None


def test_performance_stats_fields() -> None:
    s = PerformanceStats(total_calls=10, cache_hits=7, cache_misses=3, avg_time_ms=1.5, batch_speedup=2.0)
    assert s.total_calls == 10
    assert s.batch_speedup == 2.0


# ── LRUCache ────────────────────────────────────────────────────────────


def test_lru_put_and_get() -> None:
    c = LRUCache(max_size=10)
    c.put("a", 1)
    assert c.get("a") == 1


def test_lru_get_missing_returns_none() -> None:
    c = LRUCache()
    assert c.get("missing") is None


def test_lru_get_missing_counts_miss() -> None:
    c = LRUCache()
    c.get("x")
    assert c._misses == 1
    assert c._hits == 0


def test_lru_get_hit_counts_hit() -> None:
    c = LRUCache()
    c.put("a", 1)
    c.get("a")
    assert c._hits == 1


def test_lru_over_capacity_evicts_oldest() -> None:
    c = LRUCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts a
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_lru_access_moves_to_end() -> None:
    c = LRUCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # move a to end
    c.put("c", 3)  # evict b (LRU)
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_lru_update_existing_key() -> None:
    c = LRUCache(max_size=5)
    c.put("a", 1)
    c.put("a", 2)
    assert c.get("a") == 2
    assert len(c._cache) == 1


def test_lru_ttl_expiry(tmp_path) -> None:
    c = LRUCache(max_size=5, ttl_seconds=1)
    c.put("a", 1)
    # Force created_at in the past
    entry = c._cache["a"]
    entry.created_at = datetime.now() - timedelta(seconds=2)
    assert c.get("a") is None


def test_lru_ttl_not_expired() -> None:
    c = LRUCache(max_size=5, ttl_seconds=3600)
    c.put("a", 1)
    assert c.get("a") == 1


def test_lru_ttl_expired_counts_miss() -> None:
    c = LRUCache(max_size=5, ttl_seconds=1)
    c.put("a", 1)
    c._cache["a"].created_at = datetime.now() - timedelta(seconds=2)
    c.get("a")
    assert c._misses == 1


def test_lru_clear() -> None:
    c = LRUCache()
    c.put("a", 1)
    c.get("a")
    c.clear()
    assert c.get_stats()["size"] == 0
    assert c.get_stats()["hits"] == 0
    assert c.get_stats()["misses"] == 0


def test_lru_get_stats_hit_rate() -> None:
    c = LRUCache()
    c.put("a", 1)
    c.get("a")  # hit
    c.get("b")  # miss
    stats = c.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_lru_get_stats_zero_total() -> None:
    c = LRUCache()
    stats = c.get_stats()
    assert stats["hit_rate"] == 0


def test_lru_get_stats_max_size() -> None:
    c = LRUCache(max_size=42)
    assert c.get_stats()["max_size"] == 42


def test_lru_access_count_increments() -> None:
    c = LRUCache()
    c.put("a", 1)
    c.get("a")
    c.get("a")
    assert c._cache["a"].access_count == 2


# ── PerformanceOptimizer constructor ────────────────────────────────────


def test_optimizer_default_governance_dir() -> None:
    o = PerformanceOptimizer()
    assert "governance" in str(o._governance_dir)


def test_optimizer_explicit_dir(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    assert o._governance_dir == tmp_path


def test_optimizer_cache_size(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path, cache_size=7)
    assert o._cache._max_size == 7


def test_optimizer_max_workers(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path, max_workers=8)
    assert o._max_workers == 8


# ── batch_calculate_retention ───────────────────────────────────────────


def test_batch_retention_no_db_returns_default(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m1", "m2"])
    assert result == {"m1": 0.5, "m2": 0.5}


def test_batch_retention_empty_list(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    assert o.batch_calculate_retention([]) == {}


def _seed_db(db_path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE forgetting_state ("
        "memory_id TEXT PRIMARY KEY, "
        "recall_count INTEGER, "
        "last_accessed TEXT)"
    )
    conn.executemany(
        "INSERT INTO forgetting_state VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_batch_retention_from_db(tmp_path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).isoformat()
    _seed_db(db, [("m1", 5, recent), ("m2", 10, recent)])

    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m1", "m2"])
    assert "m1" in result
    assert "m2" in result
    assert 0.0 <= result["m1"] <= 1.0


def test_batch_retention_missing_id_defaults(tmp_path) -> None:
    db = tmp_path / "forgetting.db"
    _seed_db(db, [("m1", 5, datetime.now(timezone.utc).isoformat())])
    o = PerformanceOptimizer(governance_dir=tmp_path)
    # m_missing not in DB → not in results (dict only has rows returned)
    result = o.batch_calculate_retention(["m1", "m_missing"])
    assert "m1" in result
    # m_missing not in DB → key absent
    assert "m_missing" not in result


def test_batch_retention_no_last_accessed_defaults_half(tmp_path) -> None:
    db = tmp_path / "forgetting.db"
    _seed_db(db, [("m1", 5, None)])
    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m1"])
    assert result["m1"] == 0.5


def test_batch_retention_bad_timestamp_defaults_half(tmp_path) -> None:
    db = tmp_path / "forgetting.db"
    _seed_db(db, [("m1", 5, "not-a-timestamp")])
    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m1"])
    assert result["m1"] == 0.5


def test_batch_retention_older_access_lower_retention(tmp_path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=100)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()
    _seed_db(db, [("m_old", 5, old), ("m_recent", 5, recent)])

    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m_old", "m_recent"])
    assert result["m_old"] < result["m_recent"]


def test_batch_retention_uses_cache(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    r1 = o.batch_calculate_retention(["m1"])
    r2 = o.batch_calculate_retention(["m1"])
    assert r1 == r2
    # Second call should hit cache
    assert o._cache._hits >= 1


def test_batch_retention_updates_stats(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    o.batch_calculate_retention(["m1"])
    o.batch_calculate_retention(["m2"])
    assert o._stats["total_calls"] == 2


def test_batch_retention_db_exception_fallback(tmp_path, monkeypatch) -> None:
    # Create a directory where forgetting.db should be → sqlite3.connect fails
    (tmp_path / "forgetting.db").mkdir()

    def _boom(*a, **kw):
        raise sqlite3.OperationalError("nope")

    monkeypatch.setattr(sqlite3, "connect", _boom)
    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.batch_calculate_retention(["m1"])
    # Exception path → dict.fromkeys default
    assert result == {"m1": 0.5}


# ── parallel_evaluate ───────────────────────────────────────────────────


def test_parallel_evaluate_empty(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    assert o.parallel_evaluate([], lambda x: x) == {}


def test_parallel_evaluate_simple(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.parallel_evaluate(["a", "b", "c"], lambda x: x.upper())
    assert result == {"A": "A", "B": "B", "C": "C"} or result == {"a": "A", "b": "B", "c": "C"}


def test_parallel_evaluate_exception_captured(tmp_path) -> None:
    def boom(x):
        raise ValueError(f"bad {x}")

    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.parallel_evaluate(["a"], boom)
    assert "a" in result
    assert "error" in result["a"]


def test_parallel_evaluate_mixed_success_and_failure(tmp_path) -> None:
    def maybe_fail(x):
        if x == "bad":
            raise RuntimeError("boom")
        return x * 2

    o = PerformanceOptimizer(governance_dir=tmp_path)
    result = o.parallel_evaluate(["good", "bad"], maybe_fail)
    assert result["good"] == "goodgood"
    assert "error" in result["bad"]


def test_parallel_evaluate_uses_workers(tmp_path) -> None:
    # Just verify no crash with max_workers=1
    o = PerformanceOptimizer(governance_dir=tmp_path, max_workers=1)
    result = o.parallel_evaluate(["a", "b"], lambda x: x)
    assert result == {"a": "a", "b": "b"}


# ── get_stats ───────────────────────────────────────────────────────────


def test_get_stats_zero_calls(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    stats = o.get_stats()
    assert stats.total_calls == 0
    assert stats.avg_time_ms == 0


def test_get_stats_after_calls(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    o.batch_calculate_retention(["a"])
    stats = o.get_stats()
    assert stats.total_calls == 1
    assert stats.avg_time_ms >= 0


def test_get_stats_batch_speedup_hardcoded(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    assert o.get_stats().batch_speedup == 1.0


def test_get_stats_reflects_cache_hits(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    o.batch_calculate_retention(["a"])
    o.batch_calculate_retention(["a"])  # cached
    stats = o.get_stats()
    assert stats.cache_hits >= 1


# ── clear_cache ─────────────────────────────────────────────────────────


def test_clear_cache(tmp_path) -> None:
    o = PerformanceOptimizer(governance_dir=tmp_path)
    o.batch_calculate_retention(["a"])
    o.clear_cache()
    assert o._cache.get_stats()["size"] == 0


# ── get_optimizer singleton ─────────────────────────────────────────────


def test_get_optimizer_singleton(monkeypatch) -> None:
    monkeypatch.setattr(po, "_optimizer", None)
    o1 = get_optimizer()
    o2 = get_optimizer()
    assert o1 is o2


def test_get_optimizer_first_call_kwargs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(po, "_optimizer", None)
    o = get_optimizer(governance_dir=tmp_path, cache_size=10, cache_ttl=60)
    assert o._governance_dir == tmp_path
    assert o._cache._max_size == 10
    assert o._cache._ttl_seconds == 60


# ── module-level batch_calculate_retention ──────────────────────────────


def test_module_batch_calculate_retention(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(po, "_optimizer", None)
    result = batch_calculate_retention([])
    assert result == {}


def test_module_batch_calculate_retention_with_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(po, "_optimizer", None)
    o = PerformanceOptimizer(governance_dir=tmp_path)
    monkeypatch.setattr(po, "_optimizer", o)
    result = batch_calculate_retention(["x"])
    assert result == {"x": 0.5}  # no DB → default
