"""internalize/kv_cache.py 单元测试：LRU/阈值/持久化路径。"""

from __future__ import annotations

import pytest
from omnimem.internalize.kv_cache import KVCacheManager


@pytest.fixture
def mgr(tmp_path):
    m = KVCacheManager(
        data_dir=tmp_path / "kv",
        auto_preload_threshold=3,
        max_cache_size=20,
    )
    try:
        yield m
    finally:
        m.close()


def test_preload_returns_count_and_skips_invalid(mgr):
    patterns = [
        {"key": "a", "content": "A"},
        {"key": "", "content": "空 key 应被跳过"},
        {"key": "b", "content": ""},
        {"key": "c", "content": "C"},
    ]
    count = mgr.preload(patterns)
    assert count == 2
    assert mgr.is_cached("a")
    assert mgr.is_cached("c")
    assert not mgr.is_cached("b")


def test_get_increments_access_count_and_miss_returns_none(mgr):
    mgr.preload([{"key": "k", "content": "V"}])
    assert mgr.get("k") is not None
    assert mgr.get("k") is not None
    assert mgr._access_counts["k"] == 2
    assert mgr.get("missing") is None


def test_is_cached_reflects_preload(mgr):
    assert not mgr.is_cached("x")
    mgr.preload([{"key": "x", "content": "y"}])
    assert mgr.is_cached("x")


def test_get_hot_patterns_sorted_desc_with_count(mgr):
    mgr.preload(
        [
            {"key": "hot", "content": "H"},
            {"key": "cold", "content": "C"},
            {"key": "mid", "content": "M"},
        ]
    )
    for _ in range(5):
        mgr.get("hot")
    for _ in range(2):
        mgr.get("mid")
    hot = mgr.get_hot_patterns(top_k=2)
    assert [p["key"] for p in hot] == ["hot", "mid"]
    assert hot[0]["access_count"] == 5
    assert hot[1]["access_count"] == 2


def test_check_and_auto_preload_below_threshold(mgr):
    # threshold=3：前两次不触发
    assert mgr.check_and_auto_preload("k", "content") is False
    assert not mgr.is_cached("k")
    assert mgr.check_and_auto_preload("k", "content") is False
    assert not mgr.is_cached("k")


def test_check_and_auto_preload_hits_threshold(mgr):
    mgr.check_and_auto_preload("k", "content")
    mgr.check_and_auto_preload("k", "content")
    assert mgr.check_and_auto_preload("k", "content", metadata={"t": 1}, source_memory_ids=["m"]) is True
    assert mgr.is_cached("k")
    entry = mgr.get("k")
    assert entry["metadata"] == {"t": 1}
    assert entry["source_memory_ids"] == ["m"]


def test_check_and_auto_preload_no_op_when_already_cached(mgr):
    mgr.preload([{"key": "k", "content": "c"}])
    # 已缓存 → 只更新计数，返回 False
    assert mgr.check_and_auto_preload("k", "c") is False
    # 计数被 _update_access_count 更新
    assert mgr._access_counts["k"] == 1


def test_search_cache_substring_and_limit(mgr):
    mgr.preload(
        [
            {"key": "a", "content": "Apple pie is nice"},
            {"key": "b", "content": "Banana split"},
            {"key": "c", "content": "Cherry Apple"},
        ]
    )
    results = mgr.search_cache("apple", limit=5)
    keys = sorted(r["key"] for r in results)
    assert keys == ["a", "c"]

    limited = mgr.search_cache("apple", limit=1)
    assert len(limited) == 1


def test_clear_empties_cache_and_db(mgr):
    mgr.preload([{"key": "a", "content": "A"}])
    mgr.clear()
    assert not mgr.is_cached("a")
    assert mgr._access_counts == {}
    stats = mgr.get_stats()
    assert stats["cached_entries"] == 0


def test_get_stats_shape(mgr):
    mgr.preload([{"key": "a", "content": "A"}, {"key": "b", "content": "B"}])
    for _ in range(3):
        mgr.get("a")
    stats = mgr.get_stats()
    assert stats["cached_entries"] == 2
    assert stats["auto_preload_threshold"] == 3
    assert stats["max_cache_size"] == 20
    assert stats["total_preloaded"] == 2
    assert stats["total_accesses"] >= 3
    top_keys = [p["key"] for p in stats["top_patterns"]]
    assert "a" in top_keys


def test_persistence_round_trip(tmp_path):
    d = tmp_path / "kv"
    m1 = KVCacheManager(data_dir=d, auto_preload_threshold=3, max_cache_size=20)
    m1.preload(
        [
            {
                "key": "abc",
                "content": "hello",
                "metadata": {"t": 1},
                "source_memory_ids": ["m"],
            }
        ]
    )
    m1.close()

    m2 = KVCacheManager(data_dir=d, auto_preload_threshold=3, max_cache_size=20)
    try:
        assert m2.is_cached("abc")
        entry = m2.get("abc")
        assert entry["content"] == "hello"
        assert entry["metadata"] == {"t": 1}
        assert entry["source_memory_ids"] == ["m"]
    finally:
        m2.close()


def test_no_data_dir_only_in_memory():
    m = KVCacheManager(auto_preload_threshold=2, max_cache_size=10)
    try:
        assert m._conn is None
        assert m.preload([{"key": "k", "content": "v"}]) == 1
        assert m.get("k") is not None
    finally:
        m.close()


def test_restore_tolerates_malformed_metadata_json(tmp_path):
    """损坏的 metadata JSON → 恢复时降级为 {}，不抛异常。"""
    d = tmp_path / "kv"
    m = KVCacheManager(data_dir=d, auto_preload_threshold=3, max_cache_size=20)
    m.preload([{"key": "k", "content": "v", "metadata": {"t": 1}}])
    # 手动把 metadata 字段改成坏 JSON
    m._conn.execute(
        "UPDATE kv_cache_entries SET metadata = ?, source_memory_ids = ? WHERE cache_key = ?",
        ("not-json", "[bad", "k"),
    )
    m._conn.commit()
    m.close()

    m2 = KVCacheManager(data_dir=d, auto_preload_threshold=3, max_cache_size=20)
    try:
        entry = m2.get("k")
        assert entry is not None
        assert entry["metadata"] == {}
        assert entry["source_memory_ids"] == []
    finally:
        m2.close()
