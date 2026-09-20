"""QueryCacheMixin 单测（改进项 #2 覆盖，离线）。

覆盖 dict 降级缓存的读写/TTL/过期清理/LRU 淘汰，以及 ML 缓存优先路径与
其异常回退。通过一个最小 fake facade 注入 _facade 契约（_ml_cache/_query_cache/
_query_cache_ttl），无需真实 ML 缓存或 HybridRetriever。
"""
from __future__ import annotations

import time
from typing import Any

from omnimem.retrieval.cache import QueryCacheMixin


class _FakeMLCache:
    def __init__(self, *, raise_get=False, raise_set=False) -> None:
        self.store: dict[str, Any] = {}
        self.raise_get = raise_get
        self.raise_set = raise_set
        self.invalidated: list[str] = []
        self.cleared = 0
        self.last_tags: set[str] | None = None

    def get(self, key):
        if self.raise_get:
            raise RuntimeError("ml get down")
        return self.store.get(key)

    def set(self, key, results, ttl=None, tags=None):
        if self.raise_set:
            raise RuntimeError("ml set down")
        self.store[key] = results
        self.last_tags = tags

    def invalidate_memory(self, mid):
        self.invalidated.append(mid)

    def clear(self):
        self.cleared += 1
        self.store.clear()


class _Facade:
    def __init__(self, ttl: float = 60.0, ml=None) -> None:
        self._query_cache: dict[str, tuple[list, float]] = {}
        self._query_cache_ttl = ttl
        self._ml_cache = ml


class _Cache(QueryCacheMixin):
    def __init__(self, facade: _Facade) -> None:
        self._facade = facade


# ── dict 降级路径 ─────────────────────────────────────
def test_dict_set_then_hit():
    c = _Cache(_Facade())
    c.set_cache("k", [{"memory_id": "m1"}])
    assert c.check_cache("k") == [{"memory_id": "m1"}]


def test_dict_miss_returns_none():
    c = _Cache(_Facade())
    assert c.check_cache("absent") is None


def test_dict_ttl_expiry():
    c = _Cache(_Facade(ttl=60.0))
    c.set_cache("k", [{"memory_id": "m1"}])
    # 手动把写入时间拨到远超 TTL
    results, _ = c._facade._query_cache["k"]
    c._facade._query_cache["k"] = (results, time.time() - 9999)
    assert c.check_cache("k") is None


def test_cleanup_removes_expired_only():
    c = _Cache(_Facade(ttl=50.0))
    c.set_cache("fresh", [{"memory_id": "a"}])
    c.set_cache("stale", [{"memory_id": "b"}])
    res, _ = c._facade._query_cache["stale"]
    c._facade._query_cache["stale"] = (res, time.time() - 9999)
    c.cleanup_query_cache()
    assert "fresh" in c._facade._query_cache
    assert "stale" not in c._facade._query_cache


def test_set_tags_from_memory_ids():
    ml = _FakeMLCache()
    c = _Cache(_Facade(ml=ml))
    c.set_cache("k", [{"memory_id": "m1"}, {"memory_id": "m2"}, {"content": "no-id"}])
    assert ml.last_tags == {"memory:m1", "memory:m2"}


def test_invalidate_clears_dict_when_no_ml():
    c = _Cache(_Facade())
    c.set_cache("k", [{"memory_id": "m1"}])
    c.invalidate_cache_by_memory("m1")
    assert c._facade._query_cache == {}  # dict 降级：全清


def test_clear_all_cache_dict():
    c = _Cache(_Facade())
    c.set_cache("k", [{"memory_id": "m1"}])
    c.clear_all_cache()
    assert c._facade._query_cache == {}


def test_lru_eviction_bounds_growth():
    c = _Cache(_Facade())
    for i in range(2001):
        c.set_cache(f"k{i}", [{"memory_id": f"m{i}"}])
    # 超上限后按最旧淘汰约 20%，规模回到 ~1600
    assert len(c._facade._query_cache) <= 2000
    assert "k2000" in c._facade._query_cache   # 最新保留
    assert "k0" not in c._facade._query_cache  # 最旧淘汰


# ── ML 缓存优先路径 ───────────────────────────────────
def test_ml_cache_preferred_for_read():
    ml = _FakeMLCache()
    ml.store["k"] = [{"memory_id": "hit"}]
    c = _Cache(_Facade(ml=ml))
    assert c.check_cache("k") == [{"memory_id": "hit"}]


def test_ml_cache_set_writes_to_ml_not_dict():
    ml = _FakeMLCache()
    c = _Cache(_Facade(ml=ml))
    c.set_cache("k", [{"memory_id": "m1"}])
    assert ml.store["k"] == [{"memory_id": "m1"}]
    assert c._facade._query_cache == {}  # 未触碰 dict


def test_ml_get_exception_falls_back_to_dict():
    ml = _FakeMLCache(raise_get=True)
    c = _Cache(_Facade(ml=ml))
    c._facade._query_cache["k"] = ([{"memory_id": "d"}], time.time())
    assert c.check_cache("k") == [{"memory_id": "d"}]  # 回退命中 dict


def test_ml_set_exception_falls_back_to_dict_write():
    ml = _FakeMLCache(raise_set=True)
    c = _Cache(_Facade(ml=ml))
    c.set_cache("k", [{"memory_id": "m1"}])
    # ML set 抛错 → 降级写入 dict（注意：check_cache 仍走 ML 短路，不读此 dict 条目）
    assert "k" in c._facade._query_cache
    assert "k" not in ml.store


def test_ml_invalidate_and_clear_delegate():
    ml = _FakeMLCache()
    c = _Cache(_Facade(ml=ml))
    c.invalidate_cache_by_memory("m9")
    assert ml.invalidated == ["m9"]
    c.clear_all_cache()
    assert ml.cleared == 1
