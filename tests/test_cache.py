"""缓存模块测试 — Task 6：多级缓存 tag 失效优化。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from omnimem.utils.cache import (
    L1LRUCache,
    L2RedisCache,
    L3PersistentCache,
    MultiLevelCache,
)


class TestL1LRUCache:
    """L1 内存缓存行为保持不变的回归测试。"""

    def test_basic_get_set(self) -> None:
        cache = L1LRUCache(max_size=10, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_ttl_expiration(self) -> None:
        cache = L1LRUCache(max_size=10, ttl=0.05)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.06)
        assert cache.get("key1") is None

    def test_delete_by_tag(self) -> None:
        cache = L1LRUCache(max_size=10, ttl=60)
        cache.set("k1", "v1", tags={"t1", "t2"})
        cache.set("k2", "v2", tags={"t1"})
        cache.set("k3", "v3", tags={"t2"})
        deleted = cache.delete_by_tag("t1")
        assert deleted == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"

    def test_tag_update_on_overwrite(self) -> None:
        cache = L1LRUCache(max_size=10, ttl=60)
        cache.set("k1", "v1", tags={"old"})
        cache.set("k1", "v2", tags={"new"})
        assert cache.delete_by_tag("old") == 0
        assert cache.delete_by_tag("new") == 1


class TestL2RedisCacheSync:
    """L2 Redis 同步封装在降级路径下应保持安全。"""

    def test_sync_wrappers_degrade_without_redis(self) -> None:
        cache = L2RedisCache(redis_url=None)
        assert not cache.is_available()
        assert cache.get_sync("key") is None
        cache.set_sync("key", "value")
        cache.delete_sync("key")

    def test_sync_wrapper_returns_none_when_unavailable(self) -> None:
        cache = L2RedisCache(redis_url="redis://invalid:6379/0")
        assert not cache.is_available()
        assert cache.get_sync("key") is None


class TestL3PersistentCacheTagIndex:
    """L3 SQLite tag 反向索引正确性与性能测试。"""

    @pytest.fixture
    def l3(self, omni_tmp_path: Path):
        cache = L3PersistentCache(cache_dir=omni_tmp_path)
        yield cache
        cache.close()

    def test_delete_by_tag_uses_index(self, l3: L3PersistentCache) -> None:
        l3.set("k1", "v1", tags={"t1", "t2"})
        l3.set("k2", "v2", tags={"t1"})
        l3.set("k3", "v3", tags={"t2"})
        l3.set("k4", "v4", tags=set())

        assert l3.delete_by_tag("t1") == 2
        assert l3.get("k1") is None
        assert l3.get("k2") is None
        assert l3.get("k3") == "v3"
        assert l3.get("k4") == "v4"

    def test_tag_index_updated_on_overwrite(self, l3: L3PersistentCache) -> None:
        l3.set("k1", "v1", tags={"old"})
        l3.set("k1", "v2", tags={"new"})
        assert l3.delete_by_tag("old") == 0
        assert l3.delete_by_tag("new") == 1
        assert l3.get("k1") is None

    def test_tag_index_cleaned_on_key_delete(self, l3: L3PersistentCache) -> None:
        l3.set("k1", "v1", tags={"t1"})
        l3.delete("k1")
        assert l3.delete_by_tag("t1") == 0

    def test_clear_removes_tag_index(self, l3: L3PersistentCache) -> None:
        l3.set("k1", "v1", tags={"t1"})
        l3.clear()
        assert l3.delete_by_tag("t1") == 0
        assert l3.stats()["size"] == 0

    def test_delete_by_tag_performance(self, omni_tmp_path: Path) -> None:
        """反向索引应使 tag 失效时间与总数据量解耦。"""
        l3 = L3PersistentCache(cache_dir=omni_tmp_path)
        total = 1000
        for i in range(total):
            tags = {"target"} if i < 10 else {"other"}
            l3.set(f"key:{i}", f"value:{i}", tags=tags)

        start = time.perf_counter()
        deleted = l3.delete_by_tag("target")
        elapsed = time.perf_counter() - start

        assert deleted == 10
        assert l3.stats()["size"] == total - 10
        # 1000 行规模下，反向索引失效应远小于朴素全表扫描
        assert elapsed < 1.0
        l3.close()


class TestMultiLevelCacheTagInvalidation:
    """MultiLevelCache tag 失效端到端测试。"""

    def test_get_uses_l2_sync_wrapper(self, omni_tmp_path: Path) -> None:
        ml = MultiLevelCache(l1_size=10, l1_ttl=60, cache_dir=omni_tmp_path)
        ml.set("k1", "v1")
        # 同步上下文中 get 不应触发 asyncio 事件循环异常
        assert ml.get("k1") == "v1"
        ml.close()

    def test_delete_by_tag_invalidates_all_levels(self, omni_tmp_path: Path) -> None:
        ml = MultiLevelCache(l1_size=10, l1_ttl=60, cache_dir=omni_tmp_path)
        ml.set("k1", "v1", tags={"mem:1"})
        ml.set("k2", "v2", tags={"mem:1"})
        ml.set("k3", "v3", tags={"mem:2"})

        ml.delete_by_tag("mem:1")

        assert ml.get("k1") is None
        assert ml.get("k2") is None
        assert ml.get("k3") == "v3"
        ml.close()

    def test_fallback_without_l3(self) -> None:
        ml = MultiLevelCache(l1_size=10, l1_ttl=60, cache_dir=None)
        ml.set("k1", "v1")
        assert ml.get("k1") == "v1"
        ml.delete_by_tag("nonexistent")
        assert ml.get("k1") == "v1"

    @pytest.mark.asyncio
    async def test_async_interfaces_unchanged(self, omni_tmp_path: Path) -> None:
        ml = MultiLevelCache(l1_size=10, l1_ttl=60, cache_dir=omni_tmp_path)
        await ml.async_set("k1", "v1", tags={"t1"})
        assert await ml.async_get("k1") == "v1"
        ml.close()

    def test_get_inside_running_event_loop(self, omni_tmp_path: Path) -> None:
        """确保同步 get 在已有事件循环的线程中也能安全返回。"""
        ml = MultiLevelCache(l1_size=10, l1_ttl=60, cache_dir=omni_tmp_path)
        ml.set("k1", "v1")

        def run_in_loop() -> Any:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return ml.get("k1")
            finally:
                loop.close()

        result = run_in_loop()
        assert result == "v1"
        ml.close()
