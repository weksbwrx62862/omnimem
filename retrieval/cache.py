"""查询缓存管理（★ M6-9：从 hybrid_orchestrator.py 拆分）。"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class QueryCacheMixin:
    """ML 缓存优先，dict 缓存降级兑底。"""

    def check_cache(self, cache_key: str) -> list[dict[str, Any]] | None:
        """查询缓存检查（需在读锁内调用）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                cached = facade._ml_cache.get(cache_key)
                if cached is not None:
                    logger.debug("HybridRetriever ML cache hit: %s", cache_key[:50])
                    return cached
                return None
            except Exception as e:
                logger.debug("ML cache get failed, falling back to dict: %s", e)
        now = time.time()
        if cache_key in facade._query_cache:
            cached_results, cached_time = facade._query_cache[cache_key]
            if now - cached_time < facade._query_cache_ttl:
                logger.debug("HybridRetriever query cache hit")
                return cached_results
        return None

    def set_cache(self, cache_key: str, results: list[dict[str, Any]]) -> None:
        """写入查询缓存（需在读锁内调用）。"""
        facade = self._facade
        tags: set[str] = set()
        for r in results:
            mid = r.get("memory_id", "")
            if mid:
                tags.add(f"memory:{mid}")
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.set(cache_key, results, ttl=facade._query_cache_ttl, tags=tags)
                return
            except Exception as e:
                logger.debug("ML cache set failed, falling back to dict: %s", e)
        facade._query_cache[cache_key] = (results, time.time())
        # ★ 修复 L3：_query_cache 无上限，长期运行无限增长。加入 LRU 淘汰
        max_query_cache = 2000
        if len(facade._query_cache) > max_query_cache:
            # 淘汰最旧的 20% 条目（按写入时间）
            sorted_items = sorted(
                facade._query_cache.items(), key=lambda kv: kv[1][1]
            )
            evict_count = len(facade._query_cache) - max_query_cache + max_query_cache // 5
            for k, _ in sorted_items[:evict_count]:
                facade._query_cache.pop(k, None)

    def invalidate_cache_by_memory(self, memory_id: str) -> None:
        """按 memory_id 精准失效相关缓存（用于 add 时调用）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.invalidate_memory(memory_id)
                return
            except Exception as e:
                logger.debug("ML cache invalidate_memory failed: %s", e)
        facade._query_cache.clear()

    def clear_all_cache(self) -> None:
        """清空所有查询缓存（用于 delete/update/rebuild 等全量失效场景）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.clear()
            except Exception as e:
                logger.debug("ML cache clear failed: %s", e)
        facade._query_cache.clear()

    def cleanup_query_cache(self) -> None:
        """主动清理过期的查询缓存条目。"""
        facade = self._facade
        now = time.time()
        expired_keys = [
            key for key, (_, cached_time) in facade._query_cache.items()
            if now - cached_time >= facade._query_cache_ttl
        ]
        for key in expired_keys:
            del facade._query_cache[key]
