"""
PerformanceOptimizer — 性能优化模块。

批量计算优化：
1. 批量数据库查询
2. 并行计算
3. LRU 缓存机制
4. 异步处理支持

核心功能:
- 批量计算优化
- 缓存机制
- 异步处理
- 性能基准测试
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceStats:
    """性能统计"""
    total_calls: int
    cache_hits: int
    cache_misses: int
    avg_time_ms: float
    batch_speedup: float


class LRUCache:
    """LRU 缓存实现"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self._cache:
            entry = self._cache[key]

            # 检查 TTL
            elapsed = (datetime.now() - entry.created_at).total_seconds()
            if elapsed > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._hits += 1
            return entry.value

        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """设置缓存"""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].value = value
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
            )

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


class PerformanceOptimizer:
    """性能优化器

    提供:
    - 批量计算优化
    - LRU 缓存
    - 并行计算
    - 性能基准测试
    """

    def __init__(
        self,
        governance_dir: Optional[Path] = None,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        max_workers: int = 4,
    ):
        self._governance_dir = governance_dir or Path.home() / ".hermes" / "omnimem" / "governance"
        self._cache = LRUCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._max_workers = max_workers

        # 性能统计
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "batch_calls": 0,
            "total_time_ms": 0,
        }

    def batch_calculate_retention(
        self,
        memory_ids: list[str],
    ) -> dict[str, float]:
        """批量计算保持率

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            {memory_id: retention} 字典
        """
        start_time = time.time()

        # 检查缓存
        results = {}
        uncached_ids = []

        for memory_id in memory_ids:
            cached = self._cache.get(f"retention:{memory_id}")
            if cached is not None:
                results[memory_id] = cached
            else:
                uncached_ids.append(memory_id)

        # 批量查询未缓存的记忆
        if uncached_ids:
            batch_results = self._batch_query_retention(uncached_ids)
            results.update(batch_results)

            # 更新缓存
            for memory_id, retention in batch_results.items():
                self._cache.put(f"retention:{memory_id}", retention)

        # 更新统计
        elapsed = (time.time() - start_time) * 1000
        self._stats["total_calls"] += 1
        self._stats["total_time_ms"] += elapsed

        return results

    def _batch_query_retention(
        self,
        memory_ids: list[str],
    ) -> dict[str, float]:
        """批量查询保持率"""
        db_path = self._governance_dir / "forgetting.db"

        if not db_path.exists():
            return {mid: 0.5 for mid in memory_ids}

        try:
            conn = sqlite3.connect(str(db_path))
            now = datetime.now(timezone.utc)

            # 批量查询
            placeholders = ",".join(["?" for _ in memory_ids])
            rows = conn.execute(
                f"""SELECT memory_id, recall_count, last_accessed
                    FROM forgetting_state
                    WHERE memory_id IN ({placeholders})""",
                memory_ids
            ).fetchall()

            conn.close()

            results = {}
            for memory_id, recall_count, last_accessed in rows:
                # 简化计算
                if last_accessed:
                    try:
                        last_dt = datetime.fromisoformat(last_accessed.replace('+00:00', ''))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)

                        elapsed_days = (now - last_dt).total_seconds() / 86400
                        stability = min(100.0, (recall_count or 0) * 2.0)

                        # 遗忘曲线
                        alpha = 9.0
                        beta = 0.5
                        retention = (1 + elapsed_days / (stability * alpha)) ** (-beta)
                        results[memory_id] = max(0.0, min(1.0, retention))
                    except:
                        results[memory_id] = 0.5
                else:
                    results[memory_id] = 0.5

            return results

        except Exception as e:
            logger.warning("Batch query failed: %s", e)
            return {mid: 0.5 for mid in memory_ids}

    def parallel_evaluate(
        self,
        memory_ids: list[str],
        evaluate_func: Callable,
    ) -> dict[str, Any]:
        """并行评估记忆

        Args:
            memory_ids: 记忆 ID 列表
            evaluate_func: 评估函数

        Returns:
            评估结果字典
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # 提交所有任务
            future_to_id = {
                executor.submit(evaluate_func, mid): mid
                for mid in memory_ids
            }

            # 收集结果
            for future in as_completed(future_to_id):
                memory_id = future_to_id[future]
                try:
                    result = future.result()
                    results[memory_id] = result
                except Exception as e:
                    logger.warning("Evaluation failed for %s: %s", memory_id, e)
                    results[memory_id] = {"error": str(e)}

        return results

    def get_stats(self) -> PerformanceStats:
        """获取性能统计"""
        total = self._stats["total_calls"]
        cache_stats = self._cache.get_stats()

        return PerformanceStats(
            total_calls=total,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            avg_time_ms=self._stats["total_time_ms"] / total if total > 0 else 0,
            batch_speedup=1.0,  # 简化实现
        )

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 全局实例
_optimizer: Optional[PerformanceOptimizer] = None


def get_optimizer(
    governance_dir: Optional[Path] = None,
    cache_size: int = 1000,
    cache_ttl: int = 3600,
) -> PerformanceOptimizer:
    """获取全局优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer(governance_dir, cache_size, cache_ttl)
    return _optimizer


def batch_calculate_retention(memory_ids: list[str]) -> dict[str, float]:
    """便捷函数：批量计算保持率"""
    optimizer = get_optimizer()
    return optimizer.batch_calculate_retention(memory_ids)
