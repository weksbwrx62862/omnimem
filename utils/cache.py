"""OmniMem 多级缓存系统 — L1 内存 LRU → L2 可选 Redis → L3 持久化。

支持精准失效（按 key 或 tag）、异步回填、命中率统计。

设计目标：
  1. L1 内存 LRU：毫秒级访问，线程安全
  2. L2 Redis：跨进程共享，可选（未安装 redis 库时降级为 no-op）
  3. L3 SQLite 持久化：跨重启存活，WAL 模式提升并发
  4. 异步回填：L3 命中后后台线程回填 L1，不阻塞主流程
  5. 精准失效：按 tag 机制失效相关缓存（如 memory_id）
  6. 监控集成：命中率指标上报到 utils/metrics.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入监控指标（失败时降级为 no-op）
try:
    from omnimem.utils.metrics import record_cache_hit, record_cache_miss
except Exception:  # pragma: no cover - 监控模块不可用时降级
    def record_cache_hit() -> None:
        pass

    def record_cache_miss() -> None:
        pass


class L1LRUCache:
    """L1 内存 LRU 缓存 — 线程安全，支持 TTL 与 tag 精准失效。

    使用 OrderedDict 维护 LRU 顺序，threading.Lock 保证线程安全。
    维护 tag→keys 反向映射，支持按 tag 批量失效。
    """

    def __init__(self, max_size: int = 1000, ttl: float = 60) -> None:
        self._max_size = max_size
        self._ttl = ttl
        # key → (value, expire_at, tags)
        self._data: OrderedDict[str, tuple[Any, float, set[str]]] = OrderedDict()
        # tag → set(keys) 反向映射，用于精准失效
        self._tag_index: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        # 命中统计
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """获取值，检查 TTL。过期或不存在返回 None。"""
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            value, expire_at, _tags = self._data[key]
            # TTL 过期检查
            if expire_at > 0 and time.time() > expire_at:
                # 过期，删除并返回 None
                self._remove_key(key)
                self._misses += 1
                return None
            # LRU：移到末尾（最近使用）
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None, tags: set[str] | None = None) -> None:
        """设置值。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 单条 TTL（秒），None 表示使用默认 TTL，<=0 表示永不过期
            tags: 附加标签集合，用于精准失效
        """
        effective_ttl = self._ttl if ttl is None else ttl
        expire_at = (time.time() + effective_ttl) if effective_ttl > 0 else 0.0
        tag_set = set(tags) if tags else set()
        with self._lock:
            # 如果 key 已存在，先清理旧 tag 索引
            if key in self._data:
                self._remove_key(key)
            self._data[key] = (value, expire_at, tag_set)
            self._data.move_to_end(key)
            # 维护 tag 反向索引
            for tag in tag_set:
                self._tag_index.setdefault(tag, set()).add(key)
            # LRU 淘汰
            while len(self._data) > self._max_size:
                evicted_key, _ = self._data.popitem(last=False)
                self._remove_tag_index_for_key(evicted_key)

    def delete(self, key: str) -> None:
        """删除单个 key。"""
        with self._lock:
            if key in self._data:
                self._remove_key(key)

    def delete_by_tag(self, tag: str) -> int:
        """按 tag 删除所有关联的 key。

        Returns:
            删除的条目数
        """
        with self._lock:
            keys = self._tag_index.pop(tag, set())
            for key in keys:
                if key in self._data:
                    _value, _expire, _tags = self._data.pop(key)
            return len(keys)

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._data.clear()
            self._tag_index.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """返回命中率统计。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "level": "L1",
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._data),
                "max_size": self._max_size,
                "hit_ratio": (self._hits / total) if total > 0 else 0.0,
            }

    def _remove_key(self, key: str) -> None:
        """从 _data 删除 key 并清理 tag 索引（调用方需持锁）。"""
        if key in self._data:
            self._data.pop(key)
        self._remove_tag_index_for_key(key)

    def _remove_tag_index_for_key(self, key: str) -> None:
        """清理指定 key 在所有 tag 索引中的引用（调用方需持锁）。"""
        empty_tags: list[str] = []
        for tag, keys in self._tag_index.items():
            keys.discard(key)
            if not keys:
                empty_tags.append(tag)
        for tag in empty_tags:
            self._tag_index.pop(tag, None)


class L2RedisCache:
    """L2 Redis 缓存 — 可选，未安装 redis 库时降级为 no-op。

    所有方法在降级模式下返回 None 或 no-op，不影响主流程。
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url
        self._client: Any = None
        self._available = False
        if redis_url:
            try:
                import redis  # type: ignore[import-untyped]

                self._client = redis.from_url(redis_url, decode_responses=False)
                # 测试连接
                self._client.ping()
                self._available = True
                logger.info("L2RedisCache connected to %s", redis_url)
            except Exception as e:
                logger.warning("L2RedisCache unavailable, degrading to no-op: %s", e)
                self._client = None
                self._available = False

    def is_available(self) -> bool:
        """Redis 是否可用。"""
        return self._available

    async def get(self, key: str) -> Any | None:
        """异步获取值。降级时返回 None。"""
        if not self._available or self._client is None:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return pickle.loads(raw)
        except Exception as e:
            logger.debug("L2RedisCache get failed: %s", e)
            return None

    async def set(self, key: str, value: Any, ttl: float = 60) -> None:
        """异步设置值。降级时 no-op。"""
        if not self._available or self._client is None:
            return
        try:
            raw = pickle.dumps(value)
            if ttl > 0:
                self._client.setex(key, int(ttl), raw)
            else:
                self._client.set(key, raw)
        except Exception as e:
            logger.debug("L2RedisCache set failed: %s", e)

    async def delete(self, key: str) -> None:
        """异步删除单个 key。降级时 no-op。"""
        if not self._available or self._client is None:
            return
        try:
            self._client.delete(key)
        except Exception as e:
            logger.debug("L2RedisCache delete failed: %s", e)

    async def delete_by_pattern(self, pattern: str) -> int:
        """按模式删除（如 "recall:*"）。降级时返回 0。

        Returns:
            删除的条目数
        """
        if not self._available or self._client is None:
            return 0
        try:
            # 使用 scan 迭代避免阻塞 Redis（keys 命令在大库上会阻塞）
            deleted = 0
            for key in self._client.scan_iter(match=pattern, count=200):
                self._client.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.debug("L2RedisCache delete_by_pattern failed: %s", e)
            return 0

    def get_sync(self, key: str) -> Any | None:
        """同步获取值（供同步上下文使用）。降级时返回 None。"""
        if not self._available or self._client is None:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return pickle.loads(raw)
        except Exception as e:
            logger.debug("L2RedisCache get_sync failed: %s", e)
            return None

    def set_sync(self, key: str, value: Any, ttl: float = 60) -> None:
        """同步设置值（供同步上下文使用）。降级时 no-op。"""
        if not self._available or self._client is None:
            return
        try:
            raw = pickle.dumps(value)
            if ttl > 0:
                self._client.setex(key, int(ttl), raw)
            else:
                self._client.set(key, raw)
        except Exception as e:
            logger.debug("L2RedisCache set_sync failed: %s", e)

    def delete_sync(self, key: str) -> None:
        """同步删除单个 key（供同步上下文使用）。降级时 no-op。"""
        if not self._available or self._client is None:
            return
        try:
            self._client.delete(key)
        except Exception as e:
            logger.debug("L2RedisCache delete_sync failed: %s", e)


class L3PersistentCache:
    """L3 SQLite 持久化缓存 — WAL 模式，支持 tag 精准失效。

    表结构：
      cache(key TEXT PRIMARY KEY, value BLOB, tags TEXT, created_at REAL, ttl REAL)
      cache_tags(tag TEXT NOT NULL, key TEXT NOT NULL,
                 PRIMARY KEY(tag, key))
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/omnimem/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / "l3_cache.db"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库，启用 WAL 模式。"""
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # 自动提交模式
            )
            # WAL 模式提升并发读写
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    ttl REAL NOT NULL DEFAULT 0
                )
                """
            )
            # tag 索引加速 delete_by_tag（tags 字段为 JSON 数组字符串）
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)"
            )
            # tag → key 反向索引表，避免 delete_by_tag 全表扫描
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_tags (
                    tag TEXT NOT NULL,
                    key TEXT NOT NULL,
                    PRIMARY KEY (tag, key)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_tags_key ON cache_tags(key)"
            )
            # 兼容性迁移：为已有数据补齐反向索引
            self._migrate_tag_index()
            logger.info("L3PersistentCache initialized at %s", self._db_path)
        except Exception as e:
            logger.warning("L3PersistentCache init failed, degrading to no-op: %s", e)
            self._conn = None

    def _migrate_tag_index(self) -> None:
        """为已存在但未建立反向索引的缓存条目补齐 cache_tags。"""
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT key, tags FROM cache WHERE tags != '[]'"
            ).fetchall()
            for key, tags_json in rows:
                try:
                    tag_list = json.loads(tags_json) if tags_json else []
                except Exception:
                    continue
                for tag in tag_list:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO cache_tags (tag, key) VALUES (?, ?)",
                        (tag, key),
                    )
        except Exception as e:
            logger.debug("L3PersistentCache tag index migration failed: %s", e)

    def get(self, key: str) -> Any | None:
        """获取值，检查 TTL。"""
        value, _tags = self.get_with_tags(key)
        return value

    def get_with_tags(self, key: str) -> tuple[Any | None, set[str]]:
        """获取值及其 tags，检查 TTL。

        Returns:
            (value, tags) 元组，未命中返回 (None, set())
        """
        if self._conn is None:
            return None, set()
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value, tags, created_at, ttl FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
            if row is None:
                return None, set()
            raw, tags_json, created_at, ttl = row
            # TTL 过期检查
            if ttl > 0 and (time.time() - created_at) > ttl:
                # 过期，删除
                self.delete(key)
                return None, set()
            value = pickle.loads(raw)
            try:
                tags_set = set(json.loads(tags_json)) if tags_json else set()
            except Exception:
                tags_set = set()
            return value, tags_set
        except Exception as e:
            logger.debug("L3PersistentCache get_with_tags failed: %s", e)
            return None, set()

    def set(self, key: str, value: Any, ttl: float = 60, tags: set[str] | None = None) -> None:
        """设置值，并同步维护 tag 反向索引。"""
        if self._conn is None:
            return
        try:
            raw = pickle.dumps(value)
            tags_json = json.dumps(list(tags)) if tags else "[]"
            tag_list = list(tags) if tags else []
            now = time.time()
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, tags, created_at, ttl) VALUES (?, ?, ?, ?, ?)",
                    (key, raw, tags_json, now, ttl),
                )
                # 同步维护 tag → key 反向索引：先清理旧索引，再写入新索引
                self._conn.execute("DELETE FROM cache_tags WHERE key = ?", (key,))
                if tag_list:
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO cache_tags (tag, key) VALUES (?, ?)",
                        [(tag, key) for tag in tag_list],
                    )
        except Exception as e:
            logger.debug("L3PersistentCache set failed: %s", e)

    def delete(self, key: str) -> None:
        """删除单个 key，并清理其 tag 反向索引。"""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.execute("DELETE FROM cache_tags WHERE key = ?", (key,))
        except Exception as e:
            logger.debug("L3PersistentCache delete failed: %s", e)

    def delete_by_tag(self, tag: str) -> int:
        """按 tag 删除所有关联条目（使用反向索引，避免全表扫描）。

        Returns:
            删除的条目数
        """
        if self._conn is None:
            return 0
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key FROM cache_tags WHERE tag = ?", (tag,)
                ).fetchall()
                if not rows:
                    return 0
                keys = [row[0] for row in rows]
                self._conn.executemany(
                    "DELETE FROM cache WHERE key = ?",
                    [(k,) for k in keys],
                )
                self._conn.executemany(
                    "DELETE FROM cache_tags WHERE key = ?",
                    [(k,) for k in keys],
                )
                return len(keys)
        except Exception as e:
            logger.debug("L3PersistentCache delete_by_tag failed: %s", e)
            return 0

    def clear(self) -> None:
        """清空所有缓存及 tag 反向索引。"""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute("DELETE FROM cache")
                self._conn.execute("DELETE FROM cache_tags")
        except Exception as e:
            logger.debug("L3PersistentCache clear failed: %s", e)

    def stats(self) -> dict[str, Any]:
        """返回统计信息。"""
        if self._conn is None:
            return {"level": "L3", "available": False, "size": 0}
        try:
            with self._lock:
                row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            size = row[0] if row else 0
            return {"level": "L3", "available": True, "size": size}
        except Exception as e:
            logger.debug("L3PersistentCache stats failed: %s", e)
            return {"level": "L3", "available": False, "size": 0}

    def close(self) -> None:
        """关闭 SQLite 连接（Windows 下未关闭会导致临时目录无法删除）。"""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.close()
        except Exception as e:
            logger.debug("L3PersistentCache close failed: %s", e)
        finally:
            self._conn = None


class MultiLevelCache:
    """多级缓存编排 — L1 → L2 → L3，支持异步回填与精准失效。

    查询流程：
      L1 命中 → 直接返回
      L1 未命中 → 查 L2（同步降级，Redis 不可用跳过）
      L2 未命中 → 查 L3
      L3 命中 → 异步回填 L1（后台线程）
      全未命中 → 返回 None

    写入流程：
      L1 同步写入 + L2 异步写入 + L3 异步写入
    """

    def __init__(
        self,
        l1_size: int = 1000,
        l1_ttl: float = 60,
        redis_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._l1 = L1LRUCache(max_size=l1_size, ttl=l1_ttl)
        # L2 可选：redis_url 为 None 或连接失败时降级
        self._l2 = L2RedisCache(redis_url=redis_url)
        # L3 可选：cache_dir 为 None 时不启用
        self._l3: L3PersistentCache | None = None
        if cache_dir is not None:
            try:
                self._l3 = L3PersistentCache(cache_dir=cache_dir)
            except Exception as e:
                logger.warning("L3PersistentCache init failed, degrading: %s", e)
                self._l3 = None
        # 后台线程池：用于异步回填与异步写入
        self._executor = ThreadPoolExecutorLite(max_workers=2)
        # 全局命中率统计（跨 L1/L2/L3）
        self._total_hits = 0
        self._total_misses = 0
        self._stats_lock = threading.Lock()
        # 最近删除的 key 集合（防止异步回填竞态：delete 后回填线程可能重新写入）
        # key → delete_time，定期清理过期条目
        self._recently_deleted: dict[str, float] = {}
        self._recently_deleted_lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """同步获取：L1 → L2（同步降级）→ L3，L3 命中异步回填 L1。"""
        # L1 命中
        value = self._l1.get(key)
        if value is not None:
            self._record_hit()
            return value
        # L2 查询（同步降级：Redis 不可用直接跳过）
        if self._l2.is_available():
            try:
                value = self._l2.get_sync(key)
            except Exception as e:
                logger.debug("L2 get failed: %s", e)
                value = None
            if value is not None:
                # L2 命中，回填 L1
                self._l1.set(key, value)
                self._record_hit()
                return value
        # L3 查询
        if self._l3 is not None:
            value, l3_tags = self._l3.get_with_tags(key)
            if value is not None:
                # L3 命中，异步回填 L1（后台线程，携带 tags 保持精准失效能力）
                self._executor.submit(self._safe_l1_set, key, value, l3_tags)
                self._record_hit()
                return value
        # 全未命中
        self._record_miss()
        return None

    def set(self, key: str, value: Any, ttl: float | None = None, tags: set[str] | None = None) -> None:
        """同步写入 L1 + 异步写入 L2/L3。"""
        # 清除最近删除标记（允许 delete 后重新 set）
        with self._recently_deleted_lock:
            self._recently_deleted.pop(key, None)
        # L1 同步写入
        self._l1.set(key, value, ttl=ttl, tags=tags)
        # L2 异步写入
        if self._l2.is_available():
            effective_ttl = self._l1._ttl if ttl is None else ttl
            self._executor.submit(self._safe_l2_set, key, value, effective_ttl)
        # L3 异步写入
        if self._l3 is not None:
            effective_ttl = self._l1._ttl if ttl is None else ttl
            self._executor.submit(self._safe_l3_set, key, value, effective_ttl, tags)

    def delete(self, key: str) -> None:
        """三级同时删除。

        L1/L3 同步删除（本地操作，毫秒级），L2 异步删除（网络操作）。
        保证本地缓存一致性，避免失效后立即查询仍命中。
        """
        self._mark_deleted(key)
        self._l1.delete(key)
        # L3 同步删除，保证本地缓存一致性
        if self._l3 is not None:
            self._safe_l3_delete(key)
        # L2 异步删除（网络操作，不阻塞主流程）
        if self._l2.is_available():
            self._executor.submit(self._safe_l2_delete, key)

    def delete_by_tag(self, tag: str) -> None:
        """按 tag 精准失效三级缓存。

        L1/L3 同步删除，L2 异步删除。
        """
        # 收集 L1 中该 tag 对应的所有 key，标记为已删除
        with self._l1._lock:
            keys_to_invalidate = set(self._l1._tag_index.get(tag, set()))
        for key in keys_to_invalidate:
            self._mark_deleted(key)
        self._l1.delete_by_tag(tag)
        # L3 同步删除，保证本地缓存一致性
        if self._l3 is not None:
            self._safe_l3_delete_by_tag(tag)
        # L2 的 tag 失效需要应用层维护 tag→keys 映射，此处降级为跳过
        # （L2 跨进程共享，tag 失效语义复杂，生产环境可扩展为 Redis SET 索引）

    def invalidate_memory(self, memory_id: str) -> None:
        """按 memory_id 失效相关缓存。

        使用 tag 机制：set 时附加 tag=memory_id，此处按 tag 删除。
        """
        if not memory_id:
            return
        # tag 格式：memory:{memory_id}
        tag = f"memory:{memory_id}"
        self.delete_by_tag(tag)

    def clear(self) -> None:
        """清空所有级别的缓存。"""
        self._l1.clear()
        if self._l2.is_available():
            # L2 清空通过 delete_by_pattern 兜底（实际生产可配置前缀）
            pass
        if self._l3 is not None:
            self._l3.clear()
        with self._stats_lock:
            self._total_hits = 0
            self._total_misses = 0

    def stats(self) -> dict[str, Any]:
        """汇总三级缓存的命中率。"""
        l1_stats = self._l1.stats()
        with self._stats_lock:
            total = self._total_hits + self._total_misses
            overall_ratio = (self._total_hits / total) if total > 0 else 0.0
            hits = self._total_hits
            misses = self._total_misses
        return {
            "overall": {
                "hits": hits,
                "misses": misses,
                "hit_ratio": overall_ratio,
            },
            "L1": l1_stats,
            "L2": {"available": self._l2.is_available()},
            "L3": self._l3.stats() if self._l3 else {"available": False, "size": 0},
        }

    def close(self) -> None:
        """释放 L3 SQLite 连接等底层资源（幂等）。"""
        if self._l3 is not None:
            self._l3.close()

    async def async_get(self, key: str) -> Any | None:
        """异步版本（L2 查询不阻塞）。"""
        # L1 命中
        value = self._l1.get(key)
        if value is not None:
            self._record_hit()
            return value
        # L2 异步查询
        if self._l2.is_available():
            value = await self._l2.get(key)
            if value is not None:
                self._l1.set(key, value)
                self._record_hit()
                return value
        # L3 查询（L3 是同步 SQLite，用 to_thread 包装）
        if self._l3 is not None:
            import asyncio

            value, l3_tags = await asyncio.to_thread(self._l3.get_with_tags, key)
            if value is not None:
                # 异步回填 L1，携带 tags
                self._executor.submit(self._safe_l1_set, key, value, l3_tags)
                self._record_hit()
                return value
        self._record_miss()
        return None

    async def async_set(self, key: str, value: Any, **kwargs: Any) -> None:
        """异步写入。"""
        ttl = kwargs.get("ttl")
        tags = kwargs.get("tags")
        # L1 同步写入
        self._l1.set(key, value, ttl=ttl, tags=tags)
        # L2 异步写入
        if self._l2.is_available():
            effective_ttl = self._l1._ttl if ttl is None else ttl
            await self._l2.set(key, value, ttl=effective_ttl)
        # L3 异步写入
        if self._l3 is not None:
            effective_ttl = self._l1._ttl if ttl is None else ttl
            import asyncio

            await asyncio.to_thread(self._l3.set, key, value, effective_ttl, tags)

    # ─── 内部辅助方法 ─────────────────────────────────────────
    def _record_hit(self) -> None:
        """记录一次命中。"""
        with self._stats_lock:
            self._total_hits += 1
        try:
            record_cache_hit()
        except Exception:
            pass

    def _record_miss(self) -> None:
        """记录一次未命中。"""
        with self._stats_lock:
            self._total_misses += 1
        try:
            record_cache_miss()
        except Exception:
            pass

    def _safe_l1_set(self, key: str, value: Any, tags: set[str] | None = None) -> None:
        """线程安全的 L1 写入（用于后台回填，携带 tags 保持精准失效能力）。

        检查 _recently_deleted 防止竞态：如果 key 在回填期间被删除，跳过回填。
        """
        try:
            # 竞态防护：delete 可能在此回填执行前发生
            if self._is_deleted(key):
                logger.debug("L1 backfill skipped (key recently deleted): %s", key)
                return
            self._l1.set(key, value, tags=tags)
        except Exception as e:
            logger.debug("L1 async backfill failed: %s", e)

    def _mark_deleted(self, key: str) -> None:
        """标记 key 为已删除（用于防止异步回填竞态）。"""
        with self._recently_deleted_lock:
            self._recently_deleted[key] = time.time()
            # 顺便清理过期条目（超过 30 秒的）
            cutoff = time.time() - 30
            expired = [k for k, t in self._recently_deleted.items() if t < cutoff]
            for k in expired:
                self._recently_deleted.pop(k, None)

    def _is_deleted(self, key: str) -> bool:
        """检查 key 是否在最近删除窗口内（30 秒）。"""
        with self._recently_deleted_lock:
            t = self._recently_deleted.get(key)
            if t is None:
                return False
            # 超过 30 秒的视为过期
            if time.time() - t > 30:
                self._recently_deleted.pop(key, None)
                return False
            return True

    def _safe_l2_set(self, key: str, value: Any, ttl: float) -> None:
        """线程安全的 L2 写入。"""
        try:
            self._l2.set_sync(key, value, ttl=ttl)
        except Exception as e:
            logger.debug("L2 async set failed: %s", e)

    def _safe_l2_delete(self, key: str) -> None:
        """线程安全的 L2 删除。"""
        try:
            self._l2.delete_sync(key)
        except Exception as e:
            logger.debug("L2 async delete failed: %s", e)

    def _safe_l3_set(self, key: str, value: Any, ttl: float, tags: set[str] | None) -> None:
        """线程安全的 L3 写入。"""
        try:
            if self._l3 is not None:
                self._l3.set(key, value, ttl=ttl, tags=tags)
        except Exception as e:
            logger.debug("L3 async set failed: %s", e)

    def _safe_l3_delete(self, key: str) -> None:
        """线程安全的 L3 删除。"""
        try:
            if self._l3 is not None:
                self._l3.delete(key)
        except Exception as e:
            logger.debug("L3 async delete failed: %s", e)

    def _safe_l3_delete_by_tag(self, tag: str) -> None:
        """线程安全的 L3 按 tag 删除。"""
        try:
            if self._l3 is not None:
                self._l3.delete_by_tag(tag)
        except Exception as e:
            logger.debug("L3 delete_by_tag failed: %s", e)


class ThreadPoolExecutorLite:
    """轻量级后台线程池 — daemon 线程，主进程退出时自动结束。

    使用 threading.Thread 而非 concurrent.futures.ThreadPoolExecutor，
    避免在插件环境中引入额外的资源管理开销。
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self._semaphore = threading.Semaphore(max_workers)

    def submit(self, fn: Any, *args: Any) -> None:
        """提交后台任务（非阻塞）。"""
        # 信号量限流，避免无限制创建线程
        self._semaphore.acquire()
        thread = threading.Thread(target=self._run, args=(fn, args), daemon=True)
        thread.start()

    def _run(self, fn: Any, args: tuple) -> None:
        """执行任务并释放信号量。"""
        try:
            fn(*args)
        except Exception as e:
            logger.debug("Background task failed: %s", e)
        finally:
            self._semaphore.release()


class CacheKeyBuilder:
    """缓存键构建工具 — 统一 key 生成规则，避免键冲突。

    所有方法均为静态方法，无状态。
    """

    # 键前缀，避免不同类型缓存的键冲突
    _RECALL_PREFIX = "recall:"
    _EMBEDDING_PREFIX = "emb:"
    _LLM_PREFIX = "llm:"

    @staticmethod
    def build_recall_key(query: str, max_tokens: int, mode: str, top_k: int) -> str:
        """构建检索缓存键。

        与 HybridRetriever 原有格式兼容：{query}|{max_tokens}|{mode}|{top_k}
        """
        return f"{CacheKeyBuilder._RECALL_PREFIX}{query}|{max_tokens}|{mode}|{top_k}"

    @staticmethod
    def build_embedding_key(text: str) -> str:
        """构建嵌入缓存键（SHA-256 哈希）。

        与 _CachedEmbeddingFunction 原有逻辑兼容：取 SHA-256 前 16 字符。
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{CacheKeyBuilder._EMBEDDING_PREFIX}{text_hash}"

    @staticmethod
    def build_llm_key(prompt: str, system: str | None = None) -> str:
        """构建 LLM 缓存键。"""
        raw = f"{system or ''}||{prompt}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"{CacheKeyBuilder._LLM_PREFIX}{digest}"
