"""GovernanceStore — 治理统一 SQLite 存储（单 DB + 读写分离）。

替代分散的 forgetting.db / reflect.db / kv_cache.db / consolidation.db，
消除 _FORGETTING_DB_LOCK、类级共享连接、引用计数等防御代码。

设计：
  - 单 SQLite 文件 governance.db，WAL 模式
  - 读写连接分离：写连接持 _write_lock 串行化，读连接无锁并发
  - 批量提交：攒够 _BATCH_THRESHOLD 条写入才 commit
  - 一个 GovernanceStore 实例服务所有治理模块
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 10000
_BATCH_THRESHOLD = 20


class GovernanceStore:
    """治理统一存储：单 DB + 读写分离连接。

    使用方式：
      store = GovernanceStore(Path("/data/governance"))
      store.execute("INSERT INTO forgetting_state ...", (...))
      rows = store.query("SELECT * FROM forgetting_state WHERE ...")
      store.close()
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "governance.db"

        # 读写分离连接
        self._write_conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None

        # 写锁（替代 _FORGETTING_DB_LOCK 等模块级锁）
        self._write_lock = threading.RLock()
        self._pending_writes = 0
        self._closed = False

        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── 生命周期 ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """初始化数据库连接和 schema。"""
        self._write_conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, timeout=_BUSY_TIMEOUT_MS / 1000
        )
        self._write_conn.execute("PRAGMA journal_mode=WAL")
        self._write_conn.execute("PRAGMA synchronous=NORMAL")
        self._write_conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        self._write_conn.row_factory = sqlite3.Row

        # 只读连接（WAL 模式下读不阻塞写）
        self._read_conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        self._read_conn.row_factory = sqlite3.Row

        self._create_all_tables()

    def _create_all_tables(self) -> None:
        """创建所有治理相关表（幂等）。"""
        assert self._write_conn is not None

        migrator = SchemaMigrator(self._write_conn)

        # ── forgetting_state ──
        migrator.migrate(
            table_name="forgetting_state",
            create_sql="""
                CREATE TABLE IF NOT EXISTS forgetting_state (
                    memory_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL DEFAULT 'active',
                    last_accessed TEXT,
                    created_at TEXT,
                    archive_count INTEGER DEFAULT 0,
                    recall_count INTEGER DEFAULT 0
                )
            """,
            migrations=[],
        )
        _ensure_columns(self._write_conn, "forgetting_state", {
            "recall_count": "INTEGER DEFAULT 0",
            "heat": "TEXT DEFAULT 'neutral'",
            "heat_updated_at": "TEXT",
            "upgraded_to_wiki": "INTEGER DEFAULT 0",
            "wiki_page_path": "TEXT",
            "memory_type": "TEXT DEFAULT ''",
        })
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_stage ON forgetting_state(stage)"
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_heat ON forgetting_state(heat)"
        )

        # ── access_log ──
        migrator.migrate(
            table_name="access_log",
            create_sql="""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    accessed_at TEXT NOT NULL
                )
            """,
            migrations=[],
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_al_mid ON access_log(memory_id)"
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_al_time ON access_log(accessed_at)"
        )

        # ── pipeline_meta ──
        migrator.migrate(
            table_name="pipeline_meta",
            create_sql="""
                CREATE TABLE IF NOT EXISTS pipeline_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """,
            migrations=[],
        )

        # ── reflections ──
        migrator.migrate(
            table_name="reflections",
            create_sql="""
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    observation TEXT,
                    mental_model TEXT,
                    confidence REAL,
                    disposition TEXT,
                    source_ids TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ref_query ON reflections(query)"
        )

        # ── kv_cache_entries ──
        migrator.migrate(
            table_name="kv_cache_entries",
            create_sql="""
                CREATE TABLE IF NOT EXISTS kv_cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    access_count INTEGER DEFAULT 0,
                    preloaded_at TEXT,
                    last_accessed TEXT,
                    source_memory_ids TEXT
                )
            """,
            migrations=[],
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kv_access ON kv_cache_entries(access_count DESC)"
        )

        # ── consolidation_items ──
        migrator.migrate(
            table_name="consolidation_items",
            create_sql="""
                CREATE TABLE IF NOT EXISTS consolidation_items (
                    item_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_ids TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ci_stage ON consolidation_items(stage)"
        )

        self._write_conn.commit()

    # ── 写操作（持锁串行化，替代分散的模块级锁） ─────────────────────

    def execute(self, sql: str, params: tuple | None = None) -> None:
        """执行写操作（INSERT/UPDATE/DELETE），持锁串行化。

        批量提交：攒够 _BATCH_THRESHOLD 条写入才 commit。
        调用方无需持任何锁——GovernanceStore 内部管理。
        """
        with self._write_lock:
            assert self._write_conn is not None
            self._write_conn.execute(sql, params or ())
            self._pending_writes += 1
            if self._pending_writes >= _BATCH_THRESHOLD:
                self._commit()

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """批量执行同一 SQL（线程安全）。"""
        with self._write_lock:
            assert self._write_conn is not None
            self._write_conn.executemany(sql, params_list)
            self._pending_writes += len(params_list)
            if self._pending_writes >= _BATCH_THRESHOLD:
                self._commit()

    def execute_raw(self, sql: str, params: tuple | None = None) -> sqlite3.Cursor:
        """执行写操作并返回 cursor（调用方已在 _write_lock 保护内使用）。"""
        assert self._write_conn is not None
        result = self._write_conn.execute(sql, params or ())
        self._pending_writes += 1
        if self._pending_writes >= _BATCH_THRESHOLD:
            self._commit()
        return result

    def _commit(self) -> None:
        """提交写连接上的待写入数据。"""
        if self._write_conn and self._pending_writes > 0:
            self._write_conn.commit()
            self._pending_writes = 0

    def commit(self) -> None:
        """强制提交（持锁）。"""
        with self._write_lock:
            self._commit()

    def flush(self) -> None:
        """提交所有待写入（等同于 commit）。"""
        self.commit()

    # ── 读操作（只读连接，无锁并发） ──────────────────────────────────

    def query(self, sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
        """执行只读查询（无锁）。"""
        if self._closed:
            return []
        try:
            rows = self._read_conn.execute(sql, params or ()).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("GovernanceStore 查询失败: %s", e)
            return []

    def query_one(self, sql: str, params: tuple | None = None) -> dict[str, Any] | None:
        """执行只读查询，返回单行。"""
        if self._closed:
            return None
        try:
            row = self._read_conn.execute(sql, params or ()).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error("GovernanceStore 查询失败: %s", e)
            return None

    # ── 连接访问（供复杂事务场景） ────────────────────────────────

    @property
    def write_lock(self) -> threading.RLock:
        """获取写锁，供需要跨多个 execute 调用的原子事务场景。"""
        return self._write_lock

    def get_write_conn(self) -> sqlite3.Connection:
        """获取写连接（调用方已在 _write_lock 保护内使用）。"""
        assert self._write_conn is not None
        return self._write_conn

    def get_read_conn(self) -> sqlite3.Connection:
        """获取只读连接。"""
        assert self._read_conn is not None
        return self._read_conn

    # ── 关闭 ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """关闭存储。"""
        if self._closed:
            return
        with self._write_lock:
            self._commit()
            self._closed = True
            if self._read_conn:
                self._read_conn.close()
                self._read_conn = None
            if self._write_conn:
                self._write_conn.close()
                self._write_conn = None
        logger.info("GovernanceStore 已关闭")


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """确保表中存在指定列（幂等补全）。"""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col_name, col_def in columns.items():
        if col_name not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                logger.info("GovernanceStore: 添加列 %s.%s", table, col_name)
            except sqlite3.OperationalError:
                pass
